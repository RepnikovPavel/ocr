"""Two independent GPU clients: scheduling, ordering, cancellation and startup."""

import threading
from contextlib import nullcontext

import pytest

from demo import db, docstore
from demo.worker import DemoWorker
from test_demo_worker import StubParser, env, wait_for


URLS = ["http://gpu0:8000/v1", "http://gpu1:8000/v1"]


def pool_worker(env, monkeypatch, first, second, **kwargs):
    worker = DemoWorker(ckpt="/x", jobs_dir=env["jobs_dir"], engine="vllm",
                        vllm_urls=URLS, vllm_containers=[None, None],
                        parser_factory=lambda: first, **kwargs)
    monkeypatch.setattr(worker, "_default_parser_factory", lambda url: second)
    return worker


class GatedParser(StubParser):
    def __init__(self, blocked_pages):
        super().__init__()
        self.blocked_pages = blocked_pages
        self.entered = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()

    def _parse_single_image(self, *args, page_idx=0, **kwargs):
        from dots_mocr.utils.generation_stats import GenerationStats

        if page_idx in self.blocked_pages:
            self.generation_listener(GenerationStats().start())
            self.entered.set()
            while not self.release.wait(0.01):
                if self.abort_event.is_set():
                    self.exited.set()
                    return {"page_no": page_idx, "generation": {"aborted": True}}
        result = super()._parse_single_image(*args, page_idx=page_idx, **kwargs)
        self.exited.set()
        return result


def test_free_gpu_takes_next_page_and_bundle_stays_ordered(env, monkeypatch):
    first, second = GatedParser({0}), GatedParser({1})
    docstore.init(db._DB_PATH)
    docstore.remember_document("sha", "synthetic.pdf", "pdf", 3, 123)
    worker = pool_worker(env, monkeypatch, first, second)
    worker.start()
    try:
        task_id = db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0, 1, 2], {"sha256": "sha"})
        worker.notify_new_task()
        assert first.entered.wait(20) and second.entered.wait(5), "pages must overlap"
        live = worker.status()["live_generations"]
        assert {item["page_no"] for item in live} == {0, 1}
        assert {item["backend_url"] for item in live} == set(URLS)
        second.release.set()
        assert wait_for(lambda: db.get_task(task_id)["progress"]["done"] == 2)
        assert [c["page"] for c in second.calls] == [1, 2]
        assert not first.calls, "slow page must not hold up the other GPU"
        first.release.set()
        assert wait_for(lambda: db.get_task(task_id)["status"] == "done")
        result = db.get_task(task_id)["result"]
        assert [r["page_no"] for r in result] == [0, 1, 2]
        assert [r["backend_url"] for r in result] == [URLS[0], URLS[1], URLS[1]]
        cached = docstore.find_result("sha", "prompt_ocr", [0, 1, 2])
        assert cached["pages_done"] == 3
        assert cached["markdown"].index("# page 0") < cached["markdown"].index("# page 1") < cached["markdown"].index("# page 2")
    finally:
        first.release.set()
        second.release.set()
        worker.shutdown()


def test_cancel_drains_both_workers_and_keeps_finished_pages(env, monkeypatch):
    first, second = GatedParser({2}), GatedParser({1})
    worker = pool_worker(env, monkeypatch, first, second)
    worker.start()
    try:
        task_id = db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0, 1, 2], {})
        worker.notify_new_task()
        assert first.entered.wait(5) and second.entered.wait(5)
        assert wait_for(lambda: db.get_task(task_id)["progress"]["done"] == 1)
        worker.cancel_task(task_id)
        assert wait_for(lambda: db.get_task(task_id)["status"] == "cancelled")
        assert first.exited.is_set() and second.exited.is_set()
        assert [r["page_no"] for r in db.get_task(task_id)["result"]] == [0]
        # All futures have stopped before the next request reuses the parsers.
        followup = db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0], {})
        worker.notify_new_task()
        assert wait_for(lambda: db.get_task(followup)["status"] == "done")
    finally:
        worker.shutdown()


def test_secondary_failure_aborts_other_worker_and_is_not_cached(env, monkeypatch):
    first = GatedParser({0})

    class FailingParser(StubParser):
        def _parse_single_image(self, *args, **kwargs):
            assert first.entered.wait(5)
            raise RuntimeError("GPU1 unavailable")

    worker = pool_worker(env, monkeypatch, first, FailingParser())
    worker.start()
    try:
        task_id = db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0, 1, 2], {})
        worker.notify_new_task()
        assert wait_for(lambda: db.get_task(task_id)["status"] == "error")
        assert "page 1, worker 1" in db.get_task(task_id)["error"]
        assert "GPU1 unavailable" in db.get_task(task_id)["error"]
        assert first.exited.is_set()
        assert db.get_task(task_id)["result"] == []
    finally:
        worker.shutdown()


def test_running_but_cold_container_waits_beyond_old_120s_limit(env, monkeypatch):
    worker = DemoWorker(ckpt="/x", jobs_dir=env["jobs_dir"], engine="vllm",
                        vllm_containers=["gpu0"], parser_factory=StubParser)
    calls = []
    clock = [0.0]
    monkeypatch.setattr(worker, "_vllm_container_running", lambda name: True)
    monkeypatch.setattr(worker, "_docker_api", lambda *args: calls.append(args))
    monkeypatch.setattr("demo.worker.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(worker._shutdown, "wait", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def health(*args, **kwargs):
        if clock[0] < 140:
            raise ConnectionResetError("still compiling CUDA graphs")
        return nullcontext(type("Ready", (), {"status": 200})())

    monkeypatch.setattr("urllib.request.urlopen", health)
    worker._load_model()
    assert worker.model_state == "loaded"
    assert clock[0] == 140
    assert not calls, "an already running container must not be restarted"


@pytest.mark.parametrize("start_status", [204, 500])
def test_start_failure_or_timeout_is_terminal_for_queued_clients(env, monkeypatch, start_status):
    worker = DemoWorker(ckpt="/x", jobs_dir=env["jobs_dir"], engine="vllm",
                        vllm_containers=["gpu0"], vllm_startup_timeout=0.01)
    states = iter([False, True])
    monkeypatch.setattr(worker, "_vllm_container_running", lambda name: next(states, True))
    monkeypatch.setattr(worker, "_docker_api", lambda *args: (start_status, b"driver error"))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("not ready")))
    task_id = db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0], {})
    worker._load_model()
    task = db.get_task(task_id)
    assert worker.model_state == "error" and worker.paused
    assert task["status"] == "error"
    assert "gpu0" in task["error"]
    assert ("TimeoutError" if start_status == 204 else "driver error") in task["error"]


def test_both_containers_start_together_and_unload_together(env, monkeypatch):
    worker = pool_worker(env, monkeypatch, StubParser(), StubParser())
    worker.vllm_containers = ["gpu0", "gpu1"]
    barrier = threading.Barrier(2)
    started, stopped = [], []

    def startup(name, url):
        started.append((name, url))
        barrier.wait(timeout=2)

    monkeypatch.setattr(worker, "_start_one_vllm_container", startup)
    worker._load_model()
    assert worker.model_state == "loaded"
    assert set(started) == set(zip(worker.vllm_containers, URLS))
    # Test actual sleepable clients and lifecycle, not only the container helper.
    from test_demo_worker import SleepingParser
    worker.parsers = [SleepingParser(), SleepingParser()]
    worker.parser = worker.parsers[0]
    monkeypatch.setattr(worker, "_stop_one_vllm_container", stopped.append)
    worker._unload_model()
    assert all(p.asleep for p in worker.parsers)
    assert stopped == ["gpu0", "gpu1"]
    worker._load_model()
    assert all(not p.asleep for p in worker.parsers)


def test_retry_after_partial_load_rebuilds_all_clients(env, monkeypatch):
    from test_demo_worker import SleepingParser

    first, second = SleepingParser(), SleepingParser()
    worker = pool_worker(env, monkeypatch, first, second)
    calls = []

    def flaky_secondary(url):
        calls.append(url)
        if len(calls) == 1:
            raise ConnectionError("GPU1 still starting")
        return second

    monkeypatch.setattr(worker, "_default_parser_factory", flaky_secondary)
    worker._load_model()
    assert worker.model_state == "error"
    worker.request_start()
    worker._load_model()
    assert worker.model_state == "loaded"
    assert worker.parsers == [first, second], "retry must not silently fall back to one GPU"
