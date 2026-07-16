import json
import os
import threading
import time

import pytest

from demo import db
from demo.worker import DemoWorker, default_temperature


class StubParser:
    """Writes one md artifact per page; optional per-page delay."""

    def __init__(self, delay=0.0, fail_on_page=None):
        self.delay = delay
        self.fail_on_page = fail_on_page
        self.abort_event = None
        self.max_completion_tokens = 16384
        self.device = "cpu"
        self.calls = []

    def _parse_single_image(self, origin_image, prompt_mode, save_dir, save_name,
                            source="image", page_idx=0, bbox=None,
                            fitz_preprocess=False, custom_prompt=None, temperature=None):
        if self.fail_on_page == page_idx:
            raise RuntimeError(f"boom on page {page_idx}")
        deadline = time.time() + self.delay
        while time.time() < deadline:
            if self.abort_event is not None and self.abort_event.is_set():
                break
            time.sleep(0.01)
        self.calls.append({"page": page_idx, "mode": prompt_mode,
                           "temperature": temperature, "bbox": bbox,
                           "fitz_preprocess": fitz_preprocess})
        name = f"{save_name}_page_{page_idx}" if source == "pdf" else save_name
        md_path = os.path.join(save_dir, f"{name}.md")
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(f"# page {page_idx}\n\nstub markdown")
        return {"page_no": page_idx, "md_content_path": md_path,
                "input_width": 100, "input_height": 100}


@pytest.fixture()
def env(tmp_path, synthetic_pdf):
    db.init_db(tmp_path / "demo.db")
    jobs_dir = tmp_path / "jobs"
    sid = db.get_or_create_session(None)

    import shutil
    job_id = db.create_job(sid, "synthetic.pdf", "pdf", 3)
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True)
    shutil.copy(synthetic_pdf, job_dir / "input.pdf")
    return {"sid": sid, "job_id": job_id, "jobs_dir": jobs_dir}


def make_worker(env, stub=None, autostart=True, **kwargs):
    stub = stub or StubParser()
    worker = DemoWorker(
        ckpt="/nonexistent", jobs_dir=env["jobs_dir"],
        parser_factory=lambda: stub, autostart=autostart, **kwargs,
    )
    worker.start()
    return worker, stub


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_worker_executes_pdf_task(env):
    worker, stub = make_worker(env)
    try:
        task_id = db.create_task(env["sid"], env["job_id"], "prompt_layout_all_en", [0, 2], {})
        assert wait_for(lambda: db.get_task(task_id)["status"] == "done")
        task = db.get_task(task_id)
        assert [r["page_no"] for r in task["result"]] == [0, 2]
        assert task["progress"] == {"done": 2, "total": 2}
        for page in task["result"]:
            assert os.path.isfile(page["md_content_path"])
            assert "seconds" in page
        # authors' default temperature was applied
        assert stub.calls[0]["temperature"] == default_temperature("prompt_layout_all_en")
    finally:
        worker.shutdown()


def test_worker_processes_all_pages_of_pdf(env):
    # regression: pressing "run" with the default (all pages) selection must
    # parse the WHOLE pdf, not just a couple pages
    worker, stub = make_worker(env)
    try:
        job = db.get_job(env["job_id"])
        all_pages = list(range(job["num_pages"]))  # synthetic pdf has 3 pages
        assert len(all_pages) == 3
        task_id = db.create_task(env["sid"], env["job_id"], "prompt_ocr", all_pages, {})
        assert wait_for(lambda: db.get_task(task_id)["status"] == "done")
        task = db.get_task(task_id)
        assert [r["page_no"] for r in task["result"]] == all_pages
        assert task["progress"] == {"done": 3, "total": 3}
        assert {c["page"] for c in stub.calls} == set(all_pages)
    finally:
        worker.shutdown()


def test_worker_cancel_running_preserves_finished_pages(env):
    worker, _ = make_worker(env, StubParser(delay=0.5))
    try:
        task_id = db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0, 1, 2], {})
        assert wait_for(lambda: db.get_task(task_id)["status"] == "running")
        # let page 0 finish, then cancel
        assert wait_for(lambda: db.get_task(task_id)["progress"].get("done", 0) >= 1)
        worker.cancel_task(task_id)
        assert wait_for(lambda: db.get_task(task_id)["status"] == "cancelled")
        task = db.get_task(task_id)
        assert len(task["result"]) < 3
    finally:
        worker.shutdown()


def test_worker_cancel_queued_is_instant(env):
    worker, _ = make_worker(env, StubParser(delay=0.5))
    try:
        blocker = db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0, 1, 2], {})
        queued = db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0], {})
        assert worker.cancel_task(queued) == "cancelled"
        worker.cancel_task(blocker)
        assert wait_for(lambda: db.get_task(blocker)["status"] == "cancelled")
    finally:
        worker.shutdown()


def test_worker_stop_pauses_queue_and_start_resumes(env):
    worker, _ = make_worker(env, autostart=False)
    try:
        assert wait_for(lambda: worker.model_state == "stopped", timeout=2)
        worker.request_stop()  # explicit pause: tasks must NOT auto-load
        task_id = db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0], {})
        time.sleep(0.5)
        assert db.get_task(task_id)["status"] == "queued"

        worker.request_start()
        assert wait_for(lambda: worker.model_state == "loaded")
        assert wait_for(lambda: db.get_task(task_id)["status"] == "done")

        worker.request_stop()
        assert wait_for(lambda: worker.model_state == "stopped")
        assert worker.paused
    finally:
        worker.shutdown()


def test_worker_lazy_loads_on_demand_and_unloads_after_idle(env):
    worker, _ = make_worker(env, autostart=False, idle_unload_seconds=1)
    try:
        # default state: GPU is free, model not loaded
        time.sleep(0.3)
        assert worker.model_state == "stopped"
        assert not worker.paused

        # a task arrives -> the model loads by itself
        task_id = db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0], {})
        worker.notify_new_task()
        assert wait_for(lambda: db.get_task(task_id)["status"] == "done")
        assert worker.model_state == "loaded"

        # ...and unloads after the idle timeout
        assert wait_for(lambda: worker.model_state == "stopped", timeout=5)
        assert not worker.paused  # auto-unload keeps on-demand loading enabled
    finally:
        worker.shutdown()


def test_worker_keep_loaded_prevents_idle_unload(env):
    worker, _ = make_worker(env, autostart=True, idle_unload_seconds=1, keep_loaded=True)
    try:
        assert wait_for(lambda: worker.model_state == "loaded")
        time.sleep(2.0)
        assert worker.model_state == "loaded"
        worker.set_keep_loaded(False)
        assert wait_for(lambda: worker.model_state == "stopped", timeout=5)
    finally:
        worker.shutdown()


def test_worker_new_task_resumes_paused_worker(env):
    worker, _ = make_worker(env, autostart=False)
    try:
        worker.request_stop()
        task_id = db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0], {})
        worker.notify_new_task()  # server does this on POST /api/tasks
        assert wait_for(lambda: db.get_task(task_id)["status"] == "done")
    finally:
        worker.shutdown()


def test_worker_set_device_reloads_on_new_device(env):
    loaded_on = []

    def factory():
        stub = StubParser()
        stub.device = worker.device
        loaded_on.append(worker.device)
        return stub

    worker = DemoWorker(ckpt="/x", jobs_dir=env["jobs_dir"], device="cuda:0",
                        parser_factory=factory, autostart=True, keep_loaded=True)
    worker.start()
    try:
        assert wait_for(lambda: worker.model_state == "loaded")
        assert worker.status()["configured_device"] == "cuda:0"
        worker.set_device("cuda:1")
        # wait for the actual reload (factory called again), not just the flag flip
        assert wait_for(lambda: len(loaded_on) >= 2)
        assert wait_for(lambda: worker.model_state == "loaded")
        assert loaded_on[-1] == "cuda:1"
        assert worker.status()["configured_device"] == "cuda:1"
        assert worker.status()["device"] == "cuda:1"
        # setting the same device is a no-op (no extra reload)
        n = len(loaded_on)
        worker.set_device("cuda:1")
        time.sleep(0.4)
        assert len(loaded_on) == n
    finally:
        worker.shutdown()


def test_worker_load_error_pauses_instead_of_retry_loop(env):
    calls = {"n": 0}

    def failing_factory():
        calls["n"] += 1
        raise RuntimeError("no checkpoint")

    worker = DemoWorker(ckpt="/nonexistent", jobs_dir=env["jobs_dir"],
                        parser_factory=failing_factory, autostart=False)
    worker.start()
    try:
        db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0], {})
        worker.notify_new_task()
        assert wait_for(lambda: worker.model_state == "error")
        assert wait_for(lambda: worker.paused, timeout=2)
        attempts = calls["n"]
        time.sleep(1.5)
        assert calls["n"] == attempts  # no retry hammering
    finally:
        worker.shutdown()


def test_worker_marks_failed_task_as_error(env):
    worker, _ = make_worker(env, StubParser(fail_on_page=1))
    try:
        task_id = db.create_task(env["sid"], env["job_id"], "prompt_ocr", [0, 1], {})
        assert wait_for(lambda: db.get_task(task_id)["status"] == "error")
        task = db.get_task(task_id)
        assert "boom" in task["error"]
        # page 0 finished before the failure
        assert [r["page_no"] for r in task["result"]] == [0]
    finally:
        worker.shutdown()


def test_worker_image_job_uses_fitz_preprocess_map(env, tmp_path, synthetic_page_image):
    stub = StubParser()
    worker, _ = make_worker(env, stub)
    try:
        image_job = db.create_job(env["sid"], "img.png", "image", 1)
        job_dir = env["jobs_dir"] / image_job
        job_dir.mkdir(parents=True)
        synthetic_page_image.save(job_dir / "input.png")
        task_id = db.create_task(env["sid"], image_job, "prompt_ocr", [0], {})
        assert wait_for(lambda: db.get_task(task_id)["status"] == "done")
        call = stub.calls[-1]
        assert call["fitz_preprocess"] is True  # authors enable it for prompt_ocr
    finally:
        worker.shutdown()
