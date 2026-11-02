import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from demo.inference_process import InferenceProcess
from demo.worker import DemoWorker


class FakeParser:
    def __init__(self, device="cpu", fail_load=False, delay=0, **kwargs):
        if fail_load:
            raise ValueError("load failed")
        if delay:
            time.sleep(delay)
        self.device = device
        self.attn_implementation = "sdpa"
        self.memory = bytearray(16 * 1024 * 1024)

    def _parse_single_image(self, image, save_dir, save_name, custom_prompt=None, **kwargs):
        from dots_mocr.utils.generation_stats import GenerationStats

        stats = GenerationStats().start(10)
        self.generation_listener(stats)
        if custom_prompt == "fail":
            raise ValueError("page failed")
        for _ in range(20):
            if self.abort_event.is_set():
                break
            stats.record_token()
            time.sleep(0.02)
        stats.finish(aborted=self.abort_event.is_set())
        output = Path(save_dir) / (save_name + ".md")
        output.write_text("# parsed in child")
        return {"md_content_path": str(output), "pid": os.getpid(),
                "tokens": self.max_completion_tokens, "bbox": kwargs.get("bbox"),
                "generation": stats.to_dict()}


@pytest.fixture
def process_env(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, [root, root / "src", root / "tests"])))


def make_parser(**kwargs):
    return InferenceProcess(parser_class="test_inference_process:FakeParser", **kwargs)


def page_args(tmp_path):
    from PIL import Image

    path = tmp_path / "input.png"
    Image.new("RGB", (100, 200), "white").save(path)
    return dict(input_path=str(path), kind="image", page_no=0, dpi=150,
                bbox_view_size=(50, 100), bbox=[1, 2, 3, 4],
                prompt_mode="prompt_ocr", save_dir=str(tmp_path), save_name="parsed")


def test_child_parses_reports_progress_and_exits(process_env, tmp_path):
    parser = make_parser()
    child = parser.process
    progress = []
    parser.generation_listener = progress.append
    parser.max_completion_tokens = 37
    try:
        result = parser.parse_page(**page_args(tmp_path))
        assert result["pid"] == child.pid != os.getpid()
        assert result["tokens"] == 37
        assert result["bbox"] == [2, 4, 6, 8]
        assert Path(result["md_content_path"]).read_text() == "# parsed in child"
        assert progress and progress[-1].generated_tokens > 0
    finally:
        parser.close()
    assert child.poll() is not None
    assert parser.pid is None
    parser.close()


def test_cancellation_reaches_child_and_next_page_runs(process_env, tmp_path):
    parser = make_parser()
    parser.abort_event = threading.Event()
    parser.generation_listener = lambda stats: parser.abort_event.set()
    try:
        result = parser.parse_page(**page_args(tmp_path))
        assert result["generation"]["aborted"]
        parser.generation_listener = None
        parser.abort_event.clear()
        result = parser.parse_page(**page_args(tmp_path))
        assert not result["generation"]["aborted"]
    finally:
        parser.close()


def test_idle_and_shutdown_reap_child_and_allow_reload(process_env, tmp_path):
    from demo import db

    db.init_db(tmp_path / "db.sqlite")
    worker = DemoWorker("unused", tmp_path, parser_factory=make_parser, idle_unload_seconds=0)
    worker._load_model()
    first = worker.parser.process
    worker._last_used = 0
    worker._wakeup.set()
    worker._run_loop_once()
    assert worker.model_state == "stopped"
    assert first.poll() is not None
    worker._load_model()
    second = worker.parser.process
    assert second.pid != first.pid
    worker.idle_unload_seconds = 180
    worker.start()
    worker.shutdown()
    assert not worker.is_alive()
    assert second.poll() is not None


def test_startup_and_page_failures_reap_children(process_env, tmp_path):
    with pytest.raises(RuntimeError, match="load failed"):
        make_parser(fail_load=True)
    with pytest.raises(TimeoutError):
        make_parser(delay=10, startup_timeout=0.2)
    parser = make_parser()
    child = parser.process
    with pytest.raises(RuntimeError, match="page failed"):
        parser.parse_page(**page_args(tmp_path), custom_prompt="fail")
    assert child.poll() is not None


def test_dead_child_does_not_hang(process_env, tmp_path):
    parser = make_parser()
    parser.process.kill()
    parser.process.wait()
    try:
        with pytest.raises((RuntimeError, BrokenPipeError, ConnectionResetError)):
            parser.parse_page(**page_args(tmp_path))
    finally:
        parser.close()


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="requires POSIX file-cache hints")
def test_unload_releases_only_its_checkpoint_cache_after_child_exit(process_env, tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model-00001.safetensors").write_bytes(b"model weights")
    (checkpoint / "config.json").write_text("{}")
    (tmp_path / "unrelated.safetensors").write_bytes(b"another service")
    parser = make_parser(ckpt=str(checkpoint))
    child = parser.process
    dropped = []

    def advise(fd, offset, length, advice):
        assert child.poll() is not None
        assert advice == os.POSIX_FADV_DONTNEED
        dropped.append(os.read(fd, 100))

    monkeypatch.setattr(os, "posix_fadvise", advise)
    parser.close()
    assert dropped == [b"model weights"]
    assert (checkpoint / "model-00001.safetensors").read_bytes() == b"model weights"


def test_status_and_lightweight_utilities_do_not_import_torch(process_env):
    subprocess.run([sys.executable, "-c", """
import sys
from demo.worker import DemoWorker
from dots_mocr.utils.generation_stats import GenerationStats
from dots_mocr.utils.image_utils import fetch_image
w = DemoWorker('unused', '/unused')
assert w.status()['attn_implementation'] == 'flex_attention'
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""], check=True, timeout=15)


def test_child_exits_when_parent_disappears(process_env):
    parent = subprocess.Popen([sys.executable, "-c", """
from demo.inference_process import InferenceProcess
import time
p = InferenceProcess(parser_class='test_inference_process:FakeParser')
print(p.pid, flush=True)
time.sleep(60)
"""], stdout=subprocess.PIPE, text=True)
    child_pid = None
    try:
        import select

        assert select.select([parent.stdout], [], [], 15)[0]
        child_pid = int(parent.stdout.readline())
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child_pid}/status")
            if not status.exists() or "\nState:\tZ" in status.read_text():
                break
            time.sleep(0.1)
        else:
            pytest.fail("orphan inference process stayed alive")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child_pid is not None:
            import signal

            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        parent.stdout.close()
