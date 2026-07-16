import importlib
import io
import pathlib
import sys
import time

import fitz
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from test_demo_worker import StubParser, wait_for


def load_app(tmp_path, monkeypatch, variant="mocr", autostart="0"):
    monkeypatch.setenv("DEMO_STATE_DIR", str(tmp_path / f"state_{variant}"))
    monkeypatch.setenv("DEMO_VARIANT", variant)
    monkeypatch.setenv("DEMO_AUTOSTART", autostart)
    monkeypatch.setenv("CKPTDIR", "/nonexistent")
    monkeypatch.setenv("DEMO_PEER_PORT", "8602" if variant == "mocr" else "8601")
    monkeypatch.setenv("DEMO_PEER_TITLE", "peer demo")
    sys.modules.pop("demo.server", None)
    server = importlib.import_module("demo.server")
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    assert pathlib.Path(server.__file__).is_relative_to(repo_root), server.__file__
    return server


@pytest.fixture()
def mocr(tmp_path, monkeypatch):
    server = load_app(tmp_path, monkeypatch, variant="mocr")
    stub = StubParser()
    server.WORKER._parser_factory = lambda: stub
    with TestClient(server.app) as client:
        yield server, client, stub
    server.WORKER.shutdown()


@pytest.fixture()
def svg(tmp_path, monkeypatch):
    server = load_app(tmp_path, monkeypatch, variant="svg")
    stub = StubParser()
    server.WORKER._parser_factory = lambda: stub
    with TestClient(server.app) as client:
        yield server, client, stub
    server.WORKER.shutdown()


def pdf_bytes(num_pages=3):
    doc = fitz.open()
    for index in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"demo page {index}", fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


def upload_pdf(client, num_pages=3):
    res = client.post(
        "/api/upload",
        files={"file": (f"doc.pdf", io.BytesIO(pdf_bytes(num_pages)), "application/pdf")},
    )
    assert res.status_code == 200, res.text
    return res.json()


def test_healthz_and_index(mocr):
    _, client, _ = mocr
    assert client.get("/healthz").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert 'data-variant="mocr"' in page.text
    assert 'id="help"' in page.text
    assert 'id="peer-link"' in page.text


def test_state_exposes_peer_demo(mocr):
    _, client, _ = mocr
    peer = client.get("/api/state").json()["peer"]
    assert peer == {"port": "8602", "title": "peer demo"}


def test_state_has_prompts_session_and_cookie(mocr):
    _, client, _ = mocr
    res = client.get("/api/state")
    data = res.json()
    modes = [m["mode"] for m in data["prompt_modes"]]
    assert "prompt_layout_all_en" in modes
    assert "prompt_image_to_svg" not in modes
    svg_temp = [m for m in data["prompt_modes"] if m["mode"] == "prompt_ocr"][0]
    assert svg_temp["default_temperature"] == 0.1
    assert data["session"]["id"]
    # per-variant cookie: both demos share one host, names must not collide
    assert client.cookies.get("demo_sid_mocr")


def test_upload_pdf_renders_views(mocr):
    _, client, _ = mocr
    job = upload_pdf(client, num_pages=2)
    assert job["kind"] == "pdf"
    assert job["num_pages"] == 2
    assert len(job["views"]) == 2
    view = job["views"][0]
    assert view["width"] > 0 and view["height"] > 0
    image = client.get(view["url"])
    assert image.status_code == 200
    got = client.get(f"/api/jobs/{job['job_id']}")
    assert got.json()["num_pages"] == 2


def test_upload_rejects_unknown_type(mocr):
    _, client, _ = mocr
    res = client.post("/api/upload", files={"file": ("x.docx", io.BytesIO(b"zz"), "application/msword")})
    assert res.status_code == 400


def test_task_end_to_end_with_stub_model(mocr):
    server, client, _ = mocr
    client.post("/api/model/start")
    assert wait_for(lambda: server.WORKER.model_state == "loaded")

    job = upload_pdf(client)
    res = client.post("/api/tasks", data={
        "job_id": job["job_id"], "prompt_mode": "prompt_layout_all_en", "pages": "0,2",
    })
    assert res.status_code == 200, res.text
    task_id = res.json()["task_id"]

    def status():
        return client.get(f"/api/tasks/{task_id}").json()

    assert wait_for(lambda: status()["status"] == "done")
    task = status()
    assert [r["page_no"] for r in task["result"]] == [0, 2]
    md_url = task["result"][0]["urls"]["md_content"]
    assert md_url.startswith("/files/")
    assert "stub markdown" in client.get(md_url).text

    raw = client.get("/api/raw", params={"path": md_url})
    assert raw.status_code == 200
    assert "# page 0" in raw.json()["content"]

    state = client.get("/api/state").json()
    ours = [t for t in state["tasks"] if t["id"] == task_id]
    assert ours and ours[0]["own"] is True


def test_task_validation(mocr):
    _, client, _ = mocr
    job = upload_pdf(client)
    base = {"job_id": job["job_id"], "prompt_mode": "prompt_layout_all_en"}
    assert client.post("/api/tasks", data={**base, "pages": "0,99"}).status_code == 400
    assert client.post("/api/tasks", data={**base, "pages": "abc"}).status_code == 400
    # empty form value falls back to the default page 0 (FastAPI Form semantics)
    assert client.post("/api/tasks", data={**base, "pages": ""}).status_code == 200
    assert client.post("/api/tasks", data={**base, "prompt_mode": "prompt_image_to_svg", "pages": "0"}).status_code == 400
    assert client.post("/api/tasks", data={"job_id": "missing", "pages": "0"}).status_code == 404
    # grounding needs bbox and exactly one page
    grounding = {"job_id": job["job_id"], "prompt_mode": "prompt_grounding_ocr"}
    assert client.post("/api/tasks", data={**grounding, "pages": "0"}).status_code == 400
    assert client.post("/api/tasks", data={**grounding, "pages": "0,1", "bbox": "1,2,3,4"}).status_code == 400
    ok = client.post("/api/tasks", data={**grounding, "pages": "0", "bbox": "10,20,110,220"})
    assert ok.status_code == 200


def test_grounding_bbox_stored_with_view_size(mocr):
    """bbox stays in viewer coords; the worker rescales by the ACTUAL render."""
    server, client, _ = mocr
    job = upload_pdf(client)
    res = client.post("/api/tasks", data={
        "job_id": job["job_id"], "prompt_mode": "prompt_grounding_ocr",
        "pages": "0", "bbox": "72,72,144,144",
    })
    from demo import db
    task = db.get_task(res.json()["task_id"])
    assert task["params"]["bbox"] == [72, 72, 144, 144]
    view = job["views"][0]
    assert task["params"]["bbox_view_size"] == [view["width"], view["height"]]


def test_worker_scale_bbox():
    from PIL import Image

    from demo.worker import DemoWorker
    origin = Image.new("RGB", (300, 600))
    scaled = DemoWorker._scale_bbox([10, 20, 30, 40], [100, 200], origin)
    assert scaled == [30, 60, 90, 120]
    assert DemoWorker._scale_bbox(None, [100, 200], origin) is None
    assert DemoWorker._scale_bbox([1, 2, 3, 4], None, origin) == [1, 2, 3, 4]


def test_cancel_queued_task(mocr):
    server, client, _ = mocr
    job = upload_pdf(client)
    # pause the worker so the task deterministically stays in the queue
    client.post("/api/model/stop")
    assert wait_for(lambda: server.WORKER.paused)
    from demo import db
    task_id = db.create_task("other-session", job["job_id"], "prompt_ocr", [0], {})
    res = client.post(f"/api/tasks/{task_id}/cancel")
    assert res.json()["status"] == "cancelled"
    assert client.get(f"/api/tasks/{task_id}").json()["status"] == "cancelled"
    assert client.post("/api/tasks/missing/cancel").status_code == 404


def test_model_start_stop_endpoints(mocr):
    server, client, _ = mocr
    client.post("/api/model/start")
    assert wait_for(lambda: server.WORKER.model_state == "loaded")
    client.post("/api/model/stop")
    assert wait_for(lambda: server.WORKER.model_state == "stopped")
    assert server.WORKER.paused


def test_keep_loaded_endpoint_and_state_fields(mocr):
    server, client, _ = mocr
    res = client.post("/api/model/keep_loaded", data={"value": "true"})
    assert res.json()["keep_loaded"] is True
    state = client.get("/api/state").json()
    assert state["worker"]["keep_loaded"] is True
    assert state["prompt_modes"][0]["page_seconds_estimate"] > 0
    assert state["server_time"] > 0
    client.post("/api/model/keep_loaded", data={"value": "false"})
    assert client.get("/api/state").json()["worker"]["keep_loaded"] is False


def test_task_submission_lazy_loads_model(mocr):
    """Default policy: GPU is free; a submitted task loads the model itself."""
    server, client, _ = mocr
    assert server.WORKER.model_state == "stopped"
    job = upload_pdf(client, num_pages=1)
    task_id = client.post("/api/tasks", data={
        "job_id": job["job_id"], "pages": "0",
    }).json()["task_id"]
    assert wait_for(lambda: client.get(f"/api/tasks/{task_id}").json()["status"] == "done")
    assert server.WORKER.model_state == "loaded"


def test_cancel_works_from_fresh_client_after_reload(mocr):
    """The queue is server-side: a reloaded page (new client) can stop a task."""
    server, client, _ = mocr
    job = upload_pdf(client, num_pages=1)
    task_id = client.post("/api/tasks", data={
        "job_id": job["job_id"], "pages": "0",
    }).json()["task_id"]

    fresh = TestClient(server.app)  # no cookies: simulates a reloaded browser
    res = fresh.post(f"/api/tasks/{task_id}/cancel")
    assert res.status_code == 200
    assert res.json()["status"] in ("cancelled", "cancelling", "done")


def test_raw_endpoint_rejects_escapes(mocr):
    _, client, _ = mocr
    assert client.get("/api/raw", params={"path": "/etc/passwd"}).status_code == 400
    assert client.get("/api/raw", params={"path": "/files/../../etc/passwd"}).status_code == 400
    assert client.get("/api/raw", params={"path": "/files/nope.md"}).status_code == 404


def test_svg_variant_restricts_modes(svg):
    server, client, _ = svg
    state = client.get("/api/state").json()
    modes = [m["mode"] for m in state["prompt_modes"]]
    assert modes == ["prompt_image_to_svg"]
    svg_mode = state["prompt_modes"][0]
    assert svg_mode["default_temperature"] == 0.9  # authors' setting

    job = upload_pdf(client, num_pages=1)
    denied = client.post("/api/tasks", data={
        "job_id": job["job_id"], "prompt_mode": "prompt_layout_all_en", "pages": "0",
    })
    assert denied.status_code == 400
    allowed = client.post("/api/tasks", data={"job_id": job["job_id"], "pages": "0"})
    assert allowed.status_code == 200


def test_svg_variant_image_upload(svg):
    _, client, _ = svg
    import io as _io

    from PIL import Image
    buf = _io.BytesIO()
    Image.new("RGB", (256, 128), "white").save(buf, "PNG")
    buf.seek(0)
    res = client.post("/api/upload", files={"file": ("pic.png", buf, "image/png")})
    assert res.status_code == 200
    job = res.json()
    assert job["kind"] == "image"
    assert job["num_pages"] == 1
    assert job["views"][0]["width"] == 256
