import importlib
import io
import json
import sys

import fitz
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture()
def demo_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("CKPTDIR", "/nonexistent")
    sys.modules.pop("demo.server", None)
    server = importlib.import_module("demo.server")
    # make sure we did not pick up a foreign `demo` package from sys.path
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    assert pathlib.Path(server.__file__).is_relative_to(repo_root), server.__file__
    return server


class StubParser:
    """Mimics DotsMOCRParser.parse_file without a model."""

    def __init__(self):
        self.device = "cpu"
        self.temperature = 0.1
        self.max_completion_tokens = 16384
        self.dpi = 200

    def parse_file(self, input_path, output_dir="", prompt_mode="", custom_prompt=None, pages=None):
        import os
        filename = os.path.splitext(os.path.basename(input_path))[0]
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)
        results = []
        for page_no in (pages if pages else [0]):
            md_path = os.path.join(save_dir, f"{filename}_page_{page_no}.md")
            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write(f"stub page {page_no}")
            json_path = os.path.join(save_dir, f"{filename}_page_{page_no}.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump([{"bbox": [0, 0, 10, 10], "category": "Text", "text": "stub"}], fh)
            results.append({
                "page_no": page_no,
                "md_content_path": md_path,
                "layout_info_path": json_path,
            })
        return results


def make_pdf_bytes(num_pages=2):
    doc = fitz.open()
    for index in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"demo page {index}", fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


def test_healthz(demo_app):
    client = TestClient(demo_app.app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_parse_pages_selection(demo_app):
    assert demo_app.parse_pages_selection("all") is None
    assert demo_app.parse_pages_selection("") is None
    assert demo_app.parse_pages_selection("1,3") == [0, 2]
    assert demo_app.parse_pages_selection("2-4") == [1, 2, 3]


def test_prepare_then_parse_by_job_id(demo_app, monkeypatch):
    """Regression: /api/parse with job_id (no file) used to crash on file.filename."""
    client = TestClient(demo_app.app)
    stub = StubParser()
    monkeypatch.setattr(demo_app, "get_parser", lambda: stub)

    prepared = client.post(
        "/api/prepare",
        files={"file": ("doc.pdf", io.BytesIO(make_pdf_bytes()), "application/pdf")},
    )
    assert prepared.status_code == 200, prepared.text
    job = prepared.json()
    assert job["is_pdf"] is True
    assert job["num_pages"] == 2
    assert len(job["thumb_urls"]) == 2

    parsed = client.post(
        "/api/parse",
        data={"job_id": job["job_id"], "pages": "1-2", "prompt": "prompt_layout_all_en"},
    )
    assert parsed.status_code == 200, parsed.text
    payload = parsed.json()
    assert payload["num_pages"] == 2
    assert payload["results"][0]["md_content_url"].startswith("/files/")

    md = client.get(payload["results"][0]["md_content_url"])
    assert md.status_code == 200
    assert "stub page" in md.text

    meta_path = demo_app.JOBS_DIR / job["job_id"] / "job.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["filename"].startswith("input")


def test_parse_direct_file_upload(demo_app, monkeypatch):
    client = TestClient(demo_app.app)
    stub = StubParser()
    monkeypatch.setattr(demo_app, "get_parser", lambda: stub)

    response = client.post(
        "/api/parse",
        data={"pages": "all", "prompt": "prompt_layout_all_en"},
        files={"file": ("doc.pdf", io.BytesIO(make_pdf_bytes(1)), "application/pdf")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["num_pages"] == 1


def test_parse_unknown_prompt_rejected(demo_app):
    client = TestClient(demo_app.app)
    response = client.post(
        "/api/parse",
        data={"job_id": "does-not-matter", "prompt": "prompt_nonexistent"},
    )
    assert response.status_code == 400


def test_parse_missing_job(demo_app, monkeypatch):
    client = TestClient(demo_app.app)
    monkeypatch.setattr(demo_app, "get_parser", lambda: StubParser())
    response = client.post(
        "/api/parse",
        data={"job_id": "no-such-job", "prompt": "prompt_layout_all_en"},
    )
    assert response.status_code == 404
