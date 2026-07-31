"""Tests for demo.arxiv.pipeline — DAG execution against a fake OCR service.

No network, no GPU. We monkeypatch the OCR HTTP calls and source.fetch_pdf so
the full DAG (download -> submit -> wait -> bundle -> store -> index) runs end
to end against in-memory fakes, verifying the step state machine and that a
failure in one paper isolates correctly.
"""

import io
import json
import zipfile

import pytest

from demo.arxiv import db, pipeline, source
from demo import storage

OCR_URL = "http://ocr.test"
MODE = "prompt_layout_all_en"


@pytest.fixture()
def adb(tmp_path, monkeypatch):
    db.init(str(tmp_path / "p.db"))
    storage.configure_store(storage.LocalBlobStore(tmp_path / "blobs"))
    yield
    storage.configure_store(None)


def _fake_pdf_bytes(arxiv_id):
    # minimal valid PDF header so it would pass PyMuPDF if ever opened
    return (b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n" + arxiv_id.encode())


@pytest.fixture()
def fake_ocr(monkeypatch):
    """A scripted OCR service: submit returns queued, status -> done, bundle."""
    state = {"submitted": [], "sha_status": {}}

    def fake_fetch_pdf(url, timeout=120.0):
        # derive an id-ish string from the url so each paper has different bytes
        return _fake_pdf_bytes(url.rsplit("/", 1)[-1])

    def fake_submit(ocr_url, pdf_bytes, sha256, prompt_mode, agent):
        state["submitted"].append(sha256)
        state["sha_status"][sha256] = "queued"
        return {"sha256": sha256, "status": "queued", "task_id": "task-" + sha256[:6]}

    def fake_status(ocr_url, sha256, prompt_mode):
        # first poll: running; second poll: done. Drives the wait loop.
        return {"sha256": sha256, "status": "done", "cached": False,
                "progress": {"done": 3, "total": 3}}

    def fake_bundle(ocr_url, sha256, prompt_mode):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("document.md", f"# parsed {sha256[:8]}\nbody")
            zf.writestr("meta.json", json.dumps(
                {"sha256": sha256, "pages_done": 3, "generated_tokens": 100,
                 "seconds": 5.0, "task_id": "task-x"}))
        return buf.getvalue()

    monkeypatch.setattr(source, "fetch_pdf", fake_fetch_pdf)
    monkeypatch.setattr(pipeline, "submit_to_ocr", fake_submit)
    monkeypatch.setattr(pipeline, "ocr_status", fake_status)
    monkeypatch.setattr(pipeline, "ocr_bundle", fake_bundle)
    monkeypatch.setattr(pipeline, "POLL_INTERVAL_S", 0.01)
    return state


def _paper(i):
    return {"arxiv_id": f"2306.{i:05d}", "title": f"Paper {i}",
            "authors": ["A"], "summary": "s", "categories": "q-fin.TR",
            "published_at": "2023", "pdf_url": f"https://arxiv.org/pdf/2306.{i:05d}",
            "abs_url": f"https://arxiv.org/abs/2306.{i:05d}"}


def test_dag_happy_path_all_steps_done(adb, fake_ocr):
    paper = _paper(1)
    db.upsert_paper(paper)
    run_id = db.create_run("quant", 1, [paper["arxiv_id"]])
    result = pipeline.run_paper_dag(paper, OCR_URL, MODE, "agent", run_id)

    assert result["ok"] is True
    steps = {s["stage"]: s for s in db.steps_for_run(run_id)}
    for stage in db.STAGES:
        assert steps[stage]["status"] == "done", (stage, steps[stage])
    # paper landed in storage + parsed
    sha = result["sha256"]
    assert storage.has_pdf(sha)
    assert storage.get_bundle(sha) is not None
    p = db.get_paper(paper["arxiv_id"])
    assert p["parsed"] is True
    assert p["storage_status"] == "parsed"
    assert p["num_pages"] == 3


def test_dag_skips_download_when_pdf_cached(adb, fake_ocr):
    """A re-run finds the PDF already in storage → download is `skipped`."""
    paper = _paper(2)
    db.upsert_paper(paper)
    # pre-seed storage with the same bytes the fake fetch would produce
    pdf = _fake_pdf_bytes(paper["pdf_url"].rsplit("/", 1)[-1])
    sha = storage.sha256_bytes(pdf)
    storage.put_pdf(sha, pdf)
    db.set_paper_downloaded(paper["arxiv_id"], sha, len(pdf))

    run_id = db.create_run("quant", 1, [paper["arxiv_id"]])
    pipeline.run_paper_dag(paper, OCR_URL, MODE, "agent", run_id)

    steps = {s["stage"]: s for s in db.steps_for_run(run_id)}
    assert steps["download"]["status"] == "skipped"
    assert steps["store"]["status"] == "done"


def test_dag_ocr_cache_hit_skips_wait(adb, fake_ocr, monkeypatch):
    """When OCR already has the parse (status=cached), wait_ocr is skipped."""
    paper = _paper(3)
    db.upsert_paper(paper)
    # make the fake submit return cached (the real one in fake_ocr returns queued)
    def fake_submit(ocr_url, pdf_bytes, sha256, prompt_mode, agent):
        return {"sha256": sha256, "status": "cached", "task_id": "old-task"}
    monkeypatch.setattr(pipeline, "submit_to_ocr", fake_submit)

    run_id = db.create_run("quant", 1, [paper["arxiv_id"]])
    pipeline.run_paper_dag(paper, OCR_URL, MODE, "agent", run_id)

    steps = {s["stage"]: s for s in db.steps_for_run(run_id)}
    assert steps["wait_ocr"]["status"] == "skipped"
    assert steps["fetch_bundle"]["status"] == "done"


def test_dag_failure_isolates_paper(adb, fake_ocr, monkeypatch):
    """A download failure marks that paper's later stages skipped, run errors."""
    paper_ok = _paper(4)
    paper_bad = _paper(5)
    db.upsert_paper(paper_ok)
    db.upsert_paper(paper_bad)

    def failing_fetch(url, timeout=120.0):
        if "2306.00005" in url:
            raise pipeline.PipelineError("arxiv 429 rate limited")
        return _fake_pdf_bytes(url.rsplit("/", 1)[-1])
    # patch the module-level source.fetch_pdf that stage_download calls
    monkeypatch.setattr(source, "fetch_pdf", failing_fetch)
    run_id = db.create_run("quant", 2, [paper_ok["arxiv_id"], paper_bad["arxiv_id"]])
    r_ok = pipeline.run_paper_dag(paper_ok, OCR_URL, MODE, "agent", run_id)
    r_bad = pipeline.run_paper_dag(paper_bad, OCR_URL, MODE, "agent", run_id)
    assert r_ok["ok"] is True
    assert r_bad["ok"] is False

    bad_steps = {s["stage"]: s for s in db.steps_for_run(run_id)
                 if s["arxiv_id"] == paper_bad["arxiv_id"]}
    # download errored, everything after it skipped
    assert bad_steps["download"]["status"] == "error"
    for stage in ("submit_ocr", "wait_ocr", "fetch_bundle", "store", "index"):
        assert bad_steps[stage]["status"] == "skipped", stage
    assert db.get_paper(paper_bad["arxiv_id"])["storage_status"] == "failed"


def test_run_pipeline_aggregates_run(adb, fake_ocr):
    papers = [_paper(10), _paper(11), _paper(12)]
    for p in papers:
        db.upsert_paper(p)
    run_id = pipeline.run_pipeline(papers, query="quant", ocr_url=OCR_URL,
                                   prompt_mode=MODE, workers=2)
    r = db.get_run(run_id)
    assert r["status"] == "done"
    assert r["done"] == len(papers) * len(db.STAGES)
    assert r["failed"] == 0
    # all three parsed + stored
    parsed = db.count_papers()
    assert parsed["parsed"] == 3
    assert parsed["stored"] == 3


def test_run_pipeline_marks_run_error_when_some_fail(adb, fake_ocr, monkeypatch):
    papers = [_paper(20), _paper(21)]
    for p in papers:
        db.upsert_paper(p)

    def failing_fetch(url, timeout=120.0):
        if "2306.00021" in url:
            raise pipeline.PipelineError("nope")
        return _fake_pdf_bytes(url.rsplit("/", 1)[-1])
    monkeypatch.setattr(source, "fetch_pdf", failing_fetch)
    run_id = pipeline.run_pipeline(papers, query="quant", ocr_url=OCR_URL,
                                   prompt_mode=MODE, workers=2)
    r = db.get_run(run_id)
    assert r["status"] == "error"
    assert r["failed"] >= 1


# ----------------------------------------------------------------- parser-tagged + download-only

def _paper(i):
    return {"arxiv_id": f"2306.{i:05d}", "title": f"Paper {i}", "authors": ["A"],
            "summary": "s", "categories": "q-fin.TR", "published_at": "2023",
            "pdf_url": f"https://arxiv.org/pdf/2306.{i:05d}",
            "abs_url": f"https://arxiv.org/abs/2306.{i:05d}"}


def test_download_only_stores_pdf_skips_parse(adb, fake_ocr, monkeypatch):
    """--download-only fetches PDFs into storage and skips all parse stages."""
    paper = _paper(40)
    db.upsert_paper(paper)
    # download_only must NOT touch the OCR service at all
    def boom_submit(*a, **kw):
        raise AssertionError("submit_to_ocr must not be called in download-only")
    monkeypatch.setattr(pipeline, "submit_to_ocr", boom_submit)
    run_id = pipeline.run_pipeline([paper], query="quant", ocr_url=OCR_URL,
                                   prompt_mode=MODE, workers=1,
                                   parser="dots_mocr", download_only=True)
    steps = {s["stage"]: s for s in db.steps_for_run(run_id)}
    assert steps["download"]["status"] == "done"
    for stage in ("submit_ocr", "wait_ocr", "fetch_bundle", "store", "index"):
        assert steps[stage]["status"] == "skipped", stage
    sha = db.get_paper(paper["arxiv_id"])["sha256"]
    assert storage.has_pdf(sha)
    assert not storage.has_bundle(sha, "dots_mocr")
    assert db.get_run(run_id)["download_only"] == 1


def test_classic_local_parser_bypasses_ocr(adb, fake_ocr, monkeypatch):
    """A local parser runs in-process; submit/wait collapse, no OCR call.

    Uses a stub parser registered under a throwaway name so the test does not
    depend on fitz being installed.
    """
    from demo.arxiv import parsers
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)

    def stub_parse(pdf_bytes):
        return {"markdown": "# stub\nparsed text",
                "meta": {"parser": "stub", "pages": 1, "pages_done": 1},
                "images": []}
    parsers.register("stub", stub_parse, local=True)

    paper = _paper(41)
    db.upsert_paper(paper)
    def boom_submit(*a, **kw):
        raise AssertionError("local parser must not call the OCR service")
    monkeypatch.setattr(pipeline, "submit_to_ocr", boom_submit)

    run_id = pipeline.run_pipeline([paper], query="quant", ocr_url=OCR_URL,
                                   prompt_mode=MODE, workers=1, parser="stub")
    steps = {s["stage"]: s for s in db.steps_for_run(run_id)}
    assert steps["submit_ocr"]["status"] == "done"
    assert steps["wait_ocr"]["status"] == "skipped"  # local parser, no OCR wait
    assert steps["store"]["status"] == "done"
    assert steps["index"]["status"] == "done"

    sha = db.get_paper(paper["arxiv_id"])["sha256"]
    bundle = storage.get_bundle(sha, "stub")
    assert bundle is not None
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        assert json.loads(z.read("meta.json"))["parser"] == "stub"
        assert b"parsed text" in z.read("document.md")
    # paper_parses row tagged with the stub parser
    parses = db.list_parses(paper["arxiv_id"])
    assert [p["parser"] for p in parses] == ["stub"]


def test_parsers_coexist_for_same_paper(adb, fake_ocr, monkeypatch):
    """dots_mocr and a local parser each produce their own bundle+parse row."""
    from demo.arxiv import parsers
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    parsers.register("stub", lambda pb: {"markdown": "classic",
        "meta": {"parser": "stub", "pages": 2, "pages_done": 2}, "images": []}, local=True)

    paper = _paper(42)
    db.upsert_paper(paper)
    # dots_mocr run (uses the fake_ocr submit/status/bundle)
    pipeline.run_pipeline([paper], query="quant", ocr_url=OCR_URL,
                          prompt_mode=MODE, workers=1, parser="dots_mocr")
    # classic run over the SAME paper
    pipeline.run_pipeline([paper], query="quant", ocr_url=OCR_URL,
                          prompt_mode=MODE, workers=1, parser="stub")

    sha = db.get_paper(paper["arxiv_id"])["sha256"]
    # both bundles present, neither clobbered the other
    assert storage.has_bundle(sha, "dots_mocr")
    assert storage.has_bundle(sha, "stub")
    parses = {p["parser"] for p in db.list_parses(paper["arxiv_id"])}
    assert parses == {"dots_mocr", "stub"}
    counts = db.count_papers()
    assert counts["by_parser"] == {"dots_mocr": 1, "stub": 1}
