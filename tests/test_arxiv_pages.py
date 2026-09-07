"""Tests for the per-page parse cache: demo.arxiv.pages + the pipeline wiring.

No network, no GPU. PDFs are generated with PyMuPDF; the OCR HTTP calls are
monkeypatched. What is pinned here:

- page identity is a content hash (stable across re-renders, different across
  different page pixels);
- assemble_bundle rebuilds the canonical document bundle from 1-page bundles;
- a paper with SOME pages cached only sends the missing pages to OCR;
- a paper with ALL pages cached never touches OCR at all.
"""

import io
import json
import zipfile

import pytest

from demo.arxiv import db, pages as pages_mod, pipeline, source
from demo import storage

OCR_URL = "http://ocr.test"
MODE = "prompt_layout_all_en"
PARSER = "dots_mocr"


@pytest.fixture()
def adb(tmp_path):
    db.init(str(tmp_path / "p.db"))
    storage.configure_store(storage.LocalBlobStore(tmp_path / "blobs"))
    yield
    storage.configure_store(None)


def _make_pdf(marker: str, n_pages: int = 3) -> bytes:
    import fitz
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"{marker} page {i}")
    data = doc.tobytes()
    doc.close()
    return data


def _page_bundle(label: str) -> bytes:
    """A fake 1-page OCR bundle zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("document.md", f"# {label}\nbody")
        zf.writestr("meta.json", json.dumps(
            {"pages_done": 1, "generated_tokens": 10, "seconds": 1.0}))
        zf.writestr(f"images/{label}.jpg", b"jpeg-" + label.encode())
    return buf.getvalue()


# ------------------------------------------------------------------ pages module

def test_page_hashes_are_content_addressed():
    pdf = _make_pdf("alpha")
    first = pages_mod.page_hashes(pdf)
    second = pages_mod.page_hashes(pdf)
    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3  # distinct pages -> distinct hashes
    other = pages_mod.page_hashes(_make_pdf("beta"))
    assert other != first


def test_pagemap_roundtrip_and_corruption():
    shas = ["a" * 64, "b" * 64]
    assert pages_mod.parse_pagemap(pages_mod.build_pagemap(shas)) == shas
    assert pages_mod.parse_pagemap(b"not json") is None
    assert pages_mod.parse_pagemap(json.dumps({"pages": "nope"}).encode()) is None


def test_assemble_bundle_merges_pages_in_order():
    bundle = pages_mod.assemble_bundle(
        [_page_bundle("p0"), _page_bundle("p1")],
        sha256="x" * 64, prompt_mode=MODE, parser=PARSER)
    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        md = zf.read("document.md").decode()
        assert md == "# p0\nbody\n\n# p1\nbody"  # worker's "\n\n" join
        meta = json.loads(zf.read("meta.json"))
        assert meta["pages_done"] == 2
        assert meta["assembled_from_pages"] is True
        assert sorted(n for n in zf.namelist() if n.startswith("images/")) == \
            ["images/p0.jpg", "images/p1.jpg"]


# ------------------------------------------------------------------ pipeline wiring

def _paper(i):
    return {"arxiv_id": f"2306.{i:05d}", "title": f"Paper {i}",
            "authors": ["A"], "summary": "s", "categories": "q-fin.TR",
            "published_at": "2023", "pdf_url": f"https://arxiv.org/pdf/2306.{i:05d}",
            "abs_url": f"https://arxiv.org/abs/2306.{i:05d}"}


@pytest.fixture()
def fake_ocr(monkeypatch):
    """Fake OCR service that records WHICH pages it was asked to parse."""
    state = {"submitted_pages": []}
    pdf_by_id = {}

    def fake_fetch_pdf(url, timeout=120.0):
        marker = url.rsplit("/", 1)[-1]
        pdf_by_id[marker] = pdf_by_id.get(marker) or _make_pdf(marker)
        return pdf_by_id[marker]

    def fake_submit(ocr_url, pdf_bytes, sha256, prompt_mode, agent, pages="all"):
        state["submitted_pages"].append(pages)
        return {"sha256": sha256, "status": "queued", "task_id": "t"}

    def fake_status(ocr_url, sha256, prompt_mode):
        return {"status": "done", "cached": False, "progress": {"done": 1, "total": 1}}

    def fake_bundle(ocr_url, sha256, prompt_mode, pages=None):
        return _page_bundle(f"page{pages}")

    monkeypatch.setattr(source, "fetch_pdf", fake_fetch_pdf)
    monkeypatch.setattr(pipeline, "submit_to_ocr", fake_submit)
    monkeypatch.setattr(pipeline, "ocr_status", fake_status)
    monkeypatch.setattr(pipeline, "ocr_bundle", fake_bundle)
    monkeypatch.setattr(pipeline, "POLL_INTERVAL_S", 0.01)
    return state


def test_full_miss_parses_every_page_once(adb, fake_ocr):
    paper = _paper(1)
    db.upsert_paper(paper)
    run_id = db.create_run("q", 1, [paper["arxiv_id"]])
    result = pipeline.run_paper_dag(paper, OCR_URL, MODE, "agent", run_id)

    assert result["ok"] is True
    # 3 pages, each submitted individually in page order
    assert fake_ocr["submitted_pages"] == ["0", "1", "2"]
    # every page blob + the pagemap landed in storage
    shas = pages_mod.page_hashes(_make_pdf(paper["arxiv_id"]))
    assert all(storage.has_page(s, PARSER) for s in shas)
    assert storage.get_pagemap(result["sha256"], PARSER) is not None
    # assembled bundle reads as a 3-page document
    with zipfile.ZipFile(io.BytesIO(storage.get_bundle(result["sha256"]))) as zf:
        meta = json.loads(zf.read("meta.json"))
        assert meta["pages_done"] == 3
        md = zf.read("document.md").decode()
        assert md.index("# page0") < md.index("# page1") < md.index("# page2")


def test_partial_miss_only_ocrs_missing_pages(adb, fake_ocr):
    paper = _paper(2)
    db.upsert_paper(paper)
    # pre-seed pages 0 and 2 in the page cache
    shas = pages_mod.page_hashes(_make_pdf(paper["arxiv_id"]))
    storage.put_page(shas[0], _page_bundle("page0"), parser=PARSER)
    storage.put_page(shas[2], _page_bundle("page2"), parser=PARSER)

    run_id = db.create_run("q", 1, [paper["arxiv_id"]])
    result = pipeline.run_paper_dag(paper, OCR_URL, MODE, "agent", run_id)

    assert result["ok"] is True
    assert fake_ocr["submitted_pages"] == ["1"]  # only the missing page
    steps = {s["stage"]: s for s in db.steps_for_run(run_id)}
    assert steps["pages"]["status"] == "done"


def test_full_hit_never_touches_ocr(adb, fake_ocr, monkeypatch):
    paper = _paper(3)
    db.upsert_paper(paper)
    # run once to populate the cache, then re-run with a tripwire on submit
    run_id = db.create_run("q", 1, [paper["arxiv_id"]])
    assert pipeline.run_paper_dag(paper, OCR_URL, MODE, "agent", run_id)["ok"]

    def boom(*args, **kwargs):
        raise AssertionError("OCR must not be called on a full page-cache hit")
    monkeypatch.setattr(pipeline, "submit_to_ocr", boom)

    run_id2 = db.create_run("q", 1, [paper["arxiv_id"]])
    result = pipeline.run_paper_dag(paper, OCR_URL, MODE, "agent", run_id2)
    assert result["ok"] is True
    steps = {s["stage"]: s for s in db.steps_for_run(run_id2)}
    for stage in ("submit_ocr", "wait_ocr", "fetch_bundle"):
        assert steps[stage]["status"] == "skipped", stage
    assert steps["store"]["status"] == "done"
