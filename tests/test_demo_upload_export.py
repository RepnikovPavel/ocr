"""Tests for the screenshot-batch UX: multi-image upload, appending to an
image job, big-screenshot downscaling, and the zip/pdf export endpoints."""

import io
import json
import zipfile

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from test_demo_server import load_app, pdf_bytes
from test_demo_worker import StubParser, wait_for


@pytest.fixture()
def mocr(tmp_path, monkeypatch):
    server = load_app(tmp_path, monkeypatch, variant="mocr")
    stub = StubParser()
    server.WORKER._parser_factory = lambda: stub
    with TestClient(server.app) as client:
        yield server, client, stub
    server.WORKER.shutdown()


def png_bytes(size=(400, 300), color=(240, 240, 240)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    buf.seek(0)
    return buf


def png_field(name, size=(400, 300), color=(240, 240, 240)):
    return ("file", (name, png_bytes(size, color), "image/png"))


def upload_images(client, *sizes):
    files = [png_field(f"shot{i}.png", size) for i, size in enumerate(sizes)]
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200, res.text
    return res.json()


# ---------------------------------------------------------- multi-image upload

def test_multi_image_upload_makes_one_multi_page_job(mocr):
    _, client, _ = mocr
    job = upload_images(client, (400, 300), (500, 200), (320, 640))
    assert job["kind"] == "image"
    assert job["num_pages"] == 3
    assert [v["page"] for v in job["views"]] == [0, 1, 2]
    assert len({v["url"] for v in job["views"]}) == 3
    for view in job["views"]:
        assert client.get(view["url"]).status_code == 200
    got = client.get(f"/api/jobs/{job['job_id']}").json()
    assert got["num_pages"] == 3


def test_single_image_upload_still_works(mocr):
    _, client, _ = mocr
    job = upload_images(client, (256, 128))
    assert job["kind"] == "image" and job["num_pages"] == 1
    assert job["views"][0]["width"] == 256  # small images are not rescaled


def test_upload_rejects_pdf_mixed_with_images(mocr):
    _, client, _ = mocr
    files = [("file", ("doc.pdf", io.BytesIO(pdf_bytes(1)), "application/pdf")),
             png_field("shot.png")]
    res = client.post("/api/upload", files=files)
    assert res.status_code == 400


def test_upload_rejects_multiple_pdfs(mocr):
    _, client, _ = mocr
    files = [("file", ("a.pdf", io.BytesIO(pdf_bytes(1)), "application/pdf")),
             ("file", ("b.pdf", io.BytesIO(pdf_bytes(1)), "application/pdf"))]
    assert client.post("/api/upload", files=files).status_code == 400


# ---------------------------------------------------------- appending images

def test_append_images_grows_the_batch(mocr):
    _, client, _ = mocr
    job = upload_images(client, (400, 300))
    res = client.post(f"/api/jobs/{job['job_id']}/images",
                      files=[png_field("b.png", (500, 200)), png_field("c.png", (640, 480))])
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["num_pages"] == 3
    assert [v["page"] for v in data["views"]] == [0, 1, 2]
    # persisted, not just returned
    assert client.get(f"/api/jobs/{job['job_id']}").json()["num_pages"] == 3
    # appending a non-image is rejected
    bad = client.post(f"/api/jobs/{job['job_id']}/images",
                      files=[("file", ("x.pdf", io.BytesIO(pdf_bytes(1)), "application/pdf"))])
    assert bad.status_code == 400


def test_append_images_rejects_pdf_job(mocr):
    _, client, _ = mocr
    res = client.post("/api/upload",
                      files=[("file", ("doc.pdf", io.BytesIO(pdf_bytes(1)), "application/pdf"))])
    job = res.json()
    denied = client.post(f"/api/jobs/{job['job_id']}/images", files=[png_field("x.png")])
    assert denied.status_code == 400
    assert client.post("/api/jobs/missing/images", files=[png_field("x.png")]).status_code == 404


class SizeRecordingStub(StubParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sizes = []

    def _parse_single_image(self, origin_image, *args, **kwargs):
        self.sizes.append(origin_image.size)
        return super()._parse_single_image(origin_image, *args, **kwargs)


def test_worker_runs_batch_pages_in_order_with_distinct_inputs(tmp_path, monkeypatch):
    server = load_app(tmp_path, monkeypatch, variant="mocr")
    stub = SizeRecordingStub()
    server.WORKER._parser_factory = lambda: stub
    with TestClient(server.app) as client:
        job = upload_images(client, (400, 300), (500, 200))
        res = client.post("/api/tasks", data={
            "job_id": job["job_id"], "prompt_mode": "prompt_ocr", "pages": "0,1",
        })
        task_id = res.json()["task_id"]
        assert wait_for(lambda: client.get(f"/api/tasks/{task_id}").json()["status"] == "done")
        task = client.get(f"/api/tasks/{task_id}").json()
        assert [r["page_no"] for r in task["result"]] == [0, 1]
        # page 1 went through input_001, not the first screenshot again
        assert stub.sizes == [(400, 300), (500, 200)]
    server.WORKER.shutdown()


def test_worker_legacy_single_input_still_resolves(tmp_path):
    """Jobs created before the multi-image scheme carry input.png, page 0 only."""
    from demo.worker import DemoWorker

    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "legacy1"
    job_dir.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(job_dir / "input.png")
    worker = DemoWorker(ckpt="/x", jobs_dir=jobs_dir)
    assert worker._input_path({"id": "legacy1", "kind": "image"}, 0).name == "input.png"

    job_dir2 = jobs_dir / "multi1"
    job_dir2.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(job_dir2 / "input_000.png")
    Image.new("RGB", (100, 80)).save(job_dir2 / "input_001.png")
    job = {"id": "multi1", "kind": "image"}
    assert worker._input_path(job, 0).name == "input_000.png"
    assert worker._input_path(job, 1).name == "input_001.png"


# ---------------------------------------------------------- big screenshots

def test_big_screenshot_view_is_downscaled_but_input_kept(mocr):
    """A 5K screenshot must not hand the browser (or the RAM of this process)
    a full-size JPEG per page; the original bytes stay for inference."""
    server, client, _ = mocr
    job = upload_images(client, (5000, 3000))
    view = job["views"][0]
    assert max(view["width"], view["height"]) <= server.VIEW_MAX_DIM
    # aspect ratio preserved
    assert abs(view["width"] / view["height"] - 5000 / 3000) < 0.01
    # the original is untouched on disk — inference quality does not depend
    # on the viewer render
    from demo.server import JOBS_DIR
    stored = JOBS_DIR / job["job_id"] / "input_000.png"
    with Image.open(stored) as img:
        assert img.size == (5000, 3000)


def test_absurdly_large_image_is_rejected(mocr, monkeypatch):
    server, client, _ = mocr
    monkeypatch.setattr(server, "MAX_IMAGE_PIXELS", 500_000)
    res = client.post("/api/upload", files=[png_field("huge.png", (1000, 1000))])
    assert res.status_code == 400
    assert "too large" in res.text


def test_inference_image_is_capped_for_big_screenshots():
    """The pixels the model sees stay under max_pixels regardless of how big
    the pasted screenshot is — the GPU-side OOM guard."""
    from dots_mocr.utils.image_utils import fetch_image

    big = Image.new("RGB", (7680, 4320))  # 8K monitor screenshot
    image = fetch_image(big, max_pixels=2_200_000)
    assert image.width * image.height <= 2_200_000
    # extreme ultrawide: aspect kept, still capped
    wide = Image.new("RGB", (11520, 2160))
    image = fetch_image(wide, max_pixels=2_200_000)
    assert image.width * image.height <= 2_200_000
    assert image.width > image.height


def test_huge_point_page_render_never_exceeds_cap():
    """Regression: the old >4500px fallback re-rendered at 72 dpi, which for a
    big-monitor screenshot (a 1pt-per-px PDF page) is a FULL-SIZE render —
    the OOM path. The matrix is now rescaled instead."""
    import fitz

    from dots_mocr.utils.doc_utils import MAX_RENDER_SIDE, fitz_doc_to_image

    doc = fitz.open()
    page = doc.new_page(width=12000, height=6000)  # 12k x 6k pt page
    image = fitz_doc_to_image(page, target_dpi=150)
    assert max(image.width, image.height) <= MAX_RENDER_SIDE
    # and it is not the old collapse-to-72dpi either (that would be 12000 wide)
    assert image.width >= 2000
    doc.close()


# ---------------------------------------------------------- export

def _run_stub_task(client, job_pages="0"):
    res = client.post("/api/upload",
                      files=[("file", ("doc.pdf", io.BytesIO(pdf_bytes(2)), "application/pdf"))])
    job = res.json()
    res = client.post("/api/tasks", data={
        "job_id": job["job_id"], "prompt_mode": "prompt_layout_all_en", "pages": job_pages,
    })
    task_id = res.json()["task_id"]
    assert wait_for(lambda: client.get(f"/api/tasks/{task_id}").json()["status"] == "done")
    return task_id


def test_export_zip_contains_document_and_meta(mocr):
    _, client, _ = mocr
    task_id = _run_stub_task(client, "0,1")
    res = client.get(f"/api/tasks/{task_id}/export.zip")
    assert res.status_code == 200
    assert "attachment" in res.headers["content-disposition"]
    archive = zipfile.ZipFile(io.BytesIO(res.content))
    names = archive.namelist()
    assert "document.md" in names and "meta.json" in names
    assert "pages/page_001.md" in names and "pages/page_002.md" in names
    assert "stub markdown" in archive.read("document.md").decode()
    meta = json.loads(archive.read("meta.json"))
    assert meta["task_id"] == task_id and meta["pages"] == [0, 1]


def test_export_pdf_renders_markdown(mocr):
    _, client, _ = mocr
    task_id = _run_stub_task(client)
    res = client.get(f"/api/tasks/{task_id}/export.pdf")
    assert res.status_code == 200
    assert res.content.startswith(b"%PDF")
    assert res.headers["content-disposition"].endswith('.pdf"')
    import fitz
    doc = fitz.open("pdf", res.content)
    assert doc.page_count >= 1
    assert "stub markdown" in doc[0].get_text()
    doc.close()


def test_export_guards(mocr):
    server, client, _ = mocr
    assert client.get("/api/tasks/missing/export.zip").status_code == 404
    assert client.get("/api/tasks/missing/export.pdf").status_code == 404
    # a queued-but-never-run task has no results to export
    client.post("/api/model/stop")
    assert wait_for(lambda: server.WORKER.paused)
    res = client.post("/api/upload",
                      files=[("file", ("doc.pdf", io.BytesIO(pdf_bytes(1)), "application/pdf"))])
    job = res.json()
    task_id = client.post("/api/tasks", data={
        "job_id": job["job_id"], "prompt_mode": "prompt_ocr", "pages": "0",
    }).json()["task_id"]
    assert client.get(f"/api/tasks/{task_id}/export.zip").status_code == 400
    assert client.get(f"/api/tasks/{task_id}/export.pdf").status_code == 400
    client.post(f"/api/tasks/{task_id}/cancel")


# ---------------------------------------------------------- mdexport unit tests

def test_mdexport_html_subset():
    from demo import mdexport

    html = mdexport.markdown_to_html(
        "# T\n\ntext **b** *i* `c` $x^2$\n\n"
        "| a | b |\n| --- | --- |\n| 1 | 2 |\n\n"
        "- one\n- two\n\n```\ncode <>&\n```\n\n![pic](images/p.png)\n")
    assert "<h1>T</h1>" in html
    assert "<b>b</b>" in html and "<i>i</i>" in html and "<code>c</code>" in html
    assert 'class="math"' in html
    assert "<table" in html and "<td>1</td>" in html and "<th>a</th>" in html
    assert "<li>one</li><li>two</li>" in html
    assert "code &lt;&gt;&amp;" in html  # code fence content is escaped
    assert '<img src="images/p.png"' in html


def test_mdexport_pdf_never_raises_on_garbage(tmp_path):
    from demo import mdexport

    weird = "| broken\n\x00\x01 weird <tag attr='x'> $unclosed\n" * 50
    pdf = mdexport.markdown_to_pdf(weird, assets_dir=tmp_path)
    assert pdf.startswith(b"%PDF")


# ---------------------------------------------------------- upload UX (static)

def test_dropzone_click_focuses_and_never_opens_the_picker():
    """Regression (2026-09-07): a stale cached app.js kept the OLD handler
    where any click on the dropzone opened the file dialog, so the new
    click-to-focus-for-paste flow never worked. Pin both sides: the click
    handler must not touch the file input, and index.html must version-pin
    app.js so browsers cannot serve a stale copy."""
    import re
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "demo" / "static"
    app_js = (static / "app.js").read_text(encoding="utf-8")
    index = (static / "index.html").read_text(encoding="utf-8")
    assert "/static/app.js?v=" in index
    handler = re.search(r"dropzone\.onclick\s*=\s*([^;]+);", app_js)
    assert handler and "file-input" not in handler.group(1)
    # the picker is opened only by the attach button
    attach = [ln for ln in app_js.splitlines() if '$("attach-btn").onclick' in ln]
    assert attach and 'file-input' in attach[0] and '.click()' in attach[0]
    # paste is handled at document level: a non-editable div does not get
    # paste events in every browser
    assert 'document.addEventListener("paste"' in app_js


def test_pdf_export_uses_iframe_mathjax_and_post():
    """Regression (2026-09-07, twice): (a) the fitz Story GET export cannot
    typeset TeX — formulas exported as monospace source; (b) the browser
    print-window workaround added chrome garbage (URL, date, page numbers,
    title) and opened a new tab. The final design: a hidden iframe typesets
    with MathJax fontCache=none, the preview HTML + formula SVGs are POSTed,
    and a ready PDF file downloads — no new window, no print dialog."""
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "demo" / "static"
    app_js = (static / "app.js").read_text(encoding="utf-8")
    index = (static / "index.html").read_text(encoding="utf-8")
    assert 'id="export-pdf"' in index
    assert '$("export-pdf").onclick = () => exportPdf()' in app_js
    # hidden iframe MathJax, self-contained SVGs (no font dependency)
    assert "fontCache: 'none'" in app_js
    assert 'method: "POST"' in app_js and "export.pdf" in app_js
    assert "URL.createObjectURL" in app_js  # direct file download
    # neither of the rejected approaches may come back
    export_section = app_js.split("/* ------------------------------------------------ export */")[1]
    assert "window.print()" not in export_section
    assert 'window.open("", "_blank")' not in export_section


def test_export_pdf_post_embeds_typeset_formula(mocr):
    """The browser-driven export: preview HTML + MathJax SVG -> PDF file."""
    _, client, _ = mocr
    task_id = _run_stub_task(client)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="10ex" height="2.5ex" '
           'viewBox="0 -1000 500 1200"><g transform="scale(1,-1)">'
           '<path d="M10 0 L100 800 L200 0 Z" fill="black"/></g></svg>')
    html = ('<h1>Страница</h1><p>до <img data-math="0" '
            'style="height:14pt; vertical-align:-3pt;"> после</p>')
    res = client.post(f"/api/tasks/{task_id}/export.pdf",
                      json={"html": html, "math": [svg]})
    assert res.status_code == 200, res.text
    assert res.content.startswith(b"%PDF")
    assert "attachment" in res.headers["content-disposition"]
    import fitz
    doc = fitz.open("pdf", res.content)
    assert doc.page_count >= 1
    assert "до" in doc[0].get_text() and "после" in doc[0].get_text()
    assert len(doc[0].get_images()) == 1  # the formula image is embedded
    assert "layout_all" in doc.metadata["title"]  # no "about:blank"-style junk
    doc.close()


def test_export_pdf_post_validation(mocr):
    _, client, _ = mocr
    task_id = _run_stub_task(client)
    assert client.post(f"/api/tasks/{task_id}/export.pdf",
                       json={"html": "  ", "math": []}).status_code == 400
    assert client.post("/api/tasks/missing/export.pdf",
                       json={"html": "<p>x</p>", "math": []}).status_code == 404
    too_many = client.post(f"/api/tasks/{task_id}/export.pdf",
                           json={"html": "<p>x</p>", "math": ["<svg/>"] * 1001})
    assert too_many.status_code == 400


def test_html_to_pdf_sanitizes_and_degrades(tmp_path):
    from demo import mdexport

    # script/handlers are stripped; a broken svg becomes a text marker
    pdf = mdexport.html_to_pdf(
        '<p>ok</p><script>alert(1)</script><p onclick="x()">t</p>'
        '<img data-math="0" style="height:10pt">',
        assets_dir=tmp_path, math_svgs=["<not-svg"], title="t")
    assert pdf.startswith(b"%PDF")
    import fitz
    text = fitz.open("pdf", pdf)[0].get_text()
    assert "ok" in text and "alert" not in text and "формула" in text


def test_iframe_font_sizes_match_pdf_css():
    """Regression (2026-09-07): MathJax scales formulas to the surrounding
    font; the hidden typeset iframe ran at the browser default 16px while the
    PDF body is 10pt, so exported formulas came out ~1.6x too large. The two
    stylesheets must carry identical font sizes."""
    import re
    from pathlib import Path

    from demo import mdexport

    app_js = (Path(__file__).resolve().parents[1]
              / "demo" / "static" / "app.js").read_text(encoding="utf-8")
    iframe_css = re.search(r"<style>([\s\S]*?)</style>", app_js).group(1)
    pdf_css = mdexport._CSS
    for selector in ("body", "h1", "h2", "h3", "code"):
        for css in (iframe_css, pdf_css):
            assert re.search(rf"{selector}[ ,{{][^}}]*?font-size: ([\d.]+)pt", css), \
                f"{selector} font-size missing"
        sizes = [re.search(rf"{selector}[ ,{{][^}}]*?font-size: ([\d.]+)pt", css).group(1)
                 for css in (iframe_css, pdf_css)]
        assert sizes[0] == sizes[1], f"{selector}: iframe {sizes[0]}pt != pdf {sizes[1]}pt"
