"""Regression test for the oversized-embedded-image incident.

An agent parsing papers through the service (ocrc -> demo worker) hit a hard
`status=error` on ordinary arXiv PDFs — arXiv 2407.21783 (Llama 3) and
2501.12948 (DeepSeek-R1) — because `load_pdf_pages` refused to render any
page embedding an image larger than 30 MP at native size (`is_page_safe_to_render`,
ported from upstream). Those papers carry high-resolution figures
(e.g. 23666x6483 px on Llama 3 page 0), so every page selection that included
such a page failed, and the retries the agent did could never succeed.

Rendering does not need the embedded image at native resolution: get_pixmap
rasterises the page at the matrix from `get_matrix` (capped at 11.3 MP), so
the guard protected against a cost that never materialises. It was removed;
this test pins the behaviour so future updates cannot reintroduce it.
"""

import io

import fitz
import pytest
from PIL import Image

from dots_mocr.utils.doc_utils import load_pdf_pages

# Above the old 30 MP rejection threshold (23666x6483 on the Llama 3 paper);
# kept modest so the test stays fast and small in memory.
OVERSIZED_IMAGE = (6200, 5000)  # 31 MP


@pytest.fixture(scope="module")
def pdf_with_oversized_image(tmp_path_factory):
    """A one-page PDF whose only figure is a >30 MP embedded image."""
    path = tmp_path_factory.mktemp("pdf") / "oversized.pdf"
    image = Image.new("RGB", OVERSIZED_IMAGE, (200, 200, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "Page with an oversized embedded figure", fontsize=14)
    page.insert_image(fitz.Rect(72, 150, 520, 700), stream=buffer.getvalue())
    doc.save(str(path))
    doc.close()
    return str(path)


def test_page_with_oversized_embedded_image_is_rendered(pdf_with_oversized_image):
    """Must render, not raise 'not safe to render' — the incident failure."""
    pages = load_pdf_pages(pdf_with_oversized_image, dpi=100)
    assert [index for index, _ in pages] == [0]
    _, image = pages[0]
    assert isinstance(image, Image.Image)
    # the render follows the page matrix (dpi), not the embedded image's size
    assert image.width * image.height < 30_000_000


def test_oversized_page_selected_explicitly(pdf_with_oversized_image):
    """The agent's failing case: an explicit page selection that includes
    the page with the oversized figure."""
    pages = load_pdf_pages(pdf_with_oversized_image, dpi=100, page_ids=[0])
    assert len(pages) == 1
