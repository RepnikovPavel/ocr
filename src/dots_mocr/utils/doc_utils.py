import fitz
import numpy as np
import enum
from pydantic import BaseModel, Field
from PIL import Image


class SupportedPdfParseMethod(enum.Enum):
    OCR = 'ocr'
    TXT = 'txt'


class PageInfo(BaseModel):
    """The width and height of page
    """
    w: float = Field(description='the width of page')
    h: float = Field(description='the height of page')


def get_matrix(page, dpi_default=200, max_pixels=11289600):
    rect = page.rect
    if rect.width * rect.height > max_pixels:
        factor = (max_pixels / (rect.width * rect.height)) ** 0.5
    else:
        factor = dpi_default / 72
    mat = fitz.Matrix(factor, factor)
    return mat

# Upper bound for one rendered page side. get_matrix caps the AREA, so a very
# large page (a big-monitor screenshot becomes a 1pt-per-px PDF page, an A0
# poster) can still render with a side far beyond this. The old fallback for
# that case re-rendered at 72 dpi — which for a point-size page is a FULL-SIZE
# render (a 16384pt-wide page -> a 16384px pixmap, ~10x the cap) and is exactly
# the OOM path big screenshots hit. Rescale the matrix down instead.
MAX_RENDER_SIDE = 4500


def fitz_doc_to_image(doc, target_dpi=200, origin_dpi=None) -> dict:
    """Convert fitz.Document to image, Then convert the image to numpy array.

    Args:
        doc (_type_): pymudoc page
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.

    Returns:
        dict:  {'img': numpy array, 'width': width, 'height': height }
    """
    from PIL import Image
    mat = get_matrix(doc, target_dpi)
    pm = doc.get_pixmap(matrix=mat, alpha=False)
    if pm.width == 0 or pm.height == 0:
        print(f"image is empty loading from pdf, skip")
        return None

    if pm.width > MAX_RENDER_SIDE or pm.height > MAX_RENDER_SIDE:
        scale = MAX_RENDER_SIDE / max(pm.width, pm.height)
        mat = fitz.Matrix(mat.a * scale, mat.d * scale)
        pm = doc.get_pixmap(matrix=mat, alpha=False)

    image = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    return image


def load_pdf_pages(pdf_file, dpi=200, page_ids=None) -> list:
    pages = []
    with fitz.open(pdf_file) as doc:
        indices = range(doc.page_count) if page_ids is None else page_ids
        for index in indices:
            if index < 0 or index >= doc.page_count:
                raise ValueError(f"PDF page {index + 1} is out of range 1..{doc.page_count}")
            page = doc[index]
            # NOTE: pages are rendered as pages, never rejected for embedding
            # high-resolution images. An earlier upstream guard refused any page
            # whose embedded image exceeded 30 MP at native size, but get_pixmap
            # rasterises the page at the matrix above — the embedded image's
            # native resolution never becomes the render size. That guard made
            # ordinary arXiv papers with high-res figures (e.g. 2407.21783,
            # 2501.12948) unparseable, failing the whole task. Render instead.
            image = fitz_doc_to_image(page, target_dpi=dpi)
            if image is not None:
                pages.append((index, image))
    return pages


def load_images_from_pdf(pdf_file, dpi=200, start_page_id=0, end_page_id=None) -> list:
    with fitz.open(pdf_file) as doc:
        last_page_id = doc.page_count - 1 if end_page_id is None or end_page_id < 0 else min(end_page_id, doc.page_count - 1)
    return [image for _, image in load_pdf_pages(pdf_file, dpi, range(start_page_id, last_page_id + 1))]
