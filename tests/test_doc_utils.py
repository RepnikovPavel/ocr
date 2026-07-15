import fitz
import pytest
from PIL import Image

from dots_mocr.utils.doc_utils import (
    fitz_doc_to_image,
    get_matrix,
    load_images_from_pdf,
    load_pdf_pages,
)


def test_load_pdf_pages_all(synthetic_pdf):
    pages = load_pdf_pages(synthetic_pdf, dpi=100)
    assert [index for index, _ in pages] == [0, 1, 2]
    for _, image in pages:
        assert isinstance(image, Image.Image)
        assert image.width > 0 and image.height > 0


def test_load_pdf_pages_subset_keeps_original_indices(synthetic_pdf):
    pages = load_pdf_pages(synthetic_pdf, dpi=100, page_ids=[2, 0])
    assert [index for index, _ in pages] == [2, 0]


def test_load_pdf_pages_out_of_range(synthetic_pdf):
    with pytest.raises(ValueError):
        load_pdf_pages(synthetic_pdf, page_ids=[3])
    with pytest.raises(ValueError):
        load_pdf_pages(synthetic_pdf, page_ids=[-1])


def test_load_pdf_pages_dpi_scales_render(synthetic_pdf):
    low = load_pdf_pages(synthetic_pdf, dpi=72, page_ids=[0])[0][1]
    high = load_pdf_pages(synthetic_pdf, dpi=144, page_ids=[0])[0][1]
    assert high.width > low.width
    assert abs(high.width / low.width - 2.0) < 0.05


def test_load_images_from_pdf_range(synthetic_pdf):
    images = load_images_from_pdf(synthetic_pdf, dpi=100, start_page_id=1)
    assert len(images) == 2
    images = load_images_from_pdf(synthetic_pdf, dpi=100, start_page_id=0, end_page_id=0)
    assert len(images) == 1


def test_get_matrix_uses_dpi_for_normal_pages(synthetic_pdf):
    with fitz.open(synthetic_pdf) as doc:
        mat = get_matrix(doc[0], dpi_default=200)
        assert mat.a == pytest.approx(200 / 72)


def test_fitz_doc_to_image_matches_page_ratio(synthetic_pdf):
    with fitz.open(synthetic_pdf) as doc:
        page = doc[0]
        image = fitz_doc_to_image(page, target_dpi=100)
        page_ratio = page.rect.width / page.rect.height
    assert isinstance(image, Image.Image)
    assert abs(image.width / image.height - page_ratio) < 0.02
