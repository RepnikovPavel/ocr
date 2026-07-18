"""GPU integration tests against the real dots.mocr checkpoint.

Run on a CUDA machine:

    DOTS_MOCR_CKPT=/path/to/snapshot \
    DOTS_MOCR_TEST_PDF=/path/to/mobilenetv3.pdf \
    pytest tests/test_gpu_integration.py -v -m gpu

The tests prove the ported model produces valid dots.mocr outputs:
readable OCR text, layout JSON with legal bboxes and known categories,
and per-page artifacts in multi-page PDF mode.
"""

import json
import os

import pytest

pytestmark = pytest.mark.gpu

CATEGORIES = {
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text", "Title",
}

CKPT = os.environ.get("DOTS_MOCR_CKPT")
TEST_PDF = os.environ.get("DOTS_MOCR_TEST_PDF")
# A string that is known to appear on page 1 of the test document.
MARKER = os.environ.get("DOTS_MOCR_TEST_MARKER", "MobileNetV3")


@pytest.fixture(scope="session")
def parser():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from dots_mocr.cli import DotsMOCRParser

    return DotsMOCRParser(
        ckpt=CKPT,
        device="cuda:0",
        dtype="bfloat16",
        temperature=0.0,
        max_completion_tokens=8192,
        dpi=150,
        num_thread=1,
    )


@pytest.fixture(scope="session")
def test_pdf():
    if not TEST_PDF or not os.path.isfile(TEST_PDF):
        pytest.skip("DOTS_MOCR_TEST_PDF is not set or missing")
    return TEST_PDF


def test_model_on_cuda_bf16(parser):
    import torch

    param = next(parser.model.parameters())
    assert param.is_cuda
    assert param.dtype == torch.bfloat16
    total = sum(p.numel() for p in parser.model.parameters())
    assert total > 2_000_000_000, f"unexpected parameter count {total}"


def test_runs_on_the_default_attention_backend(parser):
    """Everything below asserts the model gives correct answers — this asserts it
    is the backend we think we are testing. Without it the whole file would still
    pass if the vision tower silently ran something else."""
    from dots_mocr.transformers_patch.modeling_dots_vision import VisionFlexAttention

    assert parser.attn_implementation == "flex_attention"
    assert parser.llm_attn_implementation == "sdpa"
    blocks = parser.model.vision_tower.blocks
    assert all(isinstance(block.attn, VisionFlexAttention) for block in blocks), (
        f"expected VisionFlexAttention in all {len(blocks)} vision layers, got "
        f"{ {type(b.attn).__name__ for b in blocks} }")


def test_single_page_ocr_contains_marker(parser, test_pdf, tmp_path):
    results = parser.parse_file(
        test_pdf, output_dir=str(tmp_path), prompt_mode="prompt_ocr", pages=[0],
    )
    assert len(results) == 1
    md = open(results[0]["md_content_path"], encoding="utf-8").read()
    assert len(md) > 200, f"suspiciously short OCR output: {md!r}"
    assert MARKER.lower() in md.lower(), f"{MARKER} not found in OCR output"


def test_single_page_layout_json_valid(parser, test_pdf, tmp_path):
    import fitz

    results = parser.parse_file(
        test_pdf, output_dir=str(tmp_path), prompt_mode="prompt_layout_all_en", pages=[0],
    )
    result = results[0]
    assert not result.get("filtered"), "model output was not parseable JSON"
    cells = json.loads(open(result["layout_info_path"], encoding="utf-8").read())
    assert isinstance(cells, list) and len(cells) >= 5
    # post-processed bboxes live in the coordinate space of the dpi-rendered page
    with fitz.open(test_pdf) as doc:
        rect = doc[0].rect
        page_width = rect.width * parser.dpi / 72
        page_height = rect.height * parser.dpi / 72
    for cell in cells:
        assert cell["category"] in CATEGORIES, cell
        x1, y1, x2, y2 = cell["bbox"]
        assert x2 > x1 and y2 > y1, cell
        assert x1 >= -5 and y1 >= -5, cell
        assert x2 <= page_width * 1.05 + 5, (cell, page_width)
        assert y2 <= page_height * 1.05 + 5, (cell, page_height)
    joined = " ".join(cell.get("text", "") for cell in cells)
    assert MARKER.lower() in joined.lower()
    categories = {cell["category"] for cell in cells}
    assert categories & {"Title", "Section-header", "Text"}, categories
    md = open(result["md_content_path"], encoding="utf-8").read()
    assert len(md) > 200


def test_layout_only_detection(parser, test_pdf, tmp_path):
    results = parser.parse_file(
        test_pdf, output_dir=str(tmp_path), prompt_mode="prompt_layout_only_en", pages=[0],
    )
    result = results[0]
    cells = json.loads(open(result["layout_info_path"], encoding="utf-8").read())
    assert isinstance(cells, list) and len(cells) >= 5
    for cell in cells:
        assert cell["category"] in CATEGORIES
        assert len(cell["bbox"]) == 4


def test_multi_page_pdf_mode(parser, test_pdf, tmp_path):
    pages = [0, 1, 2]
    results = parser.parse_file(
        test_pdf, output_dir=str(tmp_path), prompt_mode="prompt_layout_all_en", pages=pages,
    )
    assert [r["page_no"] for r in results] == pages
    contents = []
    for result in results:
        md = open(result["md_content_path"], encoding="utf-8").read()
        assert len(md) > 100, f"page {result['page_no']} produced almost no text"
        contents.append(md)
        assert os.path.isfile(result["layout_info_path"])
    # different pages must produce different text
    assert contents[0] != contents[1] and contents[1] != contents[2]
    jsonl_path = os.path.join(
        str(tmp_path), os.path.splitext(os.path.basename(test_pdf))[0] + ".jsonl",
    )
    assert os.path.isfile(jsonl_path)
    lines = open(jsonl_path, encoding="utf-8").read().strip().splitlines()
    assert len(lines) == len(pages)
