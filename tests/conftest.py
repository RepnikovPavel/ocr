import os
import sys
from pathlib import Path

# Anchor imports to this repo regardless of how pytest was launched:
# `demo` and `dots_mocr` must resolve to this working tree, not to another
# project that happens to be installed/editable on the machine.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import fitz
import pytest
from PIL import Image, ImageDraw


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: integration test that needs CUDA and a real checkpoint "
        "(set DOTS_MOCR_CKPT, optionally DOTS_MOCR_TEST_PDF)",
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DOTS_MOCR_CKPT"):
        return
    skip_gpu = pytest.mark.skip(reason="DOTS_MOCR_CKPT is not set")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


PAGE_TEXTS = [
    "First page unique marker ALPHA-0001",
    "Second page unique marker BRAVO-0002",
    "Third page unique marker CHARLIE-0003",
]


@pytest.fixture(scope="session")
def synthetic_page_image():
    """A document-like RGB image, dimensions divisible by 28."""
    image = Image.new("RGB", (840, 1120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([100, 60, 740, 120], outline="black", width=2)
    draw.text((110, 80), "Synthetic Title", fill="black")
    for row in range(6):
        draw.text((100, 220 + row * 60), f"Body text line {row} lorem ipsum", fill="black")
    draw.rectangle([100, 700, 500, 1000], fill=(220, 220, 220))
    return image


@pytest.fixture(scope="session")
def synthetic_pdf(tmp_path_factory):
    """A three page PDF with a distinct text marker on every page."""
    path = tmp_path_factory.mktemp("pdf") / "synthetic.pdf"
    doc = fitz.open()
    for text in PAGE_TEXTS:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), text, fontsize=14)
        page.insert_text((72, 150), "Common body paragraph for parsing.", fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)
