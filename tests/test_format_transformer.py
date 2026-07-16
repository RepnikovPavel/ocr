from PIL import Image

from dots_mocr.utils.format_transformer import (
    clean_latex_preamble,
    fix_streamlit_formulas,
    get_formula_in_markdown,
    has_latex_markdown,
    layoutjson2md,
)


def test_has_latex_markdown():
    assert has_latex_markdown(r"$$E=mc^2$$")
    assert has_latex_markdown(r"inline $x^2$ formula")
    assert has_latex_markdown(r"\frac{a}{b}")
    assert not has_latex_markdown("plain text")
    assert not has_latex_markdown(None)


def test_get_formula_in_markdown_wraps_bare_latex():
    out = get_formula_in_markdown(r"\frac{a}{b}")
    assert out.startswith("$$") and out.endswith("$$")
    assert r"\frac{a}{b}" in out


def test_get_formula_in_markdown_keeps_existing_blocks():
    assert get_formula_in_markdown("$$x + y$$") == "$$\nx + y\n$$"
    assert get_formula_in_markdown(r"\[x + y\]") == "$$\nx + y\n$$"
    assert get_formula_in_markdown("text $x$ inline") == "text $x$ inline"
    assert get_formula_in_markdown("no formula here") == "no formula here"


def test_clean_latex_preamble():
    src = r"\documentclass{article}\usepackage{amsmath}\begin{document}x=1\end{document}"
    out = clean_latex_preamble(src)
    assert "documentclass" not in out
    assert "usepackage" not in out
    assert "x=1" in out


def test_layoutjson2md_text_formula_picture_base64_fallback(synthetic_page_image):
    # without image_dir the legacy standalone base64 embed is kept
    cells = [
        {"bbox": [0, 0, 100, 40], "category": "Title", "text": "# Heading"},
        {"bbox": [0, 50, 100, 90], "category": "Formula", "text": r"\alpha + \beta"},
        {"bbox": [0, 100, 200, 300], "category": "Picture"},
        {"bbox": [0, 310, 100, 340], "category": "Page-footer", "text": "page 1"},
    ]
    md = layoutjson2md(synthetic_page_image, cells)
    assert "# Heading" in md
    assert "$$" in md and r"\alpha" in md
    assert "![](data:image/png;base64," in md
    assert "page 1" in md


def test_layoutjson2md_pictures_saved_to_folder_with_relative_links(synthetic_page_image, tmp_path):
    import os

    cells = [
        {"bbox": [0, 0, 100, 40], "category": "Title", "text": "# Heading"},
        {"bbox": [0, 100, 200, 300], "category": "Picture"},
        {"bbox": [0, 320, 180, 460], "category": "Picture"},
    ]
    image_dir = str(tmp_path / "images")
    md = layoutjson2md(synthetic_page_image, cells, image_dir=image_dir,
                       rel_prefix="images", name="p0")
    # markdown carries relative links only — no base64 blobs
    assert "data:image" not in md
    assert "![](images/p0_pic_0.png)" in md
    assert "![](images/p0_pic_1.png)" in md
    # the crops are real files on disk
    assert os.path.isfile(os.path.join(image_dir, "p0_pic_0.png"))
    assert os.path.isfile(os.path.join(image_dir, "p0_pic_1.png"))


def test_layoutjson2md_skips_degenerate_picture_bbox(synthetic_page_image, tmp_path):
    cells = [{"bbox": [50, 50, 50, 90], "category": "Picture"}]  # zero width
    md = layoutjson2md(synthetic_page_image, cells, image_dir=str(tmp_path / "im"), name="p")
    assert "![](" not in md  # degenerate crop skipped, no broken link


def test_layoutjson2md_no_page_hf_skips_header_footer(synthetic_page_image):
    cells = [
        {"bbox": [0, 0, 100, 40], "category": "Page-header", "text": "HEADER"},
        {"bbox": [0, 50, 100, 90], "category": "Text", "text": "body"},
        {"bbox": [0, 100, 100, 140], "category": "Page-footer", "text": "FOOTER"},
    ]
    md = layoutjson2md(synthetic_page_image, cells, no_page_hf=True)
    assert "HEADER" not in md and "FOOTER" not in md
    assert "body" in md


def test_fix_streamlit_formulas():
    out = fix_streamlit_formulas("before $$a+b$$ after")
    assert "$$\na+b\n$$" in out
