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


def test_layoutjson2md_text_formula_picture(synthetic_page_image):
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
