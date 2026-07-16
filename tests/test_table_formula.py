"""Synthetic test for math-inside-a-table (UI vs model-edge question).

CPU part: the scorer + the UI-side contract (sub/sup survive sanitize). GPU part
(marked `gpu`): the model actually renders in-cell math as HTML unicode+sub/sup.
"""

import os

import pytest

from benchmarks.synth.table_formula import GROUND_TRUTH, score_table_formula


def test_scorer_on_captured_model_output():
    # exact Table cell text dots.mocr produced for the diagnostic table
    model_html = (
        "<table><thead><tr><td>переменная</td><td>функция</td><td>формула</td></tr></thead>"
        "<tbody><tr><td>x</td><td>π<sub>NL</sub>(x)</td><td>1 / (1 + e<sup>-x</sup>)</td></tr>"
        "<tr><td>y</td><td>α<sub>k</sub><sup>2</sup></td><td>√(y<sup>2</sup> + 1)</td></tr>"
        "<tr><td>z</td><td>Σ<sup>n</sup><sub>i=1</sub> z<sub>i</sub></td><td>∂f / ∂z</td></tr></tbody></table>"
    )
    m = score_table_formula(model_html)
    assert m["has_table"] and m["greek_preserved"] and m["structure_present"]
    assert m["vars_found"] == 3
    # THE FINDING: the model represents table math as HTML (unicode + sub/sup),
    # not LaTeX — so nothing for the UI's $...$/MathJax path to render. Not a UI bug.
    assert m["cell_math_representation"] == "html"


def test_scorer_flags_plain_loss():
    # if a table lost all structure/greek it would score as degraded
    m = score_table_formula("<table><tr><td>x</td><td>fNL</td></tr></table>")
    assert m["structure_present"] is False
    assert m["greek_preserved"] is False


def test_ground_truth_shape():
    assert set(GROUND_TRUTH) == {"greek", "vars", "structure_markers"}


# ------------------------------------------------------------------- GPU

CKPT = os.environ.get("DOTS_MOCR_CKPT")


@pytest.mark.gpu
def test_model_renders_table_math_as_html_not_latex(tmp_path):
    """Documents the edge: dots.mocr emits table-cell math as HTML sub/sup."""
    import subprocess

    import fitz

    from benchmarks.synth.table_formula import TEX

    # compile the diagnostic doc (needs a LaTeX toolchain on PATH)
    if not any(
        os.access(os.path.join(p, "pdflatex"), os.X_OK)
        for p in os.environ.get("PATH", "").split(os.pathsep)
    ):
        pytest.skip("pdflatex not available")
    tex = tmp_path / "t.tex"
    tex.write_text(TEX, encoding="utf-8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"],
                   cwd=tmp_path, capture_output=True)
    pdf = tmp_path / "t.pdf"
    assert pdf.exists(), "diagnostic doc did not compile"

    from dots_mocr.cli import DotsMOCRParser
    from dots_mocr.utils.doc_utils import load_pdf_pages
    from dots_mocr.utils.image_utils import fetch_image
    from dots_mocr.utils.layout_utils import post_process_output

    parser = DotsMOCRParser(ckpt=CKPT, device="cuda:0", dtype="bfloat16",
                            temperature=0.0, max_completion_tokens=4096,
                            dpi=150, max_pixels=1_500_000)
    img = load_pdf_pages(str(pdf), dpi=150, page_ids=[0])[0][1]
    im = fetch_image(img, min_pixels=parser.min_pixels, max_pixels=parser.max_pixels)
    prompt = parser.get_prompt("prompt_layout_all_en", origin_image=img, image=im)
    resp = parser._inference(im, prompt, temperature=0.0)
    cells, _ = post_process_output(resp, "prompt_layout_all_en", img, im,
                                   min_pixels=parser.min_pixels, max_pixels=parser.max_pixels)
    table_text = "".join(c.get("text", "") for c in cells if c.get("category") == "Table")
    m = score_table_formula(table_text)
    assert m["has_table"], f"no Table cell in output: {resp[:400]}"
    assert m["greek_preserved"] and m["structure_present"]
    assert m["cell_math_representation"] == "html"  # the edge
