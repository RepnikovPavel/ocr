import json

import pytest

from benchmarks.synth import docs, scoring


# -------------------------------------------------------------- generators

def test_all_cases_unique_ids_and_valid_gt():
    cases = docs.all_cases()
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    assert len(cases) > 40
    for case in cases:
        assert case.kind in scoring.SCORERS
        assert case.group in docs.GROUP_PREAMBLE
        assert case.tex_body.strip()
        # ground truth round-trips through json (stored in reports)
        json.dumps(case.ground_truth)


def test_table_ground_truth_consistency():
    for case in docs.table_cases():
        gt = case.ground_truth
        assert len(gt["cell_int_set"]) == len(set(gt["cell_int_set"]))
        assert gt["rows_total"] == gt["data_rows"] + 1


def test_build_document_page_count_matches_cases():
    cases = docs.table_cases()
    doc = docs.build_document("tables", cases)
    assert doc.count(r"\clearpage") == len(cases) - 1
    assert doc.startswith(r"\documentclass")
    assert r"\begin{document}" in doc and r"\end{document}" in doc
    for case in cases:
        assert f"CASEID:{case.case_id}" in doc


# ------------------------------------------------------------------ tables

def _html_table(rows):
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return f"<table>{body}</table>"


def test_score_table_perfect():
    case = docs._table_case(rows=3, cols=3, style="grid")
    gt = case.ground_truth
    # reconstruct the exact grid: header + data rows
    rows = [["Col1", "Col2", "Col3"]]
    for r in range(3):
        rows.append([str(gt["cell_values"][f"{r},{c}"]) for c in range(3)])
    result = scoring.score_table(_html_table(rows), gt)
    assert result["cell_recall"] == 1.0
    assert result["structure_exact"] is True
    assert result["positional_accuracy"] == 1.0


def test_score_table_missing_column_degrades():
    case = docs._table_case(rows=3, cols=4, style="grid")
    gt = case.ground_truth
    rows = [["Col1", "Col2", "Col3"]]  # dropped a column
    for r in range(3):
        rows.append([str(gt["cell_values"][f"{r},{c}"]) for c in range(3)])
    result = scoring.score_table(_html_table(rows), gt)
    assert result["cols_ok"] is False
    assert result["cell_recall"] < 1.0
    assert 0.0 < result["cell_recall"] <= 0.75


def test_score_table_no_table_output():
    case = docs._table_case(rows=3, cols=3, style="grid")
    result = scoring.score_table("no table here, just prose", case.ground_truth)
    assert result["cell_recall"] == 0.0
    assert result["has_html_table"] is False
    assert result["structure_exact"] is False


def test_parse_html_table_lenient_attrs_and_wrapping():
    html = 'prefix <table border="1"><tr><td x="1">1</td><td>2</td></tr></table> suffix'
    rows = scoring.parse_html_table(html)
    assert rows == [["1", "2"]]


# ---------------------------------------------------------------- formulas

def test_score_formula_perfect_and_normalization():
    gt = {"latex": r"\frac{a}{b} + c"}
    # model wraps in $...$, uses \left/\right and \dfrac — should normalize away
    model = r"$\dfrac{a}{b} + c$"
    result = scoring.score_formula(model, gt)
    assert result["token_recall"] == 1.0
    assert result["frac_count_ok"] is True


def test_score_formula_monotonic_degradation():
    gt = {"latex": r"\frac{x + a}{b + \frac{y}{z}}"}
    perfect = scoring.score_formula(r"\frac{x + a}{b + \frac{y}{z}}", gt)["token_recall"]
    partial = scoring.score_formula(r"\frac{x + a}{b + y}", gt)["token_recall"]
    wrong = scoring.score_formula("some words", gt)["token_recall"]
    assert perfect == 1.0
    assert wrong < partial < perfect


def test_score_formula_counts_fracs():
    gt = {"latex": r"\frac{a}{\frac{b}{c}}"}
    result = scoring.score_formula(r"\frac{a}{b}", gt)
    assert result["frac_expected"] == 2
    assert result["frac_detected"] == 1
    assert result["frac_count_ok"] is False


# -------------------------------------------------------------- algorithms

def test_score_algorithm_keyword_and_identifier_recall():
    gt = {"keywords": ["while", "return", "if"],
          "identifiers": ["BinarySearch", "lo", "hi", "mid"]}
    model = "procedure BinarySearch while lo hi ... return mid"
    result = scoring.score_algorithm(model, gt)
    assert result["keyword_recall"] == pytest.approx(2 / 3, abs=1e-3)  # 'if' missing
    assert result["identifier_recall"] == 1.0
    assert "if" in result["keywords_missing"]


def test_score_algorithm_whole_word_no_false_positive():
    gt = {"keywords": ["if", "do"], "identifiers": ["Foo"]}
    # 'if' inside 'specification', 'do' inside 'domain' must NOT count
    result = scoring.score_algorithm("specification of the domain for Foo", gt)
    assert result["keyword_recall"] == 0.0
    assert result["identifier_recall"] == 1.0


def test_score_algorithm_empty():
    gt = {"keywords": ["for", "while"], "identifiers": ["Foo"]}
    result = scoring.score_algorithm("", gt)
    assert result["keyword_recall"] == 0.0
    assert result["identifier_recall"] == 0.0


# ------------------------------------------------------------------- code

def test_score_code_perfect():
    gt = {
        "lines": ["def f(x):", "    return x + 1"],
        "special_tokens": ["def", "return", ":"],
    }
    model = "def f(x):\n    return x + 1"
    result = scoring.score_code(model, gt)
    assert result["line_recall"] == 1.0
    assert result["special_token_recall"] == 1.0


def test_score_code_partial_lines():
    gt = {
        "lines": ["package main", "func main() {", "    fmt.Println(1)", "}"],
        "special_tokens": ["package", "func", "Println"],
    }
    # model drops one line and slightly garbles another
    model = "package main\nfunc main() {\n    fmt.PrintIn(1)\n"
    result = scoring.score_code(model, gt)
    assert 0.5 <= result["line_recall"] < 1.0
    assert result["lines_matched"] < result["lines_total"]


def test_score_code_special_tokens_case_sensitive():
    gt = {"lines": ["FN main"], "special_tokens": ["fn"]}
    # 'FN' upper should NOT satisfy the case-sensitive 'fn'
    result = scoring.score_code("FN main", gt)
    assert result["special_token_recall"] == 0.0


def test_primary_metric_covers_all_kinds():
    for kind in scoring.SCORERS:
        assert kind in scoring.PRIMARY_METRIC
