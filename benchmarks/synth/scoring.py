"""Scoring functions for synthetic edge cases (pure text, unit-testable).

Every scorer takes the model's text output for a case plus its ground truth
and returns a dict of metrics in [0, 1] plus diagnostic fields. The metrics
are chosen so that a monotonic difficulty sweep produces a monotonic score
trend, which is what reveals the model's "edge".
"""

import difflib
import re
from collections import Counter
from html.parser import HTMLParser


# --------------------------------------------------------------- utilities

def all_integers(text):
    return [int(m) for m in re.findall(r"-?\d+", text or "")]


class _TableExtractor(HTMLParser):
    """Lenient HTML table parser: rows of cell texts, ignoring attributes."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self._row = None
        self._cell = None
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            self._buf = []

    def handle_data(self, data):
        if self._cell is not None:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append("".join(self._buf).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def parse_html_table(text):
    """Return a list of rows (list of cell strings) from the first table."""
    if not text:
        return []
    match = re.search(r"<table.*?</table>", text, re.DOTALL | re.IGNORECASE)
    html = match.group(0) if match else text
    parser = _TableExtractor()
    try:
        parser.feed(html)
    except Exception:
        return []
    return [row for row in parser.rows if row]


# ------------------------------------------------------------------ tables

def score_table(model_text, gt):
    grid = parse_html_table(model_text)
    n_rows = len(grid)
    n_cols = max((len(r) for r in grid), default=0)

    gt_ints = set(gt["cell_int_set"])
    got_ints = set(all_integers(model_text))
    cell_recall = len(gt_ints & got_ints) / len(gt_ints) if gt_ints else 0.0

    rows_ok = n_rows == gt["rows_total"]
    cols_ok = n_cols == gt["cols"]
    structure_exact = bool(rows_ok and cols_ok)

    positional = None
    if gt["regular_grid"] and grid:
        # data rows follow the header row in the parsed grid
        correct = 0
        total = 0
        for key, value in gt["cell_values"].items():
            r, c = (int(x) for x in key.split(","))
            total += 1
            parsed_r = r + 1  # skip header
            if parsed_r < len(grid) and c < len(grid[parsed_r]):
                if value in all_integers(grid[parsed_r][c]):
                    correct += 1
        positional = correct / total if total else 0.0

    return {
        "cell_recall": round(cell_recall, 4),
        "rows_detected": n_rows,
        "cols_detected": n_cols,
        "rows_ok": rows_ok,
        "cols_ok": cols_ok,
        "structure_exact": structure_exact,
        "positional_accuracy": round(positional, 4) if positional is not None else None,
        "has_html_table": bool(grid),
    }


# ---------------------------------------------------------------- formulas

_LATEX_TOKEN = re.compile(r"\\[a-zA-Z]+|\\[^a-zA-Z]|[a-zA-Z]|\d+|[_^{}&+\-*/=<>|]")

_NORMALIZE_SUBS = [
    (r"\left", ""), (r"\right", ""),
    (r"\displaystyle", ""), (r"\dfrac", r"\frac"), (r"\tfrac", r"\frac"),
    (r"\,", ""), (r"\!", ""), (r"\;", ""), (r"\ ", ""), (r"\quad", ""), (r"\qquad", ""),
    (r"\mid", "|"), (r"\vert", "|"),
]


def normalize_latex(text):
    if not text:
        return ""
    text = text.strip()
    # strip common math delimiters/fences
    text = re.sub(r"^\$+|\$+$", "", text)
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    text = text.replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "")
    for src, dst in _NORMALIZE_SUBS:
        text = text.replace(src, dst)
    return text


def tokenize_latex(text):
    return _LATEX_TOKEN.findall(normalize_latex(text))


def score_formula(model_text, gt):
    gt_tokens = Counter(tokenize_latex(gt["latex"]))
    got_tokens = Counter(tokenize_latex(model_text))
    total = sum(gt_tokens.values())
    matched = sum(min(count, got_tokens.get(tok, 0)) for tok, count in gt_tokens.items())
    token_recall = matched / total if total else 0.0

    # structural markers that carry the formula's shape
    markers = [r"\frac", r"\sum", r"\int", "_", "^", r"\sqrt",
               r"\begin{pmatrix}", r"\begin{matrix}", r"\infty", r"\pi"]
    gt_norm = normalize_latex(gt["latex"])
    got_norm = normalize_latex(model_text)
    struct_gt = [m for m in markers if m in gt_norm]
    struct_hit = [m for m in struct_gt if m in got_norm]
    struct_recall = len(struct_hit) / len(struct_gt) if struct_gt else 1.0

    # count-sensitive: does the output have the right number of \frac / rows?
    frac_gt = gt_norm.count(r"\frac")
    frac_got = got_norm.count(r"\frac")
    frac_count_ok = frac_gt == frac_got

    return {
        "token_recall": round(token_recall, 4),
        "struct_recall": round(struct_recall, 4),
        "frac_expected": frac_gt,
        "frac_detected": frac_got,
        "frac_count_ok": frac_count_ok,
        "gt_token_count": total,
        "got_token_count": sum(got_tokens.values()),
    }


# -------------------------------------------------------------- algorithms

_WORD = re.compile(r"[A-Za-z_]+")


def _token_recall(model_text, expected, case_insensitive=True, whole_word=False):
    text = model_text or ""
    if case_insensitive:
        text = text.lower()
    found, missing = [], []
    for token in expected:
        needle = token.lower() if case_insensitive else token
        if whole_word:
            hit = re.search(r"\b" + re.escape(needle) + r"\b", text) is not None
        else:
            hit = needle in text
        (found if hit else missing).append(token)
    recall = len(found) / len(expected) if expected else 1.0
    return recall, found, missing


def score_algorithm(model_text, gt):
    # whole-word matching: short keywords like "if"/"do" must not match inside
    # natural-language words ("specification", "domain") in the OCR text
    kw_recall, kw_found, kw_missing = _token_recall(model_text, gt["keywords"], whole_word=True)
    id_recall, id_found, id_missing = _token_recall(model_text, gt["identifiers"], whole_word=True)
    return {
        "keyword_recall": round(kw_recall, 4),
        "identifier_recall": round(id_recall, 4),
        "keywords_missing": kw_missing,
        "identifiers_missing": id_missing,
        "chars": len(model_text or ""),
    }


# ------------------------------------------------------------------- code

def _norm_code_line(line):
    return re.sub(r"\s+", " ", line).strip()


def score_code(model_text, gt):
    gt_lines = [_norm_code_line(l) for l in gt["lines"] if l.strip()]
    model_lines_raw = [l for l in (model_text or "").splitlines() if l.strip()]
    model_lines = [_norm_code_line(l) for l in model_lines_raw]

    matched = 0
    per_line = []
    used = set()
    for gt_line in gt_lines:
        best_ratio, best_j = 0.0, -1
        for j, ml in enumerate(model_lines):
            if j in used:
                continue
            ratio = difflib.SequenceMatcher(None, gt_line, ml).ratio()
            if ratio > best_ratio:
                best_ratio, best_j = ratio, j
        if best_ratio >= 0.6 and best_j >= 0:
            used.add(best_j)
            matched += 1
        per_line.append(round(best_ratio, 3))
    line_recall = matched / len(gt_lines) if gt_lines else 0.0

    tok_recall, tok_found, tok_missing = _token_recall(
        model_text, gt["special_tokens"], case_insensitive=False)

    return {
        "line_recall": round(line_recall, 4),
        "lines_matched": matched,
        "lines_total": len(gt_lines),
        "special_token_recall": round(tok_recall, 4),
        "special_tokens_missing": tok_missing,
        "per_line_similarity": per_line,
    }


SCORERS = {
    "table": score_table,
    "formula": score_formula,
    "algorithm": score_algorithm,
    "code": score_code,
}


# the single headline metric per kind, used to define the "edge"
PRIMARY_METRIC = {
    "table": "cell_recall",
    "formula": "token_recall",
    "algorithm": "identifier_recall",
    "code": "line_recall",
}


def score_case(kind, model_text, gt):
    return SCORERS[kind](model_text, gt)
