"""LaTeX document generators with ground truth for edge probing.

Each generator returns SynthCase objects. Cases that share a `group` share a
LaTeX preamble and are compiled into one multi-page document, one case per
page (page index == case index within the group), which keeps the
page->case mapping trivial while amortizing pdflatex startup.

Ground truth is designed for unambiguous scoring:
  - table cells hold unique integers (OCR-friendly, position-checkable);
  - formulas store their exact LaTeX source;
  - algorithms store the keywords/identifiers that must survive rendering;
  - code listings store the exact source lines plus language-specific tokens.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SynthCase:
    case_id: str
    kind: str            # table | formula | algorithm | code
    group: str           # shared-preamble bucket -> one document per group
    tex_body: str        # LaTeX placed on a single page
    ground_truth: dict
    params: dict = field(default_factory=dict)


# Preamble per group. babel(russian) so Cyrillic renders like the diploma.
_COMMON = [
    r"\usepackage[T1,T2A]{fontenc}",
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[english,russian]{babel}",
    r"\usepackage{amsmath}",
    r"\usepackage{amssymb}",
    r"\usepackage{amsfonts}",
    r"\usepackage[a4paper,margin=15mm]{geometry}",
]

GROUP_PREAMBLE = {
    "tables": _COMMON + [r"\usepackage{booktabs}", r"\usepackage{multirow}", r"\usepackage{array}"],
    "formulas": _COMMON,
    "algo_algorithmic": _COMMON + [r"\usepackage{algorithmic}", r"\usepackage{algorithm}"],
    "algo_algpseudocode": _COMMON + [r"\usepackage{algpseudocode}", r"\usepackage{algorithm}"],
    "algo_algorithm2e": _COMMON + [r"\usepackage[ruled,vlined]{algorithm2e}"],
    "code_c": _COMMON + [r"\usepackage{listings}", r"\usepackage{xcolor}"],
    "code_cpp": _COMMON + [r"\usepackage{listings}", r"\usepackage{xcolor}"],
    "code_python": _COMMON + [r"\usepackage{listings}", r"\usepackage{xcolor}"],
    "code_go": _COMMON + [r"\usepackage{listings}", r"\usepackage{xcolor}"],
    "code_rust": _COMMON + [r"\usepackage{listings}", r"\usepackage{xcolor}"],
}


# ------------------------------------------------------------------ tables

def _cell_content(value, text_cells):
    if not text_cells:
        return str(value)
    # diploma-like: a short Russian phrase with inline math wrapping the unique
    # integer, so integer-set scoring still works on multi-token cells
    return r"знач $x_{%d}$ равно %d ед" % (value % 10, value)


def _table_body(rows, cols, style, multicol=False, multirow=False, base=1000, text_cells=False):
    """Build a tabular; data cells hold consecutive unique integers."""
    cells = {}
    value = base
    cell_col = r"p{2.2cm}" if text_cells else "c"
    header = " & ".join(f"Col{c + 1}" for c in range(cols)) + r" \\"
    lines = []
    if style == "booktabs":
        colspec = "".join(cell_col for _ in range(cols))
        lines.append(r"\begin{tabular}{" + colspec + "}")
        lines.append(r"\toprule")
        lines.append(header)
        lines.append(r"\midrule")
    else:
        colspec = "|" + "|".join(cell_col for _ in range(cols)) + "|"
        lines.append(r"\begin{tabular}{" + colspec + "}")
        lines.append(r"\hline")
        lines.append(header)
        lines.append(r"\hline")

    for r in range(rows):
        row_cells = []
        c = 0
        while c < cols:
            content = _cell_content(value, text_cells)
            if multicol and r == 0 and c == 0 and cols >= 3:
                # merge first two header-data cells of the first data row
                row_cells.append(r"\multicolumn{2}{c}{" + content + "}" if style == "booktabs"
                                 else r"\multicolumn{2}{|c|}{" + content + "}")
                cells[(r, c)] = value
                value += 1
                c += 2
                continue
            if multirow and c == cols - 1 and r % 2 == 0 and rows - r >= 2:
                row_cells.append(r"\multirow{2}{*}{" + content + "}")
                cells[(r, c)] = value
                value += 1
                c += 1
                continue
            if multirow and c == cols - 1 and r % 2 == 1:
                row_cells.append("")  # spanned by the multirow above
                c += 1
                continue
            row_cells.append(content)
            cells[(r, c)] = value
            value += 1
            c += 1
        lines.append(" & ".join(row_cells) + r" \\")
        if style != "booktabs":
            lines.append(r"\hline")
    if style == "booktabs":
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines), cells


def _table_case(rows, cols, style="grid", multicol=False, multirow=False, text_cells=False):
    tag = f"tbl_{style}_r{rows}_c{cols}"
    if multicol:
        tag += "_mc"
    if multirow:
        tag += "_mr"
    if text_cells:
        tag += "_txt"
    tabular, cells = _table_body(rows, cols, style, multicol, multirow, text_cells=text_cells)
    body = (
        r"\begin{center}" + "\n"
        + tabular + "\n"
        + r"\end{center}"
    )
    gt = {
        "data_rows": rows,
        "cols": cols,
        "rows_total": rows + 1,  # + header
        "cell_values": {f"{r},{c}": v for (r, c), v in cells.items()},
        "cell_int_set": sorted(cells.values()),
        "style": style,
        "regular_grid": not (multicol or multirow),
        "text_cells": text_cells,
    }
    return SynthCase(tag, "table", "tables", body, gt,
                     {"rows": rows, "cols": cols, "style": style,
                      "multicol": multicol, "multirow": multirow,
                      "text_cells": text_cells})


def table_cases():
    cases = []
    for cols in [2, 3, 4, 5, 6, 8, 10, 12]:
        cases.append(_table_case(rows=6, cols=cols, style="grid"))
    for rows in [3, 8, 12, 16, 20, 25]:
        cases.append(_table_case(rows=rows, cols=4, style="grid"))
    for cols in [3, 5, 8]:
        cases.append(_table_case(rows=6, cols=cols, style="booktabs"))
    cases.append(_table_case(rows=6, cols=4, style="grid", multicol=True))
    cases.append(_table_case(rows=6, cols=5, style="grid", multirow=True))
    cases.append(_table_case(rows=8, cols=6, style="grid", multicol=True, multirow=True))
    # stress: very wide tables (font auto-shrinks on A4)
    for cols in [16, 20]:
        cases.append(_table_case(rows=6, cols=cols, style="grid"))
    # stress: diploma-like text-dense cells (multi-token, inline math, wrapped)
    for (rows, cols) in [(6, 3), (8, 5), (12, 6)]:
        cases.append(_table_case(rows=rows, cols=cols, style="grid", text_cells=True))
    # de-duplicate by id, keep order
    seen, unique = set(), []
    for case in cases:
        if case.case_id not in seen:
            seen.add(case.case_id)
            unique.append(case)
    return unique


# ---------------------------------------------------------------- formulas

def _frac_nest(depth):
    expr = "x_{0}"
    for i in range(1, depth + 1):
        expr = r"\frac{" + expr + r" + a_{" + str(i) + r"}}{b_{" + str(i) + r"} + " + expr + "}"
    return expr


def _matrix(n):
    rows = []
    for i in range(n):
        rows.append(" & ".join(f"a_{{{i}{j}}}" for j in range(n)))
    return r"\begin{pmatrix}" + r" \\ ".join(rows) + r"\end{pmatrix}"


def _polynomial(terms):
    parts = [f"c_{{{k}}} x^{{{k}}}" for k in range(terms, -1, -1)]
    return " + ".join(parts)


def _formula_case(case_id, latex, level, family):
    body = r"\[" + "\n" + latex + "\n" + r"\]"
    gt = {"latex": latex, "level": level, "family": family}
    return SynthCase(case_id, "formula", "formulas", body, gt,
                     {"level": level, "family": family})


def formula_cases():
    cases = []
    for depth in [1, 2, 3, 4, 5, 6]:
        cases.append(_formula_case(f"frac_d{depth}", _frac_nest(depth), depth, "frac_nest"))
    for n in [2, 3, 4, 5, 6]:
        cases.append(_formula_case(f"matrix_n{n}", _matrix(n), n, "matrix"))
    for terms in [3, 5, 8, 12]:
        cases.append(_formula_case(f"poly_t{terms}", _polynomial(terms), terms, "polynomial"))
    cases.append(_formula_case(
        "sum_int_1",
        r"\sum_{i=1}^{n} \frac{1}{i^2} = \frac{\pi^2}{6}", 1, "sum_int"))
    cases.append(_formula_case(
        "sum_int_2",
        r"\int_{0}^{\infty} e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2}", 2, "sum_int"))
    cases.append(_formula_case(
        "sum_int_3",
        r"\sum_{k=0}^{\infty} \frac{x^k}{k!} = e^{x}, \quad "
        r"\int_{a}^{b} f(x)\,dx = F(b) - F(a)", 3, "sum_int"))
    cases.append(_formula_case(
        "nested_sub_1",
        r"a_{i_{j_{k}}}^{n^{m^{p}}} + \sqrt{\sqrt{\sqrt{x}}}", 3, "deep_nest"))
    cases.append(_formula_case(
        "mixed_1",
        r"P(A \mid B) = \frac{P(B \mid A)\,P(A)}{\sum_{k} P(B \mid A_k)\,P(A_k)}",
        2, "mixed"))
    return cases


# -------------------------------------------------------------- algorithms

def _expected_keywords(dialect, nesting):
    """Keywords that actually RENDER for a given case (not the whole dialect).

    A small algorithm without a FOR loop must not be scored against 'for' —
    the metric measures whether the keywords that ARE present survive OCR.
    """
    if dialect == "algorithmic":
        kw = {"while", "do", "end", "return"}
        if nesting >= 2:
            kw |= {"if", "then"}
    elif dialect == "algpseudocode":
        kw = {"procedure", "while", "do", "if", "then", "return", "end"}
    else:  # algorithm2e
        kw = {"for", "if", "return"}
        if nesting >= 2:
            kw |= {"while"}
    return sorted(kw)


def _algorithmic_body(n_extra, nesting):
    """Diploma-style algorithmic (uppercase \\STATE, \\WHILE)."""
    idents = ["GradientStep", "epsilon", "lambda", "theta", "grad"]
    lines = [r"\begin{algorithm}[H]",
             r"\caption{Synthetic procedure GradientStep}",
             r"\begin{algorithmic}[1]",
             r"\STATE Initialize $\theta \gets \theta_0$",
             r"\STATE Choose $\epsilon > 0$, $\lambda > 0$"]
    lines.append(r"\WHILE{$\|grad\| \geq \epsilon$}")
    lines.append(r"\STATE compute $grad \gets \nabla f(\theta)$")
    if nesting >= 2:
        lines.append(r"\IF{$\|grad\| > 10$}")
        lines.append(r"\STATE $\lambda \gets \lambda / 2$")
        lines.append(r"\ENDIF")
    lines.append(r"\STATE $\theta \gets \theta - \lambda \cdot grad$")
    for i in range(n_extra):
        lines.append(r"\STATE update auxiliary variable $z_{%d}$" % i)
    lines.append(r"\ENDWHILE")
    lines.append(r"\RETURN $\theta$")
    lines += [r"\end{algorithmic}", r"\end{algorithm}"]
    return "\n".join(lines), idents


def _algpseudocode_body(n_extra, nesting):
    idents = ["BinarySearch", "lo", "hi", "mid", "target"]
    lines = [r"\begin{algorithm}[H]",
             r"\caption{Synthetic procedure BinarySearch}",
             r"\begin{algorithmic}[1]",
             r"\Procedure{BinarySearch}{$A, target$}",
             r"\State $lo \gets 1$, $hi \gets n$",
             r"\While{$lo \leq hi$}",
             r"\State $mid \gets \lfloor (lo + hi)/2 \rfloor$",
             r"\If{$A[mid] = target$}",
             r"\State \Return $mid$",
             r"\EndIf"]
    if nesting >= 2:
        lines.append(r"\If{$A[mid] < target$}")
        lines.append(r"\State $lo \gets mid + 1$")
        lines.append(r"\Else")
        lines.append(r"\State $hi \gets mid - 1$")
        lines.append(r"\EndIf")
    for i in range(n_extra):
        lines.append(r"\State auxiliary step $s_{%d}$" % i)
    lines += [r"\EndWhile", r"\State \Return $-1$",
              r"\EndProcedure", r"\end{algorithmic}", r"\end{algorithm}"]
    return "\n".join(lines), idents


def _algorithm2e_body(n_extra, nesting):
    idents = ["Partition", "pivot", "left", "right", "arr"]
    lines = [r"\begin{algorithm}[H]",
             r"\caption{Synthetic procedure Partition}",
             r"\SetKwFunction{Partition}{Partition}",
             r"\KwIn{array $arr$, indices $left$, $right$}",
             r"\KwOut{pivot index}",
             r"$pivot \gets arr[right]$\;",
             r"$i \gets left - 1$\;",
             r"\For{$j \gets left$ \KwTo $right - 1$}{",
             r"  \If{$arr[j] \leq pivot$}{",
             r"    $i \gets i + 1$\;",
             r"    swap $arr[i]$ and $arr[j]$\;",
             r"  }",
             r"}"]
    if nesting >= 2:
        lines.append(r"\While{$i > left$}{")
        lines.append(r"  $i \gets i - 1$\;")
        lines.append(r"}")
    for k in range(n_extra):
        lines.append(r"aux step $t_{%d}$\;" % k)
    lines.append(r"\Return $i + 1$\;")
    lines = [r"\begin{algorithm}[H]", r"\caption{Synthetic procedure Partition}"] + lines[1:] + [r"\end{algorithm}"]
    return "\n".join(lines), idents


def algorithm_cases():
    cases = []
    builders = {
        "algo_algorithmic": _algorithmic_body,
        "algo_algpseudocode": _algpseudocode_body,
        "algo_algorithm2e": _algorithm2e_body,
    }
    dialect_key = {
        "algo_algorithmic": "algorithmic",
        "algo_algpseudocode": "algpseudocode",
        "algo_algorithm2e": "algorithm2e",
    }
    for group, builder in builders.items():
        for n_extra, nesting in [(0, 1), (3, 2), (8, 2), (15, 2)]:
            body, idents = builder(n_extra, nesting)
            case_id = f"{group}_x{n_extra}_n{nesting}"
            gt = {
                "dialect": dialect_key[group],
                "keywords": _expected_keywords(dialect_key[group], nesting),
                "identifiers": idents,
                "n_extra": n_extra,
                "nesting": nesting,
            }
            cases.append(SynthCase(case_id, "algorithm", group, body, gt,
                                   {"n_extra": n_extra, "nesting": nesting,
                                    "dialect": dialect_key[group]}))
    return cases


# ------------------------------------------------------------------- code

_CODE_SNIPPETS = {
    "c": {
        "group": "code_c", "lang": "C",
        "special": ["#include", "int", "printf", "return", "->", "malloc", "{", "}", ";"],
        "lines": [
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "typedef struct Node { int val; struct Node* next; } Node;",
            "int sum_list(Node* head) {",
            "    int acc = 0;",
            "    for (Node* p = head; p != NULL; p = p->next) {",
            "        acc += p->val;",
            "    }",
            "    return acc;",
            "}",
            "int main(void) {",
            "    printf(\"%d\\n\", sum_list(NULL));",
            "    return 0;",
            "}",
        ],
    },
    "cpp": {
        "group": "code_cpp", "lang": "C++",
        "special": ["#include", "template", "std::", "::", "->", "return", "<int>", "auto", "{", "}"],
        "lines": [
            "#include <vector>",
            "#include <algorithm>",
            "template <typename T>",
            "T max_element_value(const std::vector<T>& xs) {",
            "    auto it = std::max_element(xs.begin(), xs.end());",
            "    return it != xs.end() ? *it : T{};",
            "}",
            "int main() {",
            "    std::vector<int> v = {3, 1, 4, 1, 5};",
            "    return max_element_value(v);",
            "}",
        ],
    },
    "python": {
        "group": "code_python", "lang": "Python",
        "special": ["def", "return", "for", "in", "if", "self", "import", ":", "=="],
        "lines": [
            "import math",
            "def quicksort(arr):",
            "    if len(arr) <= 1:",
            "        return arr",
            "    pivot = arr[len(arr) // 2]",
            "    left = [x for x in arr if x < pivot]",
            "    mid = [x for x in arr if x == pivot]",
            "    right = [x for x in arr if x > pivot]",
            "    return quicksort(left) + mid + quicksort(right)",
            "print(quicksort([3, 6, 1, 8, 2]))",
        ],
    },
    "go": {
        "group": "code_go", "lang": "Go",
        "special": ["package", "func", "return", "range", ":=", "for", "int", "{", "}"],
        "lines": [
            "package main",
            "import \"fmt\"",
            "func sum(xs []int) int {",
            "    total := 0",
            "    for _, x := range xs {",
            "        total += x",
            "    }",
            "    return total",
            "}",
            "func main() {",
            "    fmt.Println(sum([]int{1, 2, 3}))",
            "}",
        ],
    },
    "rust": {
        "group": "code_rust", "lang": "Rust",
        "special": ["fn", "let", "mut", "->", "match", "return", "&", "::", "HashMap"],
        "lines": [
            "use std::collections::HashMap;",
            "fn classify(n: i64) -> &'static str {",
            "    match n {",
            "        x if x < 0 => \"negative\",",
            "        0 => \"zero\",",
            "        _ => return \"positive\",",
            "    }",
            "}",
            "fn main() {",
            "    let mut counts: HashMap<&str, u32> = HashMap::new();",
            "    for n in &[-2, 0, 5] {",
            "        *counts.entry(classify(*n)).or_insert(0) += 1;",
            "    }",
            "    println!(\"{:?}\", counts);",
            "}",
        ],
    },
}


def _lstlisting_body(lang_key, n_lines):
    spec = _CODE_SNIPPETS[lang_key]
    code_lines = spec["lines"][:n_lines]
    body = (
        r"\begin{lstlisting}[language=" + spec["lang"].replace("+", "").replace("#", "") +
        r",basicstyle=\ttfamily\small,columns=fullflexible]" + "\n"
        + "\n".join(code_lines) + "\n"
        + r"\end{lstlisting}"
    )
    return body, code_lines


def code_cases():
    cases = []
    for lang_key, spec in _CODE_SNIPPETS.items():
        total = len(spec["lines"])
        for n_lines in sorted({min(6, total), min(10, total), total}):
            body, code_lines = _lstlisting_body(lang_key, n_lines)
            case_id = f"{spec['group']}_l{n_lines}"
            # only score tokens that actually appear in the selected code lines
            joined = "\n".join(code_lines)
            present = [t for t in spec["special"] if t in joined]
            gt = {
                "lang": spec["lang"],
                "lines": code_lines,
                "special_tokens": present,
                "n_lines": n_lines,
            }
            cases.append(SynthCase(case_id, "code", spec["group"], body, gt,
                                   {"lang": spec["lang"], "n_lines": n_lines}))
    return cases


# ---------------------------------------------------------------- assembly

def all_cases():
    return table_cases() + formula_cases() + algorithm_cases() + code_cases()


def cases_by_group(cases):
    groups: dict[str, list] = {}
    for case in cases:
        groups.setdefault(case.group, []).append(case)
    return groups


def build_document(group, cases):
    """Wrap a group's cases into one multi-page LaTeX document (1 case/page)."""
    preamble = GROUP_PREAMBLE[group]
    parts = [r"\documentclass[12pt]{article}"]
    parts += preamble
    parts.append(r"\pagestyle{empty}")
    parts.append(r"\begin{document}")
    for index, case in enumerate(cases):
        # small machine-readable marker at the top for page<->case verification
        parts.append(r"\noindent\texttt{CASEID:" + case.case_id + "}")
        parts.append(r"\par\vspace{4mm}")
        parts.append(case.tex_body)
        if index != len(cases) - 1:
            parts.append(r"\clearpage")
    parts.append(r"\end{document}")
    return "\n".join(parts)
