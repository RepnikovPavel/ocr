"""Synthetic diagnostic: LaTeX table whose cells contain math formulas.

Reproduces the question "why does math inside a table show as unicode in the
markdown preview?". The finding (see reports/synth/table_formula_finding.md):
this is a MODEL behaviour, not a UI bug. dots.mocr follows the authors' prompt
rule "Table -> HTML" and renders in-cell math as HTML with unicode glyphs and
<sub>/<sup> tags (e.g. $\\pi_{NL}(x)$ -> "π<sub>NL</sub>(x)"), not LaTeX. The UI
renders that HTML faithfully (legible sub/superscripts).

scripts/run_synth_edge.sh compiles this doc; score_table_formula checks that the
in-cell math CONTENT survives (variable names, greek letters, sub/sup structure).
"""

import re

TEX = r"""\documentclass[12pt]{article}
\usepackage[T1,T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage[a4paper,margin=20mm]{geometry}
\pagestyle{empty}
\begin{document}
\noindent\texttt{CASEID:table_formula}
\par\vspace{4mm}
\begin{center}
\begin{tabular}{|c|c|c|}
\hline
переменная & функция & формула \\
\hline
$x$ & $\pi_{NL}(x)$ & $\frac{1}{1+e^{-x}}$ \\
\hline
$y$ & $\alpha_{k}^{2}$ & $\sqrt{y^2 + 1}$ \\
\hline
$z$ & $\sum_{i=1}^{n} z_i$ & $\frac{\partial f}{\partial z}$ \\
\hline
\end{tabular}
\end{center}
\end{document}
"""

# content that must survive OCR of the in-cell math (as unicode or LaTeX)
GROUND_TRUTH = {
    "greek": ["π", "α", "Σ", "pi", "alpha", "sum"],      # any representation
    "vars": ["x", "y", "z"],
    "structure_markers": ["sub", "sup", "_", "^", "frac", "/", "√", "sqrt"],
}


def score_table_formula(model_text):
    """Does the Table preserve its in-cell math content and use structured markup?"""
    text = model_text or ""
    low = text.lower()
    has_table = "<table" in low
    # greek content survived in SOME form
    greek_hit = any(g in text for g in ["π", "α", "Σ"]) or any(g in low for g in ["pi", "alpha", "sum"])
    # sub/superscript structure present (HTML tags or LaTeX markers)
    structure = bool(re.search(r"<sub|<sup|_\{|\^\{|[_^]", text))
    # vars present
    vars_hit = sum(1 for v in ["x", "y", "z"] if v in low)
    # representation of the in-cell math: html unicode vs latex delimiters
    representation = "html" if ("<sub" in low or "<sup" in low) else (
        "latex" if ("$" in text or "\\(" in text) else "plain")
    return {
        "has_table": has_table,
        "greek_preserved": greek_hit,
        "structure_present": structure,
        "vars_found": vars_hit,
        "cell_math_representation": representation,
    }
