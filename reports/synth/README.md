# dots.mocr — synthetic edge probing + diploma validation

Goal: find where dots.mocr's document parsing starts to fail, using
synthetic LaTeX documents with known ground truth (so the failure point is
measurable), then validate the findings on real data (a 60-page master's
thesis rich in algorithms, tables and formulas).

Harness: `benchmarks/synth/` (generators + scoring + runner), compiled with a
texlive container, rendered and parsed with `dots-mocr:bench-cu126` on an
RTX 4090. Reproduce with `scripts/run_synth_edge.sh`. Scoring is unit-tested
(`tests/test_synth_scoring.py`, 18 CPU tests). Reports: `reports/synth/*.json`.

## Edge summary

| content | tested range | edge (where it breaks) |
|---|---|---|
| **Tables** (numeric grids) | 2–20 cols, 3–25 rows, booktabs, multicol/multirow, text-dense cells | **≥16 columns**: cell recall 0.88 at 16 cols, 0.70 at 20 cols (column width / font shrink). Everything up to 12 cols × 25 rows, merged cells, and diploma-style text cells is perfect. |
| **Formulas** | frac nesting 1–6, matrices 2–6, polynomials, sums, integrals, deep subscripts | **deeply nested fractions**: exact to depth 2, under-counts from depth 3, breaks at depth 4 (½ the `\frac` recovered), loops/hallucinates at depth 6. Polynomials, sums, integrals, matrices ≤4×4 are perfect. |
| **Algorithms** (pseudocode) | algorithmic / algpseudocode / algorithm2e, up to 15 extra lines, nested | **none found**: keyword and identifier recall = 1.00 across all 3 dialects. |
| **Code listings** | C, C++, Python, Go, Rust, up to 15 lines | **none found**: line recall and language-token recall = 1.00 across all 5 languages. |

Detail per kind: `edge_tables.md`, `edge_formulas.md`, `edge_algorithms.md`,
`edge_code.md`.

## Diploma validation (real data)

12 representative pages of `DiplomaMasterDegree.pdf` (bilingual Russian + math)
scored against the PDF's own text layer: `diploma_validation.{json,md}`.

- mean content-word recall **0.90**, mean algorithm-keyword recall **1.00**.
- Algorithm pages (11, 21, 24, 26, 44, 52): every pseudocode keyword recovered;
  the model even reproduces the inline LaTeX math correctly (e.g. page 11's
  `\boldsymbol{\omega}_k`, `P_\gamma(k)`, `\int … d\boldsymbol{x}`,
  `\mathcal{C}^2`) — cleaner than the PDF's garbled text-layer extraction,
  which is why the character-similarity metric *understates* quality there.
- Table pages (15, 16, 27): word recall 0.85–0.94; the model emits an HTML
  `Table` block for the well-delimited tables.
- Formula pages (6, 7, 9): word/number recall 0.86–1.00; the diploma's
  fractions are depth 1–2, i.e. below the nested-fraction edge, so they parse
  cleanly — consistent with the synthetic prediction.

**Conclusion**: the synthetic edge predictions hold on real data. For this
document class (theses/papers with pseudocode, tables and moderately nested
math) dots.mocr is reliable; the practical failure modes to watch are tables
wider than ~12–15 columns and fractions nested 4+ deep.
