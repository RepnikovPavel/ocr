# Diploma real-data validation

12 pages of `DiplomaMasterDegree.pdf` parsed with dots.mocr (layout_all,
dpi 200), scored against the PDF text layer. See `diploma_validation.json`.

| page | kind | word recall | number recall | char sim | algo kw recall | categories |
|---|---|---|---|---|---|---|
| 6  | formula   | 0.90 | 1.00 | 0.92 | —    | Formula:2, Text:8 |
| 7  | formula   | 0.89 | 1.00 | 0.74 | —    | Formula:3, Text:8 |
| 9  | formula   | 0.89 | 0.83 | 0.72 | —    | Formula:5, Text:12 |
| 11 | algorithm | 0.93 | 1.00 | 0.60 | 1.00 | List-item:13, Text:1 |
| 15 | table     | 0.88 | 0.69 | 0.89 | 1.00 | Formula:2, Text:6, Picture:1 |
| 16 | table     | 0.94 | 1.00 | 0.75 | —    | Table:1, Formula:3, Text:5 |
| 21 | algorithm | 0.85 | 0.92 | 0.84 | 1.00 | Formula:1, Text:7 |
| 24 | algorithm | 0.92 | 0.88 | 0.77 | 1.00 | Text:10, Picture:1 |
| 26 | algorithm | 0.89 | 0.96 | 0.72 | 1.00 | Formula:3, Text:9 |
| 27 | table     | 0.85 | 0.90 | 0.83 | —    | Table:1, Formula:1, Text:8 |
| 44 | algorithm | 0.94 | 0.94 | 0.58 | 1.00 | Formula:4, Text:5 |
| 52 | algorithm | 0.92 | 0.91 | 0.87 | 1.00 | Caption:1, Text:6 |

Mean content-word recall **0.90**, mean algorithm-keyword recall **1.00**.

Note on `char_similarity`: it compares against the PDF's own text layer, which
garbles subscripts/superscripts and multi-line math. On math-heavy algorithm
pages the model's LaTeX output is actually *more* faithful than the text layer,
so this metric is a conservative lower bound there (word/keyword recall are the
reliable signals). This matches the synthetic result: algorithms and tables are
robust; the diploma's fractions are shallow (depth 1–2), below the
nested-fraction edge, so formula pages score well too.
