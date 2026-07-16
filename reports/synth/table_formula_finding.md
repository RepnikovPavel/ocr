# Finding: math inside a table renders as unicode, not typeset LaTeX

**Question:** in the MD preview, a LaTeX formula inside a table shows as unicode
(π, α, √, ∂ …) instead of typeset math — is this a UI bug or a model edge?

**Answer: it is a MODEL behaviour, not a UI bug.**

## Evidence (synthetic)

Input LaTeX table (`benchmarks/synth/table_formula.py`), cells hold real math:

| переменная | функция | формула |
|---|---|---|
| `$x$` | `$\pi_{NL}(x)$` | `$\frac{1}{1+e^{-x}}$` |
| `$y$` | `$\alpha_{k}^{2}$` | `$\sqrt{y^2+1}$` |
| `$z$` | `$\sum_{i=1}^{n} z_i$` | `$\frac{\partial f}{\partial z}$` |

dots.mocr (prompt_layout_all_en, RTX 4090) emits the Table cell as **HTML with
unicode glyphs + `<sub>`/`<sup>`**, never LaTeX `$...$`:

```html
<table>...<tr><td>x</td><td>π<sub>NL</sub>(x)</td><td>1 / (1 + e<sup>-x</sup>)</td></tr>
<tr><td>y</td><td>α<sub>k</sub><sup>2</sup></td><td>√(y<sup>2</sup> + 1)</td></tr>
<tr><td>z</td><td>Σ<sup>n</sup><sub>i=1</sub> z<sub>i</sub></td><td>∂f / ∂z</td></tr></table>
```

So `$\pi_{NL}(x)$ → π<sub>NL</sub>(x)`, `$\frac{1}{1+e^{-x}}$ → 1 / (1 + e^-x)`,
`$\sqrt{y^2+1}$ → √(y² + 1)`.

## Why

The authors' layout prompt says explicitly: *"Table: Format its text as HTML."*
The model obeys — inside the HTML table it linearizes math into unicode + HTML
sub/superscripts rather than LaTeX. Formulas **outside** tables are emitted as
LaTeX (`$$...$$` / `$...$`) and are typeset by MathJax in the preview.

## The UI is correct

The preview renders the model's HTML faithfully: the browser shows proper
subscripts/superscripts (`π` with subscript `NL`, `e` with superscript `-x`,
etc.). Verified: the table renders with 3 columns, the `<sub>`/`<sup>` are
preserved through the sanitizer, content intact. There is nothing to "fix" in
the UI — there is no `$...$` in table cells to typeset, and the unicode+HTML
representation is legible.

## Limitation (inherent to the model, not the UI)

Two-dimensional structures (fractions, integrals) are linearized inside tables
(`\frac{1}{1+e^{-x}}` → `1 / (1 + e^-x)`), so they are readable but not
beautifully typeset. Re-parsing the linearized cell back into LaTeX would be
fragile and would diverge from the model's actual output, so we keep the model's
HTML as-is. If fully-typeset in-table math is required, that is a model/prompt
change (out of scope for inference-only), not a preview change.

Reproduce: `benchmarks/synth/table_formula.py` + `tests/test_table_formula.py`
(`pytest -m gpu` for the model check).
