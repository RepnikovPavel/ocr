# dots.mocr — synthetic edge report

## edge_formulas

- config: dpi=200, max_pixels=2200000, max_new_tokens=8192
- mean primary metric by kind: formula=0.96
- cases: 20

### frac_nest

| difficulty | case | score |
|---|---|---|
| 1 | frac_d1 | 1.00 ✅ |
| 2 | frac_d2 | 1.00 ✅ |
| 3 | frac_d3 | 0.99 ✅ |
| 4 | frac_d4 | 0.50 ❌ |
| 5 | frac_d5 | 0.98 ✅ |
| 6 | frac_d6 | 0.92 ✅ |

**edge**: degrades starting at difficulty `4` (frac_d4, score 0.50).

### matrix

| difficulty | case | score |
|---|---|---|
| 2 | matrix_n2 | 1.00 ✅ |
| 3 | matrix_n3 | 1.00 ✅ |
| 4 | matrix_n4 | 1.00 ✅ |
| 5 | matrix_n5 | 0.96 ✅ |
| 6 | matrix_n6 | 0.97 ✅ |

**edge**: no degradation observed across the tested range.

### polynomial

| difficulty | case | score |
|---|---|---|
| 3 | poly_t3 | 1.00 ✅ |
| 5 | poly_t5 | 1.00 ✅ |
| 8 | poly_t8 | 1.00 ✅ |
| 12 | poly_t12 | 1.00 ✅ |

**edge**: no degradation observed across the tested range.

### sum_int

| difficulty | case | score |
|---|---|---|
| 1 | sum_int_1 | 1.00 ✅ |
| 2 | sum_int_2 | 1.00 ✅ |
| 3 | sum_int_3 | 1.00 ✅ |

**edge**: no degradation observed across the tested range.

### deep_nest

| difficulty | case | score |
|---|---|---|
| 3 | nested_sub_1 | 0.89 ⚠️ |

**edge**: degrades starting at difficulty `3` (nested_sub_1, score 0.89).

### mixed

| difficulty | case | score |
|---|---|---|
| 2 | mixed_1 | 1.00 ✅ |

**edge**: no degradation observed across the tested range.

## Findings (formulas)

Primary metric is normalized LaTeX token recall (braces treated as grouping
syntax, not content). For nested fractions token recall is a weak signal —
the tokens repeat, so a structurally wrong reconstruction still scores high;
the honest structural signal there is the **count of `\frac`** the model emits
vs the ground truth.

| family | verdict |
|---|---|
| polynomial (3..12 terms) | perfect (1.00) |
| sum / integral / limits | perfect (1.00) |
| matrix 2..4 | perfect (1.00); 5×5, 6×6 slip to ~0.96 |
| Bayes / mixed frac+sum | perfect (1.00) |
| nested subscripts `a_{i_{j_{k}}}` + nested sqrt | mild degradation (0.89) |
| **nested fractions** | **the edge** |

Nested fraction depth vs recovered `\frac` count (ground truth → detected):

| depth | fracs expected | fracs detected | token recall |
|---|---|---|---|
| 1 | 1 | 1 | 1.00 |
| 2 | 3 | 3 | 1.00 |
| 3 | 7 | 6 | 0.99 |
| 4 | 15 | 7 | 0.50 |
| 5 | 31 | 28 | 0.98 |
| 6 | 63 | 305 | 0.92 |

**Edge**: nested fractions are exact to depth 2 (3 fractions). From depth 3 the
model starts under-counting; by depth 4 it recovers only ~half the structure;
at depth 6 it loops and hallucinates repeated fractions (305 vs 63). Everything
else in the tested range — polynomials, sums, integrals, matrices up to 6×6 —
is read accurately.
