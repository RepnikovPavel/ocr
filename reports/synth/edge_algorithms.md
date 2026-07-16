# dots.mocr — synthetic edge report

## edge_algorithms

- config: dpi=200, max_pixels=2200000, max_new_tokens=8192
- mean primary metric by kind: algorithm=1.00
- cases: 12

### algo_algorithmic

| difficulty | case | score |
|---|---|---|
| 0 | algo_algorithmic_x0_n1 | 1.00 ✅ |
| 3 | algo_algorithmic_x3_n2 | 1.00 ✅ |
| 8 | algo_algorithmic_x8_n2 | 1.00 ✅ |
| 15 | algo_algorithmic_x15_n2 | 1.00 ✅ |

**edge**: no degradation observed across the tested range.

### algo_algpseudocode

| difficulty | case | score |
|---|---|---|
| 0 | algo_algpseudocode_x0_n1 | 1.00 ✅ |
| 3 | algo_algpseudocode_x3_n2 | 1.00 ✅ |
| 8 | algo_algpseudocode_x8_n2 | 1.00 ✅ |
| 15 | algo_algpseudocode_x15_n2 | 1.00 ✅ |

**edge**: no degradation observed across the tested range.

### algo_algorithm2e

| difficulty | case | score |
|---|---|---|
| 0 | algo_algorithm2e_x0_n1 | 1.00 ✅ |
| 3 | algo_algorithm2e_x3_n2 | 1.00 ✅ |
| 8 | algo_algorithm2e_x8_n2 | 1.00 ✅ |
| 15 | algo_algorithm2e_x15_n2 | 1.00 ✅ |

**edge**: no degradation observed across the tested range.

## Findings (algorithms / pseudocode)

Three LaTeX pseudocode dialects, difficulty swept by extra body lines
(0/3/8/15) and nesting depth. Metrics: recall of the control keywords that
actually render for each case (whole-word) and recall of procedure/variable
identifiers.

| dialect | keyword recall | identifier recall |
|---|---|---|
| algorithmic (uppercase \STATE/\WHILE, diploma style) | 1.00 | 1.00 |
| algpseudocode (algorithmicx) | 1.00 | 1.00 |
| algorithm2e (ruled/vlined) | 1.00 | 1.00 |

**Edge**: none observed. Across all three dialects and up to 15 extra body
lines with nested IF/WHILE blocks, every keyword and every identifier (lo, hi,
mid, pivot, theta, epsilon, lambda, grad, ...) is recovered. Pseudocode
reads as ordinary text/list layout, which the model handles reliably — this
predicts the diploma's 15 algorithmic sections should parse cleanly.
