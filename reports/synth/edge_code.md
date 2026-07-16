# dots.mocr — synthetic edge report

## edge_code

- config: dpi=200, max_pixels=2200000, max_new_tokens=8192
- mean primary metric by kind: code=1.00
- cases: 14

### code_c

| difficulty | case | score |
|---|---|---|
| 6 | code_c_l6 | 1.00 ✅ |
| 10 | code_c_l10 | 1.00 ✅ |
| 14 | code_c_l14 | 1.00 ✅ |

**edge**: no degradation observed across the tested range.

### code_cpp

| difficulty | case | score |
|---|---|---|
| 6 | code_cpp_l6 | 1.00 ✅ |
| 10 | code_cpp_l10 | 1.00 ✅ |
| 11 | code_cpp_l11 | 1.00 ✅ |

**edge**: no degradation observed across the tested range.

### code_python

| difficulty | case | score |
|---|---|---|
| 6 | code_python_l6 | 1.00 ✅ |
| 10 | code_python_l10 | 1.00 ✅ |

**edge**: no degradation observed across the tested range.

### code_go

| difficulty | case | score |
|---|---|---|
| 6 | code_go_l6 | 1.00 ✅ |
| 10 | code_go_l10 | 1.00 ✅ |
| 12 | code_go_l12 | 1.00 ✅ |

**edge**: no degradation observed across the tested range.

### code_rust

| difficulty | case | score |
|---|---|---|
| 6 | code_rust_l6 | 1.00 ✅ |
| 10 | code_rust_l10 | 1.00 ✅ |
| 15 | code_rust_l15 | 1.00 ✅ |

**edge**: no degradation observed across the tested range.

## Findings (code listings)

`lstlisting` blocks in five languages, difficulty swept by line count.
Metrics: line recall (fuzzy line match ≥0.6) and recall of language-specific
tokens that actually appear in each snippet (case-sensitive substrings).

| language | line recall | special-token recall | distinctive tokens tested |
|---|---|---|---|
| C    | 1.00 | 1.00 | `#include`, `->`, `struct`, `{ } ;` |
| C++  | 1.00 | 1.00 | `template`, `std::`, `::`, `<int>`, `auto` |
| Python | 1.00 | 1.00 | `def`, `import`, `for … in`, `==`, `:` |
| Go   | 1.00 | 1.00 | `package`, `func`, `:=`, `range` |
| Rust | 1.00 | 1.00 | `fn`, `let mut`, `->`, `match`, `::`, `&`, `HashMap` |

**Edge**: none observed. Every code line and every distinctive syntactic
token survives OCR across all five languages and up to 15 lines. Code blocks
read as monospaced text, which the model handles reliably.
