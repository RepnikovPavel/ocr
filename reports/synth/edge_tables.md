# dots.mocr — synthetic edge report

## edge_tables

- config: dpi=200, max_pixels=2200000, max_new_tokens=8192
- mean primary metric by kind: table=0.98
- cases: 25

### tables

| difficulty | case | score |
|---|---|---|
| 2 | tbl_grid_r6_c2 | 1.00 ✅ |
| 3 | tbl_grid_r6_c3 | 1.00 ✅ |
| 3 | tbl_booktabs_r6_c3 | 1.00 ✅ |
| 3 | tbl_grid_r6_c3_txt | 1.00 ✅ |
| 4 | tbl_grid_r6_c4 | 1.00 ✅ |
| 4 | tbl_grid_r3_c4 | 1.00 ✅ |
| 4 | tbl_grid_r8_c4 | 1.00 ✅ |
| 4 | tbl_grid_r12_c4 | 1.00 ✅ |
| 4 | tbl_grid_r16_c4 | 1.00 ✅ |
| 4 | tbl_grid_r20_c4 | 1.00 ✅ |
| 4 | tbl_grid_r25_c4 | 1.00 ✅ |
| 4 | tbl_grid_r6_c4_mc | 1.00 ✅ |
| 5 | tbl_grid_r6_c5 | 1.00 ✅ |
| 5 | tbl_booktabs_r6_c5 | 1.00 ✅ |
| 5 | tbl_grid_r6_c5_mr | 1.00 ✅ |
| 5 | tbl_grid_r8_c5_txt | 1.00 ✅ |
| 6 | tbl_grid_r6_c6 | 1.00 ✅ |
| 6 | tbl_grid_r8_c6_mc_mr | 1.00 ✅ |
| 6 | tbl_grid_r12_c6_txt | 1.00 ✅ |
| 8 | tbl_grid_r6_c8 | 1.00 ✅ |
| 8 | tbl_booktabs_r6_c8 | 1.00 ✅ |
| 10 | tbl_grid_r6_c10 | 1.00 ✅ |
| 12 | tbl_grid_r6_c12 | 1.00 ✅ |
| 16 | tbl_grid_r6_c16 | 0.88 ⚠️ |
| 20 | tbl_grid_r6_c20 | 0.70 ⚠️ |

**edge**: degrades starting at difficulty `16` (tbl_grid_r6_c16, score 0.88).
