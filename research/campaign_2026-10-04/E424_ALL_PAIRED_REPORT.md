# E424: eight completed architecture fits

All eight full-recipe fits, their FP32 prediction exports, and the common-FP32 paired evaluation completed successfully. No fit was repeated to obtain this report. Seed-4 combo training started after the E424 controller exited successfully.

The fixed comparison uses the 368 supplied validation sequences excluded from these final fits, grouped into 249 connected components. This is a historically exposed evaluation panel, not fresh data. Global clipped weighted Pearson is primary. All checkpoints use fixed epoch 3, the same full-data teacher lineage and fitting membership. Bootstrap intervals are conditional on the fitted seed-3 models and do not include training-seed or research-selection uncertainty.

| Architecture parent | Global WP | Difference from its parameter-budget GRU control | Paired 95% interval |
|---|---:|---:|---:|
| Learned hard specialists | 0.696446285 | +0.000353842 | [-0.001188360, +0.002638582] |
| Hash specialists | 0.696644505 | +0.000552062 | [-0.001315240, +0.003009373] |
| Uniform specialists | 0.696500615 | +0.000408171 | [-0.001032547, +0.002321731] |
| Conditional low-rank input adapter | 0.697667521 | +0.000308580 | [-0.000614844, +0.001299199] |
| Unconditional low-rank input adapter | 0.697337685 | -0.000021256 | [-0.001684901, +0.001285415] |
| Diagonal prefix modulation | 0.698127286 | +0.000768346 | [-0.000081996, +0.001731829] |

The two GRU control scores are 0.696092444 for the specialist budget and 0.697358941 for the adapter budget. The exact public-confirmed combo's equal parent/head predictor scores 0.699822159 with the same FP32 export convention. Architecture entries above are parents without the incumbent's head-fine-tuning procedure, so comparisons with the complete combo also change that procedure.

Additional mechanism contrasts were added for reporting after the automatic all-arm report was inspected. They are exploratory comparisons, not new preregistered selection gates:

| Contrast | Difference | Paired 95% interval |
|---|---:|---:|
| Learned hard routing minus uniform routing | -0.000054330 | [-0.001126962, +0.001068980] |
| Learned hard routing minus hash routing | -0.000198220 | [-0.001245168, +0.000802502] |
| Conditional low-rank minus diagonal modulation | -0.000459766 | [-0.001142758, +0.000161237] |
| Conditional low-rank minus unconditional adapter | +0.000329836 | [-0.000886793, +0.001788104] |

There is no established gain from learned hard routing, and the diagonal control has the largest observed adapter gain. Neither observation closes these mechanism families. The next already-planned intervention is soft routing, which gives every expert direct prediction gradients. Its implementation checks must pass before a scientific fit. A separate useful follow-up is a matched, fully frozen-trunk head procedure for every architecture; applying `--freeze inp,gru` blindly to specialists would leave expert recurrent cells and routing modules trainable, so that is not a valid head-only control.

Authoritative report: `checks/e424_report_all_fp32_v1.json`. Frozen screen: `checks/e424_screen_all_fp32_v1.json`. Sealed command: `specs/e424_score_all_fp32_v1.sealed.json`. Worker: `runner/attempts/e424-score-all-fp32-v1.json` (exit 0, verified teardown, 47.65 seconds including worker lifecycle, peak RAM 588,079,104 bytes, zero worker OOM events). The earlier automatic report retains its legacy baseline export and is preserved; this report uses the common-FP32 baseline instead.
