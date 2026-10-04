# Scores on the supplied validation file

E459 pools all required scored rows from **datasets/valid.parquet**, using the official clipped global weighted Pearson metric. This is the entire supplied file, not the 20-sequence packaging subset.

| Fixed predictor | All 1,873 validation sequences | 368 sequences excluded from final fit | 1,505 validation sequences used for fitting | Recorded public score |
|---|---:|---:|---:|---:|
| Combo seed3, equal parent/head | 0.700503744 | 0.699822159 | 0.700631781 | 0.6655 |
| Previous D256, equal parent/head | 0.700792587 | 0.697777394 | 0.701456363 | 0.6641 |

The full file has 37,460,000 rows; 3,528,273 satisfy `need_prediction AND is_scored`. The combo's full-file t0/t1 scores are 0.696316655 / 0.704690833. Its excluded-portion t0/t1 scores are 0.687627604 / 0.712016713.

These are cached prediction exports with verified checkpoint and prediction hashes. The combo uses FP32 exports; the legacy D256 export did not record precision. Their exact public native callback arithmetic can differ slightly. E460 is replaying the actual public-confirmed i16 ZIP over the entire validation file; no result is yet available. Its output will be `checks/e460_native_full_validation_v1.json`.

Most supplied validation sequences were used for final fitting, so their aggregate score is descriptive. The excluded 368 sequences are historically exposed, not an untouched test. The combo is about 0.000289 lower over all supplied validation, but about 0.002045 higher on the excluded portion. The independent public endpoint improves by 0.0014. Do not tune the fixed submission using these reporting-only results.

E410 measured the **recipe's fold-specific models**, not the final full-data checkpoint, under the same equal parent/head procedure:

| CV fold | Combo | Matched control | Difference |
|---|---:|---:|---:|
| 0 | 0.682068790 | 0.677455379 | +0.004613411 |
| 1 | 0.668136392 | 0.667325756 | +0.000810637 |
| 2 | 0.705765631 | 0.704782543 | +0.000983088 |
| 3 | 0.711099016 | 0.708107409 | +0.002991607 |
| Pooled | **0.693658483** | **0.691244104** | **+0.002414379** |

The pooled paired component-bootstrap interval is [+0.001565628, +0.003448539], conditional on fitted seed3 models. Existing folds are exposed, and teacher-lineage/selection limitations remain; this does not include training-seed uncertainty.

Reproduction: `scripts/validation_scorecard.py`; sealed inputs/commands in `specs/e459_validation_scorecard_v1.sealed.json`; output `checks/e459_validation_scorecard_v1.json`. CV source: `checks/combo_merged_all_folds_report.json` and `E410_complete_predictor.md`.
