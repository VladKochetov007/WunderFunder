# Exception and clock-reconstruction audit

## Finding

The strict claim that both `i0_p0` and `i0_p11` must change on every exact target-transition row is false: the full train+valid scan reproduces 1,253 raw target-pair changes outside the simultaneous `i0` price-change clock, including 92 rows in the official valid scoring mask. Those are real-sized label changes, not float jitter. At the same time, they are rare, closely bracketed by the `i0` AND clock, and carry very little clipped-absolute-target weight in the selected set. The evidence supports a near-sufficient event clock for compression and forecasting research, not an exact law and not a reason to call the residuals corrupt.

No model was trained, fitted, or run for inference. The reconstruction statistics below use labels retrospectively and are not legal predictor inputs or estimates of score change.

## Census and identity

The audit read the two target columns, four selected price-code columns, `seq_ix`, `step_in_seq`, `need_prediction`, and the valid-only `is_scored` mask, one 20,000-row Parquet row group per sequence. It verified constant `seq_ix`, steps 0–19,999, the 99-row warmup mask, valid scoring-mask inclusion within `need_prediction`, finite values, schema widths (117 train / 118 valid), and the R1 byte and row-group counts. This is a four-price-column census, not an all-112-feature census.

The full source files were hashed and remained stable during hashing. Train SHA-256 is `4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81`; valid SHA-256 is `2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782`. R1 did not record a Parquet content digest or file mtime, so the new hashes cannot establish equality to an unavailable R1 digest. Paths, byte sizes, row-group counts, schema dimensions, and invariants were checked against the R1 receipt.

The scan covered 12,480 sequences and 249,600,000 rows. `need_prediction` selected 248,364,480 rows (row 0 has no transition). There were 33,278,744 raw exact any-target transitions in those eligible rows. The exception count was 1,253: 941/211,089,907 train eligible transitions and 312/37,274,573 valid eligible transitions. That is 0.0005045% of eligible rows and 0.0037652% of eligible raw target-transition rows. Valid had 3,528,273 officially selected rows, including 418,385 raw exact target-transition rows; 92 exceptions are selected (0.002608% of selected rows, or 0.02199% of selected target-transition rows). Train has no official scoring mask; its scored counts are intentionally reported as not applicable.

| Split | Rows | Need rows | Raw any-target transitions | Clipped any-target transitions | Raw events outside i0 AND | Clipped events outside i0 AND | Officially selected exceptions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 212,140,000 | 211,089,907 | 26,810,254 | 26,685,855 | 941 | 878 | not applicable |
| Valid | 37,460,000 | 37,274,573 | 6,468,490 | 6,450,808 | 312 | 289 | 92 raw / 85 clipped |

The class labels below are the archived R1 row-group cache and are retrospective strata only. They are not inputs, do not establish an online class, and should not be generalized as class-conditional rates without sequence-level uncertainty analysis. Class 1 contains 675/941 train exceptions and 283/312 valid exceptions; 86/92 valid selected exceptions are class 1. Row group 528 has cached class `-1` and zero exceptions. Exceptions occur in all twenty 1,000-step position bins: bin counts range 34–60 in train and 8–23 in valid. There is no obvious end-of-sequence pile-up in these bins. Train exceptions occur in 688/10,607 sequences (maximum 10 in one sequence); valid exceptions occur in 203/1,873 sequences (maximum 5). The largest single-sequence shares are 1.06% train and 1.60% valid. These are descriptive counts, not IID confidence intervals.

## Event-clock confusion

The confusion tables use raw exact `t0` or `t1` changes as the positive event. They count only transition rows within the stated mask; the table cells sum to the corresponding denominator. `i0 AND` means both selected `i0` price columns changed on the same row. `i0 OR` and `any four` are wider clocks, so their increased recall comes with more false positives.

| Valid scored clock | TP | FP | FN | TN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| i0 AND | 418,293 | 307,840 | 92 | 2,802,048 | 57.61% | 99.9780% |
| i0 OR | 418,309 | 408,304 | 76 | 2,701,584 | 50.61% | 99.9818% |
| i1 AND | 69,659 | 628,544 | 348,726 | 2,481,344 | 9.98% | 16.65% |
| i1 OR | 81,610 | 688,449 | 336,775 | 2,421,439 | 10.60% | 19.51% |
| Any of four price columns | 418,335 | 997,186 | 50 | 2,112,702 | 29.55% | 99.9880% |

Among all 1,253 exception rows, 451 had a one-sided i0 price change, 316 had an i1 OR change, and 649 had at least one of the four selected price columns change; these sets overlap. The other 604 had no change in any of those four selected price columns on the exception row. For the 92 officially selected exception rows, those counts are 16, 31, and 42 respectively, leaving 50 with none of the four price changes on that row. This does not say whether another input feature changes; no other feature columns were read.

## Magnitude, clipping, quantization, and persistence

Exact raw exception counts are unchanged when a 1e-6 or 1e-4 threshold is applied. Every nonzero raw target delta recorded on an exception row is at least 0.125432968, far above either threshold. In the valid split, the per-target raw absolute-delta medians are 0.24132973 and p90 values 0.723989; in train, both medians are 0.468536 and p90 values 1.206649. Raw train maxima are 68.875 (`t0`) and 57.161 (`t1`), while valid maxima are 2.172 and 2.896. These maxima describe raw target changes, not the clipped scoring weight. Clipping reduces raw exception rows from 941 to 878 in train, 312 to 289 in valid, and 92 to 85 in valid scoring.

The archived H-002 class-specific quanta (rounded hypotheses 0.12543, 0.24133, 0.46854, 0.65687) were checked without refitting them. Across the raw exception target-delta values, 971/976 `t0` changes and 970/976 `t1` changes are within 1e-4 of an integer multiple of the corresponding candidate quantum; at 1e-6 the counts are 679/976 and 658/976. This supports a quantized/economic-sized interpretation for most changes, but the rounded candidate quanta are not proven physical ticks and the residuals are not all float-exact.

Every exception has both a previous and a next `i0 AND` event in the same sequence. Distances from exception row to previous AND are 1–3 rows in train (median 1) and 1–2 in valid (median 1); distances to the next AND are also 1–3 in train and 1–2 in valid. For the 92 scored exception rows, both distances are 1–2 (median 1). The next-event distance is retrospective and is reported only to describe persistence, never as a predictor.

The label-oracle reconstruction freezes each clipped target at its value on the most recent `i0 AND` row, then compares that value with the actual clipped target. For valid scored rows it disagrees on 61 `t0` and 59 `t1` target-rows under AND reconstruction; need-window mismatch counts are 225 and 219. Mismatch episode lengths, counted within the need mask, have median 1 and maximum 2 in valid; train maxima are 3. The mismatch mass on scored rows is 45.518 (`t0`) and 50.086 (`t1`), equal to 0.001874% and 0.002030% of the corresponding official selected clipped-absolute-target mass. The summed absolute label distortions are 23.793 and 26.541, and squared distortions 14.152 and 21.460. An i0 OR reconstruction control has 48/46 scored target-row mismatches and mass shares 0.001458%/0.001565%, with lower absolute distortion; it is still an oracle because it reads the actual label at each clock event.

The current valid selected total clipped-absolute-target mass is 2,428,465.746 (`t0`) and 2,466,842.262 (`t1`), combined 4,895,308.008. The actual clipped-absolute-target mass on raw exception rows is 71.790 and 83.490 (target-wise shares 0.002956% and 0.003385%, combined 0.003172%). These are metric-weight mass shares, not a recomputed Pearson score difference or a bound on what a trained predictor could recover. They make the exception pathway a low-mass refinement opportunity while leaving the exact simultaneous-AND law falsified.

## Interpretation and limits

The strongest practical interpretation is a near-sufficient, low-compute event clock with rare short-lag exceptions. The timing pattern is consistent with asynchronous feature/target clocks or boundary alignment, but this audit cannot distinguish target-generation horizon, source corruption, rounding, or another mechanism. The raw magnitudes and candidate-quantum alignment argue against dismissing all exceptions as floating noise; their scarcity, short persistence, and tiny selected target-weight mass argue against letting them dominate the modeling plan.

This run did not inspect other feature groups, test a predictor, recompute the pooled clipped-absolute weighted Pearson metric, estimate score gains, infer latent market states, or use external market data. It did not treat row-level counts as independent observations. The source scan never crosses sequence boundaries; retrospective `next i0 AND` and target-clock reconstruction are descriptive only. The label and class cache are never model inputs.

## Operational record

The corrected full pass ran on Beleriand with one Python CPU thread and PyArrow CPU/IO thread counts set to 1; environment thread caps were OMP/OpenBLAS/MKL/NumExpr=1. It took 485.52 s, reached 1,216,840 KiB maximum RSS, completed all splits, saved per-split checkpoints, and exited successfully. The actual final JSON writer was first tested using one synthetic record in a temporary directory; the results, JSONL, and manifest files round-tripped and their hashes verified, with zero training, optimizer, or inference steps.

An earlier complete scan computed the 1,253 exception records but failed at final JSON serialization because of a variable-name typo. Its records and failure receipt are retained as `attempt1_failed_finalization_exception_records.jsonl` and `attempt1_failed_finalization.log`; the corrected second pass produced the authoritative aggregates and matching records. No dataset or cache files were modified.
