# Frozen D1 TRAIN position exposure

Support gate **passes on all 16 selected components** (320,000 rows). Durable exit 0; 1.608 seconds, 1015.50 MiB peak RSS, one CPU, hard 2 GiB, shared campaign slice, 300-second cap. This is feature support evidence, not a model score.

## Selection and access

Metadata-only salted hashing chose one group per canonical connected component from fold 0 allowed fitting TRAIN groups. Components containing any prior inspected 64 fitting or 32 test group were excluded. The old chain IDs and current canonical component IDs differ; exclusions were converted through row-group membership. No feature, target or mask informed selection.

Launch followed independent D1 numerical v2 PASS and amended builder integration CPU proof PASS, with exact evidence hashes recorded in launch_authorization.json. Read columns were only i0_p0..21 and a6 in TRAIN. No targets, masks, VALID, external data, competition scoring or GPU execution.

Class diagnostics use the existing TRAIN-only current-code map with unknown flag. All 64 historical map-fitting TRAIN groups are allowed by fold 0. The table is historically label-derived and diagnostic only: no refit, oracle class, sample stratification or D1 inference use. Unknown rows remain in every support gate.

## Frozen support result

Every component independently has rank 96 for actual written and nonempty-history queried Walsh inputs under row, endpoint and raw-code clocks. Every event clock has at least one complete 128-write ordinal cycle. Uncentered 96x96 Grams are stored in grams.npz; eigenvalue threshold is max(1e-8, 1e-8*maximum eigenvalue). The legitimate constant column is retained. No sample replacement, period change or post-result threshold adjustment.

| Clock | Total writes | Writes per component | Lowest written Gram eigenvalue | Max distinct ordinals aliasing one modulo position |
| --- | ---: | ---: | ---: | ---: |
| row | 320,000 | 20,000–20,000 | 19968 | 157–157 |
| endpoint | 46,306 | 178–6,579 | 128 | 2–52 |
| code | 77,540 | 337–10,127 | 256 | 3–80 |

The 128-period position control aliases every row ordinal with ordinal+128. Each 20k row timeline reaches 157 distinct row ordinals per modulo address; endpoint writes reach up to 52 and raw-code writes up to 80. Full rank does not remove this aliasing or establish equivalent information to content addressing. Query histogram repetitions from quiet rows are reported separately from distinct ordinal aliases.

Endpoint writes total 46,306, raw-code writes 77,540; 6 actual endpoint origins and 1 nonorigin endpoint without raw-code change. Raw-code origins are suppressed because prior prices are unknown. No artificial event is injected.

Both clocks have initial empty-history prefixes; the longest is 2,569 rows. No component has an entirely empty event history. This confirms that training/inference must process all quiet warmup rows while leaving event S/Z empty until the first actual gate. The support test does not identify physical elapsed time or quote movement on unknown transitions.

## Per-component evidence

| TRAIN group | Component | Mapped class set | Unknown rows | Endpoint writes | Code writes | Initial empty endpoint rows | Gate |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 3045 | 1658 | 0 | 3007 | 3493 | 6574 | 11 | pass |
| 3982 | 942 | 3 | 1 | 917 | 1537 | 45 | pass |
| 5234 | 2163 | 3 | 0 | 6526 | 10127 | 0 | pass |
| 3322 | 285 | 2 | 23 | 1791 | 3326 | 35 | pass |
| 10086 | 2753 | 3 | 3 | 3815 | 5928 | 0 | pass |
| 6668 | 30 | 3 | 0 | 3842 | 6144 | 2 | pass |
| 9093 | 461 | 2 | 0 | 178 | 337 | 2569 | pass |
| 1381 | 997 | 2 | 0 | 885 | 1635 | 298 | pass |
| 10528 | 572 | 0 | 1218 | 6579 | 9919 | 0 | pass |
| 2506 | 633 | 2 | 24 | 652 | 1181 | 5 | pass |
| 5072 | 1655 | 1 | 126 | 3382 | 6054 | 23 | pass |
| 8949 | 1734 | 2 | 2 | 2010 | 3377 | 9 | pass |
| 9590 | 1430 | 2 | 309 | 1432 | 2610 | 0 | pass |
| 1382 | 998 | 2 | 6 | 2392 | 3984 | 0 | pass |
| 7219 | 1933 | 0 | 2784 | 5808 | 9654 | 0 | pass |
| 9288 | 544 | 0 | 951 | 2604 | 5153 | 40 | pass |

Current-code class mapping covers 311,546/320,000 rows (97.35813%); 8,454 unknown rows (2.64188%) are retained. Four components have mapped class0, one class1, seven class2 and four class3; these are mapped row diagnostics, not oracle labels or whole-dataset coverage.

| Mapped row class | Rows | Endpoint writes | Code writes |
| --- | ---: | ---: | ---: |
| 0 | 72,040 | 16,140 | 27,474 |
| 1 | 19,874 | 3,303 | 5,955 |
| 2 | 139,636 | 9,265 | 16,311 |
| 3 | 79,996 | 15,096 | 23,732 |
| unknown | 8,454 | 2,502 | 4,068 |

## Interpretation and replay

This fresh feature panel passes the preregistered position exposure gate. It does not independently reproduce the earlier target-derived decoded movement relation, prove physical endpoint meaning, establish forecasting gain, or authorize GPU fitting. Endpoint/raw-code comparisons change both writes and decay; content/position comparisons remain against a periodic position control. Independent raw replay is requested.

Immutable selection, registration, source hashes, authorization, JSON, Grams, logs, timestamps and exit are linked by artifact_manifest.json. command.txt records the original durable invocation. Independent replay should copy the frozen sources and authorization into a new directory, preserve ROOT resolution or explicitly supply the repository root, and write separate evidence. Do not overwrite the frozen selection/results.
