# Fixed old/new F1 student blend (R1)

This is one predeclared 50/50 prediction blend of the new old-Z4 F1 student and the strongest existing F1 alpha-0.9/e6 student. It reuses their float local predictions; it is not a ZIP score, a quantized-quality claim, or a public-score forecast.

| Endpoint | Fold-1 WP | T0 | T1 |
|---|---:|---:|---:|
| `oldz4_h230_student_f1_s3_r4@ep1` | 0.67038 | 0.66429 | 0.67646 |
| `vt_f1_dsq_dm_t3_a9_e6@ep5` | 0.67103 | 0.66617 | 0.67589 |
| Fixed 0.5 / 0.5 mixture | **0.67182** | 0.66640 | 0.67724 |

The blend is +0.00079 WP over its stronger single and +0.00144 over the old-Z4 single on the same 457 complete fold-1 evaluation sequences. The exact scoring call used the unchanged `scripts/blend_score_remote.py` and `wnn.gmetric.GlobalAccumulator`, with the official fold-1 evaluation list and rows from index 99 onward. Both input prediction sidecars match the `valid_targets.npz` row-group ordering; each array is `(1873, 20000, 2)` float32. The sidecars bind prediction hashes `ac02c0d9…457a40ce` and `bf1d4a52…e044d0ad`, respectively.

The prediction/provenance chains support fold-1 exclusion for both endpoints. Each student's fit records 10,533 train and 1,416 eligible-valid groups, with all 457 evaluation groups excluded from its valid-fit set. The old-Z4 parent, eight named teacher sources, their initializer/resume ancestry, and fold-1 propensity source were checked against the same split. The alpha-0.9/e6 student's matched `dc_98411230` producer metadata binds its fold-1 source set and ancestry; three cache source names are aliases to exact seed-3/seed-5 checkpoint hashes, not additional checkpoint identities. Exact paths, hashes, source epochs, and checks are in `EVIDENCE_R1.json`.

An 800-resample paired complete-component bootstrap (seed 271828, 149 components, 457 sequences) estimates a difference of +0.000787. Its 95% interval is [−0.000430, +0.001915] (`PAIRED_COMPONENT_BOOTSTRAP_R1.json`). The interval includes zero; this is an uncertainty summary, not a reject rule. Fold 1 is previously exposed and was used in earlier model/blend selection, so this result is robustness evidence rather than a fresh holdout.

Both source sidecars identify `fp16` local inference and no quantization preset. No Q8/native serving parity or performance claim follows from this score. The public scores and upload runtimes are not inputs to the blend or its weights. No held-out/public arithmetic was performed.

Artifacts:

- `BLEND_SCORE_STDOUT_R1.txt`: raw output of the three-spec scoring command.
- `PAIRED_COMPONENT_BOOTSTRAP_R1.json`: fixed paired component uncertainty calculation.
- `EVIDENCE_R1.json`: endpoint, source, membership, metric, code, and resource pins.
- `COMMANDS_R1.txt`: exact score invocation and bootstrap procedure.
