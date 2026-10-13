---
kind: review
status: complete
created_utc: 2026-10-13T12:00:00Z
updated_utc: 2026-10-13T12:00:00Z
hypothesis_ids: [H-TEAM-NATIVE-20261013]
experiment_ids: [E-TEAM-ARCHIVE-REPLAY-20261013-R1]
evidence_paths: [INVENTORY.json, public_results.json, INDEPENDENT_REVIEW_R1.md]
---

# Teammate methods, available evidence and useful next experiments

## Question or claim

What can the official team share, reproduce and combine? ADR0035 authorizes both code and weights. The accessible teammate branch is `origin/research/creative-iteration-20260930` at `833aaf541155c591e930bbc6c640b8817e01a536`, October6. A fresh remote/branch/PR/fork search did not locate the later FTT training push. This is an access/source finding, not evidence that no later work exists.

`team_source_critic` (gpt-6-luna/max) independently read the branch's source and reports. Root reconciles those findings with imported ZIPs below. The user describes a best two-GRU blend; the historical two-student archive is identifiable, while the supplied `pair_ftt2` and `xteam_h4` each execute four distinct recurrent trunks. We have not identified the latest two-model endpoint the user refers to.

## Mechanism and method

**Family distillation.** Wave18 final students use seed-specific `fbc1k_s{5,6}@ep3` parents, two-layer H256 GRUs, two epochs, alpha0.7 and four family target heads weighted0.1. The main cached teacher is a direct sum of `.30*fbc1k_s5 + .10*ffull_s3 + .10*ffull_s4 + .125*frawW_s3ft + .125*frawW_s4ft + .0625*orgbase`, whose coefficients total **0.8125**. The inspected code does not renormalize that single cached teacher; it clips teacher predictions and mixes `(1−alpha)*label + alpha*teacher`. Teacher amplitude is therefore part of the recipe, though no isolated amplitude benefit is established by those reports.

**Compact student ensemble.** Wave20's `distilled_h256_pair` is exactly0.5 each of distilled_s5/s6. Its NPZ hashes match the two student members in supplied pair_ftt2. The later package adds two distinct fttall_s5/s6 trunks at equal quarter weights. Xteam instead combines teammate distilled_s5/fttall_s5 with our exact Z4 and H390s3 payloads. Record shared weight identity rather than treating renamed checkpoints as new seeds.

**Heterogeneous older ensemble.** e6a averages six members: d256_s3h2, wm_s3h2, ow_s3h2, h128_s3h2, h128_s4h2 and c500_s3h2. Its source inventory has checkpoint paths and hashes. Width/layer, training-loss and sequence-length differences are plausible sources of diversity, but the source explicitly lacked local OOF for these final-data weights.

**Causal regime probes.** p_c3zero/only use the same four base predictors and two detectors. Detector probabilities accumulate during rows50–98; the decision precedes the first scored prediction at99. The two packages differ in class scaling. Their detector weights are absent from the old source branch but **present and inventoried in the supplied ZIPs**; original inference has passed short reset profiles and full replay is queued. Detector fitting/export ancestry is still missing. Complementary masked WP scores cannot be added to recover class weights.

## Evidence and interpretation

| Evidence | Reported finding | Scope |
|---|---|---|
| Historical two-student final archive | Public0.6674; exact NPZ identities match the supplied pair | Public transfer observation; final training used validation labels |
| Wave12 H256 distillation analogue | Teacher0.715325, direct pair0.716548, distilled pair0.718150; distilled−teacher+0.002825, conditional CI[+0.001957,+0.003836] | 425 valid sequences/123 chains; exposed fold0, different seeds3/4 and teacher lineage |
| Wave12 distilled−direct | +0.001602, CI[−0.000768,+0.004074] | Positive uncertain recipe evidence; not a universal rejection or final-s5/s6 result |
| Wave18 composite analogue | Distilled0.712620, teacher0.710877, direct0.709525; distilled−direct+0.003095, CI[+0.000508,+0.005893] | 436 valid sequences/129 chains; exposed fold0, restricted learned parents, uncertain orgbase pretraining ancestry |
| Wave15 head-FT addition | Valid+0.000635, CI[+0.000073,+0.001237]; train diagnostic−0.000648, CI[−0.001173,−0.000080] | Direction reverses between panels; complete matched mixture must be evaluated |
| Historical full4-int8 precision check | Quantized0.7023371 vs same-four FP16 0.7023454 | 860 exposed-fold sequences; not the six-member e6a endpoint |
| Supplied final endpoints | pair_ftt2 public0.6682, xteam0.6681, e6a0.6641 | User portal transcription, archive-name association without portal hash verification |

These historical source-report scores were not independently recomputed here. Their underlying fold checkpoints/predictions are absent at the referenced local locations; some reports refer to Beleriand storage. The actual final NPZ payloads from the imported ZIPs are available and are being evaluated separately. No common clean OOF bank is established for the current final weights.

## Limits and validity

Do not compare the historical0.71-scale reports directly with our current folds: membership, fitting scope, masks and ancestry differ. Existing folds are exposed. Conditional chain intervals exclude fitting variation and public-transfer uncertainty. Shared assets do not create independent replications. FTT docstrings alone do not establish its teacher/initializer lineage.

## Decision and next step

1. Finish original callback and individual-member diagnostics, preserving quantized versus FP32 endpoints. Measure correlations and fixed conditional contributions; label fitting overlap.
2. When source-qualified teammate CV artifacts become available, align source/group IDs and teacher ancestry with our evaluation bank. Rank complete fixed mixtures on multiple folds and fitting seeds.
3. If those artifacts remain unavailable, use the shared recipe/code to run matched fold-qualified reconstructions. Preserve the0.8125 teacher scale in the faithful arm; compare normalization, target amplitude and head-FT as explicit alternatives.
4. Distill a complete shared ensemble into one or two compact students, with family targets as a separately controlled component. This can capture six/eight-model diversity without paying six/eight recurrent passes on the host. It is a proposed experiment, not a measured improvement.

## Source records

All following source links pin the audited branch revision:

- [Wave18 final student lane](https://github.com/VladKochetov007/WunderFunder/blob/833aaf541155c591e930bbc6c640b8817e01a536/research/iterations/2026-10-05-wave18/final_student_lane.sh), [teacher manifest](https://github.com/VladKochetov007/WunderFunder/blob/833aaf541155c591e930bbc6c640b8817e01a536/research/iterations/2026-10-05-wave18/full-teacher-manifest.json), [composite report](https://github.com/VladKochetov007/WunderFunder/blob/833aaf541155c591e930bbc6c640b8817e01a536/research/iterations/2026-10-05-wave18/composite-report.json).
- [Wave12 lane](https://github.com/VladKochetov007/WunderFunder/blob/833aaf541155c591e930bbc6c640b8817e01a536/research/iterations/2026-10-05-wave12/compress256_lane.sh), [report](https://github.com/VladKochetov007/WunderFunder/blob/833aaf541155c591e930bbc6c640b8817e01a536/research/iterations/2026-10-05-wave12/wave-report256.json), [wave15 report](https://github.com/VladKochetov007/WunderFunder/blob/833aaf541155c591e930bbc6c640b8817e01a536/research/iterations/2026-10-05-wave15/wave-report.json).
- [Wave20 pair manifest](https://github.com/VladKochetov007/WunderFunder/blob/833aaf541155c591e930bbc6c640b8817e01a536/research/iterations/2026-10-05-wave20/candidate-manifest.json), [e6a source](https://github.com/VladKochetov007/WunderFunder/blob/833aaf541155c591e930bbc6c640b8817e01a536/research/iterations/2026-10-02-big-ensemble/README.md), [probe callback](https://github.com/VladKochetov007/WunderFunder/blob/833aaf541155c591e930bbc6c640b8817e01a536/research/iterations/2026-10-02-big-ensemble/probe/solution_probe.py).
- [Imported inventory](INVENTORY.json), [public transcription](public_results.json), [runtime reflection](EPYC_RUNTIME_REFLECTION_R1.md), [full comparison contract](CONTRACT.json).
