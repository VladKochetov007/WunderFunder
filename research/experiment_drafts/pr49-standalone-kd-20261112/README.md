# PR49 teacher to K16 GRU: standalone label/KD comparison

## 1. Status, lineage and owner

E-PR49-STANDALONE-KD-20261112 / H-PR49-STANDALONE-KD-20261112. Root/SemyonEpanov owns four newly admitted L4 fits. This is a separate source-qualified successor to the rejected KD configurations in [PR95](https://github.com/VladKochetov007/WunderFunder/pull/95). Canonical implementation commit is `33542502a1404a2843359b3999fa34f3a157bf3f`; the earlier active label/since source remains unchanged.

## 2. Mechanism

A clean PR49 FBC teacher may supply a smoother causal regression target than labels alone. Compare label WMSE with label WMSE plus 0.2 times teacher WMSE. Both numerators share clipped-label magnitude weights and the same globally reduced optimizer-window denominator. Raw student z is monitored and exported. No reference blend, hints, parent initialization or since weighting is enabled. The alternative is that the teacher transfers its bias without useful student improvement; the paired zero-KD control distinguishes this.

## 3. Compute advantage

Reuse the already trained teacher and admitted fitting-only cache; no new teacher fit is charged. Four cheap GRU students use L4 GPUs0-3 after the preceding fixed-E6 exports. Four simultaneous actual30-update profiles took3.64-3.87s, reserved1.78GB each, and passed exact within-profile model/optimizer/scheduler/RNG reload. Pure-update extrapolation is about78-83min for38,760 updates; full validation and I/O add time. This is an estimate, not a completion promise or public-score forecast.

## 4. Frozen comparison

GRU256x2, causal input stem16, cold initialization, FP32, AdamW LR0.0003, weight decay0.01, clip1, batch32, chunk/gradient horizon1000, accumulation1;5% warmup then warmup_cosine_v2 with0.1 floor. Fixed6epochs, final index5; all epoch checkpoints retained. Full official valid and a fixed32-member train-eval cohort are monitored each epoch, eval batch32. Masks/populations differ, so their numeric gap is not an overfitting estimate.

## 5. Bounded panel

Exactly four new scientific fits: seeds203501/203502, each with lambda0 and lambda0.2. Within each seed only kd_weight differs. Both arms require the identical teacher cache, including lambda0. Profiles are not scientific fits. Source, configuration and unchanged original7h-fit/8h-collection clock are pinned in the copied manifests. No old fit is resumed, no blind retry is allowed, and only owned process families may be stopped.

## 6. Validation and uncertainty

Official source boundary:10,325 train-only fitting sequences,282 linked train sequences excluded,1,873 valid sequences /3,528,273 scored rows. Split SHA-256 is `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`. Metric is pooled `global_wp_clip2_v1`. Valid is development-exposed. Report every scheduled-E6 seed-paired delta; two seeds do not establish significance or robust training variance. No valid-based checkpoint selection is permitted.

## 7. Leakage and coverage

Actual cache manifest is `138ef43c6ef03755a3b6195ca83f2c346f7dca4e3fe365f9e8e151ad2c71792d`. Its raw teacher checkpoint is `759165147d99b33034b74f4af63a30e8a202dafc78fd5e5699d05b6cfe9bef90`, logically aliased to the official E3 checkpoint84096689... using the preserved independent alias receipt. This is checkpoint-field equivalence, not bitwise prediction equality. Native canonical build_data verified exact train-only membership and cache file hashes before profiles; validation is absent from teacher/normalizer ancestry. seq_ix is not interpreted as time.

## 8. Decision-changing result

Use a descriptive mean paired gain of0.001 WP at fixed E6, with both deltas positive and no serving-cost increase, as the useful follow-up target declared before any new KD endpoint exists. Inconclusive or mixed results remain evidence for targeted diagnosis. A zero-crossing interval alone does not close the direction. Actual fixed-mixture performance and CPU serving cost must be measured before promotion; no public gain is inferred from a local delta.

## 9. Execution

Native root is `/data/lc-epanov_s/tabdata-final-quality-20261112-r1`. New source lives only under `kd-intake-r2/source-kd-r1`; the frozen packet is `kd-intake-r2/pr49-standalone-kd-r2`. Existing persistent SSH RPC was reused. The copied controller accepts `profile-newkd` and `fit-newkd`; full fitting requires root-approved exact profile receipts. Do not relaunch it from this note. Full-fit controller PID3962872 was detached at epoch1794510811.95; actual child progress must be checked separately from launch evidence.

## 10. Evidence and checks

[Evidence index](evidence/README.md) includes the native cache-consumer gate, four completed engineering profiles, CPU checks and source-capsule identity. [Frozen configs](configs/README.md) bind the four treatments; [controllers](controllers/README.md) explain source freezing, profiling and admission. New CPU tests passed17/17. The bounded combined suite passed71 tests; the sole frozen legacy-config identity failure reproduces unchanged on detached base390c846. The structure checker reports two existing size warnings in unchanged sequence_config.py, versus three on the base. Final source Ruff checks pass. No cache payloads, raw data or weights are copied into this PR.

## 11. Results and trace

At admission, scientific KD endpoints were unmeasured. The native profiles have zero errors and the declared fitting/valid membership, source, teacher and reload identities. The earlier rejected KD plans remain preserved in PR95. The independent reviewer found no code blocker to actual cache preflight/profiles; this is not scientific approval of a gain. Root's final formatting/import-only successor commits preserve the numerical mechanism. Final scores, checkpoints, prediction manifests and every failure must be appended rather than replacing this admission record.
