# October 18: bounded local probe and B300 handoff

## Current status — research stopped

The user explicitly stopped new research admission at 2026-10-18 08:27 UTC. The local controller barrier is recorded and drained; no healthy process was killed. Do not start a fit, score, package, upload or new PR. Let the one already-admitted remote F1/seed-1001 run finish and collect its terminal artifacts only. The detailed [online-feature closeout](online_features_r1/CLOSEOUT_20261018_R1.md), [stop receipt](online_features_r1/runtime/STOP_AUTHORITY_20261018_R1.json), and [operations note](../unified/operations/RESEARCH_STOP_20261018_R1.md) supersede the active-authority statements below.

The user says both geometry ZIPs were already submitted. R2 portal ID `14s6l2z7` scored 0.66428; R3 ID `1wmshmaz` scored 0.66456 (30m32s/627MB), as recorded in the [R3 result receipt](geometry_variations_r3/PUBLIC_RESULT.json). Both trail TL3's reported 0.66832; do not re-offer either. Existing cached TL3/R3 blends are lower than TL3 on both inspected F0 and F2 panels; see the [closeout's blend analysis](online_features_r1/CLOSEOUT_20261018_R1.md#candidate-status-and-the-users-correction).

- [Quick causal geometry probe](quick_geometry_probe_r1/README.md): user narrowed local work to one short feature/control continuation and an exploratory submission ZIP within two hours. Deep repeated comparisons move to teammate 8×B300. Local outcomes and package checks are recorded here; no scientific gain is claimed before grading.

- [Frozen-backbone input adapter](geometry_adapter_probe_r2/README.md): bounded follow-up after the negative full-continuation pair. It trains only the appended 60 input columns with AdamW decay 0, preserving old weights. Profile/fit pending; no quality claim before grading.

## Published B300 handoff

[Draft PR #15](https://github.com/VladKochetov007/WunderFunder/pull/15), branch `experiment/causal-geometry-b300` into `research-team`, contains the modular four-arm 120-fit server study, actual-device profiling gate, matched seed/fold design, training/gradient/evaluation logging, causal evidence checks, first local negative result and plots. GitHub verified OPEN / draft / MERGEABLE at published head `fafe38e262992e6f62c1dbf8439e24b9bfb65943`; CI state was UNSTABLE, so mergeability does not mean all checks passed. B300 runtimes are unmeasured; the teammate runs that campaign. The local two-hour task does not run it.

## Remaining-hour variations

The user's subsequent instruction authorizes a few further local variations while preserving the ready ZIP. [R3 feature-block probes](geometry_variations_r3/README.md) compare moments-only and dictionary-only frozen adapters under the R2 budget and same exposed F0. They reuse the reviewed B300 masking code and explicit column provenance. At most one supported candidate gets a new exact native replay; the existing file remains available.

## New continuous research authority

[Online learning and feature research](online_features_r1/README.md) records the subsequent user instruction to continue autonomously overnight and beyond until explicitpause. It retains localresource caps and first audits observabledata/labels and TL3ancestry before selecting bounded experiments. Promising larger directions require runnable8×B300draftPRhandoffs; no newpaidcompute authorized. The attempted input-only innovation-clock screen stopped at an OOM in an unbounded train-file footer parse; its exact scope and no-retry disposition are in the [incident receipt](online_features_r1/runtime/attempts/innovation-clock-train-metadata-20261018-r1.json).

The [warmup-only expert router screen](online_features_r1/warmup_expert_gate_r1/README.md) is complete. Its input-only gate stayed near uniform and underperformed the training-selected parent on all four held-out component folds, so it did not earn a separate B300 draft. The result does not close broader causal online learning or the existing Kalman teacher/student handoff.

[TL3/R3 F2 fixed-mixture robustness replay](online_features_r1/tl3_fixed_blend_f2_r1/README.md) completed exact-ZIP, label-blind prediction replay plus a fixed-weight grade on 443 sequences/144 exposed components. TL3 scored 0.723887 versus 0.715413 for R3, 0.722678 for 75/25 and 0.720950 for 50/50; conditional intervals are below zero. It repeats the F0 ordering for this exact pair, but unknown TL3 ancestry and exposed F2 prevent clean-CV or transfer claims. The scorer repair and failed first attempt are recorded; no new B300 direction was established.

The [causal Kalman output-filter screen](online_features_r1/kalman_output_filter_r1/README.md) completed on 468 exposed F0 groups. Both fixed and prefix-estimated output filters lost to raw predictions in all four descriptive component partitions; the conditional intervals are wholly negative. This narrows only output smoothing. The larger learned Kalman feature/teacher/student direction remains in draft PR #17; this inference-only screen adds no new B300 proposal.

The [early-to-late expert preference diagnostic](online_features_r1/sequence_oracle_r1/README.md) completed on 468 exposed F0 groups/141 components. The early label-informed selector scores 0.693871 late versus 0.697669 for TL3 and 0.694457 for fixed 75/25; observed choice agreement (68.16%) is below the marginal-rate expectation. Independent source and result reviews pass, and profile/grade CPU receipts show zero OOM deltas. This does not justify a new B300 router draft or close broader online learning; the separate retained-raw Kalman teacher/student handoff remains in draft PR #17.

The [groupwise fast/slow Kalman screens](online_features_r1/README.md) include a negative additive ridge proxy (raw+39 states Δ −0.013093 WP) and a two-seed matched recurrent follow-up on exposed F0. Seed3: active0.67986413 vs zero0.67981250 (Δ+0.00005163; conditional 141-component interval [−0.00013143,+0.00027018]); seed4: active0.68037958 vs zero0.68039346 (Δ−0.00001387; interval [−0.00017548,+0.00012404]). The descriptive mean delta +0.00001888 is not a reliable estimate of training variation. Both are incremental fine-tunes over a common parent with mixed position effects; they establish neither gain nor family-level failure. See the [recurrent packet](online_features_r1/kalman_recurrent_groupfast_slow_r1/README.md), [seed-3 result](online_features_r1/kalman_recurrent_groupfast_slow_r1/RESULTS_R1.md), [seed-4 result](online_features_r1/kalman_recurrent_groupfast_slow_r1/RESULTS_SEED4_R1.md), and the independent [ridge](../unified/reviews/GROUPWISE_FAST_SLOW_PROXY_RESULT_REVIEW_20261018_R1.md) and [recurrent](../unified/reviews/GROUPWISE_FAST_SLOW_RECURRENT_SCREEN_REVIEW_20261018_R1.md) reviews. The feature append itself does not warrant another B300 draft; a separate innovation-conditioned latent gain/update mechanism is being prepared alongside existing PR #17.

- [All-channel causal Kalman innovation clock](online_features_r1/innovation_clock_r1/README.md): completed a label-free, 32-component fit-side screen over 640,000 rows under a measured 1.5 GB reservation. Innovation magnitude rises across sequence bins and varies by semantic group, while the scalar CUSUM upper tail reaches its cap early. No target association or endpoint score was measured; iterate to group-aware fast/slow features. Existing broader innovation/Kalman B300 handoffs remain #14 and #17.

## Parallel research-team index snapshot — 18 October

These dated links preserve separate entries from `research-team`; the stop and
closeout above govern their historical scope and do not imply current authority.

- [Ranked25 execution](priority25_execution_r4/README.md): user-authorized ordered Beleriand/DataSphere campaign, one Beleriand GPU, initial matched labels-only architecture comparison.
- [Exploratory F0 submission](f0_submission_r1/README.md): requested package from the completed F0 pair. A fixed75/25 wide/deep mixture led the five-point exposed-fold screen at0.673850944; the native Q8 full-F0 replay was pending in this snapshot.
- [F0 residual diagnostic](quality_gap_r1/README.md): source-bound descriptive decomposition of existing wide/deep/FP32-mix/Q8 scored rows, with no training or incumbent-gap causal claim.
