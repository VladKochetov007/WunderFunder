# H512 replacement in the complete TL3 endpoint

**Status: NOT_ADMITTED.** This packet prepares a code path and fixed screen; it
does not fit a model, grade a candidate, or admit a submission. Root must first
accept the H512 CP5 terminal receipt and review each package/replay result.

The screen asks whether an H512-trained plain student can replace TL3's
`weights_s5.npz` member while keeping the other two members, their coefficients,
the original native library, and wrapper fixed. It compares the cached original
TL3 endpoint with two exact packages: the admitted H512 CP5 checkpoint and the
predeclared equal mean of CP3/CP4/CP5. No model is fitted here and no blend or
epoch is searched.

## Fixed evaluation

The shared [TL3 F0 replay](../../../campaign_2026-10-18/online_features_r1/tl3_replay/README.md)
supplies feature-only input validation, the native callback, full-sequence
replay, and A-B-A state-reset checking. The new replay script writes its own
endpoint receipts so the legacy receipt's original-TL3 label is never reused.
It verifies the candidate ZIP and build receipt before and after replay, and
requires the package's H512 checkpoint and provenance to match the accepted
CP5/SWA345 source receipt. The `production_candidate` artifact label by itself
does not pass this gate.

The grader checks the frozen 468-group, 141-component F0 panel and 808,863
selected rows. It maps the pinned `valid_targets.npz` using the existing stored
NPZ memory mapper, intersects its `is_scored` mask with the official `WARMUP=99`
prediction requirement, and copies only those selected sequences. Absolute scores use
`wnn.gmetric.global_wp`; paired deltas use
`wnn.evaluation.paired_component_moments` and
`paired_component_bootstrap` with 2,000 component resamples and seed 271828.
The fixed comparisons are:

- H512 CP5 replacement minus cached original TL3
- H512 SWA345 replacement minus cached original TL3
- H512 SWA345 replacement minus H512 CP5 replacement

F0 is exposed, and the two retained TL3 members have unknown training ancestry.
Every complete-ensemble score and delta is therefore a fitted diagnostic, not
clean cross-validation evidence. SWA345 is a dependent derivation from CP5, not
an independent fit.

## Runnable protocol

The exact commands and artifact paths are in
[ENDPOINT_SCREEN_PLAN_R1.json](ENDPOINT_SCREEN_PLAN_R1.json). Run them only after
root accepts the F0 fit terminal receipt and approves the source package path.
The two package builds, native replays, and final grade are create-only. They
write packages, prediction arrays, sidecars, and receipts on physical scratch
under `/home/vlad/development/wunder/research_tmp`.

`replay_h512_tl3_endpoint.py` reuses `f0_source.validate_f0_source` and
`f0_replay.replay_groups`, processes all 99 warmup rows, and requires the exact
468 evaluation groups and successful A-B-A equality. It does not read targets
or a score mask. `grade_h512_tl3_endpoints.py` rechecks archive, package,
prediction, target-index, split, sequence-order, and source-binding identities
before scoring. It fails closed on changed membership, coefficients, package
code, or checkpoint ancestry.

The output can establish package/native compatibility and local endpoint
behavior on this exposed fold. It cannot establish clean CV, generalization
across folds or seeds, or a public score. Root owns that interpretation and any
later submission decision.
