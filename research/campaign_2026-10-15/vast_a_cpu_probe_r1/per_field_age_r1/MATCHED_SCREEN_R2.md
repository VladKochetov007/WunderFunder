---
kind: experiment
status: proposed
created_utc: 2026-10-15T14:48:00Z
updated_utc: 2026-10-15T14:48:00Z
hypothesis_ids: [H-087, H-138]
experiment_ids: [E-FIELD-AGE-MATCHED-SCREEN-20261015-R2]
evidence_paths: [README.md]
---

# Matched field-age screen R2

This unlaunched successor replaces R1's proposed six-epoch comparison with a
bounded two-epoch whole-fitting-data C/G/F screen. The user approved continued
experiments and GPU use; root owns eventual profile, fit, export and grade.
R1's implementation and historical test/profile results remain preserved.

Use the same original warm parent for every arm:
`c20261003_combo_s3_f0@ep3`, checkpoint SHA
`d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e`,
provenance SHA
`9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f`.
This is the common original parent, not the intermediate alpha0 CP1 child
proposed in R1. It retains its documented prior distillation ancestry.

C keeps the 211-input `none` representation; G appends 112 broadcast group ages;
F appends 112 individual ages. G and F use the existing causal feature-extension
path: keep all original normalization buffers and projection columns, append
mean0/std1 and zero projection columns. Their initial mathematical function must
match the parent. Do not refit normalization independently. All output rows,
including structural/next-event auxiliary heads, must be preserved.

Hold the existing alpha0 recipe's optimizer, LR, weighting, clipping, cache,
auxiliary weights, B16/C2000/FP16 and sampler seed3 constant. Set a common
two-epoch OneCycle horizon and choose fixed CP1; do not inherit the six-epoch
schedule or select an epoch afterward. The homogeneous T3 cache remains bound
but its main-target coefficient is zero; family loss and EMA-teacher are zero.
Reuse the original source-qualified F0 split SHA
`09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`:
10,535 train-fit plus 1,405 valid-fit groups, excluding 468 valid evaluation
groups/components. Preserve source-qualified cache row mapping and teacher ancestry.
At B16/drop_last there are 746 batches and 7,460 attempted chunk updates per epoch,
14,920 total; four fitting groups are omitted per epoch by the shared sampler.
Record actual order/RNG, completed updates and AMP skips instead of assuming equal
completed updates. Fresh optimizer, identical fixed target treatment across arms.

Export **all 1,873 valid groups** in canonical target order using the full-valid,
feature-only predictor, then grade the same 468 excluded evaluation groups with
the unchanged official metric. R1's 1,405-group export instruction referred to
fit-valid membership and would omit the evaluation rows; it is superseded here.
Primary contrast F minus G; F minus C and G minus C are secondary. G has seven
unique clocks repeated across 112 coordinates, so the primary contrast also
changes input rank and optimization geometry. It is not pure information attribution.
Use paired 2,000-resample component intervals and fixed position bins, preserving
one-seed/exposed-F0 and short-budget limitations. No short negative is a universal
rejection; any longer follow-up gets its own matched horizon and budget.

Before fitting, pass the production Torch-versus-independent-oracle test (including
future causality, sequence reset and cap saturation), then an actual source-bound
trainer GPU profile. That profile must verify the intended feature mode, inherited
norm, zero-column prediction parity, nonzero new-column gradients, actual B16
memory/VRAM and the existing cache consumer against its producer fixtures. A
Torch CPU oracle pass alone does not establish complete checkpoint/native parity.

Local execution stays below two aggregate CPU cores, one GPU worker, zero loader
workers and the existing RAM/VRAM ceilings. Use physical disk scratch. The current
alpha0 chain retains priority; this note admits no simultaneous local GPU worker
or fit. Native C++ support and serving measurements remain later submission gates.
