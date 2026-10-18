# Measured strong-control comparison, 18 October 16:29 UTC

The restored historical Main T3 control scores **0.6811541566320954** on the same 808,863 scored F0 rows as the new Q8 submission's **0.6738263576698660**. The difference is **+0.00732779896223**. This is now an actual source-bound prediction comparison, rather than a comparison of unrelated historical score summaries. The control is a separate historical recipe, not the exact `tl3.zip` incumbent.

| Fixed endpoint | t0 | t1 | WP |
|---|---:|---:|---:|
| New wide | 0.663747991 | 0.683403511 | 0.673575751 |
| New deep | 0.662871519 | 0.681346016 | 0.672108767 |
| New fixed 75/25 FP32 mixture | 0.664174233 | 0.683527655 | 0.673850944 |
| Actual new Q8 package | 0.664144725 | 0.683507990 | 0.673826358 |
| Restored historical strong control | 0.674375751 | 0.687932563 | 0.681154157 |

[Canonical float64 grade](GRADE.json) independently agrees with the [diagnostic calculations](RESULT.json). The paired component bootstrap uses 141 components, 468 groups, 2,000 resamples and seed 271828. For strong control minus Q8, its conditional 95% interval is **[+0.00238446, +0.01352521]**. This conditions on these fixed predictions and this exposed F0 panel; it does not measure training-seed variation, public transfer, or CP30 performance. The previous historical score 0.6811553603333363 differs from this fresh runtime's result by about -0.000001204; exact prediction-byte parity across runtimes was not required or established.

## Where the local difference appears

- The t0 correlation difference is +0.01023103; t1 is +0.00442457. With the metric's equal target weights, t0 supplies **69.81%** of the total WP difference.
- In the additive decomposition using each endpoint's global centered moments, the common retrospective bin `|clipped target| in [1,2]` contributes **+0.00551341**, or **75.24%** of the net difference. The `[0.5,1)` bin contributes +0.00166645. These label-defined bins are diagnostic only: their future labels are not available at inference, and this is not a percentage of errors explained or a training reweighting experiment.
- Component contribution changes have both signs: **85 positive, 56 negative**, with net +0.00732780 and sum of absolute changes 0.01980108. The largest positive component (2994) contributes +0.00288837; the largest negative (995) contributes -0.00133370. These are contributions under global normalization, not per-component accuracy wins, leave-one-component-out effects, or a reason to select components.
- The Q8 loss relative to the same FP32 mixture is **0.00002458657**, about **0.34%** of the measured strong-control/Q8 gap. This local result does not support quantization as the main source of that gap. It cannot certify the bytes uploaded to the portal, which remain unverified.
- The new Q8 endpoint actually has smaller absolute weighted mean bias than the strong control, while its WP is lower. A simple mean correction is therefore not a demonstrated remedy. Raw prediction clipping affects 572/549 rows for Q8 and 1/7 for the control (t0/t1); these are model-specific memberships and descriptive counts, not matched causal clipping effects. No calibrator, blend, threshold or public-selected rule was fitted.

## Interpretation and next comparison

The new cold labels-only geometry recipe loses to a stronger existing training pipeline on exactly matched local rows. This resolves one concrete baseline-relative gap. It does not isolate KD, initialization, features, optimizer, training duration or their interactions, and does not identify methods used by leaderboard leaders.

The user identifies the public incumbent as `tl3.zip` / `1lmehfi3` / 0.66832. Its archived recipe combines distilled and tied-lead families with fixed weights 0.34/0.33/0.33, unlike the new two-model cold control. Its production models fit train+valid; evaluating those weights on F0 would not supply a clean held-out counterpart. The reviewed [clean counterpart plan](../../TL3_CLEAN_COUNTERPART_PLAN_20261018.md) therefore requires exact source/config and teacher/initializer ancestry closure, followed by an F0-excluding reconstruction with the original coefficients. That is the next diagnostic comparison; no fresh public package is selected here.

The separate frozen rank1 comparison has completed F0 and F1. Their width-versus-depth directions differ and neither conditional interval establishes a general geometry advantage. Two of thirty paired cells per arm do not complete CP30. The [F3 draft](../../../priority25_execution_r4/f3_beleriand_preparation_r1/README.md) is independently reviewed but remains unadmitted; no F3 fit is implied by this result.

## Execution and provenance

R4 [preparation](../PREPARATION_MANIFEST.json) and [independent review](../INDEPENDENT_REVIEW_R4.json) preceded execution. [Alignment](ALIGNMENT_RECEIPT.json) verifies the original labels, sequence/group/component identities and masks against the existing compact Q8 bundle, including bitwise equality and exact row order. The selected control is float32 `[808863,2]`; its file stays on Beleriand. The 299,680,128-byte full-valid export is preserved unchanged.

All three serial CPU units exited 0: alignment **8.976934 s**, diagnostic **10.214332 s**, canonical grade **13.765887 s**. Each used a 0.5 CPU / 1.5 decimal GB / 300 s bound, swap 0, CUDA hidden and physical scratch. Raw output bytes and hashes were captured before stopping each own exited unit. [Final resource status](FINAL_RESOURCE_STATUS_V1.json) shows all three units inactive, no active campaign child and unchanged parent memory events (OOM 0). No training or inference ran on the Mac. Recorded systemd MemoryPeak values after cgroup teardown are not treated as reliable compute-peak measurements.

Earlier R2 and R3 consumer failures remain preserved. R4 fixes forwarding of validated captured-evidence fields; no checkpoint, predictor, original prediction, metric or training source was changed. The root staging-wrapper correction also occurred before SSH or remote writes and is retained separately.

Exact output SHA-256:

| File | SHA-256 |
|---|---|
| ALIGNMENT_RECEIPT.json | `49198121cbe8c1787c11db7982bce687bf1d0da2083c4c7490ad4ecd725cc315` |
| DERIVED_INPUTS.json | `9ce4fd56c65779d5a32d4257536ca7a3b12d8d58c8c2b863af2fbd385507f08a` |
| RESULT.json | `aca7842c10b6c7065711c1e015efec0e2a33863338d993ace16bf8e500f78f70` |
| RESULT.csv | `1ecb1021d4bdf7942d6d109aa11875ad9230a1fb0e4d2d3bacd65c948da6a5f2` |
| GRADE.json | `2929bb4e269a73e75ed9bbdba0500eac07e72447ff901a57b813691a3577707e` |
