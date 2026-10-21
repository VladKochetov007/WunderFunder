---
kind: review
status: complete
created_utc: 2026-10-21T21:14:56Z
updated_utc: 2026-10-21T21:26:48Z
hypothesis_ids: [H-FUTURE-INPUT-DYNAMICS-20261021-R1]
experiment_ids: [E-FUTURE-INPUT-DYNAMICS-20261021-R1]
evidence_paths:
  - research/campaign_2026-10-21/next_cycle_r1/DYNAMICS_NUMERICAL_PROBE_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/DYNAMICS_NUMERICAL_PROBE_R2.json
  - research/campaign_2026-10-21/next_cycle_r1/DYNAMICS_NUMERICAL_VERIFICATION_R1.json
---

# Dynamics auxiliary FP32 window stability

## Question or claim

Does the FP32 target builder preserve small future input changes after a large earlier pulse? The original global-prefix calculation fails this synthetic check before any auxiliary profile or scientific fit.

## Mechanism and method

The target builder subtracted two long cumulative sums of squared changes. In the [original20000-row fixture](../../campaign_2026-10-21/next_cycle_r1/DYNAMICS_NUMERICAL_PROBE_R1.json), an early10000-unit pulse makes the accumulated sum too coarse to retain later0.01 transitions. Both16/256-step horizons at position16000 return−18.420681 instead of the direct-window−9.2102404:9.21044log-unit error.

The repair keeps the original FP32 normalized inputs,13feature groups, output order, epsilon placement, and start/endpoint masks. Signed means use normalized endpoint differences. Energy uses per-transition group mean-square changes and positive block-local prefix/suffix sums, so no window subtracts a distant large prefix. The mathematical52-target definition and loss are unchanged; finite-precision arithmetic is intentionally corrected. Auxiliary heads and targets remain outside the causal prediction path and are stripped for serving.

## Evidence and interpretation

[Root verification](../../campaign_2026-10-21/next_cycle_r1/DYNAMICS_NUMERICAL_VERIFICATION_R1.json) passes Ruff, the63-module training-structure checker, make quality/quality-tests, six CPU oracle/causality/head-removal tests, and an independent source/math review by recovery_audit (Luna/max). Test peak RSS791508KiB; zero swaps orGPU use. The [same original fixture after repair](../../campaign_2026-10-21/next_cycle_r1/DYNAMICS_NUMERICAL_PROBE_R2.json) differs from its direct float64 reference by only5.76e−8log units. An additional regression uses a milder1000-unit pulse; boundary tests cover unequal group widths, unaligned/partial blocks, compressed lengths and short sequences.

## Limits and validity

This is numerical implementation evidence on synthetic inputs. It does not measure error prevalence in supplied data, generalization, speed on a GPU, or auxiliary-task quality. Float64 is used only by the independent test oracle; target construction remainsFP32. No auxiliary experiment ran with the old construction. The active i1 panel uses its original frozen source.

## Decision and next step

The numerical gate is cleared. Root must freeze the corrected source and profile the complete preprocessing/training/export cost before reserving the four-fit panel inside its three-GPU-hour stage. No new fit or profile is admitted by this note.

## Source records

- [Frozen dynamics comparison](../../experiment_drafts/dynamics-aux-20261021/README.md): weight0/.05, two matched cells, four new fits, zero teacher fits.
- [Numerical fixture and execution](../../campaign_2026-10-21/next_cycle_r1/dynamics/README.md#numerical-gate-before-profiling).
- [Root check log](../../campaign_2026-10-21/next_cycle_r1/dynamics_numeric_check_r1.log).
