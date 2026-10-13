---
kind: experiment
status: inconclusive
created_utc: 2026-10-13T00:00:21Z
updated_utc: 2026-10-13T00:00:21Z
hypothesis_ids: [H-STRATEGY-MIRROR-20261013, H-317]
experiment_ids: [E-STRATEGY-S38-H317-INPUT-SYMMETRY-DIAGNOSTIC-R1]
evidence_paths:
  - research/campaign_2026-10-13/strategy_execution_r1/s38_mirror_symmetry_r1/PROPOSALS_R1.json
  - research/campaign_2026-10-13/strategy_execution_r1/s38_mirror_symmetry_r1/SYMMETRY_DIAGNOSTIC_R1.json
  - research/campaign_2026-10-13/strategy_execution_r1/s38_mirror_symmetry_r1/EXECUTION_RECEIPT_R1.json
---

# S38 / H-317 mirror-symmetry input screen R1

## Question or claim

Do simple schema-anchored bid/ask side exchange or price-axis reflection maps preserve the marginal feature distributions of source-qualified fitting sequences closely enough to justify a symmetry follow-up? This probes conditional marginal support; it cannot establish a joint data symmetry or a lawful target transform, and within-sequence preservation is not necessary for population symmetry.

## Mechanism and method

Before inspecting feature values, the proposal froze three 112-coordinate signed permutations: `side_swap`, `price_axis_reflection`, and the non-augmentation `price_sign_only_negative_control`. Each map was checked as an exact involution. Every transform was rowwise over the full sequence, with no row reorder, subsampling, cross-sequence state, warmup change, or reset change.

The diagnostic used `train.parquet` only, restricted to fold-0 `fit.train` membership (10,535 row groups). A fixed NumPy generator seed `2026101338` selected one complete 20,000-row sequence from each of 32 distinct fitting components. The proposal records every selected component, row group, and sequence ID in the result JSON. `SequenceFile.read(..., targets=False, mask=False)` requested feature columns and sequence identity only. All sampled values were finite.

For each transformed sequence, the analysis compared each transformed feature to that same coordinate's original empirical distribution. It reported the fraction outside the original q0.1%–q99.9% interval and observed min/max, plus the maximum normalized shift across q1, q5, q50, q95 and q99. The frozen coarse mismatch rule was met when at least 16 of 32 components had more than 56 of 112 features with normalized maximum quantile displacement above 1.0. A rule not being met is not a preservation or equivalence test.

## Evidence and interpretation

All three maps were involutive and finite on all 32 complete sequences. Results are equal-component summaries; the exact per-sequence and per-feature measurements are in the bound receipt.

| Map | Components crossing the frozen mismatch rule | Median features with normalized shift >1 | Median transformed fraction outside q0.1%–q99.9% | Median fraction outside observed min/max |
|---|---:|---:|---:|---:|
| `side_swap` | 18/32 | 58/112 | 50.4% | 45.1% |
| `price_axis_reflection` | 14/32 | 54/112 | 48.3% | 45.6% |
| `price_sign_only_negative_control` | 0/32 | 44/112 | 39.6% | 38.1% |

The same-index side swap crosses the predeclared mismatch cutoff, which is negative evidence for that exact marginal-preservation map. Price reflection misses that binary cutoff by two components, but its median continuous displacement is still broad: more than one normalized quantile unit on 54 features and almost half of transformed values outside the baseline central interval. The negative control also has substantial support differences despite not crossing the deliberately coarse cutoff. Neither non-crossing result is a preservation pass. Feature-specific changes are largest in several `p` coordinates; for example, the median normalized displacement is 9.11 for `i0_p8` under `side_swap` and 12.20 for `i0_p14` under `price_axis_reflection`.

This is a limited input-marginal screen, not a model-quality result. It does not establish that all H-317 symmetry ideas are false, because it tests only three frozen maps. It also does not establish that any map is a symmetry: univariate distributions can match while joint or temporal structure changes.

## Limits and validity

The [independent interpretation review](../../campaign_2026-10-13/strategy_execution_r1/s38_mirror_symmetry_r1/INDEPENDENT_INTERPRETATION_REVIEW_R1.json) qualifies two points in the original screen. The sign-only control changes at most 52 coordinates, so it cannot cross a gate requiring more than 56 displaced coordinates, regardless of the input. Its 0/32 count therefore cannot validate that gate. Also, within-sequence marginal preservation is not a necessary condition for population symmetry: a valid transform can map one sequence or latent regime into another. The measurements support a conditional marginal-mismatch finding; they do not falsify population symmetry. The frozen cutoff and raw results remain unchanged. Unknown input correspondence and target action remain the material obstacle to a justified augmentation.

The sample is small and comes from an already exposed F0 fitting partition; it is not a fresh holdout. No target, scoring mask, validation or evaluation feature was read. There was no model inference, training, GPU use, augmentation, or score measurement.

The supplied feature guide says coordinate indices identify a component but do not specify price/depth rank order. The meanings of `a0`–`a7` and the exact target law are not disclosed. The proposed conditional label swap for price reflection is valid only under a particular algebraic target definition that the available sources do not establish. Therefore this diagnostic cannot identify a legal end-to-end transformation, regardless of its marginal measurements.

The execution receipt preserves two preflight corrections and an undersized metadata-only attempt. A 400 MB PyArrow metadata scope was confirmed `oom-kill` by its systemd unit and journal; no feature rows were read in that attempt. The actual diagnostic subsequently completed under a 1.5 GB cap, one CPU, zero swap, and a 120 second runtime limit. The active GPU attempt remained healthy. This resource incident is not scientific evidence.

## Decision and next step

Do not add any of these maps as training augmentation. Keep H-317 open at its broad scope; the exact `side_swap` is inconsistent with the predeclared marginal screen, while the other two non-crossings do not establish preservation. No target map is presently justified.

The next discriminating step requires an authoritative coordinate dictionary and target definition. First establish which input coordinates correspond under a physical side/reflection operation and derive its target action algebraically. If those definitions still leave anonymous fields or labels ambiguous, no legal augmentation can be tested. If they fully specify an involution, freeze that exact feature/target map before a paired fitting-only joint/sequence check, then evaluate any augmentation in a separate matched endpoint experiment with unchanged evaluation membership.

## Source records

- [Frozen proposal](../../campaign_2026-10-13/strategy_execution_r1/s38_mirror_symmetry_r1/PROPOSALS_R1.md) and [proposal JSON](../../campaign_2026-10-13/strategy_execution_r1/s38_mirror_symmetry_r1/PROPOSALS_R1.json): three maps, deterministic sample and predeclared summary rule.
- [Actual feature-only measurements](../../campaign_2026-10-13/strategy_execution_r1/s38_mirror_symmetry_r1/SYMMETRY_DIAGNOSTIC_R1.json) and [execution/resource receipt](../../campaign_2026-10-13/strategy_execution_r1/s38_mirror_symmetry_r1/EXECUTION_RECEIPT_R1.json): selected sequences, component summaries, source hashes, exact command, resource use and preflight incidents.
- [Campaign folder README](../../campaign_2026-10-13/strategy_execution_r1/s38_mirror_symmetry_r1/README.md), [feature schema](../../../docs/data_overview.md), [first-principles proposal](FIRST_PRINCIPLES_EXPERIMENTS_20261013.md), and frozen [F0 component split](../../campaign_2026-09-29/artifacts/splits/fold0.json).
- [H-STRATEGY-MIRROR-20261013](../../hypotheses.jsonl) is a proposed child of H-317; [E-STRATEGY-S38-H317-INPUT-SYMMETRY-DIAGNOSTIC-R1](../../experiments.jsonl) records this measured input-only screen. The unchanged official metric was not evaluated.
