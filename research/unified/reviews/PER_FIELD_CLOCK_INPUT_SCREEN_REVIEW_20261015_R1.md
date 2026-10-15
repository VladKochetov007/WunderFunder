---
kind: review
status: complete
created_utc: 2026-10-15T14:07:22Z
updated_utc: 2026-10-15T14:07:22Z
hypothesis_ids: [H-087, H-138]
experiment_ids: [E-PER-FIELD-CLOCK-RESOLUTION-F0-20261015-R1]
evidence_paths:
  - research/campaign_2026-10-15/vast_a_cpu_probe_r1/per_field_clock_probe_r1.py
  - research/campaign_2026-10-15/vast_a_cpu_probe_r1/PER_FIELD_CLOCK_REDUNDANCY_SCREEN_R1.json
  - research/campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_TIMING_FIXTURE_R1.json
  - research/unified/ideas/EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md
---

# Per-field clock input screen review

## Question or claim

Does the input-only screen show that exact per-field change ages have distinct deterministic resolution from the existing three `since` features and seven raw-group ages, and what does its parity check actually cover?

## Mechanism and method

The script hashes the frozen fixture manifest and feature payload, maps the `[8, 20,000, 112]` float32 raw inputs read-only, and simulates one saturating integer age per field. Each age resets to zero at row 0 or an exact raw-value change; otherwise it increments to a 10,000-row cap. It compares reset patterns with seven clocks whose ranges match `token_groups(112)` and with the existing `StreamingEventFeatures` outputs at offsets 74–76. It also compares the existing 99-column batch feature transform against its row-wise stream, then resets that stream and checks that the first-row output repeats.

## Evidence and interpretation

The receipt pins the fixture manifest SHA-256 `941b656d…9f9f05`, raw feature payload SHA-256 `e8e8e126…5dd337`, and code hashes `8dcf9673…ba8ee`, `284c119d…12bfeee`, and `7fca0e21…878d48`. Across eight distinct train components, the screen reports 19–48 distinct field reset patterns per sequence versus exactly seven group patterns; field age differs from its corresponding group age on 7.3%–26.7% of field-row cells. Four fields match the same existing `since_i0trade` reset pattern on every slot; none of the fields match either of the other two clocks on every slot. No field is constant, no age reached the cap, and the median field had 919 distinct integer age values. The result supports added deterministic resolution on this small fitting fixture. Since `log1p` is one-to-one over the observed uncapped integer ages, that representation-resolution conclusion also applies to the proposed transformed age values. It does not show new causal information or predictive value.

The parity evidence is narrower than a feature-implementation test: full-sequence batch/row-stream equality and a sequence reset replay passed for the existing 99 event features only. The new ages are local diagnostic counters; the script never constructs the proposed appended `log1p(age)` tensor, splits computation across chunks, or compares a new feature stream to an independent oracle. The JSON receipt reports a 95,277,056-byte cgroup peak, zero OOM counters, and no Torch import. This review read source and small metadata only; it did not reopen the feature payload or rerun the screen.

## Limits and validity

The eight sequences are source-qualified F0 fitting components. They establish neither population-wide event statistics nor validation performance. The seven-clock G arm in the card broadcasts each group clock into every raw-field position, so its 112 appended columns contain only seven unique signals. That is a rank-deficient, repeated-input parameterization. With a learned input projection and matrix optimizer, F−G can reflect both signal resolution and optimizer geometry. This is not invalid if G is described as a redundant group-clock control; it is too strong to call it a pure information-only ablation. A narrower seven-clock G tensor padded with zeros to the same model input width, or explicitly shared group-clock projection weights, would make that distinction clearer.

## Decision and next step

Accept only the deterministic-resolution claim. This audit does not gate the existing A0 work. Before a fit using the new clock transform, add a small independent oracle test for the exact integer recurrence and `log1p` output, including row 0, repeated/change rows, age-cap saturation, sequence reset, and arbitrary chunk boundaries (including a reset/change on each side of a boundary). Compare both full and streaming/chunked implementations against the oracle, and verify the emitted width/order and that any new normalization statistics use fitting rows only. Pin the exact G representation and describe its repeated-clock geometry; no broader framework is needed.

## Source records

- [Screen receipt](../../campaign_2026-10-15/vast_a_cpu_probe_r1/PER_FIELD_CLOCK_REDUNDANCY_SCREEN_R1.json), [screen implementation](../../campaign_2026-10-15/vast_a_cpu_probe_r1/per_field_clock_probe_r1.py), and [frozen fixture manifest](../../campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_TIMING_FIXTURE_R1.json).
- [Per-field clock reframe](../ideas/EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md), [`StreamingEventFeatures`](../../../src/wnn/features.py), and [`token_groups`](../../../src/wnn/nn.py).
