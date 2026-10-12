---
kind: experiment
status: inconclusive
created_utc: 2026-10-12T23:28:00.694471+00:00
updated_utc: 2026-10-12T23:28:00.694471+00:00
hypothesis_ids: [H-STRATEGY-20261012-01, H-238]
experiment_ids: [E-STRATEGY-INNOVATION-ENGINEERING-20261013-R1]
evidence_paths: [INNOVATION_FULL_SEQUENCE_PARITY_R1.json, INNOVATION_FULL_SEQUENCE_PARITY_R2.json]
---

# Causal moments and innovations: implementation and long-path correction

## Question or claim

S08/S09 now have opt-in training, prediction and streaming implementations.
Engineering checks do not establish a model-quality improvement; GPU profiling,
matched fits and independent code review are still required.

## Mechanism and method

Commit `49887ee2` (integrated as `bae0e2ee`) adds a fixed signed 112-to-16 input
projection, slow/fast exponentially weighted moments with 512/32-row half-lives,
and innovation channels standardized by the previous causal slow moments.
A fitting-only prior uses 32 deterministic source-qualified fitting sequences,
sampling every 64th row. Prior means/scales, membership and source identities
are checkpointed. The prior shrinks early estimates with a 99-row half-life.

The moment representation appends 64 channels; innovations append another 16.
The first contrast uses the same 80-channel width in both arms and zeros the
innovation suffix in the control. Equal parameter count does not imply equal
effective capacity. Both start from the same seed-specific own initializer;
new input weights are zero and new normalization coordinates are mean 0/std 1.
This preserves the parent's initial function. All rows, including warmup, update
state; each sequence resets it. Training scans 512-row blocks; streaming stores
fixed-size state. ONNX export carries this state explicitly.

## Evidence and interpretation

The actor's focused CPU suite passed 18 tests with one data-dependent skip.
Root's integrated causal/streaming suite passed 13 tests. These short fixtures
did not span the complete competition trajectory.

Root then compared all 20,000 rows of supplied training group 1, using group 0
only as an engineering prior fixture. [R1](INNOVATION_FULL_SEQUENCE_PARITY_R1.json)
failed the unchanged tolerance (relative 3e-4, absolute 5e-5): batch-versus-NumPy
maximum absolute discrepancy was 0.0018363. NumPy streaming passed. This is a
numerical implementation finding, not a negative scientific result.

The batched Torch scan rounded the exponential base to float32 but formed its
complement from the unrounded Python scalar. NumPy recurrence rounded both.
Using the identical float32 coefficient and its complementary weight in the
batch and Torch-streaming paths fixes this mismatch. [R2](INNOVATION_FULL_SEQUENCE_PARITY_R2.json)
passes on the same full sequence and original tolerance: batch-versus-NumPy
maximum absolute discrepancy 0.00029087, maximum normalized tolerance ratio
0.4723; batch-versus-stream ratio 0.5216. A data-dependent full-trajectory
regression test now preserves this check. Original receipts are immutable;
later formatting has no equation change and is bound by the implementation commit.

## Limits and validity

This is CPU numerical evidence on one real sequence, not GPU numerical or
model-quality evidence. Fit priors for research models must be constructed from
their declared fitting pools; the two-group engineering fixture is not such a
model. At batch 32, full input storage alone is roughly 1.244/1.285 GB including
raw/event/final tensors, before model activations and scratch space. A full fit
needs a fresh bounded device profile. Root's actual fixture process peaked at
1.661 GB RSS (which includes shared pages); subsequent support checks reserve
2 GB rather than treating a hard memory cap as a fit estimate.

## Decision and next step

Independently inspect source and the correction, then profile the actual
training shape before admitting the matched F0/F3 student treatments. Preserve
the same parent, seed, teacher targets, auxiliary terms and schedule within each
representation comparison. Do not compare a changed Main trainer to an older
frozen source as though only the feature changed.

## Source records

- [Feature implementation](../../../src/wnn/research_features.py).
- [Causal and real-data regression tests](../../../tests/test_causal_moment_features.py).
- [Training integration](../../../scripts/train_rnn.py).
- [Streaming integration](../../../src/wnn/streaming.py) and [ONNX exporter](../../../scripts/export_onnx.py).
- [Coverage matrix](COVERAGE.json), S08/S09.
