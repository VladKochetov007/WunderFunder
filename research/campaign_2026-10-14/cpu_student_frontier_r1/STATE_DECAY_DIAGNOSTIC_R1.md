---
kind: result
status: complete
created_utc: 2026-10-14T00:14:00Z
updated_utc: 2026-10-14T00:14:00Z
hypothesis_ids: [H-STATE-INIT-20261014]
experiment_ids: [E-STATE-DECAY-20261014-R1]
evidence_paths: [STATE_DECAY_1000_PROFILE_R1.json, STATE_DECAY_20000_R1.json]
---

# Initial-state effects survive warmup in one actual fitted sequence

## Question or claim

Does a finite initial-state perturbation disappear before scoring begins? In this narrow diagnostic, it does not: the second GRU layer retains a measurable perturbation after 99 warmup rows and through several thousand rows. This is a mechanism observation on one fitting sequence, not evidence of a useful trained initializer.

## Mechanism and method

[`diagnose_state_decay.py`](../../../scripts/diagnose_state_decay.py) loads the hash-pinned clean F0 H256×2 seed-3 control at epoch5. It explicitly uses training row-group37 (`seq_ix=7598721`), the sequence actually consumed by the resource smoke. The actor subsequently found that smoke's sampler should have selected **valid** group37; its exact-sampler claim is withdrawn. This diagnostic declared train37 directly and does not claim sampler fidelity. A fixed seed734 Gaussian direction has global hidden-coordinate RMS0.1. The same vector is added either before row0 or before row99; a zero-state control runs the same chunks. All predictions, hidden states and bin calculations are FP32 inference with double-precision difference summaries. There is no training, label-based selection or change to the official metric.

The [1,000-row resource profile](STATE_DECAY_1000_PROFILE_R1.json) completed in1.09s with1.61GB max RSS. The [complete20,000-row diagnostic](STATE_DECAY_20000_R1.json) completed in4.57s with1.61GB max RSS. Both used one CPU,3GB host cap and120s timeout. The full run followed the successful bounded profile.

## Evidence and interpretation

Prediction-difference RMS against zero-state control, pooled over both output coordinates:

| Positions | Perturb before row0 | Perturb before row99 |
|---|---:|---:|
|99–199|0.005211|0.012998|
|200–999|0.004583|0.004197|
|1,000–4,999|0.001373|0.001518|
|5,000–9,999|0.00003844|0.00003976|
|10,000–19,999|0.0000007073|0.0000007205|

After row98, the row0 perturbation's RMS is0.00380 in layer1 and0.02029 in layer2. After row999 it is0.00001398 and0.004129 respectively. By row19,999 both arms are near FP32 rounding noise. Readout changes need not decay monotonically even when hidden-state norms shrink.

The later injection produces a larger immediate prediction change here, but that does not make it more accurate; neither random perturbation was optimized. The longer memory in layer2 also means the first layer's decay or an update-gate average alone would misdescribe the whole model's state memory.

## Limits and validity

One checkpoint, one fitted sequence and one random direction cannot bound the full Jacobian, describe the distribution of decay times or decide initializer quality. No public or held-out score is claimed. Different chunk shapes can introduce FP32 noise; all three arms use identical boundaries. The observations support running the already-authorized learned arms, not choosing among them or rejecting persistent conditioning.

## Decision and next step

Retain constant, first-row and prefix arms. Measure the trained initializer's decay across complete source-qualified components and the official score/position bins. Record TBPTT truncation: an effect that persists after row2000 does not mean losses there train an initializer through a detached boundary. Treat persistent prefix conditioning as a distinct successor if its added per-row cost is justified.

## Source records

[Initializer implementation and tests](INITIALIZER_IMPLEMENTATION_R1.md), [theory and primary precedent](THEORY_AND_DISCRIMINATING_TESTS_R1.md), and the [actual-data resource smoke](profiles/README.md). Raw receipts bind checkpoint, script, exact input feature hash, direction seed and observations. Independent interpretation review is pending.
