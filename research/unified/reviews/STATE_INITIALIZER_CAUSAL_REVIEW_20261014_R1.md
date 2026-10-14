# Causal initial-state implementation and decay review R1

## Decision

No blocking implementation issue found in the wrapper's declared synchronized-sequence contract. The code places the prefix residual before row 99, keeps state sequence-local, preserves zero-state behavior at initialization, and allows initializer gradients through a frozen recurrent backbone. The actual-data decay result is a useful one-sequence mechanism observation, not initializer quality or a population estimate. Fitting, trainer integration and serving remain separate work.

## Wrapper audit

I reviewed `src/wnn/state_initialization.py` and `tests/test_state_initialization.py` from commit `e114f639` (`state_initialization.py` SHA-256 `eac11200d85d1a84d54921306d4d75cbe853188036d4e1966e5530a2994c87d2`; test source `1bbd880a0821b878584b31063378a9b75119b56455b4e61733635dff796ffbef`). The wrapper is explicitly limited to the existing linear-encoder, dropout-zero GRU and synchronized complete-sequence batches. Its public input is the existing 211 causal feature vector; targets, masks, sequence IDs and future rows do not enter the initializer.

- `None` starts a sequence with zero hidden state and position. Caller-owned `InitializationState` carries hidden state, position and any prefix moments; passing `None` resets all of them.
- The constant arm learns a bounded state vector. The first-row arm predicts a bounded state from the normalized first feature row. The prefix arm computes Welford mean and population standard deviation plus the last normalized feature over positions 0–98, processes those 99 rows normally, adds the bounded residual before processing row 99, then releases the statistics. Every learned arm begins at zero residual.
- `InitializationState.detach()` detaches hidden and prefix statistics together at an explicit TBPTT boundary. Freezing the backbone parameters does not wrap recurrence in `no_grad()`, so gradients can still flow through the frozen recurrence into the initializer.

The committed JUnit receipt `INITIALIZER_TESTS_R1.xml` records ten passing checks in 2.107 seconds. They cover zero-baseline equivalence for all four arms, chunk boundaries around rows 98–100, reset replay, exact prefix context and injection timing, causal future invariance, gradients through a frozen backbone and a finite 20,000-row synthetic sequence. I inspected the receipt and tests; I did not rerun them. These are small random models and synthetic features, not native parity, trainer/export integration, actual-data initializer fitting or quality evidence.

## Actual-data decay diagnostic

I reviewed `scripts/diagnose_state_decay.py` (SHA-256 `374e5f7b39fd7dcd01f5f1a1ca0aae152d5ad24c6685ddb03702a825779b12c2`), the [1,000-row preflight](../../campaign_2026-10-14/cpu_student_frontier_r1/STATE_DECAY_1000_PROFILE_R1.json) and [20,000-row result](../../campaign_2026-10-14/cpu_student_frontier_r1/STATE_DECAY_20000_R1.json). The script hash-pins the clean-F0 H256×2 seed-3 epoch-5 checkpoint and explicitly reads train row-group 37, `seq_ix=7,598,721`; the selected feature bytes are hash-bound. A single seed-734 Gaussian hidden direction at global RMS 0.1 is added either before row 0 or before row 99. No labels, fitting or model selection enter the diagnostic.

The result supports persistence of a finite state perturbation after warmup in this one sequence: the row-0 perturbation's hidden RMS after row 98 is 0.00380 in layer 1 and 0.02029 in layer 2; after row 999 it is `1.40e-5` and 0.00413; after row 4,999 the layer-2 RMS is `1.17e-4`. Prediction-difference RMS is about `3.84e-5` over rows 5,000–9,999 and `7.07e-7` over rows 10,000–19,999; terminal hidden deltas are near FP32 rounding. This is one directional finite perturbation, not a Jacobian norm, trained-initializer benefit or quality result. Persistent hidden influence also does not imply gradients survive a detached TBPTT boundary.

The diagnostic independently declares train group 37 and checks its sequence ID, so the later B1 sampler-source correction does not invalidate this diagnostic. The observed row was actually also the row that the B1 smoke loaded, but neither artifact establishes that it was the intended first sampler item.

## Scope and provenance limits

The implementation and decay sources are commits `e114f639` and `35fdd47b`. Their campaign note and machine receipts are [initializer implementation](../../campaign_2026-10-14/cpu_student_frontier_r1/INITIALIZER_IMPLEMENTATION_R1.md), [decay analysis](../../campaign_2026-10-14/cpu_student_frontier_r1/STATE_DECAY_DIAGNOSTIC_R1.md), [test receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/INITIALIZER_TESTS_R1.xml), and the two JSON diagnostics above. I read and statically checked the sources/results; no fit, scoring, export or provider action was performed. A separately recorded correction to the B1 profile's sampler provenance is in the [profile execution review](CPU_STUDENT_PROFILE_EXECUTION_REVIEW_20261014_R1.md).

Related hypothesis and experiment: `H-STATE-INIT-20261014`, `E-STATE-INIT-IMPLEMENTATION-20261014-R1`, and `E-STATE-DECAY-20261014-R1`.
