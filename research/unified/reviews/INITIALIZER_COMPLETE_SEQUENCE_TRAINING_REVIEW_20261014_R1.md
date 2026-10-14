---
kind: review
status: complete
created_utc: 2026-10-14T02:38:41Z
updated_utc: 2026-10-14T02:38:41Z
hypothesis_ids: [H-STATE-INIT-20261014, H-MINGRU-TIMESCALES-20261014]
experiment_ids: [E-INITIALIZER-GRADIENTS-20261014-R1]
evidence_paths:
  - ../../../src/wnn/initializer_training.py
  - ../../../tests/test_initializer_training.py
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/RECURRENT_TIMESCALES_AND_INITIALIZER_GRADIENTS_R1.md
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/INITIALIZER_GRADIENT_TESTS_R1.xml
---

# Complete-sequence initializer training review

## Question or claim

Does the new initializer-only loss preserve the causal row-99 prefix treatment and carry useful gradients through complete-sequence chunking, while keeping the backbone frozen? Do the code and tests establish the target recipe or resource needs for a real fit?

## Findings

**No blocking source-level causality or checkpoint-recomputation bug found. The helper is ready for a root-controlled actual-sequence resource profile, but this review does not admit a full fit or establish initializer quality.** The reviewed uncommitted source hashes were `initializer_training.py` `5b22202d…4d81fed`, `test_initializer_training.py` `17c3ca16…ead0640`, and the method note `673e70c9…67cfbebc`.

The prefix wrapper uses only current/past input features: first-row mode reads row 0, and prefix mode accumulates normalized rows 0–98 then injects its residual before processing row 99. The helper rejects `chunk_rows <= 99` and zero-length scored input, and explicitly zeros the supplied loss weights for rows 0–98. Its first chunk therefore contains the full prefix and the first scored row. Targets and weights reach `_loss_chunk` only for the loss; they are not passed to the initializer or backbone forward. Each helper call starts with `hidden=None`, giving the batch fresh sequence-local state.

The chunk loop carries returned hidden tensors without detaching them and captures each chunk's absolute sequence position in a distinct `partial`. Non-reentrant checkpointing returns both the scalar loss and hidden state, allowing backward recomputation through earlier chunks. The wrapper rejects nonzero GRU dropout, so `preserve_rng_state=False` is deterministic for this supported RNN path. The helper also rejects trainable backbone parameters. The synthetic tests compare checkpointed chunked loss and initializer gradients with an unchunked reference for constant, first-row and prefix modes, test a late-only row-300 loss across checkpoint boundaries, and reject joint backbone updates. The provided receipt reports five tests passing; I inspected the source and receipt but did not rerun them under the 500 MB review cap.

The helper consumes, rather than constructs or verifies, the supervision tensors. For the declared final-distillation treatment, the caller must supply the exact target `0.1 * clamp(original_label, -2, 2) + 0.9 * clamp(frozen_teacher, -2, 2)` and a `[batch, time, 2]` weight tensor equal to `abs(clamp(original_label, -2, 2)) * main_row_weights(...)`. It currently checks shape and `requires_grad`, but not finite/nonnegative weights or finite targets. Actual cache/label/propensity validation and a receipt for those tensors must remain caller-side gates. The wrapper normalizes first-row/prefix context with `backbone.mean/std`; the helper itself does not bind their source or hashes. The synthetic parity tests use the default mean 0/std 1, so they do not check nontrivial normalization or actual F0 preprocessing.

There is one material objective distinction to freeze explicitly. `initializer_sequence_loss` sums all chunk numerators and divides once by total weight mass over the complete batch and sequence (`weights.sum().clamp_min(1)`). The existing trainer's cached distillation path normalizes each chunk separately by that chunk's own weight mass (`train_rnn.py` lines 3401–3402). The helper is therefore a global full-sequence weighted objective, not merely the ordinary recipe with ten updates combined into one. This is a coherent declared variant if intended; record its denominator and one-update-per-complete-sequence-batch budget in the fitted recipe. Also, the helper's floor of 1 differs from the existing loss's `1e-8` floor when total supplied weight is below 1. Record actual profile weight mass; either preserve this floor as an intentional choice or align it and add a low-mass regression.

The method note correctly labels the route as one optimizer update per complete-sequence batch, versus ten chunk updates in the baseline, and keeps it outside the frozen eight-shape student recipe. Checkpointing reduces saved activations but does not measure the actual memory bound. No actual 20,000-row CUDA profile, host/cgroup peak, allocator peak, checkpoint reload, or training call site exists in the reviewed helper. Those remain root-owned gates; the 900 MB synthetic-test result does not substitute for them.

## Evidence and interpretation

| Check | Result | Limit |
|---|---|---|
| Prefix timing and reset | Code processes 0–98, injects before 99, starts each call from empty state | Source inspection; no new full-sequence CUDA execution |
| Chunked autograd | Hidden graph is retained; non-reentrant checkpoint uses a fixed per-chunk position | Small synthetic GRUs only |
| Gradient parity | Constant, first-row and prefix losses/gradients match unchunked references; late-only loss reaches initializer | Five reported synthetic tests; not rerun here |
| Backbone freeze and recomputation | Trainable backbone rejected; supported wrapper enforces GRU dropout zero | Does not certify a future caller or other stochastic layers added later |
| Supervision fidelity | Helper applies supplied fixed two-target weights and masks warmup | Caller must construct/bind original-label, propensity and teacher tensors |
| Resource and quality | Unknown | No actual-data CUDA profile, fit, export or grade reviewed |

The mechanism claim is limited to preserving a late-loss gradient path through a frozen recurrent backbone. This is not evidence that an initializer improves prediction quality, helps every architecture, or offsets the different update and denominator semantics.

## Limits and validity

The helper is standalone and has no trainer call site. No fitting labels, teacher arrays, propensity caches, normalization source, split membership or checkpoint ancestry are encoded in its API. The new method note and ledger describe a synthetic implementation check, not a fitted result. Root's next source-qualified actual-sequence profile should bind those inputs, confirm the intended denominator, check finite loss/initializer gradients and record host and CUDA peaks before authorizing the full initializer experiment. The existing student ladder remains unchanged.

## Decision and next step

Clear the helper for a bounded actual-data, complete-sequence resource profile under root's separate admission. Before a fit, freeze the target/weight construction, the global denominator and its low-mass floor, normalization source, sequence membership/order, and optimizer/update schedule. Keep this arm separate from the matched eight-shape ladder and report its distinct budget. No quality promotion is supported.

## Source records

- [Initializer helper](../../../src/wnn/initializer_training.py), [synthetic tests](../../../tests/test_initializer_training.py), [causal state wrapper](../../../src/wnn/state_initialization.py), and [ordinary cached-distillation loss](../../../scripts/train_rnn.py).
- [Timescale and initializer-gradient method note](../../campaign_2026-10-14/cpu_student_frontier_r1/RECURRENT_TIMESCALES_AND_INITIALIZER_GRADIENTS_R1.md) and [five-test XML receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/INITIALIZER_GRADIENT_TESTS_R1.xml).
- [Experiment ledger](../../experiments.jsonl): `E-INITIALIZER-GRADIENTS-20261014-R1`.
