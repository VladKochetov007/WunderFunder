---
kind: review
status: complete
created_utc: 2026-10-18T22:59:07Z
hypothesis_ids: [H-PLATEAU-I1-INCREMENTALITY-20261020-R1, H-PLATEAU-I1-HISTORY-20261020-R1]
experiment_ids: [E-PLATEAU-I1-A-20261020-R1, E-PLATEAU-I1-B-20261020-R1, E-PLATEAU-I1-C-20261020-R1]
evidence_paths:
  - ../../campaign_2026-10-20/plateau_review_r1/IMPLEMENTATION.json
  - ../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md
---

# Causal i1 inputs: implementation and independent review

## Question and scope

Does the code implement the accepted equal-width A/B/C comparison, with common
fitting normalization and no explicitly i1-derived inputs reaching A?

The actor `plateau_i1_inputs_actor` implemented the change in an isolated worktree
from main `9defbcbc0f2a6f1b35941dc657496993b48889c6`. Independent critic
`review_direction_audit` reviewed commit
`3c1936161e72970967c66a7ae54a69c9638f564b`. Both roles used native
`gpt-6-luna`, max effort. Root integrated the same change as `0cc795070` on main.
This is a static implementation review; no model, inference or numerical check
was executed for it.

## Implementation

- [History definition and fitting scale](../../../src/wnn/i1_history.py): 52 i1
  channels, residuals from prior EMA32/EMA512, and current-delta EMA512 energy.
  Scale fitting uses a deterministic sample of at most 64 fitting sequences,
  all within-sequence transitions, RMS differences and a 1e-4 floor. Exact
  source-qualified members and the denominator are saved with the prior.
- [NumPy/Torch scans](../../../src/wnn/i1_history_scan.py) and
  [row-streaming state](../../../src/wnn/i1_history_streaming.py): means are stored
  relative to the previous raw observation. Centered 256-transition blocks avoid
  exponent growth over a full sequence. The Torch state tracks block position
  across chunks. No per-row GPU launch loop is introduced.
- [Input configuration](../../../src/wnn/training/i1_inputs.py): reject unsupported
  feature/encoder combinations, preserve width 367 and restore the model mask
  from checkpoint arguments. All three arms assemble full unmasked inputs for
  shared normalization; masking occurs after model normalization.
- A masks 52 raw i1 fields, all 26 explicitly i1 event fields and all 156 history
  slots (234 total). B masks the 156 history slots. C masks none. Correlated i0
  and additional fields remain available to A.
- Model construction, injected-model setup, prior fitting, input assembly,
  checkpoint arguments, prediction/evaluation entrypoints and source identity
  use the canonical training package. `scripts/train_rnn.py` is unchanged.

## Independent findings

The critic found no concrete blocker for the first A/B/C route. Inspection
confirmed the raw/event mask registry, common unmasked normalization, saved-prior
restoration and source hashing. The centered recurrence is algebraically
consistent across NumPy/Torch/row-streaming implementations; the first row is
zero and state resets at sequence boundaries. Without the new mask, model
normalization retains its previous expression.

The first review identified a precision concern with large offsets and inverse
powers in 512-row scan blocks. The actor revised the implementation to centered
256-transition blocks before the reviewed commit. This is an implementation
response, not evidence that the remaining numerical checks pass.

## Static checks actually run

The actor and then root ran Ruff formatting/checks on all 19 changed Python
files and the existing training structure checker. Root used `scripts/limited`
with a 1 GB memory cap, one CPU core and no swap, through `uv run --no-sync`.

- Ruff format: 19 files unchanged after integration.
- Ruff check: passed.
- Training structure: 61 files, zero errors; 300-line module and 200-line
  function limits for the enforced training package/entrypoint.

Some previously unformatted touched legacy scripts and `nn.py` were normalized
by Ruff. The existing token-encoder zip now checks equal lengths explicitly;
the first experiment uses the linear encoder. No numerical or throughput parity
claim follows from these static checks.

## Command-source audit

The same independent critic then inspected all three finite-step plans against
the integrated trainer, `predict_run.py`, `score_run.py` and `kaggle_sequence.py`.
No unsupported flags, runner-schema mismatch or artifact-path mismatch was found.
The fixed component subset overrides the required legacy `--train-seqs 10607`
flag: actual fitting membership is 1,118 train plus 118 valid sequences from 315
components, disjoint from the 468 F0 evaluation groups. Batch 8 with drop-last,
ten chunks per sequence and three epochs yields 4,620 optimizer attempts.

`--epoch 2` selects the saved `model_ep2.pt`. Campaign evaluation-only prediction
uses the same split hash and exports explicit row-group identities. The score
step does not select checkpoints, clean artifacts or tune a blend. The target
cache stores `is_scored`; the scorer also removes rows below 99. Under the
[documented need_prediction contract](../../../docs/data_overview.md), that is
the official `need_prediction & is_scored` mask. The official accumulator and
metric ID are unchanged.

Source-shape accounting found about 300 MB of target values and 37.5 MB of masks
in the full cache, about 84 MB for selected targets/masks, and about 75 MB for
the full selected prediction view. This supports a sub-1-GB scoring working-set
estimate, not a measured peak or an aggregate admission receipt. Page cache,
parent charges and worker reservation still belong in the actual budget.

The finite plans start with fitting. Their separate `profile_argv` entries are
metadata for prior execution, not automatic runner steps. This is intentional:
complete and record the separate resource/source checks before admitting a plan.
No runtime report or model directory existed for the three new names at review.

## Remaining gates and limits

1. Compare full/chunk/streaming outputs on bounded representative inputs,
   including large offset with small changes, constant inputs and resets.
2. Verify prefix invariance and exact zero influence of disabled slots.
3. Verify checkpoint/prior/mask restoration through the actual prediction path.
4. Profile actual training shapes, complete preprocessing and prediction memory
   inside the 5 GB aggregate task cap before admitting a fit. No OOM-based sizing.
5. Freeze the complete source/environment/data recipe, then admit the funded
   comparison from October 20. The prepared commands alone do not acquire the
   shared GPU lock or enforce the campaign's aggregate budget.

The existing normalizer samples its first two batches; all three arms use that
same policy. ONNX/native `StreamingGraph` support is unavailable for this mode.
The Python feature prediction route does not establish a submission-ready native
endpoint. Tests, profiles, fits, predictions and quality scores remain unrun.

## Decision and source records

Implementation is ready for the bounded prefit checks, not for a quality claim.
Use the [prepared recipe and commands](../../campaign_2026-10-20/plateau_review_r1/recipes/README.md),
[implementation manifest](../../campaign_2026-10-20/plateau_review_r1/IMPLEMENTATION.json),
and [frontier](../../campaign_2026-10-20/plateau_review_r1/FRONTIER.md).
The [meta-review reconciliation](META_REVIEW_RECONCILIATION_20261019_R1.md)
records the conditional anchoring and representation-transfer follow-ups.
