# E540 independent recipe and delivery review — V2

Date: 2026-10-08. Scope: static audit plus focused synthetic boundary tests in
an isolated worktree. No model fit, profile, data-file/footer read, full NPZ
load, provider call, or rental was performed.

**Disposition:** V1 fails recipe fidelity and its prepared prediction/export
commands were too broad for this fixed comparison. Those blockers are repaired
in the isolated V2 branch. V2 is ready for independent root review; it remains
unadmitted and has no scientific result.

## Findings

1. **V1 target objective did not match the trainer.** V1 specified WMSE against
   original unclipped targets in `RECIPE_V1.json`. The command selects the
   existing `wmse` objective, which sets `yc = clamp(y, -2, 2)`, weights by
   `abs(yc)`, and divides by the clipped weight mass within each update chunk
   ([`nn.py`](../../../../src/wnn/nn.py#L362),
   [`train_rnn.py`](../../../../scripts/train_rnn.py#L2173)). This changes
   tail-target treatment and therefore the scientific recipe. No V1 fit ran.
   Root chose the smallest recipe correction: V2 freezes clip2 WMSE identically
   in both arms and records raw-target WMSE as a separate future factor. The
   loss implementation is unchanged.

2. **V1 prediction export read more validation membership than the grade.** Its
   command called `predict_run.py ... valid`; that path loaded every valid row
   group from `valid_targets.npz`, and `Sequences.__getitem__` requested targets
   and `is_scored` by default. The staged grade is the fixed 468-group campaign
   evaluation. V2 adds the exact evaluation list to the hashed subset manifest;
   the E540 predictor checks that manifest against the checkpoint hashes,
   reads only those groups, requests feature columns only, and writes their row
   group IDs and membership digest beside predictions
   ([`predict_run.py`](../../../../scripts/predict_run.py#L26),
   [`prepare_e540_fit_subset.py`](../../../../scripts/prepare_e540_fit_subset.py#L67)).
   This is an E540 delivery restriction; it does not make a general claim about
   other campaigns' prediction exports.

3. **The V1 ONNX parity sample defaulted to validation group 0.** Its exporter
   read features only, but group 0 was not bound to the E540 fit membership and
   could be an evaluation or reserved group. V2 serializes the first selected
   fitting train group for parity; the exporter now checks that numeric-map
   parity uses a train group recorded in the checkpoint fitting provenance.
   The reader continues to request neither targets nor scored masks
   ([`export_onnx.py`](../../../../scripts/export_onnx.py#L43),
   [`build_e540_packet.py`](../../../../scripts/build_e540_packet.py#L196)).

4. **Equal parameter allocation does not equal effective capacity.** Both maps
   expand to 1,107 inputs and allocate 274,050 trainable parameters. The linear
   control's 896 ramp slots are identically zero, so their input-projection
   weights do not add data-dependent expressivity. V2 preserves this caveat;
   any PLE gain remains a map-treatment result, not proof against generic
   nonlinear capacity. A parameter-matched ordinary MLP remains a later,
   separate descendant.

## Recipe trace

| Stage | V2 implementation and audit result |
|---|---|
| Membership | Subset builder samples complete eligible components, then the trainer rechecks membership, component closure, split digest, and disjointness from evaluation. V2 also carries the exact sorted evaluation IDs and count. Packet construction refuses anything other than 512 fitting components and the frozen 468 evaluation groups. |
| Normalization and knots | The selected loader's seeded order supplies normalization; the PLE fit reads only selected train plus selected non-evaluation valid features with `targets=False, mask=False`. Sampling is 128 evenly spaced rows per sequence followed by a seeded 65,536-row reservoir and per-feature quantiles. No labels or scored masks enter knot fitting. |
| Input map | The model standardizes all 211 inputs first. PLE expands the 112 raw channels into eight clamped ramps plus an unbounded standardized raw bypass; the 99 event features pass through after that block. Constant features have no active ramps. The linear control has the same tensor shape with zero ramps. |
| Updates | Same seed and parameter layouts, one-layer GRU-128, batch 2, 256-row chunks/updates/detach, one epoch, AdamW, OneCycle, fp16 autocast, and deterministic seeded sequence order. The command disables fit-time evaluation and selects fixed epoch 0. WMSE is clipped-target, absolute clipped-target weighted, chunk normalized, matching the current trainer. |
| Checkpoint and prediction | Knot arrays/counts are model buffers in the checkpoint; load-state hook restores the fitted guard. The predictor rebuilds the model from checkpoint args and loads those buffers. V2 binds prediction membership to the exact fit-subset manifest and emits row-group order metadata. |
| Streaming export | The causal graph recomputes the existing event features row by row and carries GRU/event state. V2 parity uses a declared fitting train group. Existing synthetic checks cover full-sequence versus row-stream equivalence and reset between sequences. No ONNX export was executed in this review. |
| Evaluation | The official `global_wp_clip2_v1` metric, mask, ±2 scoring clip, and original clipped-label magnitude weights remain unchanged. Root owns independent grading over only the packet's 468 groups and must compare each prediction sidecar's row groups and digest with the frozen list before scoring. |

## Changes in the isolated branch

- Added `RECIPE_V2.json` and `EXPERIMENT_CARD_V2.json`; V1 files remain intact.
- Added exact evaluation group IDs/count to the generated subset manifest and
  trainer validation.
- Added an E540-only feature reader and checkpoint-bound evaluation membership
  validation to the prediction exporter.
- Bound ONNX parity to the first selected fitting train group and made the
  exporter reject numeric-map parity outside checkpoint fitting membership.
- Updated the V2 packet builder to hash the V2 recipe/card and audit, require
  the declared 512/468 membership sizes, and serialize the narrower commands.
- Updated the README with the V1 mismatch, V2 objective, delivery boundary and
  current review status.

## Checks and limits

- `tests/test_numeric_ple.py`: **10 passed** under `wnn-campaign.slice`, one CPU
  and a 1,000,000,000-byte memory cap. New checks reject changed evaluation
  membership, verify feature-only prediction reads, and reject numeric-map ONNX
  parity outside fitting train groups. No parquet footer or dataset was opened
  by these tests.
- `ruff check` passed for the changed packet builder, exporter, subset helper,
  predictor and focused tests. A whole-file check of the legacy
  `scripts/train_rnn.py` still reports its existing import-order and loop-style
  findings outside this patch; the modified validation block adds no reported
  finding.
- Python bytecode compilation and V2 JSON parsing passed.
- The real packet was not built because this isolated worktree has no supplied
  parquet files or campaign split/cache artifacts. Source hashes, actual
  source/data identities, 512/468 counts, Kaggle device profile, quota,
  prediction artifacts and ONNX runtime parity therefore remain root gates.

Reviewed source basis: root integration commits `e2a33b75` and `4c09ac01`,
`RECIPE_V1.json`, the active trainer/model/exporter sources listed above, and
the current project split/metric contracts. Audit code and this review are on
`audit/e540-recipe-delivery-20261008` for independent root review.
