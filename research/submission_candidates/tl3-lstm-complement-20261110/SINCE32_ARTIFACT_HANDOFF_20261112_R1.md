# Next candidate: retrieve PR89 since32 artifacts

This is an artifact collection handoff, not a training request or a ready CPU
submission. Source: [PR89](https://github.com/VladKochetov007/WunderFunder/pull/89)
at `882eaf865e9df3a56157772dbf957a45df0a6001`. Paths below are reconstructed from
the pinned producer receipts/code, **not verified on a currently connected host**.
No SSH hostname or checkpoint download URL is published. No request was sent
to a teammate by this agent.

## Exact terminal checkpoints

Producer root:
`/data2/lc-epanov_s/tabdata-all-followups-20261112-r1`

| Cell | Path relative to producer root | SHA256 |
|---|---|---|
| `since32_k16_s203301` | `runs/since32_k16_s203301/fit/model.pt` | `e8fba80e0dd6edbebc17a0cd4b38ea20637aa259b2cc8f552538e7a1fbec1ae4` |
| `since32_k16_s203302` | `runs/since32_k16_s203302/fit/model.pt` | `37cef64335aa7d3097832fe2354b6a0d11bd7d2ac72c4f9db76a9b7e51e56c26` |

For **each** cell collect, without altering a running job:

- `runs/<cell>/fit/model.pt`, `epoch_index.json`, `terminal_status.json` and
  `monitor_evaluations.jsonl` from that same fit directory.
- `configs/<cell>.json` and the corresponding profile/admission receipts.
- `control/root-fits-r4/<cell>.log`, `<cell>-result.json`, `claim-<cell>.json`
  and the associated owner/completion/resource records.
- The fitting source manifest/tree from `source-r2`, preserving its immutable
  identity. Saved normalizers and model configuration must accompany the weights.

The individual epoch filenames/sizes are not present in the copied result JSON;
inspect the collected `epoch_index.json` before claiming a specific retained E6
file. The hashes above are for the producer's terminal `model.pt`.

## Provenance and serving contract

Source commit `390c846cbdacd8e0e06e6079dd3b04312fa6bcc5`; source-manifest SHA256
`00782a64e1557a8af759e6f987038cabb6ec09f098c43ab61d6cfa6bb07e9e5f`.
Data identity `4987ed5f263cf3c655714a7af3e792605716f6cb256c1c96474eb46ae9869323`.
Producer split file:
`/data/lc-epanov_s/tabdata-stem-successors-20261111-r1/project/chronological_primary.json`,
SHA256 `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
Collect this split and the source-qualified data/membership manifest, not raw
competition data in a publication bundle.

The declared architecture is GRU256x2 with a causal K16 input stem, raw112-v1,
fixed six-epoch training, no initializer and normalization fitted on10325 training
sequences. Feature order is i0 then i1, each p0..21/v0..21/dp0..3/dv0..3, followed
by a0..7. `since_weight=32` with floor0.25 changes the training loss; it is not a
new serving feature. Serving must nevertheless preserve both recurrent hidden
state and the K16 input history. `predict_sequence.py` is the canonical export
entry in the pinned source. Do not treat these as drop-in PR76 GRU weights.

## Readout acceptance still needed

The final PR comment reports fixed-E6 WP0.686306013/0.685903598 against controls
0.683019498/0.682629372. These remain producer-reported research results. At the
pinned source head, canonical R3 readout rejected followups because
`datasets/train.parquet` was missing; the R4 layout repair/start is recorded,
but a successful terminal R4 canonical readout is not attached there.

Collect terminal readout evidence from producer root
`/data2/lc-epanov_s/tabdata-followup-readouts-20261112-r4`:

- `runs/<cell>/final-readout-r3/predictions/manifest.json` and `score.json`
  from the same `final-readout-r3` directory.
- `control/canonical-readouts-r3/<cell>-result.json`, console log and input/output
  hashes binding the accepted checkpoint, source, metric and evaluation mask.
- If the repaired readout failed or is incomplete, preserve that status and
  report it; do not substitute the monitor score for terminal acceptance.

Public provenance copies are under
`research/experiment_drafts/l4-followups-ema-aux-20261112/execution/` in PR89:
`receipts-r1/l4-publication-admission-receipts-r1/followups/<cell>-result.json`,
`ROOT-fit-admission-r2.json`, `complete.json`, the receipt manifest and
`PUBLICATION_PINS.json`. The canonical readout source is
`pinned/readout/canonical_readouts_r3.py`, SHA256
`09bd67033ba59943ac4c19a3f5ffcba4c1a9e20ca2d5ad8a2cd877d65972fae6`.

After collection, verify artifact bytes and source joins before preparing a
separate CPU predictor. Preserve both seeds, fixed endpoint and the outstanding
same-source zero-weight control. CPU parity/profile/full validation and a matched
TL3 comparison have not run; no public-score forecast or upload is authorized
by this handoff. Do not retrain or change the shared pipeline to collect files.
