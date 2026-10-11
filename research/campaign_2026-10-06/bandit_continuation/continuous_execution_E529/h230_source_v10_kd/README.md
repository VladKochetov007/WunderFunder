# H230 source V10: distillation-loss variants (kd)

Copy of `h230_source_v9_ood` (untouched) plus three opt-in loss options on the cached-distillation branch of
`scripts/train_rnn.py` and `--extra-fit-args` in `scripts/distill_pipeline.py`. Written for this project from the
cache-only diagnostics (ensemble gain is 82% in the difference channel, 66% run level). No teammate code used.
Source and CPU tests only; no real-data fit has been run. Defaults take the unchanged v9 expression
`(w (out - tgt)^2).sum() / w.sum()` (dispatch `if KD_CONFIG.is_default`), so flag-off is bit-identical:
`tests/parity_fit.py` outputs of v9 and v10 are byte-identical (checked with `cmp`; note this fit exercises the
model, not the trainer loop, so loop parity rests on the dispatch above plus `test_default_is_bit_identical_to_v9_expression`).

Notation: `tgt = (1-a) clip(y) + a teach` is the v9 blended target (`teach` = cached 0.8125-scaled teacher mean;
the 0.8125 scale lives in the cache and is untouched), `w = |clip(y)| * main_valid` per column (v9 row weights,
including `--label-weight-*`). The new options replace/extend only the main MSE; aux-struct/next, family and EMA terms are unchanged.

## Flags (need `--distill-cache-root`; rejected with the controlled macro-loss path)
| Flag | Formula |
|---|---|
| `--distill-channel-weights SUM,DIFF` (1,1) | e = out - tgt, es = (e0+e1)/2, ed = (e0-e1)/2, wm = (w0+w1)/2. loss = sum 2 wm (wS es^2 + wD ed^2) / sum w. Identity e0^2+e1^2 = 2(es^2+ed^2), so wS=wD=1 equals v9 when w0=w1 (it skips the split and runs the v9 code exactly). With unequal column weights the split uses the column-mean weight, so non-default values differ slightly from a per-column weighting. |
| `--distill-loss {mse,pearson,mse+pearson}` (mse) | pearson = mean over c in {0,1} of 1 - corr_w(out_c, tgt_c) over all rows of the chunk (B x T) with weights w_c (scale free, differentiable, eps 1e-8). `mse+pearson` = (channel-weighted) MSE + `--distill-pearson-weight` * pearson (default 1). `pearson` alone ignores the channel weights. |
| `--distill-runlevel-weight W` (0) | adds W * sum_r W_r (m_out_r - m_tgt_r)^2 / sum_r W_r. Runs: per sequence, a run starts at each row where the i0 bid and ask price both changed (`wnn.features._flags(prev,x)[0]`, the px-event flag behind `since_px`, computed from raw unsnapped x) and ends before the next such row; row 0 of the chunk always starts a run (runs are cut at chunk boundaries). m_* are w-weighted means over the run, so unscored rows (w=0) are ignored; W_r is the run weight sum per column. |

Validation: SUM,DIFF finite and >= 0; pearson/runlevel weights finite and >= 0.

## Tests
`WNN_PROJECT_ROOT=$PWD python -m pytest tests/test_kd_loss.py` (6 tests, ~4 s, CPU): default bit-identity with the v9
expression, channel-weight identity/linearity, Pearson vs numpy and finite gradient, hand-built run-level example, argparse errors.
Parity: `python tests/parity_fit.py <root> out.json` for v9 and v10, outputs identical.

## Vast deploy (same layout as v9; overlay on the v9 tree)
Files that differ from v9: `src/wnn/kd_loss.py` (new), `scripts/train_rnn.py` (modified), `scripts/distill_pipeline.py`
(new in the source dir; copy of repo `scripts/distill_pipeline.py` plus `--extra-fit-args`), `tests/test_kd_loss.py`, `README.md`.
Everything else is byte-identical to v9. The pipeline resolves `scripts/train_rnn.py` under its `WNN_PROJECT_ROOT`, so
overlay these files into that root's `scripts/` and `src/wnn/`.
Example: `distill_pipeline.py ... --extra-fit-args "--distill-channel-weights 0.5,2"`.

## Suggested queued arms (dm_t3 recipe, 3 muonall teachers; compare to dm_t3 and dm_none_e6)
1. `--distill-channel-weights 0.5,2`
2. `--distill-channel-weights 0,2`
3. `--distill-loss pearson`
4. `--distill-loss mse+pearson --distill-pearson-weight 0.5`
5. `--distill-runlevel-weight 0.5`

Interaction candidate after readout: (0.5,2) plus runlevel 0.5.
