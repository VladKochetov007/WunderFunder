# H230 source V9: out-of-sample path distillation (H-350, user idea, ledger B4)

Copy of `h230_source_v8_arch` (v6/v7/v8 untouched) plus the `--ood-*` flags. Without `--ood-teachers` the trainer
runs the unchanged v8 loop (`tests/test_ood_student.py::test_flag_off_is_bit_identical_to_v8_and_flags_are_registered`
checks the model/loss/parameter parity fit; the loop gains one `if ood_sampler is not None` branch).

Status: source and CPU unit tests only (17 tests, ~25 s, 1 thread, < 1 GB). No real-data fit, profile or GPU run.

## Idea

The user: scale student training with augmented data, form random possible input paths, train the student on the
teacher's output for them after pretraining, so that out-of-sample inputs teach the student the teacher's behaviour.
Implemented as: warm-started student (`--init`), a fraction of batches replaced by synthetic paths, target = mean of
frozen teacher GRUs run online on that same path (each teacher carries its own chunked state). Real chunks keep the
existing wMSE + `--distill-alpha` cache distillation. Student architecture is arbitrary.

New code: `src/wnn/ood_student.py` (parsing, `PathSampler`, `OnlineTeachers`, `ood_loss`, `run_synthetic_batch`),
`build_ood` and the batch branch in `scripts/train_rnn.py`, wrapper `scripts/ood_student_pipeline.py`.
Written for this project from the user's description; no teammate code used.

## Flags (all off by default)

| Flag | Meaning |
|---|---|
| `--ood-teachers RUN@EP,...` | frozen teachers from `artifacts/runs/RUN/model_epEP.pt`; their mean (first two outputs) is the soft target. Turns the feature on. |
| `--ood-mode mix\|only\|alternate` | `mix`: each batch synthetic with probability `--ood-frac` (deterministic in seed/epoch/batch); `alternate`: every round(1/frac)-th batch; `only`: all. A synthetic batch replaces a real one (the real batch is still drawn, so the real order and the schedule length are unchanged; the OneCycle schedule is not overrun). |
| `--ood-frac F` | synthetic share (default 0.5). |
| `--ood-aug LIST` | `splice`, `warp[:P/LO/HI]`, `noise[:SIGMA]`, `mixup_paths[:P]`. `sign_flip_bidask` is rejected (see below). |
| `--ood-weight teacher\|uniform`, `--ood-scale S` | row weights and teacher-mean scale (use 0.8125 to match the dsq cache coefficient sum; the wrapper does). |
| `--ood-sources N` | fit sequences read per synthetic batch (default 24). |
| `--ood-quantile-table T.npz` | optional table from `build_quantile_table.py` for `noise`; default is built from 16 seeded fit sequences. |

Augmentations (all causal, applied per path):

* `splice`: concatenate random 200-1500 row segments of different fit sequences (a draw of 24 sources per batch).
  No state reset or marker at joins, so regime jumps are visible to teacher and student. Without `splice` a path is a plain window of one source sequence.
* `mixup_paths` (prob. P): lambda~U(0.25,0.75) blend of two aligned paths on the 88 book price/volume columns
  (0-43, 52-95); trade columns and extras come from path A.
* `noise` (applied to half the paths): on rows where a source column changes, jitter in quantile space (u +
  sigma*eps, one eps per row per column group so ladders move together) and map back through the quantile table;
  static rows stay static (forward fill on the source change mask). Values can leave the tick grid; out-of-table values clamp to the fit extremes.
* `warp[:P/LO/HI]`: reuses `wnn.augment.warp_index` (repeat/drop fully static rows). Dropped rows leave a padded tail excluded by the kept-length mask.
* Event features are recomputed from the generated raw rows by the ordinary `build_inputs`.
* `sign_flip_bidask`: not implemented. The 112 columns are anonymised, there is no verified price reflection, and the
  teachers never saw reflected books, so the teacher output would not be market behaviour.

Loss on a synthetic chunk (no label): `sum(w (student - t)^2) / sum(w)` with `t = clip(scale * mean teacher, -2, 2)`,
`w = |t| * [row >= 99 and row < kept]` (the unlabelled analogue of the real weights `|clip(y)| * valid`; `uniform` drops `|t|`).
No propensity, no auxiliary heads on synthetic chunks. One optimizer step per window, like real chunks.

Validity guards: needs a clean `--split-manifest`; each teacher's provenance must carry the same manifest hash and may not
have fitted any evaluation sequence; teacher and student must share the input pipeline (`INPUT_ARGS`); the sampler is built from
`provenance["fit_row_groups"]` only, refuses overlap with the evaluation groups and asserts every read is a fit sequence.
Not combinable with controlled macro rows, origin views, thinning/warp views, intra-epoch points, curriculum.

## CLI

```sh
# trainer directly (checkpoints already on the host)
python scripts/train_rnn.py <dsq student recipe> --init c20261003_combo_s3_f1@ep3 \
  --ood-teachers vt_f1_gru256_s4@ep3,vt_f1_gru256_s5@ep3 --ood-mode mix --ood-frac 0.5 \
  --ood-aug splice,warp,noise --ood-scale 0.8125 --ood-sources 24

# pipeline wrapper, same arguments as scripts/distill_pipeline.py plus --ood-*
python scripts/ood_student_pipeline.py --arm ood_t256 --fold 1 \
  --teachers vt_f1_gru256_s4,vt_f1_gru256_s5 --parent c20261003_combo_s3_f1@ep3 \
  --ood-aug splice,warp,noise --ood-frac 0.5
python scripts/ood_student_pipeline.py --arm ood_only --fold 1 --teachers ... --parent ... --ood-mode only --real-distill none
```

Suggested first arms: `splice` alone, `splice,warp,noise`, `splice,mixup_paths`; frac 0.25/0.5; `ood_only` as the mechanism-isolating arm.

## Vast deploy list

Overlay on the host (same layout as v8): `src/wnn/ood_student.py` plus the v9 `src/wnn/*` it sits with (unchanged from v8),
`scripts/train_rnn.py`, `scripts/ood_student_pipeline.py`. The host must already have `scripts/distill_pipeline.py`,
`scripts/build_distill_cache.py`, `scripts/predict_run.py`, the teacher checkpoints `artifacts/runs/<teacher>/model_ep3.pt`
(+ `cache/pred_<teacher>_{train,valid}.*` for the cache path), the parent checkpoint, fold split manifest and propensity run.

## Cost and comparison

Expected GPU cost about 2x a dsq student fit: per synthetic batch each teacher adds a no-grad forward over the same 16 x 20000
rows (a forward is roughly a third of a train step per teacher; two teachers plus the student step), and a synthetic batch also costs ~1-2 s of
parquet reads for the 24 source sequences (host CPU, not measured). Memory: 24 float32 sequences (~220 MB) plus teacher inputs, no extra VRAM peak beyond one more forward. All numbers are estimates until a real-device profile.

Compare against, same fold/parent/teachers/seed: `dsq_t256` 0.66938 and `dsq_none` 0.66861 (teacher signal was +0.0008 over plain continuation). A positive effect
must exceed that margin; a fixed-seed repeat of the control is needed before reading differences of this size.

## Tests

`WNN_THREADS=1 scripts/limited .venv/bin/python -m pytest <this dir>/tests/test_ood_student.py`: schedule determinism and share, sampler reads
only the fit split (and refuses an overlapping set), splice segments are contiguous source rows without reset, determinism in (seed, epoch, step) with the global torch RNG untouched,
noise causality / static-row preservation / identity at sigma 0, mixup identity, warp padding mask, generated-path event features causal, online teacher targets equal `predict` on the same
input (two chunkings), checkpoint round trip through `build_ood` and `predict_run.load_model` with ancestry/compat refusals, loss weights, student training toward teachers with a kept-mask gradient check, flag-off parity with v8.
