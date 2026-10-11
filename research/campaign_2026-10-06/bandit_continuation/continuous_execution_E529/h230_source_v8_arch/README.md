# H230 source V8: new recurrent architectures (clockbank, dssm, dualtower)

Copy of `h230_source_v7_discord` (v6/v7 untouched) plus three new `--arch` values. The v7 Discord flags remain
available for the GRU path. With the default `--arch gru` the model, losses and weights are bit-identical to v7
(`tests/test_v8_arch.py::test_default_arch_is_bit_identical_to_v7`).

Status: source and CPU unit tests only. No real-data fit, profile or GPU launch was run. The ideas come from
the four independent round-6 proposals (`diversity_stack_20261009/external_ideation_20261010/round6/oc_r6_*`),
which converge on a multi-rate clock-block bank, a diagonal state-space bank and a dual-tower fusion; the code
was written here from those descriptions. All new code is in `src/wnn/arch_v8.py`.

## What overlaps with existing archs, and what is new

| Existing (`src/wnn/nn.py`) | Mechanism |
|---|---|
| `dual` | dense 2-layer row GRU (every row) + 1 event GRU on i0-price events, joined in the head |
| `twoclock` | dense row GRU + event GRU on a6 endpoint rows (with segment means) |
| `gated`, `tcn`, `structured`, `event-content`, `specialist` | sign/size gate, dilated conv, multi-decay blocks, event content, routed experts |

`dual`/`twoclock` approximate (C) only partly: both keep a dense every-row GRU that sees all columns and rewrites
its whole state each row, and add one small event stream. The new archs have no such path:

* `clockbank` (A): the whole state is partitioned into banks (per instrument price / volume / trade groups,
  extras, plus one always-on base bank). A bank holds its state EXACTLY unless one of its own columns changed on
  that row (hard clock, change flags computed causally from the raw columns with the previous row carried).
  Cells are minGRU (gate and candidate do not read h), each bank also reads its own raw column group directly in
  layer 0. Per-bank initial memory timescales differ. `--cb-layout hint` uses the proposals' fast block
  (columns 0-21, 44-47, 52-69) vs volumes vs rest vs extras. The real per-group update rates were not measured
  here; the groups are semantic and the layout table `CLOCK_LAYOUTS` is meant to be edited after EDA.
* `dssm` (B): linear time-invariant diagonal recurrence `h = a*h + g*(Bx)` with per-channel learned decay `a`
  (init log-uniform timescales 2..`--dssm-tmax` rows) and gain `g` (init sqrt(1-a^2)); no tanh and no gate in the
  state path, GELU only between layers (`y = gelu(C h) + x`). Deviation from the proposal text: gain starts at
  sqrt(1-a^2) rather than 1 so slow channels do not saturate. Not input-selective.
* `dualtower` (C): a book tower (both instruments' px/vol + matching event features) and a trade tower (trade
  px/vol, extras, trade event features), each an event-gated stack of single-layer GRUs that runs only on the
  compacted rows where its own columns changed (held in between, log gap since the previous event as input).
  Towers meet only in the readout `[book, trade, per-row skip, log1p gaps]`: late fusion, no every-row recurrence.

Recurrences are computed with an exact two-level affine scan (`affine_scan`, about T/32 + 32 sequential steps),
so no per-row Python loop. All state is a tuple of tensors (`detach_state` and TBPTT chunk carry work as for the
GRU path), zero at every sequence start. The head is `Sequential(Linear, GELU, Linear)` so `head.2.*`, aux
columns, the 8-row family head and `predict_run.py` work unchanged.

## CLI examples (append to a normal v7 command line; `...` = the usual recipe)

```sh
... --arch clockbank --hidden 384 --layers 2 [--cb-layout semantic|hint]      # 0.98M params, 8 banks of 48
... --arch dssm      --hidden 384 --layers 2 [--dssm-tmax 512]                # 0.97M params
... --arch dualtower --hidden 224 --layers 2 --event-hidden 160 [--tower-skip 128]   # 1.10M params
```

`--hidden 448` gives clockbank 1.31M. For dualtower `--hidden` is the book tower, `--event-hidden` the trade
tower, `--layers` the GRU layers per tower. `--optim muon` works (the models declare their recurrent matrices;
`--muon-scope` rnn/hidden/all as before). Not supported (rejected at startup): `--enc != linear`, dropout,
`--head highway`, `--rnn-init != default`, QAT, factorization, QMC init, next-row pretraining, controlled macro
distillation losses. `--freeze` prefixes are validated (`inp`, `layers`, `book_gru`, `trade_gru`, `head`...).

CPU 1-thread fwd+bwd on a 4x1000 batch (indicative only): GRU-256x2 1.6 s, clockbank-384 1.3 s, dssm-384 0.9 s,
dualtower 2.4 s; MACs per row about 1.0M for the first two, GPU speed is untested.

## Vast deploy file list (like v7)

- `scripts/train_rnn.py`, `scripts/predict_run.py`
- changed: `src/wnn/optim.py` (Muon groups for the new archs)
- new: `src/wnn/arch_v8.py`
- plus the v7 changes (`src/wnn/nn.py`, `training_sampling.py`, `augment.py`, `extractor.py`, `input_act.py`,
  `pretrain.py`) unchanged from v7; everything else under `src/wnn` is identical to v7.

## Serving: the Rust engine cannot run these yet

The trainer records `native_export_supported=false` for the new archs. They are research/teacher models until
native work is done, per arch:

* `clockbank`: change-flag computation on 7 column groups (needs the previous raw row), per-channel masked
  minGRU update (no recurrent matmul; two GEMVs per layer), bank-tick gather, own-column projections, new head
  input layout. Closest to the existing GRU kernel; int8 GEMV reuse is straightforward.
* `dssm`: diagonal decay vector, softplus gain, two GEMVs per layer and GELU between layers. Simplest kernel; no
  recurrent GEMV at all.
* `dualtower`: two independently gated GRU stacks (an update runs only on that tower's event rows, the state
  and the held output persist otherwise), explicit gap counters, previous-row carry, skip projection and the new
  head. Per-row cost is variable and lower on quiet rows.

Alternatively these can be used only as teachers and distilled into the served GRU student (cache path).

## Tests

`WNN_THREADS=1 scripts/limited .venv/bin/python -m pytest <this dir>/tests/test_v8_arch.py` (35 tests, about
35 s, CPU, under 1 GB): scan vs sequential loop, parameter count and shapes (aux columns, family head), causality,
per-sequence reset, chunk-carried-state equals the full sequence for several chunkings, exact bank freezing
between ticks, tower state held between events, gradients and Muon steps, checkpoint round trip via
`predict_run`, rejected flag combinations, and default-arch bit parity against v7. `test_discord_flags.py` is
inherited and still passes; `test_family_head_control.py` is inherited and needs real data.
