# H230 source V7: Discord-trick flags

Copy of `h230_source_v6_lead` (unchanged) plus six OFF-by-default research flags from
`diversity_stack_20261009/DISCORD_TRICKS_DEEP_DIVE_20261011.md`. The ideas come from the sibling-challenge
writeup; the implementation is written here from the descriptions only. Every flag is causal (no look-ahead,
state reset at each sequence start), touches fitting inputs only (never labels or evaluation components in
pretraining/extractor/quantile fits), and leaves the official metric alone. With all flags off the model,
losses and weights are bit-identical to v6 (`tests/test_discord_flags.py::test_flags_off_is_bit_identical_to_v6`).

Status: source and CPU unit tests only. No real-data fit, profile or GPU launch was run.
Serving: flags 1, 3 and 6 change the network or its inputs and the Rust engine only supports the plain MLP
head, so they need native work before a submission (the trainer records `native_export_supported=false`).
Flags 2, 4 and 5 are training-only and the resulting checkpoints serve with the existing engine.

## Flags and CLI examples

Append to a normal v6 command line (`...` = the usual recipe).

1. `--head highway`: head becomes Linear(512,256) -> 2 highway layers (`g*GELU(Wx)+(1-g)*x`, sigmoid gate,
   gate bias -2) -> Linear(256,outputs). `head.0`/`head.2` keep their roles, so aux columns, the family head
   and `predict_run.py` work unchanged. `... --head highway`
2. `--pretrain-next-epochs N [--pretrain-next-frac F] [--pretrain-next-lr LR] [--pretrain-next-in-budget]`:
   N epochs of next-row MSE on the 104 standardised raw channels using fitting inputs only, then encoder + GRU
   are kept and the head is left at its fresh initialisation; the optimizer and schedule start afresh.
   `--pretrain-next-in-budget` makes supervised epochs = `--epochs` - N. Fresh plain-GRU fits only.
   `... --epochs 12 --pretrain-next-epochs 2 --pretrain-next-in-budget`
3. `--frozen-extractor PATH`: PATH is a checkpoint from `scripts/train_extractor.py` (GRU-128 next-step model,
   32-d projection). Its projected state is appended to the inputs (211 -> 243); whole sequences are run so
   its state starts at zero per sequence.
   `python scripts/train_extractor.py --out ext_s0.pt --groups fit_groups.npy --split train --epochs 2 --device cuda`
   then `... --frozen-extractor ext_s0.pt`
4. `--aug-warp P,LO,HI`: with probability P per training sequence, fully static rows (all 104 price/volume
   channels equal to the previous row, after the 99-row warm-up) get floor(f) copies plus one with
   probability frac(f), f ~ U[LO,HI]; labels move with rows; truncated/padded tails are masked by the existing
   kept-length path. Stays inside one sequence, so it never crosses a component boundary. Not combinable with
   `--thin-prob`. `... --aug-warp 0.3,0.6,1.6`
5. `--curriculum anti [--curriculum-difficulty running|FILE.npz] [--curriculum-strength 1.0]`: hard half
   (difficulty >= median) sampled with weight `1 + strength*(1 - e/(E-1))` (x2 at epoch 0, uniform plain
   permutation in the last epoch), with replacement, epoch size unchanged. `running` learns per-sequence
   weighted MSE online (first epoch uniform); a file needs `row_group` and `loss` arrays covering every
   fitting group. `... --curriculum anti --curriculum-difficulty running`
6. `--input-act sigmoid|tanh|silu|gelu --input-act-table FILE.npz`: appends the activation of a
   quantile-normal copy of the 112 raw columns (112 extra inputs). Build the table from fitting rows only:
   `python scripts/build_quantile_table.py --out q.npz --groups fit_groups.npy --split train`.
   `... --input-act tanh --input-act-table q.npz`

`scripts/predict_run.py` is the v6 script plus `load_model`/`configure_inputs` helpers (so the new heads and
appended inputs rebuild from checkpoint args), `--device` and `WNN_PROJECT_ROOT`.

## Vast deploy file list (like v6)

- `scripts/train_rnn.py`, `scripts/predict_run.py`
- changed: `src/wnn/nn.py`, `src/wnn/training_sampling.py`
- new: `src/wnn/augment.py`, `src/wnn/extractor.py`, `src/wnn/input_act.py`, `src/wnn/pretrain.py`
- only when used: `scripts/train_extractor.py` (flag 3), `scripts/build_quantile_table.py` (flag 6), plus the
  extractor checkpoint / quantile table files, whose paths are stored in the checkpoint args.
- everything else under `src/wnn` is identical to v6.

## Tests

`WNN_THREADS=1 scripts/limited .venv/bin/python -m pytest <this dir>/tests/test_discord_flags.py`
(24 tests, about 40 s, CPU): causality (future rows leave earlier outputs unchanged), per-sequence reset and
chunk-carried-state equality, flag-off bit parity against v6, warp alignment, sampler annealing and
predict_run checkpoint round trips. `tests/test_family_head_control.py` is inherited and needs real data.
