# H218 block-memory trainer integration

Code-only handoff. No fit ran. No profile ran. No speed or score is claimed.
The prototype remains unfit and unprofiled. This note records an implementation
that a later actual-device run can launch. It is not experimental evidence.

## Changed files

- `src/wnn/block_memory_attention.py`
- `scripts/train_rnn.py`
- `tests/test_block_memory_trainer_integration.py` (written, not executed)
- `H218_IMPLEMENTATION_HANDOFF.md`

No evaluator, resource controller, ADR, `AGENTS.md`, source snapshot, or
serving package was edited.

## API and defaults

`--arch block-memory` builds `BlockMemoryAttention` through `make_model`.
The same normalization buffers, membership, fixed-checkpoint, AMP, and profile
path stay in `scripts/train_rnn.py`. Prediction and training still call
`forward_chunk`, which calls `model(inputs, state)`.

| Control | Default for this arch |
|---|---|
| `--hidden` | 256 (existing trainer default; width) |
| `--layers` | 4 when the command omits `--layers`; an explicit `--layers` or `--layers=N` is kept. Other architectures still default to 2 |
| `--block-heads` | 8 |
| `--block-rows` | 256 |
| output width | the `n_out` already computed by the trainer, including ordinary auxiliary columns |

`make_model` requires positive `--hidden` divisible by positive `--block-heads`,
and positive `--block-rows`. The module still requires positive input, layer,
and output sizes.

State is one `BlockState` per layer: completed-block `memory`, unfinished-block
`inputs`, and `output_sum`. `BlockState.detach()` detaches all three.
`detach_state` already calls `.detach()` on each element of a tuple, so the
existing TBPTT boundary works without a second detach implementation. Chunk
edges do not need to land on a block boundary; a partial block stays in the
state and the next chunk continues it.

Normalization is unchanged: registered `mean` and `std` buffers, applied as
`(values - mean) / std` before the input projection.

## Head change

The previous prototype head was `LayerNorm` then a single `Linear`, so the
output weight was `head.1.weight`. Fixed checkpoints and `predict_run` read
output width from `head.2.weight`. The prototype head is now:

`Linear(width, width)` -> `GELU` -> `Linear(width, n_out)`

That is `head.0`, `head.1`, and `head.2`. The block stack and the mean/std
normalization were not otherwise redesigned. This is a change to our own
prototype. Older checkpoints of the LayerNorm head will not load.

## Unsupported flags

`block_memory_rejection` runs immediately after argument parsing, before any
dataset read. `make_model` also rejects geometry, dropout, cell, encoder, and
unknown `--freeze` prefixes. The trunk prefixes are `project` and `layers`.

Rejected for this arch:

- `--numeric-map` (also rejected later because numeric maps require a plain GRU)
- `--qat` (also rejected later unless the model is a plain GRU; `configure_qat` raises on this class if a preset is applied)
- `--recurrent-rank` (also rejected later; `configure_factorized` raises unless rank is 0)
- `--privileged-mode` other than `none` (previously reached input concatenation when a fit-groups manifest was present)
- `--init` parent weight continuation
- `--rnn-init` other than `default` (would look up a missing `model.gru`)
- `--cell` other than `gru`, `--enc` other than `linear`, and nonzero `--dropout` (these would otherwise be ignored)
- `--optim muon` and any `--optim-switch` (Muon groups require `model.gru`)
- `--credit-mode` other than `deterministic` (the credit autograd function expects tensors, and `BlockState` is not a tensor)
- `--readout-activation fixed_mix` (also rejected later for non-GRU models)
- sentinel value, indicator, or base-width arms (also rejected later for non-GRU models)
- `--init-qmc`
- non-positive hidden, layers, heads, or block rows
- hidden not divisible by `--block-heads`

`--resume-state` is still the trainer's same-recipe training-state path.
A mismatched state dict fails in `load_state_dict`. It is not a GRU warm start.
Raw scratch means a new block-memory initialization with fitted `mean` and `std`,
because `--init` is closed.

## Proposed actual-device profile

Not run. The existing profile exit still performs `--profile-warmup` plus
`--profile-steps` optimizer updates on real training batches, prints `PROFILE`
and `PROFILE_UPDATE_EVIDENCE`, and returns before checkpoint selection. Authorize
that separately. One shape that uses the integrated defaults:

```sh
scripts/train_rnn.py \
  --name h218_block_memory_profile \
  --arch block-memory \
  --hidden 256 \
  --block-heads 8 \
  --block-rows 256 \
  --device cuda \
  --precision fp16 \
  --batch 4 \
  --chunk 256 \
  --loss wmse \
  --epochs 1 \
  --checkpoint-policy fixed \
  --checkpoint-epoch 0 \
  --no-eval-during-fit \
  --profile-warmup 1 \
  --profile-steps 2
```

Omitting `--layers` selects 4. CUDA fp16 uses the trainer's existing autocast.
A second profile with `--chunk 200` would cross a 256-row block inside the
carried `BlockState`. Neither command has a measured runtime.

## Tests written, not executed

`tests/test_block_memory_trainer_integration.py` covers:

- `make_model` head layout, `head.2` output width, and `mean`/`std` reload
- `forward_chunk` plus `detach_state`: partial-block contents survive, and
  backward from the next row does not fill the pre-detach output gradient
- full-sequence parity for uneven chunks, including a detach at every edge
- `make_model` errors for bad divisibility, dropout, and an unknown freeze prefix
- layer default 4 unless `--layers` is present, and a clean rejection result
- one message per unsupported modifier above
- `main` exits with status 2 on `--init` before creating `datasets/`

`tests/test_block_memory_attention.py` was not edited and was not run. Its
causal, reset, chunk, and gradient checks still target the same block state.
The head change means those checks no longer describe the old LayerNorm readout.

Syntax was checked with `ast.parse` on the three Python files. Torch and pytest
were not imported or run in this session.

## Constraints

- No serving path, submission ZIP, or native export was added.
- The per-layer memory update remains an internal `GRUCell`, not `model.gru`.
- There is no dropout inside the block-memory module.
- This delivery did not read competition data, start a fit, or touch a GPU lane.
