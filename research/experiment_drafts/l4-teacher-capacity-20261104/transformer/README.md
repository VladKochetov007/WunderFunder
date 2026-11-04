# Small causal Transformer teacher adapter

This packet adapts the existing `scripts/train_sequence.py` / `wnn.training`
runtime. It does not define a second trainer. The small core patch lets the
existing model factory size the Transformer projection and fitting-only
normalization buffers to the already-supported, deterministic `event211`
sequence input. The builder computes all 211 columns from each complete raw112
sequence before chunking, so event clocks and Transformer KV state both reset at
whole-sequence boundaries.

## Frozen teacher recipe

`configs/causal_transformer_d384_l8_h8_s1001.json` and `_s1002.json` are matched
fresh-scratch runs. Both use d384, 8 causal RoPE layers, 8 heads, a 1024-row
rolling window, FFN width 1536, AdamW at 3e-4, weight decay .01, global gradient
clip 1, strict FP32, four fixed epochs, and all epoch snapshots. The FFN width
uses the existing integer multiplier field: `384 * 4 == 1536`. Attention uses
the existing math SDPA path to keep strict FP32; no AMP, TF32,
quantization, initializer, all-corpus checkpoint, or teacher parent is used.

`context_rows=1024` is a rolling per-layer attention window of at most 1024 total
rows including the current row: up to 1023 preceding rows from the KV cache plus
current rows. Across the eight layers, attention can propagate through up to
8,185 assembled feature rows. The event211 builder also carries causal held-event
and recency values from earlier raw rows, resetting both with each new sequence.
KV state and absolute RoPE positions pass across chunks, and the cache is trimmed
to 1,023 rows. Training detaches that cache after every chunk, so accumulation
sums objective gradients across chunks without extending the backpropagation
graph across them. Each batch contains whole 20,000-row sequences and begins
with fresh model state.

The initial resource recipe is batch 8, 1024-row chunks, and four accumulated
chunks, or up to 32,768 forwarded rows per full update window. Selected objective
rows are lower in the warmup-containing window and the final partial chunk. This
yields 25,820 total updates for 10,325 train sequences over four epochs; the 5%
warmup is 1,291 updates, then `warmup_cosine_v2` follows an endpoint-aligned
cosine to zero on the final update. The first engineering profile is 50 real-data updates so the
declared 50-step diagnostic runs once. Its GPU memory/throughput result must be
reviewed before authorizing either full fit; this packet contains no profile or
fit result.

The training objective mask is every row from index 99 onward. The fixed train
diagnostic uses the same mask on 32 fitting sequences. Each completed epoch
reports all 1,873 official validation sequences using the official `is_scored`
mask and row index >=99; the pooled metric is `global_wp_clip2_v1`. The split
manifest is the official chronological manifest with SHA256
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`. Fit and
normalization membership must be the 10,325 train sequences only. This validation
has been inspected repeatedly and remains exposed development evidence.

## CPU-only adapter checks

Run from the receiving source checkout after copying this packet and the two
core adapter changes:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/l4_teacher_capacity_20261104/transformer_cpu_check.py
```

The check validates both frozen configs through the production parser, exact
update/scheduler counts, 211-wide model and normalizer, causal suffix isolation,
chunk continuation, whole-sequence state reset, warmup/scored-row masks, and
parity between the streaming official metric and its reference formula. It reads
no competition data and performs no fit or GPU operation.

## Receiving commands

From the exact `source-tf-r1` checkout, set `WNN_RUN_ROOT`, `WNN_DATA_ROOT`,
`WNN_SPLIT`, and one `CUDA_VISIBLE_DEVICES` assignment to the owned L4 selected
by the root operator. Keep the same device assignment for each profile and its
fit because the profile receipt binds the runtime device identity. First run one
50-update profile; independently review the receipt before marking it approved.
Only then run the same seed's four-epoch fit. The profile receipt is a mandatory
trainer admission gate.

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CONFIG=research/experiment_drafts/l4-teacher-capacity-20261104/transformer/configs/causal_transformer_d384_l8_h8_s1001.json
PROFILE="$WNN_RUN_ROOT/transformer/s1001/profile"
FIT="$WNN_RUN_ROOT/transformer/s1001/fit"
torchrun --standalone --nnodes=1 --nproc-per-node=1 scripts/train_sequence.py \
  --config "$CONFIG" --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT" \
  --output "$PROFILE" --profile-updates 50
```

After the profile receipt has been reviewed and `approved` is true:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 scripts/train_sequence.py \
  --config "$CONFIG" --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT" \
  --output "$FIT" --profile-receipt "$PROFILE/profile_receipt.json"
```

Repeat with the seed1002 config and separate output directories. Do not turn a
failed profile into a fit. A changed batch/chunk recipe requires its own exact
profile receipt before fitting.
