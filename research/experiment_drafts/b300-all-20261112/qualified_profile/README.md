# PR75 B300 placement profile

This is the first receiving profile for the new B300 wave. It uses the already
qualified canonical PR75 source capsule at commit
`8626000a4b965763a70011a0291a891aa6535d7e`; it does not import the unaccepted
PR81 experiment adapters. The source archive used by the existing receiver is
`b300-qknorm-20261111/source-r1.tar.gz`, SHA256
`08f8e31d18a523ba7c543fe71fa3b31799cfe1af566672d6b903e30beac639f5`.

The two configs isolate placement while preserving the same 256-sequence global
microbatch, four 256-row chunks per update, 3,280 planned optimizer updates,
four-epoch endpoint, LR `3e-4`, Muon/AdamW roles, BF16/BF16x9 math, FP32 master
parameters and optimizer state, full-train normalization, and the official
train/valid file boundary. The one-GPU config uses local batch 256; the reference
two-GPU config uses local batch 128 on each rank. This is a new engineering
placement comparison, not a historical PR75 quality result.

The one-GPU update receives the same shuffled sequence group as the two-GPU
update: rank 0 and rank 1 take alternating indices from one deterministic
permutation, so their two 128-sequence batches contain the same 256 examples as
the single 256-sequence batch. The last batch has 85 real sequences; the old
distributed form pads one inert sentinel, excluded from the objective mask.
Floating-point reduction order differs. The profile receipt must confirm the
actual update count, resource use and optimizer/reload contract on the receiver.

## Config identities

| Placement | Config | SHA256 | Devices |
|---|---|---|---:|
| One B300 | `transformer_world1_batch256_s204101.json` | `475aba168f52df0305553bccf7954048a3c02bf362c9b34633ce4c696d1343db` | 1 |
| Two B300s | `transformer_world2_batch128_s204101.json` | `a2be8ca9169f2edd879b4c2406c4c302bec3e461ea10c98b60262efcdda3f335` | 2 |

Both configs use seed `204101`, train membership SHA256
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`,
10,325 fitting sequences and the full 1,873-sequence / 3,528,273-scored-row
official validation endpoint. No extra temporal order is inferred from `seq_ix`.

## Receiver command

On the clean PR75 worktree, set the actual persistent paths and selected free
device IDs, then run one bounded 50-update profile. For one GPU:

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUBLAS_EMULATE_SINGLE_PRECISION=1
export CUDA_VISIBLE_DEVICES=<one-free-b300-index>
uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  scripts/train_sequence.py \
  --config /path/to/transformer_world1_batch256_s204101.json \
  --data-root /path/to/data-root \
  --split-manifest /path/to/frozen-split.json \
  --output /path/to/persistent/profile-world1-b256 \
  --profile-updates 50
```

The paired reference uses the same command with `--nproc-per-node=2`, two free
B300 IDs, and `transformer_world2_batch128_s204101.json`. Keep the complete
output and inspect `profile_receipt.json`, `run_metadata.json`, `rank0.jsonl`,
`rank1.jsonl` when present, and canonical reload diagnostics. This profile is an
admission artifact; it starts no full fit.
