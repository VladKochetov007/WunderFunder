# Native GPU/TPU comparison contract R2

R2 is the identified continuation of the preserved R1 contract. It compares
our independently initialized GRU on native PyTorch/cuDNN/CUDA using two T4
workers and JAX/XLA using one eight-device TPU allocation. It measures the
implemented hardware and software stacks; it does not claim a chip-to-chip
speed ratio.

The correctness anchor is matched FP32: raw 112 plus 99 causal event features,
GRU 256×2, linear projection, GELU, 350 complete fit sequences, and 468
exposed evaluation sequences. Both backends consume the same fresh initial
state, normalization, ordered sequence IDs, and 220-row OneCycle learning-rate
and beta1 trace. AdamW, weight decay, clipping, detached recurrent carry,
normalization, update count, and sequence masks remain fixed. The two objective
arms are weighted MSE and weighted Huber with delta 1. Targets are clipped to
±2 for the objective; absolute clipped-target magnitude supplies the original
weight. Evaluation uses the unchanged `global_wp_clip2_v1` metric and its fixed
scored mask. The exposed fold-0 screen supports a local comparison only.

Each update uses 32 sequences and chunk length 2,000. Loss numerators are
summed over the global batch and divided by the global weight sum before one
global gradient clip at 1 and one AdamW update. The final two of 350 sequences
are padded on TPU with zero weights; their padding contributes neither to
normalization, objective, nor evaluation. GPU R4's actual completed producer
contract supplies the initialization, normalization, order, schedule, and
hash-bound first-batch fixture. Native TPU startup must consume the mounted
producer cache through the pinned R4 cache reader and prove that its first
batch exactly matches that producer fixture before accelerator parity work.

The producer binding for this revision is GPU R4:

- Kernel: `vladkochetov07/wnn-loss-pair-gpu-20261009-v4`.
- Code packet: `vladkochetov07/wnn-loss-pair-code-20261009-r4`.
- Source commit: `1bc0fbacf53cdc08a8c4664c6f66f38f8980092c`.
- Kernel SHA-256: `86346f9c1addefbd38d3dcec52b18f514698a9514e7917aeb6081cc32f7eb039`.
- Candidate packet SHA-256: `dd1f57dc288b8b75df349f150170c6fc60cdff474f2f6f14897baa60b9fcc16a`.
- Actual completed training-contract SHA-256: pending root review and a
  successful admitted GPU R4 run.

The cache-reader repair is the exact module in R4 commit above, independently
bound by its SHA-256 in the sealed source manifest. Its focused bridge test
exercised `ParsedSequenceCache.open → preverified_sources → load_campaign_split`
against producer receipts, split membership, and sparse header-valid mappings.
The R4 owner reports five focused tests passing in 3.18 seconds; this is
source-level evidence, not proof of a successful GPU run.

Before any fit, the TPU source must pass the complete tiny lifecycle suite and
an actual eight-device gate using the producer's real first-batch fixture and
initial OneCycle row. The gate checks forward values, gradients, AdamW update,
save/reload, recurrent reset, prefix causality, global replica reduction,
the 30-sequence remainder mask, and full batch32/chunk2000 memory. Actual
device kind/count and versions must be recorded from runtime objects. Timings
include the host call through `jax.block_until_ready`; separate cold compile
from repeated synchronized warm updates.

Native BF16 is a separate TPU arm, gated by the matched FP32 actual-data gate.
It uses FP32 master parameters and AdamW accumulators. Record first-update
loss, global gradient norm, clip scale, and FP32/BF16 prediction drift on the
same actual input fixture. Compare identical OneCycle row and initialization.
This is not equivalent to GPU FP16. GPU native FP16 remains a separately
identified source successor and is not part of the current R4 FP32 producer.

Retain per-arm final weights, optimizer/checkpoint state, normalization,
predictions, exact evaluation membership, per-update gradient/LR/beta1/clip
and skip traces, timings, resource peaks, device/software inventory, and
artifact hashes. Report single-fit latency separately from aggregate
two-worker GPU pair throughput and sequential eight-device TPU pair wall time.
Separate setup and cache validation, cold compile, warm updates, complete fit,
prediction/export, and total worker wall time. Time-to-comparable-score may be
reported only from observed saved predictions and timestamps on the same 468
IDs; no FLOPS estimate or endpoint interpolation stands in for a measured
curve.

Root owns compute admission, provider actions, allocation, and final synthesis.
The prepared packet keeps provider push and fitting disabled until the exact
R4 training contract and source manifest are root-bound. A successful parity
gate is necessary evidence, not spending authority.
