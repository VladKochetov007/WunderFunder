# Native GPU/TPU comparison contract R3

R3 preserves the R2 native TPU implementation and changes its producer binding
to the completed R5 export-only recovery of the R4 GPU controls. R4 completed
the matched WMSE and Huber fits; R5 exported FP32 validation predictions and
the actual shared training contract without retraining or rewriting those
checkpoints. R3 consumes the exact R5 `training_contract.json` bytes with SHA-256
`5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7`. Root's
terminal review is `ROOT_GPU_R5_TERMINAL_REVIEW_R1.json`, SHA-256
`60e9770de432a6b39d3ae41c818d60b2d47b5298f630b33ee3483a0e7fecc7f9`.

R2 remains preserved as V2. Its contract's ambiguous sentence about padding is
corrected here: the 350 real fit sequences form ten full batches of 32 and one
final batch with 30 real sequences plus two zero-weight padding rows. The
padding does not change fitting membership and contributes neither to the
global weight denominator, objective, nor gradient. Evaluation still covers
the fixed 468 exposed fold-0 sequences.

The matched FP32 anchor remains raw 112 plus 99 causal event features, a two-
layer GRU with hidden size 256, linear input projection and GELU, 32-sequence
batches, 2,000-row chunks, 220 updates per arm, AdamW, OneCycle, shared
initialization, normalization, sample order, schedule, masks, and one global
gradient clip at 1. WMSE and Huber delta 1 remain separate arms. Targets are
clipped to ±2 and weighted by the absolute clipped-label magnitude. The official
evaluation metric remains `global_wp_clip2_v1` with its fixed mask. The exposed
screen supports local comparison only.

R5's actual producer contract and all 14 shared roles must pass byte, path,
schema, recipe, initialization, normalization, order and first-batch checks
before JAX accelerator setup. The TPU worker then requires exactly eight
runtime TPU devices and runs lifecycle, gradient, save/reload, recurrent reset,
prefix-causality, replica-reduction, remainder-mask and full batch32/chunk2000
memory gates on the actual producer fixture. It records device identity and
synchronized host-plus-device timing, with first compile separated from five
warm updates. R5 contract, 14 shared-role hashes, cache identity, fold0
membership and the first ordered batch are persisted with each run.

Native TPU BF16 remains a separate, later admission. It retains FP32 master
parameters and AdamW accumulators and records the same initial schedule row,
loss, gradient norm, clip scale, prediction drift and save/reload evidence. GPU
FP16 is a separate source successor and is not inferred from TPU BF16. A passing
parity gate is correctness evidence, not fit or spending authority.

Retain arm weights, optimizer/checkpoint state, predictions, exact evaluation
membership, gradient/LR/beta1/clip and skip traces, phase timings, device and
software inventory, resource peaks, and artifact hashes. Report single-fit
latency separately from two-T4 aggregate throughput and eight-device TPU pair
wall time. Measure startup, input verification, compilation, warm updates, full
fit, prediction/export and collection. Do not infer a speed ratio from FLOPS or
interpolate an unobserved time-to-score curve.

This is a pending source candidate. Root owns source review, exact producer
control admission, provider push, profile/fit allocation and result synthesis.
The sealed R3 candidate keeps push, fit and BF16 permissions disabled.
