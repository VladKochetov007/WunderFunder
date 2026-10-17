---
name: b300-training
description: "Prepare, adapt, profile or review full precision PyTorch training on NVIDIA B300, including causal time series, GRU teachers/students, DDP, logging and reproducible artifact handoff."
---

# b300-training

Read AGENTS.md and research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md. This is a project-authored skill, not NVIDIA certification or a measured hardware recipe. Read references/sources.md for primary sources.

1. Inspect actual allocated hardware, nvidia-smi topology, driver, CUDA runtime, torch build and supported kernels. Record versions and memory; do not install arbitrary nightly versions or assume advertised throughput. Respect current local CPU/RAM and disk scratch rules. No rental/billing changes.
2. Scientific runs use strictFP32: model/input/state/loss/optimizer FP32, no autocast/scaler, TF32 disabled, float32 matmul precision highest. Quantization is a later submission phase. FlashAttention and fused paths require verified FP32 support; use math SDPA otherwise. A user precision change must be an explicit new arm.
3. Choose one GPU per small independent fit or torchrun/DDP for large teachers based on a bounded real-shape profile. DDP is not automatically faster. Record topology and effective global batch, LR schedule and fixed row/update budget. Use NCCL one process/device, distributed sampler component boundaries, equal collective counts and explicit reset. Handle unequal-length sequences without silently repeating evaluation rows.
4. For weighted/masked objectives all-reduce denominators per objective over the full accumulation window. With DDP gradient averaging scale local numerator by world_size/global_denominator; normalize each auxiliary objective separately. Reduce sums/counts before nonlinear official metrics, never average per-rank correlations. Use no_sync for nonfinal accumulated microbatches consistently across ranks.
5. Causal attention masks and output AR use only observed inputs/model prior predictions. Never teacher-force true future-horizon competition labels. Preserve component exclusions across teacher/initializer/preprocessing ancestry. Full20k-contextFP32 attention can be quadratic and huge: measure memory and use explicit chunk/memory semantics, not a guessed batch.
6. Start from the exact frozen optimizer/schedule; changes are arms. AdamW with clipping and WSD/cosine are candidates, not universal best choices. Keep quality-first model/context/epoch ladders explicit; use the AR proposal for concrete starting settings. Choose batch using measured memory and optimization quality, not GPU occupancy alone.
7. Write config/source/ancestry/environment before training. Record successful/attempted steps, per-objective loss, preclip grad norm, LR, nonfinite/clipping events, validation scores, throughput/timing and peak memory; save resumable checkpoints. Rank-local logs are separate. Hook research_reporting explicitly; old trainers are not automatically instrumented.
8. Return evidence and weights archives separately with file/archive hashes, manifests, source commits and private storage locations. Update unified notes and all30 paired cells, including failures. Never claim estimated wall time or external B300 benchmark as our measured result.
9. Profile only when a measured bottleneck warrants it: use the installed Nsight Systems skill if present, capture warmed steady work and NCCL/data/checkpoint phases. Nsight Compute is for a specific kernel question. Do not import external sudo/privileged/container flags as project authorization.
