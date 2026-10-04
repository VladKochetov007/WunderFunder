# E465: full-fold eight-TPU precision continuation

Register before any E465 scientific fit. User authorized JAX experiments on
TPU; the real-data one-update E463 preflight is a prerequisite, not the research
endpoint. This experiment uses all eight measured v5-lite devices for training.

Question: can the verified causal GRU/objective port train full 20,000-row
sequences efficiently on eight TPUs, and does BF16 training change predictions
relative to a matched FP32 continuation? Neither arm reproduces Muon. Both
start from the exact same fixed fold-0 combo epoch-3 inference/EMA checkpoint
and a fresh AdamW state. This changes optimizer/history relative to the original
parent and cannot be called a continuation of its original optimizer state.

## Fixed scientific design

- Two arms: FP32 and BF16 computation. Identical FP32 master parameters,
  initialization bytes, shuffled fitting-group order, auxiliary targets,
  teachers, weights, schedule and exported evaluation precision.
- Fold-0 component manifest and its exclusion of TRAIN row group 528; same
  verified fold-specific propensity and two fixed since-teacher caches.
- Global batch 32, four sequences on each of eight devices. One complete epoch:
  373 batches, 7,460 attempted updates, fixed final checkpoint epoch 0. Each
  sequence spans 20,000 rows; carry numerical hidden/EMA state across all
  twenty 1,000-row chunks and detach gradient history at each chunk boundary.
- Same stopped-bootstrap horizon 8, alpha 0.5, since weighting, auxiliary
  structural/next weights 0.3, and EMA consistency objective as the audited JAX
  bridge. Freeze and record learning-rate/scheduler and EMA conventions before
  admitting the fit. No evaluation-based stopping or checkpoint selection.
- Each weighted objective term pools its global numerator and denominator
  across replicas before normalization. Global gradient reduction and clipping
  must reproduce the unsharded global-batch update. Averaging separately
  normalized shard losses or clipped local gradients is not equivalent.
- Export each fixed final model and full excluded-fold predictions in FP32;
  evaluate with the existing frozen clipped global-WP/component scorer.
  Preserve the unchanged starting parent's comparison as an additional anchor.

## Correctness and admission

First verify gradients and an optimizer step against an unsharded reference
with deliberately unequal and zero-mass shards. Include the main, EMA and
auxiliary terms and two successive chunks with detached, carried state.
Verify exact group/count identities, fresh sequence resets, FP32 master-state
finiteness, distributed replica agreement, save/load prediction parity, and
export back to the Torch model before any submission claim.

Keep this source in a separate staging directory. Do not alter the successful
E463 CPU source or its fresh TPU preflight bundle. Root review and successful
real-data TPU preflight precede a full scientific launch.

Resources: eight measured devices, 80% allocator target and 90% sampled HBM
stop per device; initially 12 GiB process-tree host RSS and eight host CPU
threads with 20% physical-host reserve. Record device HBM and host RSS at
phase boundaries and periodically. If production-shape profiling contradicts
these limits, preserve the attempt and revise resources under a fresh identity
before running science. Do not silently change batch, precision or sequence
length. Reserve four TPU-provider hours for recovery; at most eight provider
hours for both precision arms including setup/compilation/exports. Physical
TPU-chip time and provider quota time must be recorded separately from the
cumulative 80-device-GPU-hour campaign budget.

Interpret all local scores as exposed-fold evidence conditional on these
models. A framework port or precision difference is new systems/numerical
knowledge, not automatically a forecasting gain or a public-score forecast.
