# GALAM local monitoring and L4 comparison preparation

Status: implemented and checked on CPU/API/browser fixtures; no scientific fit,
GPU workload or measured quality. Parent is the source-traced [GALAM reconstruction](../galam-reconstruction-20261027/README.md),
not an exact historical reference. The user requested actual online train/valid
monitoring, a local dashboard, recurrent alternatives and a realistic L4 plan.
Lineage: H-GALAM-L4-20261028 / E-GALAM-L4-20261028, parent H-GALAM-20261027.

- [Contract](CONTRACT.json) freezes current engineering scope, telemetry interchange
  and verification boundaries. New fits are not dispatched by this packet.
- [L4 inventory](L4_PREFLIGHT.json) records one persistent read-only SSH session:
  eight L4 GPUs, 192 logical CPUs and728GiB host memory. GPU memory is separate per
  device, not a single192GB pool. Both data mounts are full and only an incomplete
  train download is present in the inspected project; a full fit is not ready.
- [Implementation and commands](IMPLEMENTATION.md) map the actual modules,
  dashboard services, telemetry/retention semantics and future receiver steps.
- [Prospective panel](PLAN.json), [six recipes](configs/README.md) and
  [eleven-section draft](PR.md) prepare3arms x2seeds. [Inspection](configs/INSPECTION.json)
  records loaded recipes, effective topology and parameter counts; no fit ran.
- [Validation](VALIDATION.json), [independent review](INDEPENDENT_REVIEW.json),
  [browser checks](UI_CHECK.json) and [sparse-history check](SPARSE_UI_CHECK.json)
  qualify correctness claims. The [preview](plots/dashboard-overview.png) is
  synthetic, not a fitted model's result.
- [Browser captures](plots/README.md) explain desktop/mobile artifact provenance.

## Measurement and interpretation

The model preserves recurrent time dependence. GPU kernels can batch independent
sequences and fuse gates, but ordinary LSTM/GRU do not remove their dependence on
the preceding hidden state. GRU has three gate projections versus four for LSTM;
this does not imply25% less total training time. At equal input/hidden widths,
recurrent affine parameter counts are4H(I+H+2) and3H(I+H+2); other model work,
launch overhead, caches, data and validation affect the elapsed ratio.

The current reference graph also iterates temporal steps in Python. A future
vectorized implementation must pass causal/chunk/state/gradient checks before
being treated as numerically equivalent. A measured warmed one-GPU profile is
needed before promising an ETA or increasing workers per GPU. Compare completed
declared work/hour, not a utilization percentage alone.

The prospective initial screen is3arms x2matched seeds =6new fits: full LSTM
GALAM, explicit full GRU adaptation, and LSTM+attention local-only. The same
official chronological primary boundary and fixed training recipe apply. GRU
has no cell state, so its3-channel decoder changes more than the recurrent
cell; report parameter counts and do not attribute its entire delta to GRU gates.
No Cartesian hyperparameter grid or implicit30-repeat matrix is proposed.

## Monitoring semantics

Comparable train-eval/valid loss share one plot; train-eval/valid official scores
share another. Online fitting loss has its own label because its row mask and
training-time BatchNorm behavior differ. Score per target, optimizer diagnostics,
branch activation/gradient norms, gates, attention and relative-bias diagnostics,
resources and timings remain separate selectable panels. Absent measurements
stay absent, and a demo is labelled synthetic rather than a scientific result.

Primary evaluation is the complete official valid membership, using the existing
pooled official clip2 weighted Pearson scorer and scoring mask. All transforms,
initializers and teachers exclude valid. The protocol is [ADR0044](../../../specs/adr/0044-chronological-primary-validation.md);
shuffled seq_ix must never be used to reconstruct chronology. Repeated use of
valid is exposed selection, not an untouched confirmation set.

## Outstanding receiving-host work

Recover or mount an already verified complete official train file without
deleting other projects or treating a sparse.partial file as completed data.
Provide sufficient physical-disk headroom, prepare and verify actual component
and target indexes, generate the chronological manifest, validate the exact
runtime, run bounded correctness/profile checks and record a declared fit budget.
No remote installation, data repair, publication or training was performed here.
