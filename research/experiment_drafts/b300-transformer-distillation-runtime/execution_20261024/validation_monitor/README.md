# Intermediate checkpoint validation

User-requested correction on 2026-10-24: online training loss alone does not
reveal overfitting. The original trainer evaluates only its fixed epoch-30
endpoint. This external observer adds diagnostics without changing the live
`f1beeb4c210a9da01735e03b1ff6e27be878b873` source, training data, model,
optimization, or endpoint selection.

The first retained checkpoints are epoch 28 for both 630M fits and epoch 15
for both 1.54B fits. Earlier checkpoint files were atomically overwritten;
their validation scores cannot be reconstructed from the online loss log.

- `checkpoint_io.py` reads trusted, locally produced resume snapshots on CPU,
  validates their original profile contract and copies only model weights.
- `evaluate.py` runs causal inference with the original normalization and
  precision, resetting state between complete sequences. It evaluates the entire
  original held-out fold and a deterministic hash-selected 64-sequence fitting
  probe frozen before validation outcomes. Probe loss is a subset estimate.
- `metrics.py` uses the canonical official pooled WP implementation. WMSE uses
  raw predictions, targets clipped to [-2, 2], and absolute clipped-target
  weights. The comparable loss mask covers all rows after the 99-row warmup;
  official validation WP uses the original scored mask.
- `test_metrics.py` checks pooling, clipping, masking, shape and nonfinite handling.
- `watch.py` captures immutable checkpoint inodes through hardlinks after the
  trainer's atomic replacement. One observer per cell evaluates snapshots in
  order. Capturing is blocked below 225 GiB of disk headroom. Each inference
  worker has a separate 16 GiB GPU allocation cap and a bounded lifetime;
  observer cleanup targets only its own child process groups.
- `report.py` updates diagnostic comments on PR32 and PR40 when new complete
  validation results arrive, using the already configured reporting account.

The monitor runs in its own cron session and survives SSH disconnection. A
36-hour watch is bounded; failed checkpoints retain their failure receipt and
are not retried automatically. A worker may share a GPU with training and
therefore adds inference overhead. No optimizer is moved to the observer GPU.
Snapshots retain optimizer state on physical disk for reproducibility.

**Interpretation:** compare same-checkpoint fitting-probe WMSE and held-out
WMSE, then inspect successive validation WP and loss. The logged online epoch
loss used changing weights and is labeled separately. A growing gap, increasing
validation loss and decreasing validation WP provide evidence of overfitting.
These diagnostics expose the outer folds; they do not create a fresh holdout or
authorize early stopping. The frozen final30 endpoint remains the primary result.

Runtime outputs are under
`/data/epanov_s/tabdata/runs/pr32-validation-monitor-20261024`.
Prediction files and weights stay off Git. Each evaluation records checkpoint,
source, config, split, membership and metric provenance beside its scores.
