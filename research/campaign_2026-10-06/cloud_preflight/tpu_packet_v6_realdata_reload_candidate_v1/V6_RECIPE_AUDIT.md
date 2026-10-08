# V6 recipe-to-code audit

## Frozen inputs and ancestry

The JAX parameters come from own run `c20261003_combo_s3_f0`, fixed epoch 3.
The packet verifies the E463 source manifest and direct file inventory, then
checks the Torch checkpoint, JAX NPZ, NPZ sidecar, and `src/wnn_jax/rnn.py`
against the hashes in `packet_manifest.json`. The embedded parent provenance
binds epoch 3 to the same fold manifest and exact source-qualified fit and
evaluation group sets. The raw parquet pair and the selected fold assets are
hash-checked before the TPU child starts.

The selected update order is `bridge.seeded_group_batches(valid_fit, seed=3,
epoch=0, batch_size=16)[0]`. It contains 16 unique `("valid", group)` pairs.
The runtime gate requires every pair to be in both the frozen 1,405-group
fit-VALID set and the parent checkpoint's fit-VALID ancestry; evaluation
membership is checked separately. These are previously exposed fit examples,
not a held-out sample.

## Objective and optimizer mapping

The bridge constructs the same production supervision with SBD `alpha=0.5`,
horizon 8, fixed since-s3/since-s4 teacher mean, propensity weights, structural
labels, and next-change targets. `collate_sequences` applies the valid-mask
correction and since-feature scale. The probe takes rows `[0:2000]` from each
of the 16 complete 20,000-row sequences, shards them as eight replicas of two
sequences, and runs one causal GRU chunk from zero recurrent state.

For local replica numerators `N_r` and global masses `D`, the reused E465 helper
reports

```text
L = N_primary / D_primary
  + N_consistency / D_rows
  + 0.3 * N_structural / D_rows
  + 0.3 * N_next / D_next
```

Each replica differentiates `8 * sum(coefficients * N_r / D)`. The copied
`pmean_and_clip_gradient` averages these local gradients before the global
norm-1 clip, which recovers the gradient of the globally pooled objective
before clipping. The copied E465 OneCycle implementation supplies schedule
index 0 from the full 7,460-update B16/C2000 epoch schedule. The reused JAX
AdamW update uses a fresh optimizer state, weight decay 0.01, beta2 0.999, and
the scheduled beta1. The EMA update is `0.999 * old + 0.001 * updated` for
trainable tensors; mean/std buffers remain fixed.

These helper files are byte-identical to the owned E465 sources and their
expected hashes are fixed in the builder and packet manifest. The V5 synthetic
parity/profile path and its 20% initial HBM headroom and 80% peak HBM ceiling
remain intact; the actual update is included before the final HBM measurement.

## Scope and admission

The probe saves and bitwise reloads updated student parameters, updated EMA
parameters, and AdamW moments/count. It runs inference from the reloaded EMA,
checks prediction identity, future-tail prefix invariance, and zero-state
reset. It exports no predictions and computes no metric, grade, or forecast.
The result only establishes one-batch data-pipeline, update, checkpoint, and
inference compatibility if the provider run passes; it cannot establish model
quality or full-epoch throughput.

This is an offline source candidate. No current Kaggle quota, provider input
version, remote asset availability, or remote kernel inventory was queried
while building it. Root review and fresh quota plus remote-input identity
admission are required before any push.
