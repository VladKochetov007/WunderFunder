# R5 export-only recovery

R4 completed both 220-update fits and its root review verified their unchanged
epoch-1 checkpoints. The kernel then stopped at prediction startup because its
command supplied an unsupported `--device cuda` option. R5 recovers those
already-produced checkpoints from the private recovery asset, validates them
against the R4 receipts, reruns only FP32 validation prediction export, and
builds the shared training contract. It does not fit or grade a model.

The prediction sidecar remains the predictor's original output. R5 verifies its
checkpoint hash, original run name, epoch 1, prediction hash, exact 468-group
membership, FP32 inference mode, cache pins, and new runtime cache receipt. A
separate membership sidecar records the R4 checkpoint receipt and the fresh
cache receipt used during inference so the relocation is explicit.

The output arrays for independent grading are
`out/arms/wmse/predictions_valid.npy` and
`out/arms/huber_d1/predictions_valid.npy`; each arm also has a JSON prediction
sidecar and source-qualified membership sidecar. `training_contract.json`
records the producer/cache identities and leaves grading pending. No labels or
target archive are downloaded by this packet.

Fold-0 startup discovery reads the pinned asset manifest and delegates payload
verification to the existing four-file auxiliary validator. Large mounted
prediction artifacts outside that required set are not opened or hashed.

`gpu_loss_pair_export_recovery.packet.json` pins the private source, entrypoint,
caps, checkpoint receipts and expected outputs. Provider dataset creation and
kernel push require root admission against those exact hashes.
