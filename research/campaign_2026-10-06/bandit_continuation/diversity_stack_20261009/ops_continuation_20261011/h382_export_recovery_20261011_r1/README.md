# H382 export recovery R1

This packet repeats only the three validated fold-0 prediction exports from
Kaggle H382 R3. R3's fits and exports completed, but its final collector glob
matched `pred_kg_h382_*` while the actual run IDs use `pred_kg_h382r3_*`; the
prediction arrays and sidecars were therefore absent from the downloaded
terminal inventory. The original checkpoints, provenance, histories, status,
input identity and terminal file hashes remain preserved under the R3 receipt.

The recovery has **zero fit commands and three export commands**. Each export
uses the exact R3 `python /root/wnn/scripts/predict_run.py RUN valid --epoch E`
argv, after checking the two-T4 device gate, original source/data/split inputs,
the rescued checkpoint archive digest and all three per-checkpoint provenance
records. Outputs are collected by exact run name into `out/predictions/` and
validated for the full-valid shape `(1873, 20000, 2)`, sidecar bindings and
checkpoint SHA. The prior R3 prediction hashes are recorded for comparison;
the recovery does not claim a score or treat a provider COMPLETE state as
prediction delivery.

The private checkpoint dataset contains only the three selected checkpoints,
their R3 provenance and training histories plus a manifest. `asset_stage/` is
an upload staging directory outside the committed packet; do not add the
checkpoint archive to git. The builder verifies the collected R3 files and
refuses to overwrite an existing stage archive.

To prepare a fresh private stage after reviewing this packet, run the builder
under the campaign's bounded support wrapper with the main repository
interpreter. This is local packet construction only; it does not create a
dataset, push a kernel or authorize a new provider attempt. Before any upload
or push, the operator must independently review the final bytes and refresh
the actual Kaggle quota, session inventory and two-T4 availability.

From the main repository root, the bounded builder invocation is:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=700000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
  /path/to/h382_export_recovery_20261011_r1/build_export_recovery.py \
  --asset-stage /tmp/wnn-h382-export-recovery-private-stage
```

The intended private assets are:

- Kernel `vladkochetov07/wnn-h382-export-recovery-20261011-r1`, requested as
  two Nvidia T4 GPUs, with a 7,200-second worker wall.
- Checkpoint dataset
  `vladkochetov07/wnn-h382-rescued-checkpoints-20261011-r1`.
- Existing private H382 R3 source bundle, fold-0 assets and `wunder-funder`
  dataset, all rechecked by the worker before checkpoint staging or export.

