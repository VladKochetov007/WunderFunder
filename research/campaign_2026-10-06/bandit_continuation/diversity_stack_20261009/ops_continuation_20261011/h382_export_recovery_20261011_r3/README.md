# H382 export recovery R3

R3 repeats only the three prediction exports from the completed H382 R3
checkpoints. The first recovery packet fixed the output collector path but was
rejected for missing finite-value checks and late terminal receipt binding.
The second added those checks, but review found it incorrectly required the
new NPY byte digests to equal outputs produced by the prior Kaggle worker.
R3 keeps all fresh input, checkpoint, provenance, sidecar, shape, byte-count,
finite-value and collection checks; the previous prediction digests are
recorded as a comparison diagnostic only. Cross-device output hash equality
does not decide whether a fresh export is valid.

The worker has **zero fit commands and three export commands**, each using the
exact completed R3 command
`python /root/wnn/scripts/predict_run.py RUN valid --epoch E`. It validates the
original sources, mounted inputs, split, rescued checkpoints and their
provenance. Exported `(1873, 20000, 2)` float32 arrays are scanned for NaN/Inf
in 16-row mmap chunks; finite results and scanned-row counts are written to
per-run receipts. Before `terminal_inventory.json` is serialized, it records
the expected and newly collected prediction digests and whether they match.
Completion requires all three exports to pass fresh validation and all six
NPY/sidecar files to be collected, while digest equality remains diagnostic.

The prior worker measured export commands at 160.5, 82.2 and 160.6 seconds;
worker startup/setup time was not separately captured. The frozen 1,800-second
per-export command limits and 3,600-second worker wall remain. Root owns fresh
quota/session admission and any dispatch decision. This packet is prepared
only; it has not uploaded its new private checkpoint dataset or pushed a
kernel.

The dataset stage contains only the three selected checkpoints, their R3
provenance and histories plus a manifest. Never add the checkpoint archive to
git. The bounded builder revalidates the actual collected R3 outputs and
refuses to overwrite an existing stage archive.

Run the builder with the main repository interpreter under the bounded
support wrapper:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=700000000 WNN_RUNTIME_MAX=120s scripts/limited .venv/bin/python \
  /path/to/h382_export_recovery_20261011_r3/build_export_recovery.py \
  --asset-stage /tmp/wnn-h382-export-recovery-private-stage-r3
```

The identified private assets are kernel
`vladkochetov07/wnn-h382-export-recovery-20261011-r3` and checkpoint dataset
`vladkochetov07/wnn-h382-rescued-checkpoints-20261011-r3`. The kernel requests
two Nvidia T4 GPUs and enforces exactly two visible T4s before it stages the
rescued checkpoints or runs any export.
