# H382 export recovery R2

R2 repeats only the three prediction exports from the completed H382 R3
checkpoints. R3's fits and exports succeeded, but its final collector glob did
not match the actual run IDs, so the NPY arrays and sidecars were not included
in the downloaded terminal inventory. R1 corrected that collector path but
was rejected on review because it did not scan prediction values for NaN/Inf
and attached reference hashes after writing the terminal inventory. This new
packet preserves R1 and fixes both defects.

There are **zero fit commands and three export commands**. Each uses the exact
R3 `python /root/wnn/scripts/predict_run.py RUN valid --epoch E` argv after
validating the original sources, mounted inputs, split, rescued checkpoints
and their provenance. Exported `(1873, 20000, 2)` float32 arrays are checked
for sidecar/checkpoint identity and scanned for finite values in 16-row mmap
chunks. The finite-scan result and row count are recorded in each export
receipt. Before `terminal_inventory.json` is serialized, it records the
expected and collected NPY digests plus a match flag for each run, and terminal
completion requires all three matches.

The measured R3 export command durations were 160.5, 82.2 and 160.6 seconds.
The source receipt did not separately record worker startup/setup time. R2
keeps the frozen 1,800-second per-export command limit and uses a 3,600-second
worker wall; this allows setup and validation headroom without reserving the
previous 7,200-second wall. Root remains responsible for fresh quota/session
admission and any dispatch decision.

The private checkpoint dataset contains only the three selected checkpoints,
their R3 provenance and histories plus a manifest. The upload stage is outside
the committed packet; never add the checkpoint archive to git. The builder
revalidates the collected H382 R3 outputs and refuses to overwrite an existing
stage archive. Building this packet does not upload the dataset or push a
kernel.

Use the main repository interpreter under the bounded support wrapper:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=700000000 WNN_RUNTIME_MAX=120s scripts/limited .venv/bin/python \
  /path/to/h382_export_recovery_20261011_r2/build_export_recovery.py \
  --asset-stage /tmp/wnn-h382-export-recovery-private-stage-r2
```

The identified private assets are kernel
`vladkochetov07/wnn-h382-export-recovery-20261011-r2` and checkpoint dataset
`vladkochetov07/wnn-h382-rescued-checkpoints-20261011-r2`. The kernel requests
two Nvidia T4 GPUs and also enforces exactly two visible T4s before it stages
the rescued checkpoints or runs any export.
