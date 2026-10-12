# Shared Q8 clean-fold quality consumer R2

R2 preserves the R1 student/HFT checkpoints, split and provenance inputs,
exported weights, native Py311 ELF, feature data, callback runner, and official
metric. Its identified consumer correction wraps the raw native ensemble in
the existing `WrappedPredictor` adapter from `cpu_shared_backbone_r1` because
the pinned callback runner invokes `model.model.reset()` and then
`model.predict(DataPoint)`.

The R2 callback fixture exercised one full 20,000-row feature-only evaluation
sequence from each fold, twice per sequence through the real callback runner.
It verified all-row advancement, `None` during warmup, finite required
predictions, exact `q8vnni` precision, and bitwise-equal reset replays. The
fixture opened no labels and made no quality claim. See
`CALLBACK_FIXTURE_R2.json` for exact row-group, feature, output, and source
hashes.

The complete evaluation remains pending root admission. It uses the fixed
0.5/0.5 output mean, both already exposed source-qualified folds, and the
unchanged official `global_wp_clip2_v1` accumulator. The bounded plan is one
thread, one CPU at 100% quota, 1,000,000,000 bytes memory maximum, and 1,800
seconds wall time. Prior callback timings imply roughly 555 seconds of
callback-only work at 30 microseconds per row for 18.5 million rows; decode,
metric, and system overhead make this a planning estimate, not a runtime
guarantee.

Run the label-free fixture with the pinned Py311 interpreter and native package:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
env PYTHONPATH=/tmp/wnn-shared-f0-py311-package-r1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/tmp/wnn-projection-precompute-r1/py311/bin/python run_quality_r2.py \
  --mode callback-fixture --output CALLBACK_FIXTURE_R2.json
```

Full scoring requires `--mode full-quality` and an independent root admission
receipt bound to `CONTRACT_R2.json`, the unchanged R1 weight archive and
receipt, the native ELF, and the validation Parquet footer identity. There is
no full-fold score or submission recommendation in this packet.
