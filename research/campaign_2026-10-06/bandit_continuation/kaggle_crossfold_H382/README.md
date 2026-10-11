# H382 fold-0 Muon-all crossfold cohort

This packet prepares three fresh fold-0 seed-3 GRU-256x2 fits using the reviewed
`h230_source_v8_arch` tree and the predictor helper used by the completed Kaggle
V6 cohort. It varies propensity power and fixed OneCycle schedule length while
keeping the data split, initialization, batch, chunk, loss, optimizer scope,
precision, learning rate, weight decay, and auxiliary losses fixed.

| Job | Propensity power | Epochs | Fixed checkpoint | Role |
|---|---:|---:|---:|---|
| `kg_h382_f0_muonall_prop2_e3_s3` | 2 | 3 | 2 | Fresh all-scope anchor |
| `kg_h382_f0_muonall_prop0_e3_s3` | 0 | 3 | 2 | Propensity contrast against anchor |
| `kg_h382_f0_muonall_prop0_e4_s3` | 0 | 4 | 3 | Longer fresh OneCycle schedule |

The prior fold-0 propensity-0/2 runs used `--muon-scope rnn`; they are not exact
duplicates of these all-scope fits. The all-scope choice can route eligible
parameters through Muon while fallback parameters use AdamW at the shared
learning rate. This packet does not claim a pure-Muon optimizer.

The fixed fold-0 manifest has 10,535 fitting train sequences and 1,405 fitting
validation sequences, with 468 separate evaluation sequences. The worker also
exports predictions over all 1,873 validation sequences for downstream grading.
Fold 0 is an exposed first-screen fold; the packet is a crossfold confirmation,
not a fresh holdout claim. Parent-controlled grading and candidate admission
remain outside this packet.

The source is frozen by a per-file SHA-256 manifest. The data bundle contains
source, the split and portable registry, the fold-0 asset manifest, and exact
argv JSON; large raw data and content-addressed fold assets stay in their
existing private Kaggle datasets. The kernel validates the bundle and every
mounted asset digest, raw parquet sizes and footers, propensity cache NPY
headers, fit membership, prediction NPY layout, sidecars and checkpoint hashes.
It records input observations incrementally before fitting, bounded phases and
terminal inventory. The three fit steps are capped at 5,400 seconds each,
exports at 1,800 seconds each, and the two-slot cohort at 14,400 seconds.

Completed V6 runs on the same Kaggle 2xT4 GRU-256x2 batch-32/chunk-1000 recipe
took 2,899–3,545 seconds to fit and 97–142 seconds to export. These are actual
fold-1 reference timings, not a fold-0 profile or a speedup estimate. The root
controller must refresh Kaggle quota and review the packet before uploading or
dispatching. `build_packet.py` performs local deterministic packaging only.

Rebuild in this isolated checkout with the project-limited CPU wrapper:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=800000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/kaggle_crossfold_H382/build_packet.py
```

The generated upload assets are `dataset/bundle.wnn`,
`dataset/dataset-metadata.json`, `kernel/kernel.py`, and
`kernel/kernel-metadata.json`. No provider upload or launch is performed here.
