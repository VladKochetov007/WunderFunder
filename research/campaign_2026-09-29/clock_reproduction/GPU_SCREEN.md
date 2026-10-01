# Frozen quote-clock auxiliary falsifier

Status: design frozen before any GPU run. The small fold-2 target-law panel was
used to choose the **8-event** correction, so fold 2 is not a fresh confirmation
set for this mechanism. No score gain is claimed from that panel.

## Evidence and mechanism

`results.json` uses 160 TRAIN groups and 32 distinct fold-2 components selected
from feature-only chain metadata. Among 12 retrospective class-3 sequences, the
four-quote-event rule improves exact target-pair reconstruction over the
four-representation-code-event control on identical scored rows by 7.51
percentage points. It still fails on 36.54% of eligible clipped-target mass.
The 8-event auxiliary labels differ on 81.89% of full class-3 scored target mass.
`label_alignment.json` shows equal-row weighted correlation with clipped target
contrast improves from 0.9374 to 0.9558 at horizon 8, but falls from 0.9404 to
0.9089 at horizon 24. This is retrospective label quality, not a causal
prediction gain. The TRAIN-label decoder does not independently identify
physical quotes.

## Fixed comparison

Use root clean fold-0 manifest
`research/campaign_2026-09-29/artifacts/splits/fold0.json`. Fit a class-3
price-code transition/spread decoder from **only its fit.train groups**. Build
two separate caches for fit.train and fit.valid. Do not read evaluation groups.
Both variants use the same decoder, same training examples, same finite masks,
same code-clock horizon-24 label, and same 0.1 auxiliary loss weight:

- control: mean future midpoint displacement over 8 representation-code changes;
- candidate: mean future midpoint displacement over 8 known nonzero decoded
  quote changes.

The paired cache builder is `build_aux_cache.py`. It invalidates intervals with
unknown decoder transitions. The trainer must accept an optional
`--quote-cache-root` directory so the two runs read these cache files without
overwriting the historical cache. Its split, seed, architecture, data ordering,
parent loss, optimizer, EMA policy, propensity, batch 32, epoch 3 checkpoint,
and lack of evaluation during fitting must be identical. Use the fold-specific
propensity `c20260929_prop_f0` and its cache. No checkpoint selection uses the
evaluation fold.

The first full cache build failed with the existing 0.001 max lattice tolerance
on TRAIN group 2315, a class-2 sequence with extreme labels (max residual
0.00151332). `lattice_audit_f0.json` scanned all 10,535 fit TRAIN groups and
found this as the sole failure. The decoder auxiliary fit excludes group 2315;
both variants leave its auxiliary cache zero. The GRU still trains on its main
targets in the unchanged clean manifest. The failed job and empty cache
directory remain preserved. A second CPU build was deliberately interrupted
after 90.67 seconds while a GPU parent ran because file cache approached the
shared 36 GiB memory cap; its record and empty directory remain too. A
TRAIN/fit-VALID target-only class index now precedes the third build, so its
decoder and cache passes read only class-3 groups.

Build the class index and caches under one core and 3 GB after the concurrent
GPU parent finishes:

```bash
WNN_SLICE=wnn-campaign.slice WNN_CPU_PCT=7 WNN_THREADS=1 \
WNN_CPU_QUOTA=100% WNN_MEM_MAX=3G OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1 \
scripts/limited .venv/bin/python \
research/campaign_2026-09-29/clock_reproduction/aux_class_index.py

WNN_SLICE=wnn-campaign.slice WNN_CPU_PCT=7 WNN_THREADS=1 \
WNN_CPU_QUOTA=100% WNN_MEM_MAX=3G OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1 \
scripts/limited .venv/bin/python \
research/campaign_2026-09-29/clock_reproduction/build_aux_cache.py \
--split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json \
--out research/campaign_2026-09-29/clock_reproduction/cache_f0_v3
```

GPU commands below are the exact model arguments. Submit them sequentially
through `scripts/campaign_job.py launch --kind gpu` with a 2400-second hard
timeout each, so failures and device-wide VRAM peaks are recorded. Apply the
same root 10-core cap while this one-core research job remains active; otherwise
the campaign's 12-core cap is sufficient. `scripts/limited` does not enforce
VRAM, so the job monitor must stop either run above 5 GB total observed usage.

```bash
COMMON="--train-source train+valid --n-valid-folds 4 --eval-source vfold --full-valid \
--loss wmse --aux-struct 0.3 --aux-next 0.3 --aux-quote 0.1 \
--propensity c20260929_prop_f0 --prop-power 2 --prop-eps 0.03 \
--valid-true-mask 0.03 --hidden 128 --epochs 4 --ema-teacher 1.0 \
--ema-decay 0.999 --eval-ema --fast --since-weight 32 --since-floor 0.25 \
--valid-fold 0 --seed 3 --batch 32 --checkpoint-policy fixed \
--checkpoint-epoch 3 --no-eval-during-fit \
--split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json"

scripts/limited .venv/bin/python scripts/train_rnn.py \
--name c20260929_auxclock_code8_f0 \
--quote-cache-root research/campaign_2026-09-29/clock_reproduction/cache_f0_v3/code \
$COMMON

scripts/limited .venv/bin/python scripts/train_rnn.py \
--name c20260929_auxclock_quote8_f0 \
--quote-cache-root research/campaign_2026-09-29/clock_reproduction/cache_f0_v3/quote8 \
$COMMON
```

If the frozen clean seed-4 no-aux parent is available, make the primary screen
the equal blend of that parent with each model's fixed epoch-3 output. Also
report each single model and the no-aux clean seed-3/seed-4 pair. Predict the
entire validation file only **after** both fits, but score only the manifest's
468 fold-0 evaluation row groups. Apply `need_prediction AND is_scored`, clip
targets/predictions to [-2,2], and compute Global Weighted Pearson. Keep row
order, exact masks and the clean component labels identical. Prediction files
must carry checkpoint and cache hashes; use existing `predict_run.py` at epoch 3.

## Decision before seeing fold-0 scores

Pass the screen only if the quote8 candidate's **pair** exceeds the code8
control pair by at least +0.0005 exact Global WP and is above the no-aux clean
pair; no parity, data provenance or VRAM failure is allowed. No epoch or weight
selection is permitted. Then freeze the candidate and repeat on an independent
clean fold with both parents and propensity rebuilt. A positive fold-0 screen
alone cannot support the 0.6660 public expectation.

The clean seed-3 parent completed in 1095.65 seconds at 4.563 GB peak observed
device usage. Two auxiliary parents are provisionally 35–45 GPU minutes total;
extra output dimensions may raise VRAM, so this is a forecast, not measured
fitness. Both retain batch 32. The complete paired cache build should use
roughly 5–12 CPU minutes and about 2 GB apparent sparse cache storage; measure
actual runtime and RSS in the durable job record. Do not run the GPU comparison
until the cache metadata and trainer flag are independently audited.
