# Kaggle independent reproduction: Muon vs AdamW (fresh seeds, true FP32, T4) — 2026-10-25 R1

**Why.** The local claim (strict-FP32 Muon +0.0038/+0.0043 over AdamW) used seeds 1001/1002,
an RTX 3060 and — as the Opus red-team found — TF32 cuDNN GRU kernels in both arms. This lane
reproduces it independently: new seeds (2001 on F0, 2002 on F2), Tesla T4 (no TF32 hardware)
plus `--strict-fp32-kernels`, `NVIDIA_TF32_OVERRIDE=0`, same fast recipe otherwise. Kaggle
arms are compared only to Kaggle controls (hardware/precision differ from local).

- [build_specs.py](build_specs.py) → 4 specs in [specs/](specs/) (`krepro_{f0_s2001,f2_s2002}_{adamw,muon}_r1`),
  code from commit `864cffcd0` tree (192 files, hash-pinned), [portable registry](portable_inputs.json)
  for the four cache identities (Kaggle mounts change mtimes).
- Inputs: parquets from `vladkochetov07/wunder-funder`; `chains`, `chain_folds`,
  `train_targets`, `valid_targets` from existing content-addressed
  `vladkochetov07/wnn-c20261004-fold0-assets` (no new large upload).
- Preflights (local, isolated root, CPU, 2 profile steps): hardlinked inputs pass; **copied
  inputs with fresh mtimes + registry + `--strict-fp32-kernels` + Muon pass** (30 finite
  updates, 3.3 GB RSS). No other files read or created.
- Bundle: private dataset `vladkochetov07/wnn-muon-repro-20261025-r1` ([manifest](bundle-manifest.json),
  archive 738,219 B, READY with exact sizes); kernel `vladkochetov07/wnn-muon-repro-20261025-r1-runner`
  v1, T4×2, no internet, pushed 12:10 UTC, status RUNNING (not yet proof of fitting).
- Kaggle returns run directories only (`model_ep2.pt`, provenance, history); export and the
  official paired grade happen locally. Two fits run concurrently (one per T4).

## R1 terminal: ERROR before training (execution evidence, not a scientific result)

Kernel v1 ended `ERROR` after ~2 h ([summary](terminal_r1/summary.json), [failure](terminal_r1/failure.json),
[input identity](terminal_r1/input_identity_f0_adamw.json)). Runner verified all sealed code and
the four cache mounts; parquets were deferred as declared. Both F0 workers ran 5,400 s with
**empty worker logs, peak GPU 0 B, peak RSS 556 MB**, then timed out; F2 specs never started.
`train_rnn` prints its first line only in `initialize_progress`, after split/membership
validation, loader construction, model build and normalization (≈6 s locally), so the stall is
inside data setup on Kaggle's mounts, location unknown.

## R2 (diagnostic + results attempt)

`scripts/kaggle_traced_train.py` runs `train_rnn` unchanged via `runpy` with
`faulthandler.dump_traceback_later(240, repeat=True)`, so any stall writes all thread stacks to
the worker log every 4 minutes. Only the F0 pair (`krepro_f0_s2001_{adamw,muon}_r2`), 3,600 s
timeout. Dataset `vladkochetov07/wnn-muon-repro-20261025-r2` (READY, 555,980 B archive), kernel
`…-r2-runner` v1 pushed 14:13 UTC.

## R2 diagnosis (traced) and R3

R2 v1 failed at provider startup (empty log); an identical v2 ran. Stack dumps every 4 minutes
([worker log](terminal_r2/worker_f0_adamw_r2.log)) show both workers for the whole hour in
`splits.load_campaign_split → portability.verify_portable_source → sha256_file`: the fold
manifests' `sources` include `datasets/train.parquet` (29.2 GB) and `valid.parquet` (5.6 GB), so
each worker re-hashed ~35 GB through the input mount concurrently. The runner had deferred those
files (its own check of the 2.2 GB caches took ~54 s). This likely also explains the Oct-24
"Kaggle CPU profile timed out at 855 s" incident.

**Fix (R3):** `scripts/kaggle_traced_train.py` wraps `wnn.portability._content_digest` with a
lock-protected cross-process cache (`WNN_SHARED_DIGEST_CACHE`): each relocated source is hashed
once per kernel and still compared with the sealed registry. Local two-run test: first run
computes all six digests (35 GB in ≈31 s on NVMe), second run hits the cache; both reach
training. R3: all four fits (`krepro_{f0_s2001,f2_s2002}_{adamw,muon}_r3`), 5,400 s, dataset
`vladkochetov07/wnn-muon-repro-20261025-r3` (READY, 739,008 B), kernel `…-r3-runner` pushed
16:17 UTC, RUNNING.
