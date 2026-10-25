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
