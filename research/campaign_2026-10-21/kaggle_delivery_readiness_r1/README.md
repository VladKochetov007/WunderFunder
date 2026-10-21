# Kaggle delivery readiness — October 21

**Status: metadata-only; no cloud fit admitted.** This receipt evaluates the candidate F1 WMSE/NLL pair and the cold H512 fallback against the current Kaggle quota, existing data assets, and the provider's actual kernel metadata interface. It records no score or profile result.

## Findings

A fresh read-only SDK quota query at `2026-10-20T23:09:10Z` found **18,076.427 GPU seconds (5.0212 hours) unreserved**, zero provider-reserved seconds, and refresh at October 24 UTC. The WNN kernel search returned no refs stamped October 20 or 21; three exact historical status queries returned `COMPLETE`. The installed Kaggle SDK can list notebook metadata and query one known kernel's status, but does not expose an account-wide running-session roster. The current campaign has no recorded planned reservation from another agent; the quota snapshot cannot establish plans that were never submitted to Kaggle.

The Kaggle dataset `vladkochetov07/wunder-funder` already contains the exact root inputs: `train.parquet` at 29,240,219,011 bytes and `valid.parquet` at 5,615,509,128 bytes, plus `valid_mask.parquet` at 91,972,846 bytes. The dataset is **public**, not private. Dataset detail and the per-file sum agree at 34,947,700,985 bytes; the dataset-list aggregate is 33,761,236,515 bytes, so use the per-file sizes for identity.

No F1 target cache, propensity, or initializer was found in that Kaggle dataset inventory. The F1 parent checkpoint, target cache and propensity assets are recorded as local assets in the existing [fold asset catalog](../../campaign_2026-10-20/submission_delivery_r1/nll_adaptation/FOLD_ASSET_CATALOG_R1.md). The target cache is about 2 GB. A Kaggle F1 run therefore needs those exact local assets staged as private inputs through the existing bundle flow.

The existing [bundle builder](../../../scripts/kaggle_bundle.py) requests `machine_shape=NvidiaTeslaT4`. Kaggle's current [kernel metadata documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md) identifies that shape as **two T4 GPUs**, and lists `NvidiaL4` without a device count. The installed root SDK's supported-value list instead contains T4, P100, and the older TPU v3 shape, omitting L4; this is interface-version drift. Neither the SDK nor the quota response reports which GPU shapes are currently available to this account. P100 is an accepted identifier, but availability was not established; no single-GPU shape was verified as available. The quota response has aggregate GPU seconds and no per-device billing multiplier, so this audit does not treat a dual-T4 hour as either one or two quota hours.

## Candidate lane and gates

The **F1 pair is the nearest candidate**: one WMSE and one NLL fit, with a plan estimate of 2.5 hours of fit time plus 45 minutes for exports. At 3.25 hours total, that would leave about 1.77 hours if quota tracks session wall time one-for-one. If two T4s were charged independently, the same wall-time estimate would consume 6.5 hours and exceed the current 5.02-hour remainder. Both are planning scenarios: actual Kaggle throughput and quota treatment remain unmeasured. F1 remains `NOT_ADMITTED` until the local F0 pair is scored and independently graded.

Before reconsidering F1, the controller needs to:

1. Wait for the active R2 exports and score, then apply the frozen F1 value-of-information gate.
2. Stage the exact parent, target, and propensity assets privately. Reuse `kaggle_bundle.py`, `kaggle_assets.py`, `kaggle_gpu_runner.py`, and `kaggle_common.py`; use the canonical trainer, with no new trainer.
3. Confirm a currently accessible machine shape, then run a bounded representative profile and reconcile before/after quota seconds.
4. Reserve the measured profile, two fits, exports, artifact collection, and recovery margin against the **observed** quota charge before admission.

The H512 alternative is less ready. Its plan describes a 360-second local profile and a six-epoch H512×1 recipe, but has no Kaggle actual-shape profile or Kaggle fit-cost estimate. It is not a quota-feasible lane yet; the same sealed runner/bundle path can be used after profiling.

## Scope and provenance

The SDK checks used the root `.venv` Kaggle package (version 2.2.4) and the existing read-only [`poll_provider.py`](../../campaign_2026-10-06/cloud_preflight/poll_provider.py) client-timeout helper. API work ran under a 20% CPU quota, 300 MiB memory maximum, zero swap, and `TMPDIR=/home/vlad/development/wunder/research_tmp`. No credentials were printed or changed, no package was installed, and no dataset was uploaded or kernel launched.

The [machine-readable receipt](KAGGLE_DELIVERY_READINESS_20261021_R1.json) contains the exact quota values, dataset file sizes, asset hashes, kernel-status sample, assumptions, and blockers. The accepted [local/Kaggle plan](../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md), current [frontier](../../campaign_2026-10-20/plateau_review_r1/FRONTIER.md), [F1 plan](../../campaign_2026-10-20/submission_delivery_r1/nll_followup_f1/F1_FOLLOWUP_PLAN_R1.json), and [H512 profile plan](../../campaign_2026-10-20/submission_delivery_r1/h512_replication_profile/PROFILE_PLAN_R1.json) remain authoritative for scientific scope and admission.
