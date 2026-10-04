# E419 dual-T4 QAT bundle inventory

Status: **draft_only_not_pushed**. This directory contains an unpushed draft; no Kaggle write or job launch occurred.

Dataset: `vladkochetov07/wnn-e419-qat-dualt4-20261004-v2`  
Kernel: `vladkochetov07/wnn-e419-qat-dualt4-20261004-v2-runner`  
Bundle SHA-256: `818b90818aed37b7a89bdefbfa13cc2f8f708eb95046fb587e682a4de96f73c3`  
Sealed code aggregate SHA-256: `b86e83eb78317ad5031e7f5ba924e1b6c74a3aa0e7441735ae921c65e4e38d47`  
Scientific source HEAD recorded at preparation: `112f4fd3c8cca53eb7001c3636516af1af1d8e2d`  

Provider timeout: 22500 s. Each worker timeout: 21600 s. Two workers run concurrently on separate Tesla T4s.

Each worker runs its QAT one-batch production-shape probe, then the matched FP32 control, then the QAT treatment. It writes fixed epoch-0 full-validation predictions and one single-arm score for each control and treatment. The fixed control uses the same initialization, preprocessing, data, seed, batch, learning rate, one-epoch schedule, teachers and objective; only the QAT mode differs. The score files are diagnostics, and the controller computes paired comparisons after download.

| Arm | FP32 control | QAT treatment |
|---|---|---|
| recurrent | `c20261004_e419_fp32_control_recurrent_s3_f0_t4v2` | `c20261004_e419_qat_recurrent_s3_f0_t4v2` |
| recurrent_head | `c20261004_e419_fp32_control_recurrent_head_s3_f0_t4v2` | `c20261004_e419_qat_recurrent_head_s3_f0_t4v2` |

Bundle manifest declares 89 dependencies, with 26 mounted from existing private inputs and 63 archived in the private runner dataset. The combo epoch-3 parent and provenance are embedded because the E413 shared fold-0 dataset does not contain them. The archive is 3,863,763 bytes; expanded transport files total 4,655,114 bytes.

Per worker: 2 CPUs, 10 GiB RAM, 11 GiB allocator target, 12 GiB measured operating limit and 13 GiB sampled stop. The measured local production-shape QAT probe peaked at 1.782 GiB VRAM and 6.588 GiB RAM. E413's two T4 workers peaked at 6.441 GiB VRAM and 8.59 GiB RAM. The 21,600 s worker and 22,500 s provider limits preserve the complete control-plus-QAT sequence; they are prospective watchdog limits, not a completion claim.

Validation artifact: `bundle_verification_20261004T0645Z.json` confirms archive digest `818b90818aed37b7a89bdefbfa13cc2f8f708eb95046fb587e682a4de96f73c3`, all embedded file digests, both sealed specs, and no missing/undeclared members. Kernel metadata marks it private, selects `NvidiaTeslaT4`, and disables internet. The current runner writes resource-only heartbeat JSON every 30 seconds.

The parent process must review this bundle, inputs, code hashes, provider timeout handling, and resource contract before any push.

The sealed source closure includes `scripts/screen_score.py` and its dynamic dependency `research/campaign_2026-09-30/audit/campaign_score.py`.
