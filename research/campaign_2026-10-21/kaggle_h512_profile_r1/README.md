# Kaggle H512 engineering profile delivery

This folder preserves two bounded, private Kaggle engineering profile attempts for the frozen H512×1, batch 8, chunk 2000, FP32, fold 0, seed 1003 recipe. They were intended to establish a device and phase cost before considering a scientific fit. Both attempts ended before training; there is no profile measurement, model result, or fit admission.

The terminal receipt [KAGGLE_H512_PROFILE_TERMINAL_20261021_R1.json](KAGGLE_H512_PROFILE_TERMINAL_20261021_R1.json) records provider quota before and after each attempt, requested and actual machine shapes, exact worker deadlines, preserved terminal collections, full raw input identities, and the second failure diagnosis. The original profile recipe is [h512_profile_spec.json](h512_profile_spec.json); attempt 2 added the minimal logging wrapper at [collected wrapper source](kaggle_profile_wrapper_attempt2.py) and reused the canonical trainer through the existing bundle runner.

## Outcome

Attempt 1 requested `NvidiaTeslaP100`, but Kaggle allocated two Tesla T4 GPUs. The worker was pinned to device 0 and had a 297-second effective deadline. It timed out after 270.556 worker seconds with zero GPU allocation and no worker log. Its raw input hashes were deferred and never observed. The provider quota delta was 305.803 seconds; the conservative dual-device bound is 611.606 seconds, which exceeds the then 600-second bound by 11.606 seconds. The complete collection is in [archived terminal_collection](archives/README.md).

Attempt 2 explicitly requested `NvidiaTeslaT4` and again received two T4s. One worker used device 0. Its 898-second effective deadline was under the 1200-second outer cap with a 300-second finalizer reserve. The wrapper verified the original and derived manifest hashes and confirmed their membership and evaluation fields were equal. It entered canonical `wnn.splits.load_campaign_split`; that call completed portable SHA checks for both raw parquets, then failed because `cache/chains.npz` was absent. Worker wall time was 482.874 seconds, peak GPU allocation was zero, and no training import or profile update occurred. The exact terminal output and worker files are preserved in [archived terminal_collection_attempt2](archives/README.md), with provider polling in [provider_polls_attempt2.jsonl](provider_polls_attempt2.jsonl).

The actual provider quota decrement in attempt 2 was 492.972 seconds, while the conservative two-device budget bound is 985.944 seconds. Across both attempts the measured quota decrement was 798.775 seconds. The post-attempt-2 account snapshot showed 90,722.348 of 108,000 GPU seconds used, zero reserved, 17,277.652 unreserved, and refresh at 2026-10-24 00:00 UTC. The provider response exposes aggregate seconds and no device multiplier; the attempt-specific charge is recorded without assuming future dual-T4 billing. The exact owner+slug status poll returned `KernelWorkerStatus.ERROR`; Kaggle did not expose a session ID.

The installed SDK accepts T4 and P100 shape identifiers, but the P100 request was actually allocated dual T4s. A current single-GPU shape was not verified. The SDK has no account-wide live-session listing endpoint, and zero reserved seconds does not prove all unrelated kernels are terminal. No current-cycle reservation record was found. The earlier read-only asset, quota, and SDK-interface inventory is [KAGGLE_DELIVERY_READINESS_20261021_R1.json](../kaggle_delivery_readiness_r1/KAGGLE_DELIVERY_READINESS_20261021_R1.json).

## Required repair before another profile

The profile-only manifest correctly contains exactly the canonical raw `train.parquet` and `valid.parquet` sources, preserving all non-source membership and evaluation fields bit for bit. The canonical split loader also needs two small structural graph caches that are not source-membership entries: `cache/chains.npz` (650,142 bytes, SHA-256 `39aba46f…`) and `cache/chain_folds.npz` (550,072 bytes, SHA-256 `a2cfe843…`). Stage those exact files at the expected root paths and verify their hashes before calling the canonical loader. Keep the profile manifest sources unchanged. A future worker deadline must include the observed roughly 483-second raw-source preflight, profile setup and updates, serialization, and finalizer reserve.

The second diagnostic attempt exhausted the authorized profile retry. A future profile needs a new controller decision after review of this diagnosis; any fit also remains uncosted and unadmitted until a genuine finite profile and its quota delta are recorded.

## Provenance and limits

The packet uses the existing `scripts/kaggle_bundle.py`, `scripts/kaggle_gpu_runner.py`, `scripts/kaggle_common.py`, `scripts/kaggle_kernel.py`, and unchanged canonical `scripts/train_rnn.py`/`src/wnn` code. No copied trainer, target cache, initializer, propensity asset, credential, or scientific fit was included. The requested training source revision was `68ff682977b822f38351aac0f621146f5f13b440`; the bundled checkout was `d3055423a185d2ecfe7baf1c20206abeb743ad8b`, with no diff in the listed training and Kaggle runner paths. This receipt is engineering and provider-delivery evidence only, not evidence about quality or expected public score.

## Lossless source/terminal archive

The original input bundles, terminal trees and derived split manifest are preserved in [the verified snapshot archive](archives/README.md). Historical paths in the immutable terminal receipt are archive member paths relative to this packet. Every file digest was checked after packing; no raw bytes or results were edited. The small [attempt2 diagnostic wrapper](kaggle_profile_wrapper_attempt2.py) is available directly for source review.
