# Independent review: seed-4 QAT Kaggle packet

**Verdict: conditional offline packet pass.** The corrected packet has no remaining code or binding blocker to a bounded profile dispatch. This review is not provider authorization: the controller must take a fresh quota/device reading, run only the profile packet, and inspect all terminal receipts before admitting any fit.

## Finding and correction

The [frozen packet](../../../../research/campaign_2026-10-13/qat_renewable_r1/kaggle_f0_s4_r1/README.md) supersedes packet revision `867445d15937d5a13c8b76302a6d36f54071c1b0`, which was not ready for profile dispatch. [`train_rnn.py`](../../../../scripts/train_rnn.py#L1946) counts `--profile-warmup` and `--profile-steps` in full-sequence batches; each batch performs 20 optimizer updates. The old `5 + 100` arguments therefore represented 2,100 updates rather than the declared 100. The frozen successor `8f0c2a92e99cb98959c0d3edef2c964ff85ba8b6` changes every profile arm to two warmup batches plus three timed batches: 40 warmup updates and 60 timed updates, 100 total. Static inspection confirms the full-fit argument vectors are byte-identical to the predecessor.

## Checks on the frozen successor

The review inspected packet/spec metadata, generated bootstrap files, the existing staged-hash receipt and bounded file metadata; it did not reread the 1.97 GB staged array payloads. The receipt reports `pass` for 22 assets / 1,972,041,441 bytes, with hardlinks preserved. The root reports the private asset dataset is READY, with manifest `0ebf4803…`; this review did not query or mutate Kaggle.

- All three candidate profiles have the same seed-4 CP5 parent `e6be82ff799f4ea2b62e770e9c340a8cfd8f98780d1c7dec4a76eab5a853d35c`, target-cache metadata `1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8`, fold-0 split `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`, and full fit-subset recipe. The arms are FP32 (`--qat none`), uniform recurrent W4, and mixed input W4/recurrent W8. Their full-fit vectors are unchanged from `867445d`.
- The preflight spec is the sole job in its own runner invocation. The bootstrap requires that invocation to return a complete terminal summary and collect a passing `input_preflight.json` before invoking the three-candidate runner. The profile and fit stages are in separate bundles; the bootstrap rejects mixed stage kinds.
- Generated private bundle slugs and mount mappings agree with the exact IDs in the packet. Bootstrap SHA-256 bindings are profile `ddd775d794edb59c71f0e051668cfa5962ba99908d3b25eb9d461f504a776ce1` and fit `276ff43061a33a7eca783c53cf31e17f1154af833e8be76612b6d24a8feecc77`. Archive bindings are profile `88def98be42e360d7f435a35eb0fe8c9d339a1d26a58a563da7583ecfa103815` and fit `d87b698a367b6dd3f28a7bc3aac01c85d47945de090ad35d180df57e081158bb`. The generated bootstraps are below 1 MB and their internal wall limits match their packet bindings.
- The profile push is bound to an 1,800-second Kaggle timeout, fit to 10,800 seconds, each with a 300-second finalization reserve. Bootstrap and runner code enforce the outer wall deadline and reduce worker timeouts to current headroom. Per-worker declared timeouts do not override that wall deadline.
- Preflight worker limits are 2 CPUs, 6 GiB host RAM, 3.5 GB GPU admission / 4.5 GB stop threshold, and a 3,600-second local worker cap. Profile workers use 2 CPUs, 8 GiB host RAM, the same GPU bounds, and the same declared cap; the generated provider deadline reduces that cap at launch. Actual device availability and concurrent memory headroom remain controller admission questions. The three profiles may be serialized by the runner if two-device admission is not safe.
- Packet metadata identifies 62 source files, and the bounded reviewer check matched their recorded hashes. Full recipe membership includes 10,607 train sequences and 600 validation sequences; the fit-subset manifest is the fixed F0 S3 100% whole-component subset. Fits use `--no-eval-during-fit`; downstream F0 evaluation covers 468 groups. These groups/folds and shared CP5/cache ancestry are exposed, so the packet supports a matched engineering comparison, not a pristine holdout claim.

## Validation performed

The bounded offline static review checked packet bindings, generated bootstrap/archive hashes and sizes, source-file hashes, stage separation, the preflight barrier, private IDs/slugs, fit-vector invariance, profile accounting, and the existing staged asset receipt without hashing the large arrays again. The focused [preflight barrier test](../../../../tests/test_qat_kaggle_preflight_barrier.py) passed:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=500000000 WNN_RUNTIME_MAX=90s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
-m pytest -q tests/test_qat_kaggle_preflight_barrier.py
```

Result: `2 passed in 0.13s`. Kaggle CLI 2.2.4 help and installed command handling were inspected for the recorded `--timeout` and accelerator options; no push, upload, launch, or provider mutation was performed.

## Scope and remaining gate

This review covers offline packet integrity and control flow only. It does not verify a live Kaggle mount, refreshed quota, device count, provider-side slug resolution, profile completion, fit feasibility, score, speed, package size, or model quality. The profile recipe is intentionally a 100-update diagnostic rather than a full trajectory. Root must record the fresh quota/resource check, require terminal PASS for the isolated preflight and all three profile jobs, collect their outputs, and make a separate fit admission decision. No profile result or fit authorization follows from this pass.

Reviewed packet source revision: `8f0c2a92e99cb98959c0d3edef2c964ff85ba8b6`. Training source pin: `173cce08e9c7f13caccd1c3090419dd044aa702d`. Fit ancestry uses seed-4 CP5 continued from the documented seed-4 combo lineage; reused weights and caches do not create an independent replication.
