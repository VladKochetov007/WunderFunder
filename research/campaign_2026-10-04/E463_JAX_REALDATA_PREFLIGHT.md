# E463 real-data JAX preflight

This is a bounded framework/objective preflight, not a model-selection experiment. It runs one AdamW update from the same fixed initialization in FP32 and BF16 on four actual fitting sequences, then reports finite predictions and a short-prefix diagnostic on four excluded validation sequences. It does not reproduce Muon or provide a full-sequence validation score.

Attempt 1 failed before fitting at the launcher's first external-input hash: `sha256_file` was called with `deadline=` but the outer helper did not accept that keyword. The provider kernel is `vladkochetov07/wnn-jax-real-data-preflight-e463-v1-runner` (the title-derived slug differed from the metadata ID). Its source, log, provider output and failed identity are retained under `kaggle/jax_realdata_preflight_v1/`. There were no completed model updates or scores from this attempt.

Attempt 2 preserves the unchanged 17-member scientific source archive, fixes only the launcher's hash-helper deadline binding, and uses matching dataset/kernel titles and IDs. Root inspected its exact kernel, metadata and manifest hashes and approved the private CPU retry. The launcher verifies all mounted input hashes before data updates, restricts CPU affinity and threads, uses a 2 GiB aggregate RSS stop and an 1,800-second whole-job bound. GPU, TPU and internet are disabled for this preflight.

The independent bounded stdlib harness passed in 0.20 seconds at 36.9 MB peak RSS: actual small-file SHA, expired-deadline rejection, outer and embedded helper call-signature binding, all 17 source-member hashes, source/manifest/spec identity, and matching provider slugs. Neither JAX nor Torch was imported by that harness. These checks justify a retry; they do not prove the real-data JAX workload fits its memory bound.

Attempt-2 review and checks: `kaggle/jax_realdata_preflight_v2/review_packet.json` and `staticcheck_result.json`. The actual provider launch/status record is authoritative once created. The existing TPU probe remains a separate kernel; this retry does not allocate a TPU or establish its HBM capacity.

## Attempt 2 failure and attempt 3

Attempt 2 verified all 18 external input hashes, then its controlled aggregate RSS guard killed the real-data worker at 2,306,338,816 bytes against a 2 GiB limit. There was no completed update or score artifact. This is an observed resource-guard termination, not an inferred operating-system OOM. The raw provider outputs, terminal status and failure identity are preserved under `kaggle/jax_realdata_preflight_v2/`.

Attempt 3 preserves the four fitting sequences, B4/T1000 shape, objectives, initialization and evaluation prefixes. It raises the separate CPU VM's RSS cap to 4 GiB, adds flushed phase/memory milestones, and requires measured host headroom before importing JAX. GPU, TPU and internet remain disabled. Root verified the scientific AST after removing milestone calls and independently reran the bounded stdlib harness: all 17 source members, 18 external inputs and four resource-boundary cases passed. The private provider's exact 17-member file inventory and downloaded hashes were verified before pushing `vladkochetov07/wnn-jax-realdata-e463-20261004-v3-runner`. The current provider status is RUNNING; no successful model-update claim is yet available.

Separately, the original TPU hardware probe completed and returned eight TPU v5 lite devices with 16,909,332,480 bytes of reported memory limit per device. Its small FP32/BF16 framework checks passed. This is hardware/framework smoke, not real-data model training. The next TPU real-data preflight needs its own resource limits because JAX's observed TPU-host import footprint was already above the CPU attempt's 4 GiB cap.

## Verified completion and TPU launch

CPU attempt 3 completed successfully: worker exit 0, all 18 external input hashes verified, elapsed 224.965 seconds and peak RSS 3,068,354,560 bytes under its 4 GiB guard. FP32 and BF16 each completed one AdamW update on the fixed B4/T1000 batch. The short-prefix scores cover only 252 selected rows and are framework diagnostics, not campaign validation scores. Raw evidence is in `kaggle/jax_realdata_preflight_v3/provider_output_v3/`.

The CPU phase log contains a malformed first physical JSONL line caused by a literal newline escape. Its raw bytes remain unchanged. The new TPU launcher corrects that telemetry issue; root independently checked archive hashes, unchanged scientific inputs, actual two-record newline parsing, and the revised memory sampler.

The private TPU real-data kernel `vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1-runner` was pushed at 17:23 UTC and remains QUEUED at the 17:31 UTC provider poll. Its bound is 12 GiB host RSS, eight host CPUs, a 3,600-second worker deadline and per-device HBM monitoring. The initial workload is a one-device objective preflight; eight-device scientific training is separately preregistered as E465. Provider records and the independently approved packet are in `kaggle/jax_tpu_realdata_preflight_v1/`. No TPU real-data update has yet been observed.
