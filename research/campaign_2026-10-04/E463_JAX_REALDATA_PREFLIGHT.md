# E463 real-data JAX preflight

This is a bounded framework/objective preflight, not a model-selection experiment. It runs one AdamW update from the same fixed initialization in FP32 and BF16 on four actual fitting sequences, then reports finite predictions and a short-prefix diagnostic on four excluded validation sequences. It does not reproduce Muon or provide a full-sequence validation score.

Attempt 1 failed before fitting at the launcher's first external-input hash: `sha256_file` was called with `deadline=` but the outer helper did not accept that keyword. The provider kernel is `vladkochetov07/wnn-jax-real-data-preflight-e463-v1-runner` (the title-derived slug differed from the metadata ID). Its source, log, provider output and failed identity are retained under `kaggle/jax_realdata_preflight_v1/`. There were no completed model updates or scores from this attempt.

Attempt 2 preserves the unchanged 17-member scientific source archive, fixes only the launcher's hash-helper deadline binding, and uses matching dataset/kernel titles and IDs. Root inspected its exact kernel, metadata and manifest hashes and approved the private CPU retry. The launcher verifies all mounted input hashes before data updates, restricts CPU affinity and threads, uses a 2 GiB aggregate RSS stop and an 1,800-second whole-job bound. GPU, TPU and internet are disabled for this preflight.

The independent bounded stdlib harness passed in 0.20 seconds at 36.9 MB peak RSS: actual small-file SHA, expired-deadline rejection, outer and embedded helper call-signature binding, all 17 source-member hashes, source/manifest/spec identity, and matching provider slugs. Neither JAX nor Torch was imported by that harness. These checks justify a retry; they do not prove the real-data JAX workload fits its memory bound.

Attempt-2 review and checks: `kaggle/jax_realdata_preflight_v2/review_packet.json` and `staticcheck_result.json`. The actual provider launch/status record is authoritative once created. The existing TPU probe remains a separate kernel; this retry does not allocate a TPU or establish its HBM capacity.
