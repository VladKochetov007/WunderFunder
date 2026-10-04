# E461 JAX objective bridge checks

The private Kaggle CPU-only v2 kernel completed all six implementation checks: no failures, errors, or skips. JAX/JAXLIB 0.11.1 and Torch 2.11.0+cpu were used. Worker duration was 15.92 seconds, peak aggregate RSS 491,790,336 bytes, below the declared 4 GiB limit. All ten expanded source members were verified against the manifest. The archive was auto-unpacked, so the observed archive SHA is unavailable and is not claimed as measured.

The preserved v1 failure concerned a fixture's expectation for masked final-row labels. The corrected fixture compares all label values and masks against the actual frozen production `next_change_labels` function, including masked placeholders. The bridge implementation was not altered to make that fixture pass.

These checks cover synthetic objective/update alignment and are complementary to E458's eleven recurrent/state/gradient/AdamW parity checks. They do not establish TPU execution, production memory fit, full-sequence accuracy, or a forecasting gain. JAX AdamW remains distinct from the public combo's Muon optimizer. The next admission requires a bounded real-data preflight and observed memory before any larger JAX fit.

Evidence: `kaggle/jax_bridge_e461_v2/provenance_result_v2.json`, `output_v2/output_hash_manifest.json`, the recorded JUnit/runtime output, and the preserved v1 attempt under `kaggle/jax_bridge_e461_v1/`.
