# E476 exported-score reproduction

I independently recomputed the official scores from the immutable selected-row export and checked the labels against the source parquet groups. All 12 arms reproduce the saved global and per-target weighted Pearson scores exactly at the stored precision (maximum absolute difference: 0 for global WP, t0, and t1). The export has 26,990 finite selected rows across the fixed 16 test components; each exported label matches the parquet labels after the official mask and clip. The fit/test split remains 24/16 with no component overlap.

I also independently reconstructed centered per-component weighted moments and reused the preregistered 2,000 component-resample draws (seed 271828) to reproduce two AdamW warm-cosine-versus-constant-mean schedule contrasts and both activation-by-schedule interactions. All four point estimates, standard deviations, percentile intervals, and sign frequencies match `scores.json` within `2e-9`. The component IDs are numeric in the scorer and strings in the NPZ; the reproduction converts them back to numeric IDs before sorting, matching the original draw order.

The reproduced schedule contrasts were +0.067328 (base; 95% interval [0.027565, 0.104393]) and +0.072700 (sine embedding; [0.031514, 0.110653]). The activation-by-schedule differences were +0.005372 for AdamW ([0.002671, 0.007006]) and +0.001626 for Muon+AdamW ([0.000045, 0.003083]). These are descriptive results on the previously exposed 16-component cohort, with one fit seed; they do not support public-score inference or selecting a general winning recipe.

Reproduction command, run under `scripts/limited` with a 1-CPU/2-GiB cap:

```bash
WNN_CPU_PCT=8 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G WNN_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 \
scripts/limited python research/campaign_2026-10-05/activation_schedule/reproduce_exports.py \
  --output research/campaign_2026-10-05/activation_schedule/eda/twelve_v1/RESULT_INDEPENDENT_REVIEW.json
```

The run completed in about one second. `py_compile` passed. The reproduction script SHA-256 is `6e9037da88ff4edf5cc5ba070ee57e7456eb019c709fa8a0ed70a275a7860e0a`; the result receipt SHA-256 is `b977af2b93340c1e05f4fbf61fa017ec3b5ebb1ec8291ee6499c557bb38432e6`. The source export SHA-256 matches the grader receipt: `49e6a36713bb55b7abd70dd46630d36ebfc1397251ceb577e86f7eec60b72ae9`.
