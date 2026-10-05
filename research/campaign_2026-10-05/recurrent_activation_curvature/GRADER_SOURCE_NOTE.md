# E481 independent scorer source note

`score_recurrent_curvature.py` independently implements the registered
reset-after, two-layer H64 recurrent candidate scan for baseline `u`, bounded
sine `u + 0.1 sin(pi u)`, and matched-center cubic
`u + 0.1 pi u (1-u²)`. It validates the fixed E481 sealed spec and future
runner attempt, all 18 checkpoint hashes/shapes/identity buffers, 24 fit and
16 test cohort roles, seeded fit traversal, 474 updates, fit-only normalization,
source and CPU-preflight provenance, FP32 IEEE metadata, telemetry, and runner
OOM/teardown/artifact receipts before any scoring.

The scorer reports all 18 official global/t0/t1 scores and 30 preregistered
contrasts: 18 within-shape comparisons, 9 Muon-minus-AdamW comparisons, and 3
joint `(cubic − sine)_Muon − (cubic − sine)_AdamW` interactions. It uses 2,000
common bootstrap resamples over numeric-sorted component IDs separately for
each seed, without pooling seeds. The no-overwrite selected-export NPZ retains
raw selected labels/predictions, masks, component IDs, row groups, shared
component multiplicities, all 18 arm bootstrap WP traces, and all 30 contrast
bootstrap traces.

Validation was bounded and synthetic only. No test-label rows were loaded or
scored, and no GPU or fit was run. The sealed spec identity was checked against
the current code/input map and hashes; this validates provenance only. Checks
run:

```text
ruff check research/campaign_2026-10-05/recurrent_activation_curvature/score_recurrent_curvature.py research/campaign_2026-10-05/recurrent_activation_curvature/test_score_recurrent_curvature.py
python -m py_compile research/campaign_2026-10-05/recurrent_activation_curvature/score_recurrent_curvature.py research/campaign_2026-10-05/recurrent_activation_curvature/test_score_recurrent_curvature.py
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=600 scripts/limited python research/campaign_2026-10-05/recurrent_activation_curvature/test_score_recurrent_curvature.py
cd /tmp && WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=120 /home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited python /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-05/recurrent_activation_curvature/score_recurrent_curvature.py --help
```

All six source-only synthetic checks passed. The E481 spec identity check
resolved the exact expected seal, 14 code entries, and 10 input entries.

Scorer SHA-256:
`52b91f9a3eaead30c71ef824feed8370b5a1e339ad7f8005c509b115b903c4d5`.
Test SHA-256:
`b233bf363ce111d96ffb81194dd54249e1d4528a8eeecdbc889fe56932b67379`.
