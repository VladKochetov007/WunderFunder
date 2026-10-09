# H230 actual source seal, successor V3

V1 and V2 source-seal attempts are preserved with their exact maps and failure
receipts. V2 validated four teachers, then found that the V1 snapshot's raw
seed-4 provenance hash predated the completed E521 fit. The completed frozen
E521 V6 artifact report binds the current raw seed-3/seed-4 checkpoint,
provenance, validation prediction and sidecar hashes.

V3 keeps every matching V1 hash strict. For the two `bd26_w384raw` members only,
it permits an old snapshot hash to be superseded only when the current bytes
exactly match the E521 V6 artifact map. Pending train-export hashes are filled
from the terminal E521 train-export sequence and the pinned builder's sidecar
and full-byte checks. The final bundle records any V1-to-E521 reconciliation
explicitly, the grade-report/spec hashes, and the exact current hashes. The
E521 metric or score is not used to choose or validate H230 sources.

All other source checks remain delegated to the pinned worktree
`build_distill_cache.py` `_audit_ancestor_closure` and `_inspect_teacher`,
including checkpoint/provenance identity, component-safe fold membership,
ordered row groups, sidecars, full prediction hashes, shapes, dtypes and
finite values. This step remains CPU-only, does not materialize prediction
arrays, and does not build the cache, fit, or score an H230 student.

```sh
CUDA_VISIBLE_DEVICES='' WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=3600s PYTHONPATH=/home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007/src scripts/limited /home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007/.venv/bin/python /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/seal_h230_source_bundle_v3.py --root /home/vlad/development/wunder/wnn_connectome_starterpack --bundle /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SOURCE_HASH_BUNDLE_V1.json --output /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SOURCE_HASH_BUNDLE_V2.json --receipt /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SOURCE_SEAL_PROGRESS_V4.json
```

The source map binds the E521 V6 artifact report and grading spec. The source
seal is admitted as complete only when its V4 progress receipt says `finished`,
all eight sources and sixteen exports pass, and the output V2 bundle hash is
recorded. Cache build remains a separate admitted stage with its own live
resource/disk checks and independent source-map review.
