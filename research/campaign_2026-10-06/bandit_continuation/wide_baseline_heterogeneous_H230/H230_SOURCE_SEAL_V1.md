# H230 actual source seal

This CPU-only step completes the pre-fit source bundle after the E521 paired
train exports reached terminal success. It does not build the H230 cache, fit a
student, score predictions, or change the H230 recipe.

The sealer verifies the immutable V1 source-bundle hash and all 32 pinned
worktree source files, requires the E521 pair-export plan and live receipt to
show terminal success, loads the frozen fold through the pinned worktree
split-loader, and verifies target-index order. For each of eight teacher
members it calls the pinned `build_distill_cache.py` ancestry and teacher
inspectors. Those inspectors bind checkpoint and provenance, fit membership
inside the declared fold with no fit/evaluation component overlap, ordered row
groups, sidecars, full prediction SHA-256, shape, float32 dtype, and finiteness.
Teacher prediction exports stay read-only memory maps and are released before
the next teacher.

The V1 bundle predates the completed E521 raw-family exports. Known hashes are
still required to match; only V1 null fields are filled from the terminal
producer artifacts and their bound sidecars. The output V2 bundle records the
actual E521 receipt hash, all checkpoint/provenance/prediction/sidecar hashes,
fold and target identities, ancestry, and builder hash. Raw parquet full-file
SHA-256 is not computed by this step; each cache/profile/fit stage must refresh
its live data identity through the pinned split loader.

Frozen worker command (one CPU, 1.5 GB hard memory cap, no GPU; 3,600-second
local runtime ceiling):

```sh
CUDA_VISIBLE_DEVICES='' WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=3600s PYTHONPATH=/home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007/src scripts/limited /home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007/.venv/bin/python /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/seal_h230_source_bundle.py --root /home/vlad/development/wunder/wnn_connectome_starterpack --bundle /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SOURCE_HASH_BUNDLE_V1.json --output /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SOURCE_HASH_BUNDLE_V2.json --receipt /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SOURCE_SEAL_PROGRESS_V2.json
```

The result is admitted only when the progress receipt says `finished`, all
eight sources and sixteen prediction exports are present, the output bundle
hash is recorded, and an independent reviewer verifies the resulting exact
source map. Any mismatch or resource kill leaves the partial progress receipt
as failure evidence and blocks cache admission.
