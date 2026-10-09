# H230 actual source seal, successor V2

V1 attempt is preserved with its exact map and failed progress receipt. It
successfully validated three teachers and then stopped on an unsupported
equality check between a legacy source-bundle seed label and the trainer seed.
That `s4` run is initialized from its `s4` parent but records trainer seed 3.
The pinned cache builder validates its checkpoint, ancestry, fold membership,
fixed epoch, prediction sidecars and bytes; it does not require a source run-name
seed label to equal the fine-tune invocation seed.

This successor removes only that extra equality gate. It preserves the bundle's
original `seed` label and adds `seed_identity_diagnostic` to the sealed source
entry, reporting the label, provenance seed and checkpoint seed separately. It
does not change family membership, recipe, split, source artifacts, grader or
cache semantics. All validation continues through the pinned worktree
`build_distill_cache.py` `_audit_ancestor_closure` and `_inspect_teacher`.

The V2 command uses the same one-CPU, 1.5-GB hard memory cap, disables CUDA,
and reads teacher arrays only as read-only memory maps. It writes a distinct
progress receipt V3 and refuses to overwrite source bundle V2. The V1 failed
receipt remains at `H230_SOURCE_SEAL_PROGRESS_V2.json`.

```sh
CUDA_VISIBLE_DEVICES='' WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=3600s PYTHONPATH=/home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007/src scripts/limited /home/vlad/development/wunder/worktrees/wnn-bandit-r1-familyloss-20261007/.venv/bin/python /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/seal_h230_source_bundle_v2.py --root /home/vlad/development/wunder/wnn_connectome_starterpack --bundle /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SOURCE_HASH_BUNDLE_V1.json --output /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SOURCE_HASH_BUNDLE_V2.json --receipt /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_SOURCE_SEAL_PROGRESS_V3.json
```

Cache admission still needs a new independent review of the completed V2
source bundle, a refreshed live input/resource receipt, and the H230 cache disk
floor/budget check. This source seal is not scoring or a claim about student or
competition performance.
