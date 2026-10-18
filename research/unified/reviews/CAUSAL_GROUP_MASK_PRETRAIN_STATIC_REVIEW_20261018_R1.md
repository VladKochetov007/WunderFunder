---
kind: independent-static-review
status: completed_before_publication
created_utc: 2026-10-18T06:00:51Z
hypothesis_ids: [H-CAUSAL-GROUP-MASK-PRETRAIN-20261018-R1]
experiment_ids: [E-CAUSAL-GROUP-MASK-PRETRAIN-20261018-R1]
evidence_paths:
  - ../../experiment_drafts/causal-group-mask-pretraining-b300/README.md
---

# Independent static review: causal group-masked pretraining

**Review status:** final follow-up passed before PR publication. **Scope:** source and recipe inspection only. The critic did not run code, load competition data, start a GPU process, or train a model.

## Findings and repairs

1. Resume provenance originally hashed the pretraining entrypoint but omitted canonical input, model, data, and loop dependencies. The recipe now hashes these project sources as well as the pretraining modules and entrypoint.
2. Reused statistics and initializer artifacts were not fully checked against the current fold and source. Existing statistics now verify the fixed sample membership and source hashes; initializer conversion verifies the pretraining source, statistics and metadata, converter, and both checkpoint hashes before reuse.
3. The first run command required a stats file without showing how to create it; the fold-specific statistics step is now part of the pair runner and has a one-thread cap. The frozen seed is `880301 + fold`.
4. C1 was called compute-matched although its design only matches two additional supervised sequence passes. Recipe and PR now say row-pass matched, not FLOP matched.
5. The pair runner attempted to rename a manifest after the converter had already written its final `_pair.json` path. The stale rename was removed.
6. The PR body and unified index entries are included with the code handoff.
7. Concurrent seed cells could race while creating shared fold statistics. `run_pair.sh` now holds a per-fold `flock` across statistics creation or verification.
8. Existing pretraining checkpoints could match source and fold while carrying different training hyperparameters. The initializer converter now checks exact fit/evaluation membership plus the full frozen model and optimization recipe, and the checkpoint records the optimizer name.

## Remaining boundary

The trainer and prediction/endpoint-grading commands are present, but this PR does not implement the cross-cell seed/fold bootstrap aggregator or Holm adjustment. That adapter must be supplied before summarizing the 30-cell results. No hardware profile or quality execution is represented as complete.

## Critic outcome

The independent critic found no direct label or evaluation-group path in the masked feature loader or state-transfer layout. Its first follow-up identified the shared-stat race and incomplete checkpoint-recipe validation; both were corrected, and the final follow-up returned **PASS with no remaining high-severity static blocker to publishing the draft** on 2026-10-18. It did not run Python, tests, data scans or training. This is a source review, not a reproduction or hardware validation.
