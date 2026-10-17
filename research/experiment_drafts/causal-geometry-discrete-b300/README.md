---
kind: experiment_draft_packet
status: proposal_implementation_complete_profile_pending_no_matrix_fit
created_utc: 2026-10-18
updated_utc: 2026-10-18
hypothesis_ids:
  - H-CAUSAL-GEOMETRY-DISCRETE-B300-20261018-R1
experiment_ids:
  - E-CAUSAL-GEOMETRY-DISCRETE-B300-20261018-R1
evidence_paths:
  - research/campaign_2026-10-17/moment_geometry_r1/README.md
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/README.md
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/summary_r3.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_quick_probe_comparison.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_quick_probe_results.png
---

# Causal geometry and discrete state features: B300 handoff

This packet proposes a four-arm, 30-cell-per-arm comparison on eight independent B300 workers. It includes the causal feature implementation, an indexed equal-width mask, a clean single-cell runner, per-card profile receipts, a frozen 120-fit/15-wave matrix, and exact launch instructions. **No B300 profile or scientific fit has run from this packet.** The one local warm-start probe belongs to root's separate two-hour exploratory task and is not a full-matrix cell.

## Contents

- [PR.md](PR.md): English draft PR with the required eleven sections, mechanism, comparison, uncertainty and limits.
- [recipe.json](recipe.json): machine-readable arms, folds, seeds, optimizer/update budget, contrasts and lineage.
- [matrix.json](../../../scripts/geometry_b300/matrix.json): all 30 crossed fold/seed pairs and all 15 waves with eight explicit GPU-slot assignments per wave. The adapter validates this file before each invocation.
- [LAUNCH.md](LAUNCH.md): exact commands, read-only data bindings, writable output/scratch mounts, profile gate, wave launch, resume and archives.
- [prior_evidence/README.md](prior_evidence/README.md): linked geometry EDA and copied, hash-bound R3/prior-probe evidence.
- [native serving receipt](../../unified/operations/geometry_native_serving_20261018/README.md): corrected native channel order, ABI scan, synthetic full-feature parity and fixed-precision callback profiles; actual-weight/package checks remain pending.
- [scripts/geometry_b300/](../../../scripts/geometry_b300/): independent one-GPU cell runner, eight-card profile launcher, profile receipt validator and eight-fit wave launcher.
- [src/wnn/moment_geometry/online.py](../../../src/wnn/moment_geometry/online.py) and [src/wnn/research_features.py](../../../src/wnn/research_features.py): causal transform, fitting-only prior, fixed feature order and indexed controls, reused by the canonical modular trainer.
- [native geometry serving path](../../../native/src/python_geometry.rs), [single-member callback](../../../submission/solution_geometry.py), [reference checker](../../../scripts/check_geometry_submission.py) and [packager extension](../../../scripts/package_submission.py): native build/ABI and corrected synthetic parity passed; actual-weight parity, complete competition runtime and package checks remain pending.
- [H/E ledger records](../../hypotheses.jsonl) and [experiment ledger](../../experiments.jsonl): append-only proposal registration, zero scientific fits.

## Provenance and implementation status

The isolated branch is based on origin/research-team at SHA e9a4c16c848e09d87d279a3dd64d7db376a8551d. Shared policy ancestor is origin/main at SHA 0f4796609781da2b0886e2c5f9c52139987ba120. Selected training/feature files were copied individually from the dirty local main worktree at 35cf36a337a23ecc0e86159a5cb1867a851ac06e; the original publication source SHA is recorded in the [draft PR](https://github.com/VladKochetov007/WunderFunder/pull/15) and [catalogue](../README.md), while runtime receipts bind their exact source SHA. No root logs, local admission JSON, dataset payload, cache, prediction, or weight file is copied.

The source changes are integrated into the canonical src/wnn/training/ architecture and compatibility CLI. The fresh H512×1 comparison is implemented. The optional warm-start direction requires one matching fold-clean parent for each fold/seed identity; the current root result is one exposed F0/seed3 probe only. Native serving source commits bdf996f9d1040936e2f84c837a39fc39e7431e40, ce5f8f63c1c495dac3c7f4a3ce46226b6efd7aba, channel-order correction 03e6db635977d9514f46ec990738112075c588de, indexed evidence 04252f6a35bc99711edf57397b775b66f6f11fe3, and callback reset-compatibility fix 52e7ed6bb6f5861846bb11eac29ceeaf3669dc0b are included. The [indexed serving receipt](../../unified/operations/geometry_native_serving_20261018/README.md) records a fixed native build/ABI pass, corrected synthetic 20,000-row feature/prediction parity, callback reset compatibility, and fixed-precision profiles. Actual-weight parity, full competition runtime and package checks remain pending. No B300 launch-ready claim is made until a clean checkout, four portable source manifests, all eight measured cards and the sealed B300 profile receipt are present.

## Research trace

Start at the current [unified ideas index](../../unified/ideas/README.md), follow the [geometry validation record](../../unified/validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md), and use the [experiment draft catalogue](../README.md) for the published PR URL and original source SHA. The source evidence is exposed development CV; it does not establish a pristine holdout, market semantics, serving readiness, or public-score impact.
