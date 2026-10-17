---
kind: experiment_draft_packet
status: proposal_implementation_complete_profile_pending_no_matrix_fit
created_utc: 2026-10-18
updated_utc: 2026-10-18
hypothesis_ids:
  - H-CAUSAL-GEOMETRY-DISCRETE-B300-20261018-R1
experiment_ids:
  - E-CAUSAL-GEOMETRY-DISCRETE-B300-20261018-R1
  - E-GEOMETRY-QUICK-PROBE-20261018-R1
  - E-GEOMETRY-ADAPTER-PROBE-20261018-R2
  - E-GEOMETRY-ADAPTER-NATIVE-SERVING-20261018-R2
  - E-GEOMETRY-VARIATIONS-20261018-R3
  - E-GEOMETRY-CORRECTION-SCALE-20261018-R3S
  - E-GEOMETRY-ADAPTER-PUBLIC-RESULT-20261018-R2
evidence_paths:
  - research/campaign_2026-10-17/moment_geometry_r1/README.md
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/README.md
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/summary_r3.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_quick_probe_comparison.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_quick_probe_results.png
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_comparison.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_frozen_parent_check.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_recipe.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_native_q16x/README.md
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_native_q16x/RELEASE.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_adapter_probe_r2_native_q16x/PUBLIC_RESULT.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_geometry_variations_r3/README.md
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_geometry_variations_r3/geometry_r3_comparison.png
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_geometry_variations_r3/RELEASE.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_geometry_variations_r3/geometry_r3_reverse_half_q16x_20261018_v2_f0_serving.json
  - research/experiment_drafts/causal-geometry-discrete-b300/prior_evidence/root_geometry_variations_r3/geometry_r3_reverse_half_q16x_20261018_v2_python311_smoke.json
---

# Causal geometry and discrete state features: B300 handoff

This packet proposes a four-arm, 30-cell-per-arm comparison on eight independent B300 workers. It includes the causal feature implementation, an indexed equal-width mask, a clean single-cell runner, per-card profile receipts, a frozen 120-fit/15-wave matrix, and exact launch instructions. **No B300 profile or matrix fit has run from this packet.** Root's separate local R1/R2 direction probes and later R3 block/scale follow-up are exposed F0/seed3 evidence, not B300 cells or independent seed replications. R1 and R2 adapter training were negative for their exact recipes; the two R3 block fits also scored below their parent. The inference-only R2 alpha=−0.50 alias was slightly above the parent, with a conditional component interval crossing zero; its selected q16x endpoint scored 0.6814234 on the same exposed F0 and is marked ready for user exploratory upload. The [prior evidence index](prior_evidence/README.md) keeps these scoped findings and their limitations separate; none rejects the broader feature, teacher/student or distillation family.

## Contents

- [PR.md](PR.md): English draft PR with the required eleven sections, mechanism, comparison, uncertainty and limits.
- [recipe.json](recipe.json): machine-readable arms, folds, seeds, optimizer/update budget, contrasts and lineage.
- [matrix.json](../../../scripts/geometry_b300/matrix.json): all 30 crossed fold/seed pairs and all 15 waves with eight explicit GPU-slot assignments per wave. The adapter validates this file before each invocation.
- [LAUNCH.md](LAUNCH.md): exact commands, read-only data bindings, writable output/scratch mounts, profile gate, wave launch, resume and archives.
- [prior_evidence/README.md](prior_evidence/README.md): linked geometry EDA and copied, hash-bound evidence. The earlier H128×2 R3 screen is distinct from the later root-owned F0/seed3 R3 block fits and inference-only alpha screen; both retain their separate figures and receipts. The R2 q16x bundle also records the user's public result.
- [native serving receipt](../../unified/operations/geometry_native_serving_20261018/README.md): corrected native channel order, ABI scan and synthetic full-feature parity; root R2 and selected R3 q16x endpoints have separate full exposed-F0 scores and exact-ZIP Python 3.11 receipts. The R2 FP32 native/Torch tolerance failed; complete target-host runtime remains unmeasured.
- [scripts/geometry_b300/](../../../scripts/geometry_b300/): independent one-GPU cell runner, mount and frozen-matrix preflight, eight-card profile launcher, profile receipt validator and eight-fit wave launcher.
- [src/wnn/moment_geometry/online.py](../../../src/wnn/moment_geometry/online.py) and [src/wnn/research_features.py](../../../src/wnn/research_features.py): causal transform, fitting-only prior, fixed feature order and indexed controls, reused by the canonical modular trainer.
- [src/wnn/training/feature_adapter.py](../../../src/wnn/training/feature_adapter.py): optional guarded warm-parent adapter that freezes old weights and trains only the appended causal-feature input columns; its R2 local probe was negative for one exposed recipe.
- [native geometry serving path](../../../native/src/python_geometry.rs), [single-member callback](../../../submission/solution_geometry.py), [reference checker](../../../scripts/check_geometry_submission.py) and [packager extension](../../../scripts/package_submission.py): native build/ABI and corrected synthetic parity passed; the separate root R2 q16x endpoint has full exposed-F0 grading and exact-ZIP Python3.11 checks. FP32 native/Torch tolerance failed; full target-host runtime remains unmeasured.
- [H/E ledger records](../../hypotheses.jsonl) and [experiment ledger](../../experiments.jsonl): append-only proposal registration, root-owned local R1/R2/R3 evidence, the user-reported R2 public result, and separate q16x native serving evidence; no B300 profile or matrix fit.

## Provenance and implementation status

The isolated branch is based on origin/research-team at SHA e9a4c16c848e09d87d279a3dd64d7db376a8551d. Shared policy ancestor is origin/main at SHA 0f4796609781da2b0886e2c5f9c52139987ba120. Core training/feature files were copied individually from the dirty local main worktree at 35cf36a337a23ecc0e86159a5cb1867a851ac06e. The optional `feature_adapter.py` guard was copied separately from root’s dirty post-R2 helper; the portable R2 receipts do not capture that run’s exact pre-guard source revision. The original publication source SHA is recorded in the [draft PR](https://github.com/VladKochetov007/WunderFunder/pull/15) and [catalogue](../README.md), while runtime receipts bind their exact source SHA. No root logs, local admission JSON, dataset payload, cache, prediction, or weight file is copied.

The source changes are integrated into the canonical src/wnn/training/ architecture and compatibility CLI. The fresh H512×1 comparison is implemented. An optional guarded adapter mode freezes a compatible raw/event parent and trains only the zero-initialized appended60 input-weight columns; the R2 one-pair result is negative and exposed, and its local source revision was not captured in the portable training receipts. The B300 matrix does not use this adapter path. Any broader warm-start comparison requires a matching fold-clean parent for each fold/seed identity. Native serving source commits bdf996f9d1040936e2f84c837a39fc39e7431e40, ce5f8f63c1c495dac3c7f4a3ce46226b6efd7aba, channel-order correction 03e6db635977d9514f46ec990738112075c588de, indexed evidence 04252f6a35bc99711edf57397b775b66f6f11fe3, and callback reset-compatibility fix 52e7ed6bb6f5861846bb11eac29ceeaf3669dc0b are included. The [indexed serving receipt](../../unified/operations/geometry_native_serving_20261018/README.md) now records the selected q16x package's exact exposed-F0 score, Python 3.11 smoke, signed-zero-sensitive q16x cache replay, and failed FP32 native/Torch tolerance. Complete target-host runtime remains unmeasured. No B300 launch-ready claim is made until a clean checkout, four portable source manifests, all eight measured cards and the sealed B300 profile receipt are present.

## Research trace

Start at the current [unified ideas index](../../unified/ideas/README.md), follow the [geometry validation record](../../unified/validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md), and use the [experiment draft catalogue](../README.md) for the published PR URL and original source SHA. The source evidence is exposed development CV; it does not establish a pristine holdout, market semantics, serving readiness, or public-score impact.
