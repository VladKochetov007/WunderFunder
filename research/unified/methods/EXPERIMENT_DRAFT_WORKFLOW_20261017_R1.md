---
kind: method
status: proposed
created_utc: 2026-10-17T12:15:27Z
updated_utc: 2026-10-17T12:15:27Z
hypothesis_ids: []
experiment_ids: []
evidence_paths: []
---

# Research-team experiment draft workflow

## Question and authority

The October 17 user instruction establishes `main → research-team → experiment/<id>`: each scientific idea is a draft PR targeting **research-team**, with English code, numbered rationale and execution instructions. Prepare locally first. The subsequent user instruction explicitly authorizes pushing and creating draft PRs through gh, superseding the initial pre-push hold. This is authority for documentation, branches and implementation, not an implicit restart of paused fits, a new rental or a billing change. No queue manager is introduced.

Canonical future branch: [research-team](https://github.com/VladKochetov007/WunderFunder/tree/research-team). This link becomes available only after publishing. A local branch is not a published PR.

## Mechanism and method

### 1. Branches and ownership

Keep shared instructions, skills and evidence conventions on main. Fork research-team from that foundation. Fork each experiment from research-team; put its PR body, recipe, adapters and README under `research/experiment_drafts/<id>/`. One idea may contain necessary matched controls and ablations; independent mechanisms get separate PRs. Record the source main SHA, experiment SHA, H/E IDs, parent IDs and owner. Rebase or merge current research-team before integrating; retain frozen run source SHAs.

Use `.agents/skills/create-experiment-draft/SKILL.md` and the numbered template in `research/unified/templates/EXPERIMENT_DRAFT_PR.md`. Before modifying a mechanism use conduct-creative-research and source-grounded arxiv-search; use b300-training for accelerator code. Independent critique is proportionate to the scientific claim; code completion is not empirical approval.

```bash
# Read-only fetch is permitted; inspect before integrating upstream changes.
git fetch origin
MAIN_SHA=$(git rev-parse main)
# First setup only: git branch research-team "$MAIN_SHA"
git merge-base --is-ancestor "$MAIN_SHA" research-team
TEAM_BASE=$(git rev-parse research-team)
git switch -c experiment/<id> "$TEAM_BASE"
# Record MAIN_SHA and TEAM_BASE in recipe.json. If ancestry fails, reconcile first.
# Before publication also inspect origin/main and origin/research-team if present.
# Implement, document, record sources and commit. Once publication is authorized:
git push -u origin experiment/<id>
gh pr create --draft --base research-team --head experiment/<id>   --title "[Experiment] <id>" --body-file research/experiment_drafts/<id>/PR.md
```

Do not create PRs targeting main for individual experiments. Add the actual PR URL to the packet and its unified index when it exists. Review/merge completed or useful negative/inconclusive work into research-team, preserving IDs and evidence; a separate integration PR promotes a coherent snapshot to main. Retain receipts for rejected experiments through an archive/evidence commit rather than losing them when branches are deleted. Merging notes does not promote a candidate model. Resolve common-file conflicts by retaining both research entries, never by replacing append-only ledgers. Branch-specific source is pinned per run even after merge.

### 2. Thirty paired cells per scientific comparison

Freeze 30 `(fold, fitting seed)` cells before results: F0/F1 have seeds1001–1008, F2/F3 seeds1001–1007 (8/8/7/7). Candidate and control each run all30 cells: **60 fits for a two-arm comparison**, not30 fits split between arms. Additional arms also need30 fits when claiming a confirmed comparison. Resource-only smoke/profile/parity checks are engineering work, not30 scientific replications. Existing runs do not silently count toward the new matrix.

Use existing source-qualified component folds, not new row splits. Every cell requires matching fitting membership, preprocessing, update/epoch budget and checkpoint rule, with fold-clean teachers, initializers, normalizers and propensity models. A F0-trained parent cannot initialize another outer fold merely because the CLI fold flag changed. If ancestry cannot be reconstructed, train a clean parent or mark the cell ineligible. Do not train on held-out targets or select a checkpoint on the outer evaluation. Any development split/checkpoint selection stays within fitting components. All existing folds remain exposed.

### 3. Leakage and representativeness

Trace the supplied columns and target horizon to the data contract; do not assume price-like values are physical prices. Use raw causal rows, sequence reset and past-only feature transformations. Test the intended prefix-invariance/reset property before costly runs. Statistics and imputers fit only on fitting membership. Keep complete component/sequence grouping; audit duplicate/source aliases across train and valid. Distillation targets for evaluated groups must come from teachers and ancestors excluding those groups. Output AR uses previous model predictions, never true lagged competition labels whose horizons may overlap the future.

Report score by source, fold, sequence age and target-magnitude regime in addition to pooled official `global_wp_clip2_v1`. Inspect train/valid covariate drift, family/sequence-length coverage and fitting-vs-evaluation chronology where available. Anonymous/missing timestamps limit claims about future-regime generalization; component disjointness alone does not establish temporal representativeness. Repartitioning exposed folds creates no fresh holdout.

### 4. Decision and uncertainty

Before fits record a practically useful score improvement, absolute or cost-adjusted, with its justification; do not invent a universal cutoff. Specify resource opportunity cost and useful downstream teacher/student/blend outcomes. Report every cell, failures and any exclusion. Pair candidate and control predictions on identical scored rows. Compute pooled official scores, per-fold deltas, seed variation and component-cluster paired resampling; never treat millions of rows or30 overlapping-fit cells as independent observations. Reuse seed identities across folds as a crossed factor. Report evaluation-unit uncertainty separately from fitting variation and selection/public-transfer uncertainty; a conditional bootstrap over fixed predictions covers only the first.

Freeze the primary estimand as `D = (1/4) Σ_f [(1/n_f) Σ_s (WP(candidate_f,s) − WP(control_f,s))]`, with n_f=8,8,7,7 and equal quarter weights for folds. Compute WP separately for each fitted endpoint on its full outer scored rows, then pair; do not average predictions across seeds for this estimand. Report pooled OOF and seed-ensemble endpoints as separately named secondary statistics; neither replaces D for recipe ranking without a declared new endpoint comparison.

For fitting variability, draw10000 bootstrap samples of the eight seed identities with replacement, using the SAME sampled seed multiplicities in all folds; within each fold renormalize over its available cells, redraw a sample if any fold has zero available seed multiplicity. Recompute D and report the percentile95% interval as an approximate seed-bootstrap interval conditional on these four exposed folds, fixed data and ancestry. This preserves shared-seed structure and the unbalanced eighth seed. Eight seed identities are few; do not claim exact coverage or uncertainty over new market regimes. For evaluation-unit uncertainty keep fits fixed and resample source-qualified components within each fold, with identical component multiplicities for every seed and both arms, recomputing official scores and D. A joint sensitivity can nest these two resamplings, clearly labeled approximate; keep the two marginal intervals as well. Freeze a resampling RNG seed in the packet. Report all four fold means/ranges as robustness evidence, not an inferred population of unseen folds. For a claim of statistical significance, define the null, confidence procedure and comparison family before reading results; account for shared components/seed effects and multiple arms (e.g. Holm within the frozen primary comparison family). Thirty repetitions alone do not guarantee power or calibrated coverage. Inspect paired variance to plan follow-up precision. Do not report a CI that resamples folds as if four folds were a large independent population. Success means useful effect with robustness and compatible student transfer; uncertainty can justify targeted follow-up. A zero-crossing interval does not close a mechanism. A negative result narrows the tested recipe, not every descendant. Never add local delta to a public score.

### 5. Precision and B300 execution

For this campaign full precision means FP32 weights, states, activations, gradients and optimizer state; autocast/GradScaler off and TF32 off. No FP16/BF16/FP8 or quantization in scientific comparisons. Explicitly set float32 matmul precision to highest and disable CUDA matmul/cuDNN TF32. Inspect actual kernels: FlashAttention fast paths often do not support FP32, so use a verified FP32 math implementation unless support is measured. Do not silently enable reduced precision for advertised B300 throughput.

Use one B300 per independent GRU fit when it fits and measure GPU occupancy. Allocate multiple GPUs to a large teacher through torchrun/DDP; eight cards need not all train the same small model. State actual GPU count, topology, CUDA/driver/PyTorch build, effective batch, accumulation, context, optimizer/scheduler, memory and time. Source commands must be executable for the packet's implementation stage; mark adapters and trainer integration still pending. No wall-time estimate is a measurement. Profile a bounded real-shape step before committing a long run. Quality-first schedules and architecture ladders are in the [AR teacher proposal](../ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md).

### 6. Training receipts and portable archives

Record immutable config + source SHA/dirty patch, package lock/container digest, hardware/environment, source/data/split hashes, all ancestor hashes and run/cell/arm IDs. Log optimizer attempts and successful steps separately; epoch, rows/tokens, loss and components, LR for every param group, gradient norm before clipping, clipping/nonfinite counts, weight/update norms when useful, train and validation official scores, throughput, elapsed/compile/load/evaluation/checkpoint time, peak RAM/VRAM and device utilization/power when available. Log checkpoints and fixed selection policy; outer held-out score is reporting, not checkpoint feedback. Log validation mask, sample/unit counts and exact metric version. Preserve RNG/optimizer/scheduler/sampler and sequence-position state for resume; restart from a safe boundary if stream-state restoration is incomplete.

One writer per rank file; rank0 records aggregate metrics only after correct reductions. Save rank-local diagnostics separately. Never log credentials or whole private rows by default. Shared recording/bundling APIs are engineering helpers; trainers must explicitly call them. They do not instrument old scripts automatically.

Produce separate evidence archive (config, environment, progress JSONL, terminal status, predictions/grades or their manifests, source/provenance) and weights archive (checkpoint and resume states). SHA256 every file and archive, preserve relative paths, and include size/precision/format metadata. Keep large payloads off Git and record approved private storage locations. Do not bundle supplied datasets automatically. Treat an archive as untrusted: bound metadata and forbid traversal/symlink escape on import. After receiving teammates' archives, verify hashes and producer formats, reproduce metric from predictions, record missing fields and add a result note even for a failed run.

### 7. Unified trace and merge contract

Required trace: unified topic README → experiment packet README/PR → H/E IDs and exact source commit → run manifest → progress/evaluation receipts → archive hashes/private access locations. A result note states mechanism, observed result, limitations and decision; narrative prose is welcome if these links remain recoverable. Every useful child file is described in its folder README. Cross-link results from ideas, models/validation and operations where relevant without duplicating canonical claims. Store run records append-only; corrections are successor notes/ledger events. Record PR number, branch, merge SHA and promoted-main SHA when known, never fabricated URLs. Preserve historical recipes and teacher exports in their original directories.

## Evidence, interpretation and next step

This workflow is a user-requested preparation protocol. No B300 has been inspected, no30-cell comparison has run, and no new leaderboard impact is measured. The [draft inventory](../ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) distinguishes interrupted fits, completed sources and new proposals. Publish through gh under the subsequent explicit user authorization, then report actual PR links, code readiness, source search and limitations.
