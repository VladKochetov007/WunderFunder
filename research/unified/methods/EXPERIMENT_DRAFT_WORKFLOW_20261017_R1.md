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

### 2. Fast matched screens — prospective October 19 revision

Future drafts budget **4–6 total newly trained fits for an initial idea screen**.
For two arms this normally means 2–3 matched `(fold, fitting seed)` cells, one
control and one candidate fit each. It does not mean4–6 fits per arm. Three arms
with two cells cost six fits. Include teacher and student fits in the inventory;
a teacher-only screen does not establish student transfer. Exact source-clean
cached controls may be reused; name their IDs, recipe, ancestry and old versus
new fitting costs. One shared control is not a fresh replication per contrast.

Freeze explicit cells before outcomes. Prefer distinct existing folds and fitting
seeds; report which axis is covered. A default two-cell panel is F0/seed1001 and
F2/seed1002; an optional third is F3/seed1001. These identities are examples,
not reservations: verify ownership before admission. Two or three cells cannot
separate fold and seed effects or establish significance. Useful gains, mixed
mechanisms and informative uncertainty can justify separately budgeted follow-up.
There is no automatic expansion to30. Full-data fits and larger teachers remain
eligible under actual compute authority.

The plan helper accepts explicit `cells`, reports per-arm and total fitting
counts and defaults to two cells. For an unchanged historical v1 CP30 recipe,
`--legacy-source-commit <full-original-source-SHA>` explicitly enables its old
panel; the source reference is declared, not verified by the plan printer. A new
recipe cannot bypass the screen budget merely by declaring30. Cached endpoints
need matching arm/fold/seed plus `artifact_ref`/`ancestry_ref`; follow-up needs
`followup_allocation` with `reference` and `new_fit_budget`. Verify those references
and actual resources at admission, not just their syntax. Preserve frozen recipes and results;
changing future admission in an existing study needs a dated amendment.
Resource-only smoke/profile/parity checks are engineering work, not scientific
replications. See [ADR0043](../../../specs/adr/0043-fast-screens-and-transformer-teacher-scale.md).

Use existing source-qualified component folds, not new row splits. Every cell requires matching fitting membership, preprocessing, update/epoch budget and checkpoint rule, with fold-clean teachers, initializers, normalizers and propensity models. A F0-trained parent cannot initialize another outer fold merely because the CLI fold flag changed. If ancestry cannot be reconstructed, train a clean parent or mark the cell ineligible. Do not train on held-out targets or select a checkpoint on the outer evaluation. Any development split/checkpoint selection stays within fitting components. All existing folds remain exposed.

### 3. Leakage and representativeness

Trace the supplied columns and target horizon to the data contract; do not assume price-like values are physical prices. Use raw causal rows, sequence reset and past-only feature transformations. Test the intended prefix-invariance/reset property before costly runs. Statistics and imputers fit only on fitting membership. Keep complete component/sequence grouping; audit duplicate/source aliases across train and valid. Distillation targets for evaluated groups must come from teachers and ancestors excluding those groups. Output AR uses previous model predictions, never true lagged competition labels whose horizons may overlap the future.

Report score by source, fold, sequence age and target-magnitude regime in addition to pooled official `global_wp_clip2_v1`. Inspect train/valid covariate drift, family/sequence-length coverage and fitting-vs-evaluation chronology where available. Anonymous/missing timestamps limit claims about future-regime generalization; component disjointness alone does not establish temporal representativeness. Repartitioning exposed folds creates no fresh holdout.

### 4. Decision and uncertainty

Before fitting, state the useful effect, compute cost, baseline and decision that
would change. Pair official scores on identical scored rows and retain every
cell, failure, exclusion and tried configuration. Report each paired delta and
their descriptive mean. With unequal fold counts, freeze equal-fold weighting
over INCLUDED folds and name them; an unrun fold is not a zero effect. Report
pooled OOF and seed-ensemble predictions as distinct endpoints with full ancestry.

Component-cluster paired resampling holds the fits fixed and describes evaluation
sensitivity only. Freeze its RNG seed and keep component multiplicities identical
for both arms and shared seed endpoints. With2–3 cells, do not claim a reliable
training-variance estimate or significance from bootstrapping fit scores. List
fold/seed coverage and confounding instead. Repeated rows and overlapping fitting
sets are not independent datasets. More seeds/folds are decision-driven follow-up,
not a quota. A later formal significance claim needs a declared null, comparison
family, adequate design and selection accounting.

Historical CP30 recipes retain their frozen equal-four-fold estimand, shared-seed
resampling and endpoints. The new policy does not rewrite their statistics or
turn partial matrices into completed ones. Success includes student and complete
mixture contribution. A zero-crossing interval alone does not close an idea;
negative evidence narrows the tested recipe. Never add a local delta to a public
score or relabel exposed folds as fresh.

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

This protocol supplies no measured gain by itself. The
[draft inventory](../ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) preserves
original proposals. The [October19 B300 review](../reviews/B300_PR_RUN_UPDATE_REVIEW_20261019_R1.md)
records partial endpoints from stopped jobs. Publish through gh under the user's
standing authority, then record actual PR links, source, code readiness and limits.
The prospective fitting-count revision and large Transformer-teacher priority are
recorded in [ADR0043](../../../specs/adr/0043-fast-screens-and-transformer-teacher-scale.md).
