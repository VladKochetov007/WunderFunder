# Project agent instructions

## October 29 session: TabM-style GRU, custom Muon, idea triage

User: "think about custom optimization, about TabM like architecture but for this competition with recurrency" plus an idea-triage reflection; 30 GB RAM, 5 GB VRAM, all Kaggle. Done: TabM-mini GRU (`--ensemble-k`, `--ensemble-init`), NorMuon (`--muon-update normuon`) and train-only column view masks (`--view-mask-p`) all **screen_failed** on the cold Muon panel (12 new fits); the k-member blend law ensemble−mean member ≈ 0.336·(1−1/k)·(1−ρ̄) holds exactly, and only seeds/small variants sit below break-even deficit < 2·0.168·(1−ρ). Reframe (Opus + Grok) cards triaged against prior evidence; quadratic output defect closed. [Campaign](research/campaign_2026-10-29/README.md) · [triage reflection](research/unified/methods/IDEA_TRIAGE_REFLECTION_20261029_R1.md). Loop stopped; no jobs running. Next lever: teacher quality (coordinate with teammate PRs 46/47/49). `wnn-campaign.slice` raised to 28G/30G/10 cores.

## October 26 resume: Muon submissions

User: "Go on with research. form the best submissions with new information … make best possible single GRU and blend"; up to 20 GB RAM, 5 GB VRAM, 10 CPU cores. Supersedes the October 25 pause. Kaggle R3 replicated Muon (+0.0036/+0.0033, fresh seeds, T4, true FP32). Done (04:52 UTC): stage-6 Muon transfer +0.00026; parent FP16/bf16 Muon numerics cost nothing (Kaggle); final students `vt_all_h390m_s{3,4}_r1` built; four checked ZIPs in `submissions/muon_20261026/` ([results](research/campaign_2026-10-26/submissions_r1/README.md)). Muon students correlate 0.9995 with AdamW siblings: expect public within noise of Oct-24 quota ZIPs. No jobs running. See the [campaign](research/campaign_2026-10-26/README.md) and [resume receipt](research/campaign_2026-10-26/RESUME_R1.json). No automatic upload.

## October 25 session: paused by user, jobs landing

User: "let current experiments land gracefully, wrap all the research progress … stop research loop, just finish with already running jobs". The research loop is deleted; admit nothing new. Two jobs land on their own: local `wnn-muon6` (auto-graded) and Kaggle `wnn-muon-repro-20261025-r3-runner` (collect manually). Read the [session synthesis](research/campaign_2026-10-25/SESSION_SYNTHESIS_R1.md) and [pause receipt](research/campaign_2026-10-25/PAUSE_R1.json) first. New recipes must use `--strict-fp32-kernels` (torch 2.14 cuDNN RNN defaults to TF32). Resume only on an explicit user instruction.

## October24 bounded cycle: complete

The four-hour cycle has closed new scientific admission. All admitted diagnostics are terminal and evidence is preserved in draft PR #41; no follow-up fit, provider retry or competition upload is authorized by remaining worker accounting. The user-requested additional single Astra/ultra review is complete. Collection completed before14:38:12UTC; the notebook server is stopped and its app/figures are saved. Final publication commit56f6e369dfa2dd6f172bd0ce7e94c8a3b9dbf857 passes CI. Preserve the original resume below as history and use the [collection receipt](research/campaign_2026-10-24/resume4h_r1/COLLECTION_AND_PUBLICATION_R1.json) and frontier for current status. No new allocation is inferred.

## Active resume — user instruction, 2026-10-24

Latest steering: **"Give me some submissions to utilize submission quota"** requests plural exploratory ZIPs, with honest expected-boost evidence; [release scope](research/campaign_2026-10-24/resume4h_r1/QUOTA_SUBMISSION_AUTHORITY_R1.json). Prepare and technically check fixed endpoints without claiming an established public gain or automatically uploading. After the user's93°C CPU reading, root reduced the operational research CPU cap to one core within the authorized ten-core ceiling; [thermal response](research/campaign_2026-10-24/resume4h_r1/LOCAL_THERMAL_RESPONSE_R1.json). This is not a research stop; the remote profile and bounded packaging continue.

Subsequent steering: **"Make some eda $marimo-pair use up to 10 cpu cores"** raises the aggregate CPU ceiling to ten cores and adds live marimo EDA. The 1-GiB/no-swap limit and four-hour deadline remain unchanged; see [CPU/EDA amendment](research/campaign_2026-10-24/resume4h_r1/MARIMO_CPU10_AMENDMENT_R1.json).

The user explicitly resumes experiments for **four more hours**, requests checking **all updated PRs**, pulling useful remote changes and loading uploaded artifacts, using **gh, Kaggle and local compute**, and invokes conduct-creative-research. This supersedes the October23 stop below. The later CPU amendments above supersede the initial quarter-core cap; 1-GiB/no-swap aggregate limits remain binding. No paid compute or automatic competition upload. Preserve dirty work, frozen evidence, historical allocations and the production reserve. See [resume contract](research/campaign_2026-10-24/resume4h_r1/RESUME_R1.json) and [active frontier](research/campaign_2026-10-24/resume4h_r1/FRONTIER.md). Native delegates revert to Luna/max; the prior ten Astra tasks are complete.

## Active stop — user instruction, 2026-10-23

The latest user explicitly orders: **"stop all the experiments now. it overloads laptop. kill all"**. This supersedes every earlier resume. No fits, profiles, imports/tests, exports, packaging, worktree operations, provider launches or automatic research continuation until an explicit user resume. Root killed/stopped research services, disabled the remaining collector timer, interrupted all helpers and runtime-masked the WNN slices. See [stop receipt](research/experiment_drafts/b300-fast-local-kaggle-20261023/USER_STOP_R1.json). Preserve existing files; do not start cleanup, commit/push or another job as a continuation of the stopped task.

## Active resume — user instruction, 2026-10-23

The user explicitly resumes research and requests continuous work until explicitly stopped, invoking conduct-creative-research and research-reframe. This supersedes the reboot pause below while preserving its record. Original allocation/resource limits and production reserve remain binding. See [postboot resume](research/campaign_2026-10-21/next_cycle_r1/POSTBOOT_RESUME_R1.json) and [frontier](research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md).

## Active pause — user instruction, 2026-10-22

Research is **paused for reboot** by the latest user instruction. No new fits, exports or engineering profiles until an explicit user resume. Completed checkpoint/export, unstarted R3 profile and disabled research observers are recorded in [pause receipt](research/campaign_2026-10-21/next_cycle_r1/PAUSE_FOR_REBOOT_R2.json) and [frontier](research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md). Commit/push preservation is authorized; reboot is performed by the user. Older resume/allocation entries below remain history.

## Active cycle amendment — user instruction, 2026-10-21

Implement the [accepted next-cycle plan](specs/plans/2026-10-21-next-local-kaggle-cycle.md) and resume from its [frontier](research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md). This is a new **30-local-GPU-hour** allocation plus verified available Kaggle, local/Kaggle only; **B300 deferred**. Recover saved artifacts, then six total i1 fits (A/B/C × F0/1001,F2/1002), four total dynamics fits; strong-parent follow-up and production are conditional. Preserve previous15hcycle costs and frozen evidence. Retain oneGPUworker,twoaggregateCPUcores,5GBtaskRAM/VRAM,noTaskSwap,diskScratch,strictFP32. Recheck shared quota; admit complete panels using sustained throughput. No paid compute or competition upload. The dated earlier strategy and stop records below remain history.


## Prospective experiment policy — user instruction, 2026-10-19

Future experiment drafts use **4–6 total new scientific fits for an initial idea
screen**, normally 2–3 matched fold/seed cells for a two-arm comparison. There is
no mandatory 30-run matrix and no default 4–6 runs per arm. State per-arm counts,
total new fits, reused exact controls, and teacher/student fitting costs explicitly.
Use multiple source-qualified folds and distinct seeds where affordable; a small
screen cannot disentangle both axes or establish significance. Report every
paired delta, exposure and uncertainty; allocate targeted follow-up by information
and complete-endpoint value. Preserve frozen historical CP30 recipes/results;
amend a continuing experiment explicitly rather than silently changing its plan.
See [ADR0043](specs/adr/0043-fast-screens-and-transformer-teacher-scale.md) and the
[updated draft workflow](research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md).

The new research priority is a large causal Transformer teacher on the team's
eight B300 GPUs. Latest steering prioritizes teacher quality and empirical
scaling; distillation is optional and is not a promotion gate. Prepare architecture,
scale, optimization and profiling handoffs; this instruction does not restart
the stopped remote jobs or change the existing local start/budget. Strict FP32,
autocast/TF32 off, causal inputs, clean ancestry and official scoring still apply.
Large teachers may exceed submission limits. Scale by measured teacher quality
and generalization, with optional later transfer; see the
[teacher-scaling amendment](research/unified/ideas/TEACHER_QUALITY_SCALING_20261019_R1.md).

## Active continuation plan — user instruction, 2026-10-19

After compaction, read the [accepted local/Kaggle plan](specs/plans/2026-10-20-plateau-review-local-kaggle.md)
and [campaign frontier](research/campaign_2026-10-20/plateau_review_r1/FRONTIER.md)
before choosing work. The user requested implementation and durable preservation.
Prepare now; the agreed fitting start is October 20 Kyiv. This cycle has 15 local
GPU-hours plus verified available Kaggle quota in parallel, with **no mandatory
30 paired cells per arm**. Full-data fits remain allowed within that budget.
Retain two aggregate local CPU cores, one GPU worker, 5,000,000,000-byte aggregate
task RAM, 5 GB VRAM, no swap and physical-disk scratch. No paid compute. Preserve
the October 18 stop as history; this scoped authorization does not reopen
unrelated jobs or consume another agent's Kaggle reservation. Record actual
progress and evidence links in the frontier before handoff.

## Python training architecture — user instruction, 2026-10-17

Use `$python-standards` proactively for Python changes. The [training component map](src/wnn/training/README.md) is canonical: reuse its datasets, input assembly, model factory, objectives, update, checkpoint and provenance components. `scripts/train_rnn.py` is a compatibility entrypoint; do not add orchestration or model/feature/loss implementations there. New training logic belongs in a focused semantic component, never in a copied trainer.

Run `uv run ruff format`, `uv run ruff check` and `uv run python scripts/check_training_structure.py` for changed trainer code. The latter enforces300lines/module and200lines/function, typed signatures, public docs and no global/nonlocal; static checks and stdlib checker tests run in CI on main/research-team and PRs. `make quality quality-tests` runs those same checks; `make training-tests` runs the bounded CPU compatibility suite in an already provisioned environment (see [Development](docs/DEVELOPMENT.md)). Use Ruff's configured line length from pyproject.toml. Preserve numerical/update order and source-qualified provenance; do not claim measured parity or performance from static checks. See [ADR0041](specs/adr/0041-modular-training-code.md) and the [refactor/review receipt](research/unified/methods/MODULAR_RNN_TRAINER_REFACTOR_20261017_R1.md). Frozen historical source snapshots are preserved; new research code reuses the canonical package.


## Experiment draft PR workflow — user instruction, 2026-10-17

Use the [research-team workflow guide](research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md) and [English PR template](research/unified/templates/EXPERIMENT_DRAFT_PR.md). Shared policy lives on main; [research-team](https://github.com/VladKochetov007/WunderFunder/tree/research-team) is the integration base; each idea gets an `experiment/<id>` draft PR targeting research-team. The remote link is prospective until first push. The subsequent October17 user instruction explicitly authorizes `gh`, pushing the prepared branches and creating draft PRs; it supersedes the initial pre-push hold. Documentation/implementation preparation does not resume paused model fits.

Proactively use `$create-experiment-draft` for experiment proposals/code/handoffs, `$conduct-creative-research` for mechanisms and independent criticism, `$arxiv-search` for primary literature, and `$b300-training` for accelerator implementation. Canonical skill folders are `.agents/skills/`; Claude discovers the same skills through `.claude/skills/` symlinks. `CLAUDE.md` is a symlink to this file.

Initial scientific screens budget 4–6 total new fits, with an explicit matched panel and any reused controls; the prospective policy above supersedes the October17 CP30 requirement. Preserve clean teacher/initializer ancestry, strictFP32 with autocast and TF32 off, and the unchanged official metric. Engineering profiles/parity checks are not quality comparisons. No quantization until submission work; no queue manager. Every PR answers mechanism, compute advantage, validation/uncertainty, lineage, decision-changing effects and leakage/coverage. Preserve progress metrics, preclip gradient norm, LR, scores, source/config/ancestry and separate evidence/weights archives. Trace all results from unified READMEs through packet/run/archive manifests and PR/merge SHAs; retain negative and inconclusive evidence. See the guide for precise statistics, execution and merge rules.


Latest local instruction (2026-10-15): **two aggregate CPU cores**, replacing
earlier five-core allocations, including support jobs. GPU research continues
with one worker; serialize CPU support at low quota. Temporary files/caches must
use physical disk at `/home/vlad/development/wunder/research_tmp`, not RAM-backed
`/tmp` or `/dev/shm`. See [ADR0039](specs/adr/0039-two-core-local-budget-and-disk-scratch.md).

## Mission and session entry

This repository's root task is Wunder Connectome Challenge2. Maximize long-term expected competition performance with traceable team models, reproducible experiments and valid deterministic submissions. **Form and rank submissions from our research and valid local evaluation evidence. Public leaderboard feedback confirms transfer; it is not the primary selection objective or a prerequisite for packaging a locally supported candidate.** The user wants top-1 and public score above 0.700 as active, achievable research objectives. Do not prune promising directions because current scores are lower or a public forecast is unavailable; report claims and uncertainty honestly. A confidence interval crossing zero is not a universal reject/stop rule; follow [ADR0009](specs/adr/0009-uncertainty-is-not-a-kill-switch.md) and the research methods when allocating uncertain arms. Follow [ADR0008](specs/adr/0008-research-scale-beyond-serving-envelope.md) when distinguishing competition rules from relaxable serving limits. Exclude `wunder/challenge-3`; nested challenge packages can have different formats.

At the start of a session or agent task:

1. Read [specs/README.md](specs/README.md), [objective](specs/research-objective.md) and [current state](specs/current-state.md).
2. Inspect `git status`, the active campaign's frontier and its newest allocation/resume/stop receipts. Preserve unrelated changes.
3. Read the relevant ADRs and skill instructions below before executing their workflows.
4. Check actual model/provider access, resource usage and current quota before allocating work.
5. Give each delegated task its basis, scope, owner, exact deliverable and permitted budget.

The latest user instruction overrides older instructions, specs and campaign allocations. Record its effective scope. An explicit stop closes new admission and drains healthy jobs; a subsequent explicit resume supersedes it while preserving the original record. Do not mistake a historical stop snapshot for current authorization.

Latest submission instruction (2026-10-11): **Do not select from fold 1 alone. Diversify evaluation across source-qualified folds and distinct fitting seeds, with clean teacher/initializer ancestry. After the current exploration results land, rank complete deterministic endpoints using reliable cross-validation and deliver only the strongest supported submissions.** Keep fold exposure and uncertainty explicit; cross-fold evaluation is robustness evidence, not a newly pristine holdout. Preserve existing frozen ZIPs as historical artifacts, but do not recommend a stream of weaker variants. Public feedback remains transfer confirmation. See [ADR0031](specs/adr/0031-cross-fold-top-tier-submission-release.md) and the [current wave/exploitation record](research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/IDEA_STATUS_AND_EXPLOITATION_PLAN_20261011_R1.md). The subsequent exploitation phase should use measured serving headroom up to competition limits; Q4 must earn its place through actual quantized quality, package size and recurrent runtime measurements.

Latest Vast instruction (2026-10-11): auto top-up is disabled. Remaining prepaid credit and provider expiry can be stricter than the $70 campaign ceiling. Reserve time for targeted collection before depletion, collect useful checkpoints/provenance/logs and evaluation predictions while work lands, and do not authorize a top-up, new rental or billing change. Jobs admitted under an older credit assumption need a recorded deadline/collection reassessment; no blind retry or late queued start.

Latest submission exception (2026-10-13): the user explicitly requests one exploratory cross-team blend submission even if it is not best locally, to test public transfer. Prepare and check that package now; the earlier complete-comparison/local-ranking timing gate does not block this release. Keep fixed coefficients, ancestry, deterministic serving, size/ABI checks and honest evidence labels. This does not turn fitted-data diagnostics into clean CV or authorize an external upload. See [release authority](research/campaign_2026-10-13/team_models_r1/EXPLORATORY_SUBMISSION_AUTHORITY_R1.json).

## Research note organization

Start at the repository [research index](RESEARCH.md), then follow the topic READMEs under `research/unified/`. Put new ideas in `research/unified/ideas/`, methods and metric references in `methods/`, independent reviews and audit receipts in `reviews/`, useful data sources, papers and imported external writeups in `sources/`, and operational logs or incidents in `operations/`. Use the shared `templates/NOTE_TEMPLATE.md` for new narrative research notes; record scientific hypotheses and experiments in the append-only JSONL ledgers as well.

**Every useful research item must be linked and described in the README of its corresponding folder under `research/unified/`.** This includes every review, plan, idea, useful data or literature source, evidence receipt, positive or negative result, analysis, plot, model or submission artifact, and operational finding that is created, discovered, imported or used. A chat message, ledger row, generated catalog entry or bare link alone does not satisfy this requirement.

- Each README entry must explain what the item contains, why it matters, and its finding or current status. Include source/provenance, related hypothesis or experiment IDs, and material evidence limitations where applicable; distinguish proposals, external claims and measured results.
- Add the link and description in the same change or handoff as the item. When using an older unindexed item, add its entry then. A task is not complete while its useful outputs or sources are absent from this navigation.
- Give each new research subfolder a README and link that README from its parent, so every item is reachable from the unified index. Choose one canonical description and cross-link other relevant topics when an item spans domains.
- Keep frozen campaign recipes, run receipts, logs and artifacts in their originating folders; link directly to them from the corresponding unified README. Keep large or private data and weights in their approved storage and index their provenance manifest and access location instead of copying payloads or exposing credentials. Preserve frozen archives and append-only history.

Root compatibility aliases have been removed; link directly to canonical files. The unified directory is the mandatory navigation and explanation layer even when the source artifact lives elsewhere.

## Crucial specifications and decisions

| Subject | Read |
|---|---|
| Long-term objective and exploration | [ADR0001](specs/adr/0001-long-term-winning.md), [objective](specs/research-objective.md) |
| Context management without information loss | [ADR0002](specs/adr/0002-layered-evidence-memory.md) |
| Diverse LLMs and random/directed literature | [ADR0003](specs/adr/0003-independent-diverse-ideation.md), [idea generation](specs/idea-generation.md) |
| Team assets, ancestry and target treatments | [ADR0035](specs/adr/0035-official-team-code-and-weight-sharing.md); historical target-treatment decisions in [ADR0004](specs/adr/0004-independent-training-and-target-variants.md) |
| Producer/consumer compatibility and delivery recovery | [ADR0021](specs/adr/0021-validate-consumers-against-producer-artifacts.md); actual producer fixtures, source-qualified group IDs and real binary-format accounting |
| Reuse of teammate methods, code and weights | [ADR0035](specs/adr/0035-official-team-code-and-weight-sharing.md); source and training ancestry remain explicit; supersedes the sharing restrictions in ADR0012 |
| Faithful method reconstruction before variants | [ADR0023](specs/adr/0023-faithful-method-reconstruction-before-variants.md); match teacher stages, seed-specific parents and student blends; optimizer, baseline, normalization and schedule changes are explicit arms |
| End-to-end result ownership | [ADR0005](specs/adr/0005-result-ownership-and-independent-arms.md), [agent contract](specs/agent-contract.md) |
| Continuous execution and repair | [ADR0010](specs/adr/0010-continuous-experiment-delivery.md); keep an active owner for runnable next steps and overlap the next recipe with cloud execution |
| OOM prevention and autonomous repair | [ADR0006](specs/adr/0006-resource-safe-autonomous-execution.md), [execution](specs/experiment-execution.md) |
| Uncertainty and stop/allocation decisions | [ADR0009](specs/adr/0009-uncertainty-is-not-a-kill-switch.md), [research methods](research/RESEARCH_METHODS.md) |

The [ADR index/template](specs/adr/README.md) makes this extendable. Add a decision for consequential changes to objective, allocation, data/recipe semantics or execution ownership. Preserve superseded decisions and link successors. Frozen experiments and raw artifacts are never silently rewritten.

## Research basis and rules

The user confirms the official team merge and explicitly authorizes direct reuse, modification and combination of team source code, native libraries, model weights, checkpoints, teachers and initializers for research and submissions. Independent reimplementation or retraining is no longer required solely because an asset originated with a teammate. This includes training, preprocessing, serving, native libraries, packaging and delegated work. See [ADR0035](specs/adr/0035-official-team-code-and-weight-sharing.md).

Record the author, source revision or archive hash, imported and modified files, training membership, and all teacher/initializer ancestry. Preserve original team assets and identify adaptations separately. Shared weights, repeated exports, and different agents do not constitute independent training replications or remove validation exposure. Independent criticism and clean evaluation requirements remain in force. Older no-copy and own-weights-only instructions are superseded for this official team; unrelated third-party assets still require permitted provenance.

Use only permitted supplied data and allowed publicly available pretrained sources. Official competition documentation lives in `docs/`; verify current rules online when a new rules question arises. Permitted training includes supplied `train.parquet` and `valid.parquet`; data used for fitting cannot support a held-out claim. Keep teacher/source/propensity ancestry out of the evaluated fitting groups for local recipe comparisons.

The official metric is `global_wp_clip2_v1` in `src/wnn/gmetric.py`: fixed scored/prediction mask, clipping ±2 and original clipped-label magnitude weights. Training-target clipping is a separate experimental variable. Never change official evaluation clipping to make an experiment appear better. Predictions/serving must be causal and deterministic with correct sequence-state reset.

Valid local, Kaggle and Vast cross-validation is the primary evidence for research **and submission** selection; public validation confirms transfer. Rank recipes on their mean across distinct seeds and measure each fixed mixture against its matched local endpoint. Do not choose members solely by their public scores, optimize blends from leaderboard feedback, or require an expected-public numerical forecast before preparing a submission. When a strong local run (including a 0.67+ run) is omitted, name the run and give the concrete evidence, ancestry, checkpoint, serving or final-training reason; lack of public confirmation is not such a reason. Hardware location does not make a comparison more or less valid: fitting membership, teacher/initializer ancestry, evaluation exposure and selection determine validity. Investigate public/local disagreement rather than automatically killing a mechanism. See [validation evidence](specs/research-pipeline.md#validation-evidence-and-public-confirmation).

Keep permitted-data, pretrained-source and declared team-asset provenance rules binding. Research-time models may exceed final serving limits in size, context, precision, inference cost, or ensemble count if the active compute allocation permits it. Include explicit model-family/scale ladders and a student/compression/blend path for larger teachers; do not assume a submission constraint is a training constraint.

Previously inspected validation folds and holdouts remain exposed. Different agents, seeds or repartitioning do not make them fresh. Separate evaluation-unit resampling, training variation and public transfer uncertainty. Do not add a local delta to a public score. Public best and current candidate status belong in the live state, not hardcoded promises in prompts. When the user asks for a “new best,” report the strongest candidate supported by our research and local evidence, its comparison to the current local endpoint, and the reason for selecting its model or blend. Report public confirmation separately when available. Label exposure and uncertainty; never manufacture a public score from a local delta or let an unavailable public forecast prevent a locally supported submission.

The teammate-inspired direction is heterogeneous recurrent-teacher distillation with family targets; the audited successful ZIP does not establish a Transformer teacher. [Replication reflection](research/campaign_2026-10-06/bandit_continuation/reviews/MAXIM_REPLICATION_REFLECTION.md) records exact differences and what never ran. Ordinary C0 continuation is not a test of that mechanism.

## How the bandit-like research tree works

Historical diagnostic exception: [ADR0016](specs/adr/0016-exact-reference-diagnostic-training.md) records the earlier fresh-weight reference comparison. [ADR0035](specs/adr/0035-official-team-code-and-weight-sharing.md) now permits team code and weights throughout the research and production submission pipeline. Preserve the old experiment's actual ancestry and do not retroactively change its claim.

Use **objective → mechanism family → direction → hypothesis → experiment**. Give combinations explicit interaction nodes and parent links. Preserve ancestor evidence without assuming gains add or a failed parent makes descendants impossible.

The controller allocates bounded effort across leading lineages, independent exploration and confirmation/serving. Unknown arms may have null numerical priors and qualitative priorities. Update measured cost, effect and uncertainty only from actual relevant runs. Infrastructure/access failure provides execution evidence, not a negative scientific result. An individually weaker model may still improve a teacher target, residual, representation or served blend; measure that contribution.

Treat confidence intervals as uncertainty summaries, not automatic arm-kill gates. Keep validity/resource admission, scientific interpretation, and allocation as separate decisions. Positive but imprecise results can justify a bounded follow-up when expected information value warrants it; defer an arm for opportunity cost without declaring its mechanism false. Preserve an exploration reserve across families. A stop applies to the exact recipe or spending path supported by relevant evidence, while materially different descendants remain eligible.

Deepen promising directions and escape repeated uninformative loops. Move a stale recipe out of active context while preserving its archive. Assign a fresh-context agent a different source/lens/framing, without requiring the old failed explanation. Reopen a direction on a concrete new representation, interaction or contradictory observation.

The authoritative reusable workflow is [research/RESEARCH_METHODS.md](research/RESEARCH_METHODS.md). Existing tree tooling:

The compact [research pipeline](specs/research-pipeline.md) connects portfolio
renewal, independent ideation, interaction experiments and warm compute delivery.
Use `research-portfolio-control` when selecting/reframing lineages and
`research-compute-throughput` when loading/setup or idle lanes limit measured
results. They complement the actor/critic research workflow and do not add an
approval layer. Treat the task as a quant brainteaser when useful: interrogate
measurement, anonymous encodings, causal clocks and invariants as well as models.

- `research/hypotheses.jsonl`, `research/experiments.jsonl`: append-only scientific records.
- `research/bandit_tree/config.json`, `journal.jsonl`: curated lineage/allocation decisions.
- `scripts/research_tree.py --root . build`, `check`, `rank`: generated navigation, integrity and optional ranking.
- `scripts/bandit.py`, `research/bandit/arms.json`: earlier quantitative allocation machinery with standalone/additive reward assumptions. Use only compatible reviewed system-level beliefs; do not use it to reject low-scoring components with proposed conditional value or interactions. Register combinations as explicit nodes and evaluate the complete served endpoint. Never manufacture numbers for untested arms.
- `research/bandit_tree/generated/INDEX.md`: entry point; specs are included in its document catalog.

Mutable monitoring logs can make a generated catalog stale. Rebuild after meaningful changes and distinguish snapshot freshness from scientific integrity. Preserve historical rows, notes and raw artifacts.

## Agent roles and invocation

Native subagents currently use **gpt-6-luna, reasoning effort max**.

Dated user exception (2026-10-11): four independent `gpt-6-sol`/medium ideation reviews plus Grok/OpenCode/Freebuff access attempts were explicitly requested and completed/diagnosed in `ideation_exploitation_20261011_r1/INVOCATIONS_R1.json`. This was a bounded ideation round, not standing Sol implementation/review permission. Default remains Luna/max, at most three active subordinate roles; preserve independent first proposals before synthesis. Do not use Sol or Astra unless the user changes this preference. Use isolated branches/worktrees for writers and minimal role-specific context. Keep at most three subordinate roles active; avoid recursive discussion swarms.

The 2026-10-08 user instruction authorized one Astra ideator for LR scheduling, plateau/noise alternatives and combinations (ADR0023); that task completed. The latest user instruction requires explicit confirmation before any further Astra task. Let an already-running Astra task finish; do not spawn or restart an Astra task autonomously. Default native implementation/ideation roles remain Luna/max.

| Role | What it owes |
|---|---|
| Controller/root | Contract, allocation, admission, source freezing, integration and evidence synthesis |
| Lead scientific actor | Runnable recipe through admitted fit, checkpoints and predictions; code is an intermediate milestone |
| Cloud operator | Actual Kaggle execution, phase/resource monitoring, diagnosed repairs and terminal downloads |
| Ideator/scout | Source-grounded committed hypothesis cards; no fits unless explicitly assigned |
| Independent critic | Concrete checks and pass/fail findings in a bounded review window |
| Grader | Frozen metric/comparison and independently evaluated results |
| Reproducer | Claim-appropriate independent implementation/rerun, with limits stated |

In a Codex harness, use `collaboration.spawn_agent` with a distinct task name, `model="gpt-6-luna"`, `reasoning_effort="max"`, and `fork_turns="none"` or a small explicit history slice when selecting those overrides. Provide a self-contained assignment following [agent-contract.md](specs/agent-contract.md). Use `send_message` for an active agent, `followup_task` to restart an idle/completed role, and `list_agents` to inspect actual live status. Names like `distill_builder` or `cloud_operator` identify session roles, not permanent background processes.

Other harnesses should implement these same boundaries using their native delegation tools. Give critics only necessary sources; hide competing proposals/private checks when useful. An actor must not approve itself or rewrite its grader, frozen recipe or budget authority. Explicitly label code-only assignments so scientific execution is not left ownerless.

## OpenCode, Freebuff and Grok

External models are authorized for diversity, literature ideation, criticism and isolated implementation. Record the actual tool/provider/model, version, prompt/source sections, public output, code and measured result. Do not archive private reasoning transcripts or expose credentials. Keep at most one active native CLI session per credential profile unless supported by the provider.

| Tool | Verified interface and current limitations |
|---|---|
| OpenCode | Installed at `~/.opencode/bin/opencode`; version 1.18.35 at inspection. `opencode models` inspects available routes; `opencode run --help` documents invocation. MiMo-v2.6-flash-free supplied a proposal; Nemotron supplied criticism. Later free coding calls were denied, so verify each route before depending on it. |
| Freebuff | Installed at `~/.npm-global/bin/freebuff`; version 0.2.26 observed 2026-10-12. Interactive CLI: `freebuff --cwd <isolated-workspace>`; `--continue [conversation-id]` resumes a conversation. Use its actual model picker. SolarPro4 produced proposals/draft code; a GLM attempt timed out. Do not invent headless flags. |
| Grok | Native CLI `~/.grok/bin/grok`, version 1.0.50 observed 2026-10-12. Earlier on 2026-10-07, `grok models` listed `grok-4.7` and a bounded native request using that model completed successfully. Check `grok models` and access before each assignment; do not treat that historical route as permanent. See [access receipt](research/campaign_2026-10-06/bandit_continuation/reviews/GROK_NATIVE_ACCESS_20261007.json). |

A bounded OpenCode invocation, after verifying the exact model and preparing an isolated context, is:

```sh
timeout --signal=TERM --kill-after=10s 10m \
  opencode run --dir /path/to/isolated/workspace \
  --model verified-provider/verified-model \
  --format json --file /path/to/public-task.md 'Perform the assigned task.'
```

Run the CLI itself inside the campaign's CPU/RAM boundary. Permission configuration must match the role: read-only source access for ideators, scoped isolated writes for builders, no grader/budget mutation. Do not enable sharing or indiscriminate auto-approval. Freebuff's `--trust-agents` loads `.agents`/MCP configuration; inspect that configuration before trusting it. Keep private competition code/data in authorized private workspaces.

For a bounded native Grok review, use `grok --cwd /path/to/isolated/workspace --model grok-4.7 --no-subagents --permission-mode plan --max-turns 4 --prompt-file /path/to/task.md` inside the same resource wrapper. Inspect `grok --help` for the installed interface. Use the existing grok.com authentication; do not switch to a paid API route to work around a CLI problem. Recheck access after a concrete failure rather than carry historical provider status as a permanent restriction.

Mix directed papers with randomly selected books/articles/passages from the user's library. Record selection seeds and actual passages. Ask for several distinct ideas and a real effort to extract at least one transferable component, including an unconventional one. A small algorithmic detail can matter when the headline method is irrelevant. Do not fabricate applicability. Keep first proposals independent until committed.

## Skills to use

Read each selected `SKILL.md`; user instructions override its generic defaults. Announce first application in the session.

| Skill | Location / purpose |
|---|---|
| `$conduct-creative-research` | `.agents/skills/conduct-creative-research/SKILL.md`; research contract, independent actor/critic/grader roles, evidence and stopping. Read its empirical-validation reference for score interpretation. |
| `$research-portfolio-control` | [.agents/skills/research-portfolio-control/SKILL.md](.agents/skills/research-portfolio-control/SKILL.md); mechanism diversity, explicit combinations, evidence-driven deepening and fresh-context renewal. |
| `$research-compute-throughput` | [.agents/skills/research-compute-throughput/SKILL.md](.agents/skills/research-compute-throughput/SKILL.md); phase measurements, verified warm cohorts, end-to-end delivery and bounded scaleout. |
| `$wnn-connectome-top1` | `.agents/skills/wnn-connectome-top1/SKILL.md`; root competition metric/exposure/serving constraints. Historical resource numbers defer to the latest user allocation. |
| `$marimo-pair` | `~/.agents/skills/marimo-pair/SKILL.md`; paired EDA through a running notebook kernel. Discover sessions rather than assuming an old port/service still exists. |
| `$python-standards` | `.agents/skills/python-standards/SKILL.md`; meaningful Python changes and focused checks. |
| `$research-recipe-audit` | [.agents/skills/research-recipe-audit/SKILL.md](.agents/skills/research-recipe-audit/SKILL.md); equation-to-code and exact-command fidelity before fits/replication claims. |
| `$research-evidence-review` | [.agents/skills/research-evidence-review/SKILL.md](.agents/skills/research-evidence-review/SKILL.md); independent leakage/comparison/uncertainty and deliverable review. |
| `$research-experiment-delivery` | [.agents/skills/research-experiment-delivery/SKILL.md](.agents/skills/research-experiment-delivery/SKILL.md); end-to-end authorized computation and failure repair. |
| `$research-reframe` | [.agents/skills/research-reframe/SKILL.md](.agents/skills/research-reframe/SKILL.md); fresh-context, source-grounded independent tree arms and combinations. |
| `$arxiv-search` / paper-reading skills | Available skills catalog; primary literature/source verification when a scout needs it. |
| `$autoresearch` | `.agents/skills/claude-autoresearch-skill/SKILL.md`; optional bounded iterative search. Existing authorized data/compute need not be requested again. |

## Local execution and existing tools

Latest memory instruction (2026-10-12): avoid OOM rather than use process kills to learn task sizes. For future admissions, target **17 decimal GB** total committed working memory within the unchanged18GB working/20GB hard allocation, leaving1GB operating headroom. Include current parent memory, unused admitted worker reservation and the new support task's complete bound without double-counting worker memory. Healthy admitted jobs continue. Unknown allocations, file formats and staging sizes require bounded actual-producer checks first; never launch-and-see. Metadata readers must bound lengths before reading, and large immutable data stays on physical disk using hardlinks/memmaps/streaming. A hard memory cap is containment, not evidence that a task fits. Plan support-task margin above measured non-reclaimable working arrays; if the bound does not fit, wait or stream instead of retrying a kill. Keep the no-swap, CPU and VRAM caps; do not alter frozen running sources.

Support-task prevention update (2026-10-12): follow the [process-kill prevention plan](research/campaign_2026-10-12/KILL_PREVENTION_PLAN_20261012.md). Before large staging, verify the physical filesystem and reject RAM-backed `/tmp` or `/dev/shm`; prefer hardlinks for immutable inputs on the same filesystem, without chmod or source mutation. Reuse validated NPY readers and bound header reads before payload access. Budget cgroup `memory.current`, including file/shmem charges and active-worker reservation headroom, rather than process RSS alone. Test consumers against actual producer fixtures, derive pins from trusted manifests, and diagnose the first failed bounded command before retrying. These are task-specific checks using existing tools, not permission to raise caps or redesign the controller.

Latest user scope (2026-10-08): CPU is for analysis, EDA and execution support; do not allocate large CPU model fits. Model training uses admitted GPU/TPU workers. Small-data screens follow [the subset protocol](specs/plans/2026-10-08-small-data-screening.md): preserve complete sequences/components and causal histories, keep fixed evaluation membership, record fidelity and optimization budgets, and confirm scale-dependent directions without a universal small-screen kill gate.

Use `rg`/`rg --files`, never `grep`. Batch independent reads/checks, keep dependent operations sequential. Preserve unrelated dirty files. Commit/push meaningful contract, implementation, results and packaging steps separately. Frozen source worktrees must survive a running job; vanished `/tmp` paths must be recreated from commits rather than assumed present.

Temporary user allocation: **12 CPU cores until approximately 2026-10-08 10:00 Europe/Kyiv**, then **5 CPU cores**. Research continues with no time cutoff until explicit user stop; RAM/VRAM limits remain unchanged. See [latest dated authorization](research/campaign_2026-10-06/bandit_continuation/runtime/LOCAL_CPU_CONTINUOUS_20261008_V2.json) and [ADR0017](specs/adr/0017-continuous-research-five-core-morning-cap.md). The verified user timer restores the parent quota; local admission uses the same dated window. Reconcile controller guards before changing parent properties so healthy jobs are not terminated by stale policy.

Local aggregate limits after 10:00 are **5 CPU cores,20 decimal GB RAM,5 decimal GB observed VRAM**, one GPU worker. The 2026-10-08 morning transition has been observed: the effective parent quota is500%CPU. Reaching10:00 never closes research admission; continue until the user explicitly stops. Use persistent `wnn-campaign.slice` with18 GB memory-high,20 GB maximum and no swap. Recheck effective properties after service/daemon changes. Support CLIs, tests and notebooks count toward the same aggregate. New user caps supersede these values.

For small CPU support tasks, set explicit limits rather than the wrapper's historical defaults:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python scripts/research_tree.py --root . build
```

Use `scripts/bandit_resources.py` for durable launch/admission/status/reconciliation and `scripts/bandit_sequence.py` for frozen sequences. Read their CLI help and current campaign operations before launching. Shared runtime contains the single GPU lock, attempt receipts and admission barrier. Profile actual batch/chunk shapes before a fit; stream/memmap caches and start with zero loader workers. Observe RAM including page cache, VRAM, heartbeat, exit status andOOM counters. `scripts/reclaim_bandit_export_cache.py` is a bounded file-cache handoff helper, not permission to raise limits.

Model/data tools already include `scripts/train_rnn.py`, `predict_run.py`, `build_distill_cache.py`, `validate_bandit_campaign_recipe.py` and `evaluate_bandit_continuation.py`, plus modules for splits, metrics, evaluation and teacher-cache provenance. Reuse these primitives where appropriate. Check whether an evaluator actually compares the intended candidate/control rather than only warm parents.

## Kaggle execution

Use existing authorized Kaggle credentials privately; CLI 2.2.4 was verified. User dataset is `vladkochetov07/wunder-funder`; other private source/assets datasets are specified by current packets. Discover actual nested `/kaggle/input` mounts rather than assuming flat paths.

Existing tooling: `scripts/kaggle_bundle.py`, `kaggle_assets.py`, `kaggle_gpu_runner.py`, `kaggle_kernel.py`, `kaggle_sequence.py`, `kaggle_prepare_probe.py`, `kaggle_tpu_probe.py`. The current campaign has reviewed E508 actual-data profile and E510 own-baseline packet builders. Read their manifests/README before reuse; old broken packets remain evidence, not current launch templates.

Kaggle delivery preflight must check the actual title-derived slug against the intended new ID. Keep kernel bootstrap source below1MB as a conservative project envelope and mount larger source/checkpoint payloads through private, versioned, hash-bound assets. Validate the exact mounted payload and nested path offline; do not embed multi-megabyte model bytes in kernel source.

Before dataset uploads, stage only the frozen upload files in a short physical private directory, then verify their hashes. Kaggle CLI 2.2.4 failed with `ENAMETOOLONG` when its upload-cache filename encoded a deep campaign path in both H232 R7 and E540 V10. Preserve the original failure and reconcile exact owner inventory/readiness before a corrected create. An immediate read-only download 404 after creation may precede READY; verify the exact version before launch. A provider RUNNING status and an empty stdout stream do not expose a worker that records phases only to output files; collect those receipts at terminal status and distinguish startup from fitting/export.

Before pushing to slow-start Kaggle or renting Vast, check the actual sealed packet, not only the working source: archive/source/config bindings, exact entry-point import and CLI bootstrap, actual producer artifacts consumed by its validators, CUDA initialization order, and terminal/export interfaces. Follow [ADR0021](specs/adr/0021-validate-consumers-against-producer-artifacts.md). Cover actual NPY headers and nested optimizer precision records; a synthetic fixture alone is insufficient when the producer is available. Keep checks bounded and relevant so preparation does not become an idle-compute loop. Hardware-specific numerical and memory claims still require a short actual-device profile before a long fit. Preserve failed packets and identify corrections. Reuse completed profiles/checkpoints when a postprocessing consumer fails rather than repeat valid computation.

For paid launch admission, also exercise the applicable reservation and cohort-stage checks against a private copy of the current ledger, without provider mutations. Use the spending path authorized by the current ADR; a valid per-attempt ceiling does not establish aggregate admission. After dispatch, inspect the supervisor's actual startup and ledger/provider receipt immediately: a successful `systemd-run` return does not prove the worker started. Stop only a diagnosed campaign supervisor's futile restart loop, preserve its failure, and bind a corrected attempt before retrying.

Each admitted cloud cohort needs:

1. A new identified packet with exact source/config/data/recipe hashes, private inputs and output path. Run the packet's own offline startup validators against the final embedded payload, including conditional-fit authorization receipts; an external controller approval does not populate a pending embedded receipt.
2. Live quota, separate cloud headroom and representative actual-device timing/memory. Local ceilings do not define cloud memory capacity. Paid Vast execution is authorized only under the bounded plan below; no subscription or credit top-up is authorized.
3. Input identity and split/membership checks, including expected/observed fingerprints persisted before failure. Correct producer/consumer hash ranges; diagnose mismatches rather than suppressing them.
4. Phase/resource heartbeat, bounded worker execution, terminal status and complete output/log/checkpoint downloads.
5. Diagnosed corrected retry under a new ID when necessary. Provider RUNNING is not proof fitting started; startup delays are not bounded by timeout-from-push assumptions.

`research/campaign_2026-10-06/cloud_preflight/poll_provider.py`, `provider_targets.json` and the `wnn-campaign-kaggle-poller.timer` provide durable read-only monitoring. Add new targets and terminal receipts, refresh the allocation, and reconcile provisional quota at terminal status. A monitor alone does not collect/validate results or rerun failures; that belongs to the cloud operator/controller. TPU production requires real-device numerical, gradient/reload and full-shape checks before spending on fits.

The user authorized autonomous cloud troubleshooting. Continue with concrete repairs rather than repeatedly asking for routine permission. Preserve diagnostic evidence and actual hardware/resource observations. Do not mark an experiment successful from provider status alone.

The specialist-skill design and MFTE comparison are recorded in
[ADR0007](specs/adr/0007-specialized-research-skills.md). Use these roles within
`conduct-creative-research`; they do not replace the controller or current contract.

## New tooling, EDA and delivery

The user requests scaling toward **5–6 distinct evaluated idea screens per hour** when Kaggle and Vast are operational. Follow [the throughput scaleout plan](specs/plans/2026-10-08-throughput-scaleout.md). Measure cold-start and warm rates separately; count completed mechanism comparisons rather than seed repeats or startup jobs. Use measured cycle time to allocate additional owned cloud capacity and warm multi-recipe sessions, while retaining full-scale confirmation and memory headroom. This is a target, not evidence of current throughput.

### Standing compute-utilization instruction

The user's latest instruction is to keep **Kaggle GPUs/TPUs, the local GPU and authorized Vast workers continuously supplied with useful experiments until the user explicitly says to stop**. This supersedes historical clock cutoffs and stop snapshots. Follow [ADR0014](specs/adr/0014-continuous-useful-compute.md) and the [live delivery estimates](specs/plans/2026-10-07-live-research-estimates.md).

The controller owns actual execution, not only a prepared queue. Give every admitted lane an end-to-end operator and prepare its reviewed successor while its current job runs. Check actual phase/heartbeat, resource consumption and terminal outputs; provider RUNNING alone is insufficient. Diagnose errors and launch identified corrected attempts within existing authority. Do not leave a healthy runnable scientific arm idle waiting for unrelated ideation, infrastructure or ablations. If a lane is idle, record its concrete dependency, current repair owner and next action, then work to remove that dependency. Refresh live quota and allocate bounded production TPU work after numerical/real-data parity; old probe-only budgets are not a permanent research cap.

The user explicitly permits multiple campaign-owned Vast workers simultaneously or a larger multi-GPU worker when measured throughput, host headroom and all-in cost justify it. The initial one-worker pilot is a measurement step, not a permanent concurrency restriction. Prefer independent and diverse mechanism families before detailed sweeps of one arm; account for replicated data staging, shared input/CPU bottlenecks and artifact collection. Keep aggregate paid reservations inside the approved cohort/campaign budget. After results land, or ready useful work is insufficient for admitted capacity, generate fresh directed/random-literature ideas with diverse providers, register them as distinct or interaction nodes, and prioritize bounded informative tests using the bandit-tree allocation. Never fill a lane with an uninformative repeat solely to keep it busy.

Continuous utilization means useful computation: never spin, duplicate completed runs, relax causality/data/metric checks, exceed local aggregate caps or spend on idle rentals to manufacture utilization. Existing credit and the $20 cumulative Vast ceiling, no top-up, independent review requirements and strict campaign-created-instance ownership remain binding. ADR0015 supersedes the historical $4 initial-cohort release gate. Collect results, update evidence and teardown our paid worker when its useful queue is empty. An explicit stop drains healthy admitted jobs and closes new admission; a later explicit resume records its supersession.

Prioritize different approaches over additional seeds. Keep already admitted seed/control work for its declared comparison, but use new exploration capacity for mechanisms and interactions. Parallelize already planned independent runs when it reduces time to useful evidence within quota and all-in budget. Runner work must remove a concrete execution bottleneck: reuse validated packets and checks, overlap staging/review with execution, and avoid repeated infrastructure design before delivering the first scientific result on a lane.

### Approved continuous execution and Vast ownership

Follow [the saved execution plan](specs/plans/2026-10-07-continuous-research.md). The initial user allocation was $4 for the first Vast cohort and $20 cumulative. The latest flexible-spending instruction supersedes the first-cohort release gate: follow [ADR0015](specs/adr/0015-flexible-credit-bounded-parallel-compute.md), using existing credit with a conservative $20 cumulative operating ceiling, bounded reviewed reservations, and no top-up. Historical allocation receipts remain immutable. Load the private `.env` `vastai` credential only inside the API client; never log, commit, include it in agent prompts or upload it to workers.

The user's strict ownership rule is: **“touch only machines you rented. do not hurt existing, they are not mine.”** Never stop, delete, reconfigure, connect to, collect from or adopt pre-existing/unrelated Vast instances. Every such operation requires the campaign's durable creation receipt binding attempt ID, account scope and provider instance ID. Matching labels alone are insufficient. An ambiguous create must be reconciled without duplicate creation or touching an unproven instance. No bulk cleanup. Preserve this boundary in delegated tasks and tests.

Before renting with optional signed-input staging, run `kaggle_vast_prefetch.prepare_prefetch` locally against the exact private plan, public admission and frozen input identities. Preserve signed URLs privately and validate their permitted representation before spending; a hash-bound plan alone does not establish that its URL schema passes the downloader. Never print the URLs or put them in prompts, commits or process arguments.

Keep successors prepared and switch only after valid terminal evidence or a diagnosed repair. Reassess the research allocation every two hours. Provider/tool access errors are execution evidence, not negative scientific results; a prepared queue is not a running job.

When a preferred experiment is still being implemented, run an already-reviewed useful fallback on available compute rather than wait for the preferred recipe. Each lane's queue must distinguish runnable code with exact commands and source/data identities from ideas needing implementation. Replenish the runnable queue before it empties. Reuse processes or compatible loaded data across successors where existing tools support it, with fresh declared model/optimizer state; repeated full-data loading is a measured cost to avoid, not a prerequisite for every run. Never change a live frozen recipe to insert a successor.

Follow [ADR0022](specs/adr/0022-prepared-successors-and-paid-warm-cohorts.md): prioritize suitable powerful cloud accelerators, maintain a rolling executable queue per lane, and prepare/review successors while predecessors run. Start an admitted successor on valid completion without waiting for another user message; keep independent fallback arms ready when possible. Reuse compatible verified loaded data across finite multi-model cohorts, preserving recipe membership/preprocessing and independent training state. Distinguish proposals from executable/admitted entries. Paid Vast sessions must run useful bounded queued work rather than wait for ideas merely to keep data loaded; collect and destroy only our receipt-proven worker when its useful queue is exhausted. Track completed comparisons per hour, phase/loading time and all-in cost; preserve headroom, current quota/budget and ownership rules.

The user explicitly authorizes terminating an inefficient or bugged job when a diagnosed corrected successor has better long-term value. Preserve its partial outputs, failure/resource evidence and costs; do not classify loading failure as a negative scientific result. Continue healthy informative runs. Paid-worker termination remains limited to instances proven campaign-owned by creation receipts.

Build a new helper only when an existing tool cannot perform the required concrete task. Give it a narrow schema, bounded inputs/resources, explicit failure behavior, provenance and focused tests for material invariants. Independently review before production. Avoid constructing a generic framework as a prerequisite for one runnable experiment.

Use marimo-pair for EDA and save PNG/SVG images inside each research lineage's `plots/` subdirectory. Edit a running notebook through its kernel APIs; keep reactive cells from launching fits. Preserve standalone artifacts for humans and later agents.

After meaningful results, update hypothesis/experiment records, frontier, specs/current-state and tree/index. Bind claims to original commands/checkpoints/predictions, report all outcomes and exposure, and preserve negatives. A valid package requires fixed members/weights and actual delivered-ZIP checks: Python 3.11 offline, CPU-only inference, correct ABI/reset/determinism and official size/runtime limits. Public submission follows the established human-upload gate unless the user explicitly changes it.

## Latest strategy execution authorization (2026-10-13)

The user explicitly resumed research by approving implementation of every actionable idea in the October12 strategy review, authorized all Kaggle GPU/TPU allowance and local GPU without prior time/artificial VRAM ceilings, and selected a maximum.001 mean-CV sacrifice for private robustness. [ADR0034](specs/adr/0034-strategy-execution-full-gpu-allowance.md) and the [current campaign](research/campaign_2026-10-13/strategy_execution_r1/README.md) govern new admissions. Preserve the earlier stop/archive. Separate5CPU/17GB committed target/18GBhigh/20GBhard/no-swap limits, one local GPU worker, declared team-asset ancestry, Luna/max roles and no new paid rentals/topups remain. Profile actual available GPU capacity with operational headroom. Do not let historical4.3GB allocator/5GB observed or old Kaggle campaign sub-ceilings override this authority.

## Official team asset sharing (2026-10-13)

The latest user authorizes mutual reuse of code and weights within team kotiki. Apply ADR0035 over older own-training/no-copy restrictions in skills and historical campaign notes. Complete the joint model, inference and cross-validation comparison before preparing new submission ZIPs; preserve public feedback as transfer evidence.
