# Project agent instructions

## Mission and session entry

This repository's root task is Wunder Connectome Challenge2. Maximize long-term expected competition performance with independently trained models, reproducible experiments and valid deterministic submissions. The user wants top-1 and public score above 0.700 as active, achievable research objectives. Do not prune promising directions because current scores are lower or a public forecast is unavailable; report claims and uncertainty honestly. A confidence interval crossing zero is not a universal reject/stop rule; follow [ADR0009](specs/adr/0009-uncertainty-is-not-a-kill-switch.md) and the research methods when allocating uncertain arms. Follow [ADR0008](specs/adr/0008-research-scale-beyond-serving-envelope.md) when distinguishing competition rules from relaxable serving limits. Exclude `wunder/challenge-3`; nested challenge packages can have different formats.

At the start of a session or agent task:

1. Read [specs/README.md](specs/README.md), [objective](specs/research-objective.md) and [current state](specs/current-state.md).
2. Inspect `git status`, the active campaign's frontier and its newest allocation/resume/stop receipts. Preserve unrelated changes.
3. Read the relevant ADRs and skill instructions below before executing their workflows.
4. Check actual model/provider access, resource usage and current quota before allocating work.
5. Give each delegated task its basis, scope, owner, exact deliverable and permitted budget.

The latest user instruction overrides older instructions, specs and campaign allocations. Record its effective scope. An explicit stop closes new admission and drains healthy jobs; a subsequent explicit resume supersedes it while preserving the original record. Do not mistake a historical stop snapshot for current authorization.

## Crucial specifications and decisions

| Subject | Read |
|---|---|
| Long-term objective and exploration | [ADR0001](specs/adr/0001-long-term-winning.md), [objective](specs/research-objective.md) |
| Context management without information loss | [ADR0002](specs/adr/0002-layered-evidence-memory.md) |
| Diverse LLMs and random/directed literature | [ADR0003](specs/adr/0003-independent-diverse-ideation.md), [idea generation](specs/idea-generation.md) |
| Independent weights and target treatments | [ADR0004](specs/adr/0004-independent-training-and-target-variants.md) |
| Independent implementation of shared teammate methods | [ADR0012](specs/adr/0012-independent-teammate-method-implementation.md); inspect ideas and details, write our own code and builds |
| End-to-end result ownership | [ADR0005](specs/adr/0005-result-ownership-and-independent-arms.md), [agent contract](specs/agent-contract.md) |
| Continuous execution and repair | [ADR0010](specs/adr/0010-continuous-experiment-delivery.md); keep an active owner for runnable next steps and overlap the next recipe with cloud execution |
| OOM prevention and autonomous repair | [ADR0006](specs/adr/0006-resource-safe-autonomous-execution.md), [execution](specs/experiment-execution.md) |
| Uncertainty and stop/allocation decisions | [ADR0009](specs/adr/0009-uncertainty-is-not-a-kill-switch.md), [research methods](research/RESEARCH_METHODS.md) |

The [ADR index/template](specs/adr/README.md) makes this extendable. Add a decision for consequential changes to objective, allocation, data/recipe semantics or execution ownership. Preserve superseded decisions and link successors. Frozen experiments and raw artifacts are never silently rewritten.

## Research basis and rules

The user confirms the competition hosts approved the team merge. Recipe and code inspection within that approved collaboration is authorized. Train our base models, teachers and students independently; do not use teammate model weights as our initializers/teachers. Reusing our own audited independently trained checkpoints is allowed when the recipe explicitly declares their ancestry.

Teammate source code may be read to understand ideas, algorithms and implementation details, but must never be copied as-is or used as the basis of a cosmetic rewrite. Share methods and independently implement them in this project's code style, libraries and build system. This applies to training, preprocessing, serving, native libraries, packaging and delegated work. Record the inspected reference and the independently written implementation separately. The user reports that identical teammate code can be banned; treat independent implementation as a binding project requirement. This supersedes older requests to directly run or adapt a teammate's code.

Use only permitted supplied data and allowed publicly available pretrained sources. Official competition documentation lives in `docs/`; verify current rules online when a new rules question arises. Permitted training includes supplied `train.parquet` and `valid.parquet`; data used for fitting cannot support a held-out claim. Keep teacher/source/propensity ancestry out of the evaluated fitting groups for local recipe comparisons.

The official metric is `global_wp_clip2_v1` in `src/wnn/gmetric.py`: fixed scored/prediction mask, clipping ±2 and original clipped-label magnitude weights. Training-target clipping is a separate experimental variable. Never change official evaluation clipping to make an experiment appear better. Predictions/serving must be causal and deterministic with correct sequence-state reset.

Keep data, pretrained-source and weight-independence rules binding. Research-time models may exceed final serving limits in size, context, precision, inference cost, or ensemble count if the active compute allocation permits it. Include explicit model-family/scale ladders and a student/compression/blend path for larger teachers; do not assume a submission constraint is a training constraint.

Previously inspected validation folds and holdouts remain exposed. Different agents, seeds or repartitioning do not make them fresh. Separate evaluation-unit resampling, training variation and public transfer uncertainty. Do not add a local delta to a public score. Public best and current candidate status belong in the live state, not hardcoded promises in prompts.

The teammate-inspired direction is heterogeneous recurrent-teacher distillation with family targets; the audited successful ZIP does not establish a Transformer teacher. [Replication reflection](research/campaign_2026-10-06/bandit_continuation/reviews/MAXIM_REPLICATION_REFLECTION.md) records exact differences and what never ran. Ordinary C0 continuation is not a test of that mechanism.

## How the bandit-like research tree works

Latest diagnostic exception: [ADR0016](specs/adr/0016-exact-reference-diagnostic-training.md) records the user's explicit request to train the identified Maxim source unchanged with our fresh weights for comparison and EDA. This supersedes the earlier no-direct-run rule only for isolated reference diagnostics. No teammate weights or source code enter our production submission pipeline.

Use **objective → mechanism family → direction → hypothesis → experiment**. Give combinations explicit interaction nodes and parent links. Preserve ancestor evidence without assuming gains add or a failed parent makes descendants impossible.

The controller allocates bounded effort across leading lineages, independent exploration and confirmation/serving. Unknown arms may have null numerical priors and qualitative priorities. Update measured cost, effect and uncertainty only from actual relevant runs. Infrastructure/access failure provides execution evidence, not a negative scientific result. An individually weaker model may still improve a teacher target, residual, representation or served blend; measure that contribution.

Treat confidence intervals as uncertainty summaries, not automatic arm-kill gates. Keep validity/resource admission, scientific interpretation, and allocation as separate decisions. Positive but imprecise results can justify a bounded follow-up when expected information value warrants it; defer an arm for opportunity cost without declaring its mechanism false. Preserve an exploration reserve across families. A stop applies to the exact recipe or spending path supported by relevant evidence, while materially different descendants remain eligible.

Deepen promising directions and escape repeated uninformative loops. Move a stale recipe out of active context while preserving its archive. Assign a fresh-context agent a different source/lens/framing, without requiring the old failed explanation. Reopen a direction on a concrete new representation, interaction or contradictory observation.

The authoritative reusable workflow is [research/RESEARCH_METHODS.md](research/RESEARCH_METHODS.md). Existing tree tooling:

- `research/hypotheses.jsonl`, `research/experiments.jsonl`: append-only scientific records.
- `research/bandit_tree/config.json`, `journal.jsonl`: curated lineage/allocation decisions.
- `scripts/research_tree.py --root . build`, `check`, `rank`: generated navigation, integrity and optional ranking.
- `scripts/bandit.py`, `research/bandit/arms.json`: earlier quantitative allocation machinery with standalone/additive reward assumptions. Use only compatible reviewed system-level beliefs; do not use it to reject low-scoring components with proposed conditional value or interactions. Register combinations as explicit nodes and evaluate the complete served endpoint. Never manufacture numbers for untested arms.
- `research/bandit_tree/generated/INDEX.md`: entry point; specs are included in its document catalog.

Mutable monitoring logs can make a generated catalog stale. Rebuild after meaningful changes and distinguish snapshot freshness from scientific integrity. Preserve historical rows, notes and raw artifacts.

## Agent roles and invocation

Native subagents currently use **gpt-6-luna, reasoning effort max**. Do not use Sol or Astra unless the user changes this preference. Use isolated branches/worktrees for writers and minimal role-specific context. Keep at most three subordinate roles active; avoid recursive discussion swarms.

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
| Freebuff | Installed at `~/.npm-global/bin/freebuff`; version 0.2.15. Interactive CLI: `freebuff --cwd <isolated-workspace>`; `--continue [conversation-id]` resumes a conversation. Use its actual model picker. SolarPro4 produced proposals/draft code; a GLM attempt timed out. Do not invent headless flags. |
| Grok | Native CLI `~/.grok/bin/grok`, version 1.0.46, is authenticated through grok.com. On 2026-10-07, `grok models` listed `grok-4.7` and a bounded native request using that model completed successfully. Default is `grok-4.6`, so select `--model grok-4.7` explicitly. Earlier alternate-route failures do not establish native CLI unavailability. See [access receipt](research/campaign_2026-10-06/bandit_continuation/reviews/GROK_NATIVE_ACCESS_20261007.json). |

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

Latest user scope (2026-10-08): CPU is for analysis, EDA and execution support; do not allocate large CPU model fits. Model training uses admitted GPU/TPU workers. Small-data screens follow [the subset protocol](specs/plans/2026-10-08-small-data-screening.md): preserve complete sequences/components and causal histories, keep fixed evaluation membership, record fidelity and optimization budgets, and confirm scale-dependent directions without a universal small-screen kill gate.

Use `rg`/`rg --files`, never `grep`. Batch independent reads/checks, keep dependent operations sequential. Preserve unrelated dirty files. Commit/push meaningful contract, implementation, results and packaging steps separately. Frozen source worktrees must survive a running job; vanished `/tmp` paths must be recreated from commits rather than assumed present.

Temporary user allocation: **12 CPU cores until approximately 2026-10-08 10:00 Europe/Kyiv**, then 6. RAM/VRAM limits remain unchanged. See [dated authorization](research/campaign_2026-10-06/bandit_continuation/runtime/LOCAL_CPU_WINDOW_20261008_V1.json). The verified user timer restores the parent quota; local admission uses the same dated window. Reconcile controller guards before changing parent properties so healthy jobs are not terminated by stale policy.

Current local aggregate limits are **6 CPU cores,20 decimal GB RAM,5 decimal GB observed VRAM**, one GPU worker. Use persistent `wnn-campaign.slice` with18 GB memory-high,20 GB maximum and no swap. Recheck effective properties after service/daemon changes. Support CLIs, tests and notebooks count toward the same aggregate. New user caps supersede these values.

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

Each admitted cloud cohort needs:

1. A new identified packet with exact source/config/data/recipe hashes, private inputs and output path.
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

Continuous utilization means useful computation: never spin, duplicate completed runs, relax causality/data/metric checks, exceed local aggregate caps or spend on idle rentals to manufacture utilization. Existing $4 initial/$20 cumulative Vast limits, no top-up, independent review requirements and strict campaign-created-instance ownership remain binding. Collect results, update evidence and teardown our paid worker when its useful queue is empty. An explicit stop drains healthy admitted jobs and closes new admission; a later explicit resume records its supersession.

Prioritize different approaches over additional seeds. Keep already admitted seed/control work for its declared comparison, but use new exploration capacity for mechanisms and interactions. Parallelize already planned independent runs when it reduces time to useful evidence within quota and all-in budget. Runner work must remove a concrete execution bottleneck: reuse validated packets and checks, overlap staging/review with execution, and avoid repeated infrastructure design before delivering the first scientific result on a lane.

### Approved continuous execution and Vast ownership

Follow [the saved execution plan](specs/plans/2026-10-07-continuous-research.md). The initial user allocation was $4 for the first Vast cohort and $20 cumulative. The latest flexible-spending instruction supersedes the first-cohort release gate: follow [ADR0015](specs/adr/0015-flexible-credit-bounded-parallel-compute.md), using existing credit with a conservative $20 cumulative operating ceiling, bounded reviewed reservations, and no top-up. Historical allocation receipts remain immutable. Load the private `.env` `vastai` credential only inside the API client; never log, commit, include it in agent prompts or upload it to workers.

The user's strict ownership rule is: **“touch only machines you rented. do not hurt existing, they are not mine.”** Never stop, delete, reconfigure, connect to, collect from or adopt pre-existing/unrelated Vast instances. Every such operation requires the campaign's durable creation receipt binding attempt ID, account scope and provider instance ID. Matching labels alone are insufficient. An ambiguous create must be reconciled without duplicate creation or touching an unproven instance. No bulk cleanup. Preserve this boundary in delegated tasks and tests.

Keep successors prepared and switch only after valid terminal evidence or a diagnosed repair. Reassess the research allocation every two hours. Provider/tool access errors are execution evidence, not negative scientific results; a prepared queue is not a running job.

The user explicitly authorizes terminating an inefficient or bugged job when a diagnosed corrected successor has better long-term value. Preserve its partial outputs, failure/resource evidence and costs; do not classify loading failure as a negative scientific result. Continue healthy informative runs. Paid-worker termination remains limited to instances proven campaign-owned by creation receipts.

Build a new helper only when an existing tool cannot perform the required concrete task. Give it a narrow schema, bounded inputs/resources, explicit failure behavior, provenance and focused tests for material invariants. Independently review before production. Avoid constructing a generic framework as a prerequisite for one runnable experiment.

Use marimo-pair for EDA and save PNG/SVG images inside each research lineage's `plots/` subdirectory. Edit a running notebook through its kernel APIs; keep reactive cells from launching fits. Preserve standalone artifacts for humans and later agents.

After meaningful results, update hypothesis/experiment records, frontier, specs/current-state and tree/index. Bind claims to original commands/checkpoints/predictions, report all outcomes and exposure, and preserve negatives. A valid package requires fixed members/weights and actual delivered-ZIP checks: Python 3.11 offline, CPU-only inference, correct ABI/reset/determinism and official size/runtime limits. Public submission follows the established human-upload gate unless the user explicitly changes it.
