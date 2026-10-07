# Project agent instructions

## Mission and session entry

This repository's root task is Wunder Connectome Challenge2. Maximize long-term expected competition performance with independently trained models, reproducible experiments and valid deterministic submissions. The user wants serious research, diverse ideas, useful submissions and eventual winning, rather than short-term convenience. The 0.7000 public milestone is an ambition, not an evidence-supported forecast. Exclude `wunder/challenge-3`; nested challenge packages can have different formats.

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
| End-to-end result ownership | [ADR0005](specs/adr/0005-result-ownership-and-independent-arms.md), [agent contract](specs/agent-contract.md) |
| OOM prevention and autonomous repair | [ADR0006](specs/adr/0006-resource-safe-autonomous-execution.md), [execution](specs/experiment-execution.md) |

The [ADR index/template](specs/adr/README.md) makes this extendable. Add a decision for consequential changes to objective, allocation, data/recipe semantics or execution ownership. Preserve superseded decisions and link successors. Frozen experiments and raw artifacts are never silently rewritten.

## Research basis and rules

The user confirms the competition hosts approved the team merge. Recipe and code inspection within that approved collaboration is authorized. Train our base models, teachers and students independently; do not use teammate model weights as our initializers/teachers. Reusing our own audited independently trained checkpoints is allowed when the recipe explicitly declares their ancestry.

Use only permitted supplied data and allowed publicly available pretrained sources. Official competition documentation lives in `docs/`; verify current rules online when a new rules question arises. Permitted training includes supplied `train.parquet` and `valid.parquet`; data used for fitting cannot support a held-out claim. Keep teacher/source/propensity ancestry out of the evaluated fitting groups for local recipe comparisons.

The official metric is `global_wp_clip2_v1` in `src/wnn/gmetric.py`: fixed scored/prediction mask, clipping ±2 and original clipped-label magnitude weights. Training-target clipping is a separate experimental variable. Never change official evaluation clipping to make an experiment appear better. Predictions/serving must be causal and deterministic with correct sequence-state reset.

Previously inspected validation folds and holdouts remain exposed. Different agents, seeds or repartitioning do not make them fresh. Separate evaluation-unit resampling, training variation and public transfer uncertainty. Do not add a local delta to a public score. Public best and current candidate status belong in the live state, not hardcoded promises in prompts.

The teammate-inspired direction is heterogeneous recurrent-teacher distillation with family targets; the audited successful ZIP does not establish a Transformer teacher. [Replication reflection](research/campaign_2026-10-06/bandit_continuation/reviews/MAXIM_REPLICATION_REFLECTION.md) records exact differences and what never ran. Ordinary C0 continuation is not a test of that mechanism.

## How the bandit-like research tree works

Use **objective → mechanism family → direction → hypothesis → experiment**. Give combinations explicit interaction nodes and parent links. Preserve ancestor evidence without assuming gains add or a failed parent makes descendants impossible.

The controller allocates bounded effort across leading lineages, independent exploration and confirmation/serving. Unknown arms may have null numerical priors and qualitative priorities. Update measured cost, effect and uncertainty only from actual relevant runs. Infrastructure/access failure provides execution evidence, not a negative scientific result. An individually weaker model may still improve a teacher target, residual, representation or served blend; measure that contribution.

Deepen promising directions and escape repeated uninformative loops. Move a stale recipe out of active context while preserving its archive. Assign a fresh-context agent a different source/lens/framing, without requiring the old failed explanation. Reopen a direction on a concrete new representation, interaction or contradictory observation.

The authoritative reusable workflow is [research/RESEARCH_METHODS.md](research/RESEARCH_METHODS.md). Existing tree tooling:

- `research/hypotheses.jsonl`, `research/experiments.jsonl`: append-only scientific records.
- `research/bandit_tree/config.json`, `journal.jsonl`: curated lineage/allocation decisions.
- `scripts/research_tree.py --root . build`, `check`, `rank`: generated navigation, integrity and optional ranking.
- `scripts/bandit.py`, `research/bandit/arms.json`: earlier quantitative allocation machinery. Use only compatible reviewed beliefs; do not manufacture numbers for untested arms.
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
| Grok | Requested Grok4.7 was unavailable through previously configured routes. Recheck actual provider catalog/authentication when needed; use an existing authorized route if available. Never claim Grok ran from a requested label or guess a model ID. |

A bounded OpenCode invocation, after verifying the exact model and preparing an isolated context, is:

```sh
timeout --signal=TERM --kill-after=10s 10m \
  opencode run --dir /path/to/isolated/workspace \
  --model verified-provider/verified-model \
  --format json --file /path/to/public-task.md 'Perform the assigned task.'
```

Run the CLI itself inside the campaign's CPU/RAM boundary. Permission configuration must match the role: read-only source access for ideators, scoped isolated writes for builders, no grader/budget mutation. Do not enable sharing or indiscriminate auto-approval. Freebuff's `--trust-agents` loads `.agents`/MCP configuration; inspect that configuration before trusting it. Keep private competition code/data in authorized private workspaces.

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

Use `rg`/`rg --files`, never `grep`. Batch independent reads/checks, keep dependent operations sequential. Preserve unrelated dirty files. Commit/push meaningful contract, implementation, results and packaging steps separately. Frozen source worktrees must survive a running job; vanished `/tmp` paths must be recreated from commits rather than assumed present.

Current local aggregate limits are **10 CPU cores,20 decimal GB RAM,5 decimal GB observed VRAM**, one GPU worker. Use persistent `wnn-campaign.slice` with18 GB memory-high,20 GB maximum and no swap. Recheck effective properties after service/daemon changes. Support CLIs, tests and notebooks count toward the same aggregate. New user caps supersede these values.

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
2. Live quota, separate cloud headroom and representative actual-device timing/memory. Local ceilings do not define cloud memory capacity. No new paid rental/subscription is authorized.
3. Input identity and split/membership checks, including expected/observed fingerprints persisted before failure. Correct producer/consumer hash ranges; diagnose mismatches rather than suppressing them.
4. Phase/resource heartbeat, bounded worker execution, terminal status and complete output/log/checkpoint downloads.
5. Diagnosed corrected retry under a new ID when necessary. Provider RUNNING is not proof fitting started; startup delays are not bounded by timeout-from-push assumptions.

`research/campaign_2026-10-06/cloud_preflight/poll_provider.py`, `provider_targets.json` and the `wnn-campaign-kaggle-poller.timer` provide durable read-only monitoring. Add new targets and terminal receipts, refresh the allocation, and reconcile provisional quota at terminal status. A monitor alone does not collect/validate results or rerun failures; that belongs to the cloud operator/controller. TPU production requires real-device numerical, gradient/reload and full-shape checks before spending on fits.

The user authorized autonomous cloud troubleshooting. Continue with concrete repairs rather than repeatedly asking for routine permission. Preserve diagnostic evidence and actual hardware/resource observations. Do not mark an experiment successful from provider status alone.

The specialist-skill design and MFTE comparison are recorded in
[ADR0007](specs/adr/0007-specialized-research-skills.md). Use these roles within
`conduct-creative-research`; they do not replace the controller or current contract.

## New tooling, EDA and delivery

Build a new helper only when an existing tool cannot perform the required concrete task. Give it a narrow schema, bounded inputs/resources, explicit failure behavior, provenance and focused tests for material invariants. Independently review before production. Avoid constructing a generic framework as a prerequisite for one runnable experiment.

Use marimo-pair for EDA and save PNG/SVG images inside each research lineage's `plots/` subdirectory. Edit a running notebook through its kernel APIs; keep reactive cells from launching fits. Preserve standalone artifacts for humans and later agents.

After meaningful results, update hypothesis/experiment records, frontier, specs/current-state and tree/index. Bind claims to original commands/checkpoints/predictions, report all outcomes and exposure, and preserve negatives. A valid package requires fixed members/weights and actual delivered-ZIP checks: Python 3.11 offline, CPU-only inference, correct ABI/reset/determinism and official size/runtime limits. Public submission follows the established human-upload gate unless the user explicitly changes it.
