# Research specifications and decisions

Start here when resuming research or assigning an agent. Our objective is long-term competition winning through reproducible improvements, useful exploration and reliable execution. A completed code task is not a completed experiment.

[Project AGENTS.md](../AGENTS.md) describes agent invocation, model/provider routes,
skills, existing tooling and Kaggle execution.

Read in order:

1. [Objective and constraints](research-objective.md).
2. [Evidence basis and current state](current-state.md).
3. [Idea generation and research-tree allocation](idea-generation.md).
4. [Agent assignments and result handoffs](agent-contract.md).
5. [Experiment execution](experiment-execution.md).
6. [Decision index](adr/README.md).

Current approved execution plan: [continuous local/Kaggle/Vast research](plans/2026-10-07-continuous-research.md), including the strict campaign-owned-instance boundary. [ADR0015](adr/0015-flexible-credit-bounded-parallel-compute.md) supersedes the initial cohort release gate with bounded useful parallel spending within existing credit and the $20 cumulative ceiling.

Status terminology: **recipe locked** means the settings, sources and evaluation are fixed for reproducibility; it does not pause an experiment. **Backbone weights frozen** means that training updates only the specified heads or adapters. **Deferred** means no current compute allocation, with the reason and condition for reconsideration recorded. **Execution blocked** means a concrete missing resource, access route or implementation dependency. Avoid the standalone status “frozen,” which conflates these distinct meanings. Historical records retain their original wording.

For experiment allocation, do not treat a confidence interval crossing zero as
a universal stop rule; see [ADR-0009](adr/0009-uncertainty-is-not-a-kill-switch.md).

These files are compact specifications, not replacements for research evidence. [General research methods](../research/RESEARCH_METHODS.md) explain the full workflow; [research tree](../research/bandit_tree/generated/INDEX.md) and campaign records index individual hypotheses, experiments and artifacts. Follow raw evidence links before making claims.

[Research pipeline](research-pipeline.md) is the compact portfolio/compute entry
point: diverse independent proposals, explicit combinations, fresh-context renewal,
warm cohorts and scaling from useful ready work and measured costs.

The latest user instruction overrides older specifications and allocations. Record that change and its effective scope. Current-state entries are mutable; frozen recipes, results and raw artifacts are immutable. ADRs explain consequential decisions: append a new ADR when changing one, mark the old decision superseded and link its successor. Do not silently change an experiment already admitted.

Maintain this folder after a meaningful change to objective, evidence basis, allocation policy, scientific recipe or execution ownership. Update current state after a result or failure changes the next action. Do not fill the main agent's context with the entire historical archive.
