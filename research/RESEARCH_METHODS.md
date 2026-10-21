# Long-horizon creative research with diverse agents and a bandit tree

This document describes a reusable research process. It is independent of any
particular competition, model architecture, dataset, or evaluation measure. Its
purpose is to sustain useful exploration, deepen promising directions, and escape
loops caused by stale assumptions or an unproductive task framing.

Project-specific prospective fit counts follow
[ADR0043](../specs/adr/0043-fast-screens-and-transformer-teacher-scale.md):4–6 total
new fits per initial idea comparison, explicit matched cells and teacher/student
costs, followed by targeted rather than quota-driven confirmation. Historical
frozen protocols remain preserved. This small-screen budget does not establish
significance or relax causal/ancestry/evaluator requirements.

The operating cycle is:

**frame → generate → challenge → probe → implement → evaluate → reproduce → update → reframe**.

Research should produce inspectable artifacts and evidence, not just persuasive
conversations. Ambitious ideas deserve a way to be tried. Uncertainty should guide
the next experiment rather than become a reason to dismiss a whole direction.

The user's [plateau guideline](unified/methods/AUTORESEARCH_PLATEAU_GUIDELINE_20261019_R1.md)
is preserved verbatim with its source hash. Apply its useful questions about
operator diversity, measurement, memory and screen fidelity; distinguish its
external claims from verified evidence. The current concrete allocation is the
[October 20 local/Kaggle plan](../specs/plans/2026-10-20-plateau-review-local-kaggle.md).

## A common contract, several independent ways of thinking

Give every research stream a compact common contract: the actual objective,
allowed information, deliverable, evaluation rules, resource limits, known
observations, and requirements for reproducibility. Distinguish requirements
from hypotheses about how the problem works. An inherited explanation must not
quietly become part of the specification.

Use different LLMs, model families, coding agents, and tools for diversity in
representation, implementation, and criticism. Rotate their roles across research
rounds. A model that generates a candidate in one branch can critique another;
another coding tool can independently implement the same minimal specification.
Honor the user's model selections and exclusions. Model rotation is a deliberate
research choice, not permission to silently substitute providers.

Changing a model name alone does not ensure independent thought. Vary source
material, problem framing, allowed context, implementation strategy, and the kind
of failure an agent is asked to look for. Several agents with the same history
and prompt can reproduce the same blind spot. Their agreement is not evidence
that a mechanism works.

Separate responsibilities:

- A controller maintains the contract, resources, research tree and allocation.
- Ideators commit hypotheses before seeing competing proposals.
- Builders turn selected hypotheses into bounded, reproducible implementations.
- Critics predict concrete failure modes and turn them into executable checks.
- Evaluators measure candidate behavior against the declared comparison.
- Reproducers independently reconstruct or replay a claim, stating what their
  verification actually establishes.

An agent should not be the sole judge of its own candidate. Give each role the
context it needs rather than the entire campaign conversation. Share conclusions
and evidence after independent proposals are committed.

## Literature as a source of transferable pieces

Use both directed and random literature selection. Directed reading follows a
specific bottleneck or neighboring problem. Random reading samples the user's
library or a recorded source pool to introduce concepts that the current framing
would not normally retrieve. Mix familiar and distant domains. Record the actual
source, sampled section, selection procedure, and random seed when applicable.

Assign different articles, books, chapters, or passages to different subagents.
Do not give every agent the same paper and ask the same broad question. A useful
assignment combines a concise task description, a source, a creativity lens and
an explicit requested artifact.

The central instruction is **extract something that might transfer**. The paper's
main contribution may be unusable while a small detail is valuable: a sampling
procedure, decomposition, state representation, normalization, approximation,
constraint, proof trick, diagnostic, optimization schedule, or way to construct
a control. Do not ask only whether the whole paper can be applied. Ask which
part could alter an assumption, intervention, representation, or experiment.

Require a real attempt at idea generation, including unusual possibilities.
When the obvious application fails, ask for a smaller transferable component or
an indirect analogy. An agent should explain which parts transfer, which do not,
and how to test the proposed connection. It can report that no defensible
connection was found after attempting alternatives; it should not manufacture
evidence or force an unsafe or impossible implementation to satisfy the prompt.

A literature ideation prompt can be:

> Read the task contract and the assigned source. Generate several distinct
> hypotheses, including unconventional ones. Look for small transferable parts,
> not just the headline method. For each proposal, name the source concept,
> explain the task analogy, specify the intervention, predict an observable
> consequence, give the cheapest discriminating experiment and a falsifier.
> Try to find at least one usable component before concluding the source has
> nothing to offer. Commit your proposals before reading other agents' ideas.
> Do not train models or launch expensive jobs in this ideation task.

Keep source-grounded ideas separate from unsupported speculation. A surprising
analogy earns an experiment, not immediate acceptance. Preserve unsuccessful
ideation jobs as well as successful ones so future agents can see what was tried.

## Turn proposals into a research tree

Organize research as an extendable tree:

**objective → mechanism family → direction → specific hypothesis → experiment**.

A branch can deepen whenever a child makes a concrete change. A failed parent
recipe is not a ceiling for its descendants. A better representation, revised
training protocol, different information pathway, or corrected implementation
can justify a new child with better prospects. State exactly what it changes.

Use stable identifiers and retain parent relationships. Use cross-links for
shared evidence, dependencies and interactions between branches. A combination
can have a navigation parent while linking all contributing directions; do not
pretend that its scientific ancestry is a simple chain. Give combinations their
own controls and complete-endpoint evaluation.

An arm card should contain:

- The question, mechanism and expected observable consequence.
- Parent direction, related arms, source material and precise change.
- Current status and claim scope: proposed, screened, incomplete, supported,
  reproduced, or contradicted for a specified recipe.
- Baseline or comparator, cheapest useful next test and falsifier.
- Existing supporting and contradicting evidence, including provenance.
- Expected value and uncertainty, which can remain unknown or qualitative.
- Estimated resource cost, prerequisites and stopping conditions.
- Agent, model, tool, prompt and context provenance.

Split a broad idea when it hides distinct mechanisms. Merge duplicate proposals
as navigation while preserving their original source cards and evidence. Add a
new family when a direction does not fit the existing taxonomy. Do not force
novel ideas into an old category simply because the index already has that label.

The tree represents an evolving search. It should be possible to add siblings,
deepen a branch, reopen an archived direction, or register an interaction without
rewriting the history or changing the orchestration code for ordinary additions.

## Use bandit allocation to choose where to learn next

Treat mechanism lineages as arms competing for limited research attention and
compute. Allocate using a combination of plausible benefit, uncertainty,
information gained, experiment cost, reusable artifacts, and evidence quality.
A cheap experiment that resolves an important ambiguity can be more valuable
than another expensive attempt on the latest fashionable branch.

Preserve unknown beliefs as unknown. If numerical beliefs are later used, record
their units, comparator, origin and review status. Subjective priors are planning
assumptions, not experimental findings. Old priors require reconsideration when
the task, baseline, evaluator or training protocol changes.

Balance exploitation and exploration. Deepen promising lineages while keeping
room for independent families and unusual proposals. Allocate across families
as well as within them, so one busy branch cannot consume the queue merely
because it has the most recent notes. A promising child may deserve extended
work; diversity does not require abandoning depth after every experiment.

Do not add parent, child, or sibling improvements together. Shared models,
examples and evaluation choices make observations dependent. A combination must
be evaluated as its own endpoint. Agent diversity also does not create new,
statistically independent evaluation data.

Bandit output nominates the next research decision or preparation task. It does
not bypass resource admission, authorize a launch, or replace scientific review.
Record why an arm was chosen, deferred, reopened, split, or archived. Update
allocation from measured evidence rather than from how confidently an agent
writes about its idea.

### Uncertainty informs the next experiment; it does not automatically reject an arm

A confidence interval that crosses zero is an uncertainty description, not a
universal hypothesis test or bandit kill rule. In a small or costly study, a
promising mechanism may remain unresolved even when its useful effect is real.
Do not relabel “not resolved” as “no effect.” Record the point estimate,
interval width, evaluation-unit count, exposure, training variation and what
mechanism was actually tested.

Keep three decisions separate. First, validity and resource safety are hard
gates: the recipe, data identity, split, serving semantics and OOM protections
must pass. Second, interpret scientific evidence at the level of the exact
recipe and its mechanism. Third, allocate using expected long-term value,
information gain, cost, reversibility, reusable artifacts and opportunity
cost. A positive but imprecise observation can justify a cheap independent or
discriminating test; it does not justify a public claim. A repeated uninformative
test can be deferred for low value-of-information without declaring its whole
lineage false.

Keep an exploration reserve across distinct families so uncertainty in the
current leader does not consume the entire research budget. Stop an exact recipe
when relevant evidence or the cost of another attempt warrants it; preserve
siblings and materially different descendants unless their mechanism was
tested. Never count new seeds, agents or partitions of exposed data as fresh
evaluation evidence. See [ADR-0009](../specs/adr/0009-uncertainty-is-not-a-kill-switch.md)
for the decision record.

### A scalar bandit reward is not the value of every research component

The older `scripts/bandit.py` knowledge-gradient calculation assumes arms have
standalone gain against “adopt nothing” and that adopted gains add. That is a
useful approximation only for compatible, reviewed system-level arms. It is not
the current universal allocator and must not be used to discard a weak
standalone model, teacher, feature or representation whose proposed value is
conditional on another model or training recipe. Record that role in the tree
and create an explicit interaction node with matched controls. Estimate its
value at the complete served endpoint while also tracking the component's
conditional contribution and information value. A component's negative
standalone score is evidence about standalone use, not proof of zero value as a
target, residual source or expert.

For allocation, ask what the next action can teach across the lineage: a cheap
ancestry/representation check, a conditional-signal screen, a matched fit, a
replication, or a serving comparison. Do not turn a confidence interval into a
universal acquisition function. Use numerical knowledge-gradient ranking only
when the reward scope, compatible arm structure, current priors and cost model
match its assumptions; otherwise use the curated tree, qualitative evidence,
explicit exploration reserve and recorded controller judgment.

## Escape loops by changing the context, not deleting the evidence

A research loop is suspect when agents repeatedly rename the same intervention,
retune around an unsupported explanation, cite the same weak evidence, or inherit
a framing that makes alternatives invisible. Repeated effort is justified when
it answers a new question; repetition without a discriminating change has low
expected value.

Distinguish these cases:

- An implementation failed, but the mechanism was not adequately tested.
- A specific recipe was tested and failed its spending gate.
- A mechanism received relevant counterevidence.
- A task framing itself is unsupported or misleading.
- An expensive test could not resolve the uncertainty within available resources.

These cases imply different next steps. None automatically proves that every
related idea is impossible. Avoid rejecting proposals solely through pessimism
or a previous recipe's outcome. First identify what the proposed child changes
and whether a bounded test can learn anything new.

Sometimes the best move is to forget an idea **in the working context**. Archive
its card, evidence and status, remove it from the active brief, and start a fresh
subagent from the task specification and necessary observations. Do not supply
the discarded explanation, the whole failed conversation, or the controller's
preferred answer. Long-lived creative agents should be able to work from this
clean brief and build their own representation of the problem.

A context reset must retain hard constraints, evaluator definitions, known
information leakage, data exposure, resource limits and decisive facts. It must
not hide counterevidence needed to interpret a claim. One useful sequence is to
let an ideator first commit a hypothesis from a specification and minimal facts,
then give a critic the relevant archived failures and ask whether this is a real
new mechanism or an old failure in different language.

Maintain an explicit distinction between archived history and active context.
The archive is complete; the prompt is selective. Record the reset, the excluded
assumptions, the facts retained and the reason. Reintroduce an archived idea only
when new evidence or a materially changed child makes it relevant. Forgetting
for exploration is not erasing provenance or declaring a direction permanently
dead.

A clean-context assignment can be:

> Work from this task contract and these necessary observations. Build your own
> framing and several distinct solution paths. Commit hypotheses before reading
> prior proposed explanations. You may pursue a promising path over several
> bounded iterations. Preserve all attempts and explain what each test can
> distinguish. A separate reviewer will later compare your committed proposals
> with archived evidence and check for rediscovery, leakage and invalid claims.

## Run sustained research through bounded experiments

Give a long-running agent a clear deliverable, allowed tools, mutation scope,
resource envelope, reporting cadence and conditions for continuing or stopping.
Use bounded iterations rather than an unlimited conversational swarm. A long
horizon allows deeper implementation and unexpected findings; it does not remove
accountability for spending.

For each predictive or empirical claim, declare the comparison before inspecting
its result. Keep evaluation sampling, implementation correctness, training
variation, adaptive selection and external transfer separate. Repartitioning
previously inspected data, changing an agent, or rerunning fixed predictions
does not restore an untouched evaluation set. If fresh evidence is unavailable,
label the work exploratory and describe what future evaluation could establish.

Use a cheap probe when it can meaningfully discriminate the idea. Confirm a
promising probe at the relevant scale and endpoint. Do not let an uninformative
toy failure veto a whole family, or a toy win become proof of a production gain.
Write down the full tried set and keep failed gates and revised protocols.

Keep the evaluator and permission layer outside the actor's control. Critics
should convert concerns into checks. Reproducers should state whether they
verified arithmetic, implementation dependence, training robustness, or transfer;
these are different accomplishments.

When sharing results between streams, compare declared endpoints and conditions.
Use disagreements to identify missing controls. Blend compatible ideas only
through explicit interaction experiments. Competition between streams can
motivate stronger candidates without becoming a contest of agent confidence.

## Resources, operational reliability and human-readable artifacts

Treat CPU, RAM and accelerator memory as aggregate budgets across all active
jobs. Local and cloud workers can have separate documented headroom, while each
still obeys its applicable cap. Check available capacity and provider quotas
before admission. Profile conservatively, serialize heavy work when needed, and
respect time cutoffs for new runs.

Monitor durable jobs for memory pressure, process exits, provider errors and
missing outputs. Preserve interrupted attempts and diagnose the cause before
rerunning. Distinguish a crash, timeout, cancellation, budget failure and successful
completion. A planned or admitted job is not evidence of a completed experiment.
Do not start duplicate workers because a monitor temporarily lost visibility.

Store code, commands, configurations, source identities, environment, seeds,
raw outputs and receipts alongside the experiment. Keep original evidence
append-only or immutable. Store EDA images and plots in that research directory's
subdirectories, with captions and source/context information. Inspect plots as
part of reasoning rather than producing them only for presentation.

Use inspectable notebooks, including reactive notebooks such as marimo when
appropriate, to connect data exploration, code and saved figures. Preserve a
runnable artifact rather than relying on a transient kernel or conversation.
Summaries should help a human or future agent find the actual evidence.

Maintain compact Markdown/JSON notes, an experiment ledger, a decision journal,
an index and the tree. Preserve repeated identifiers as separate attempt records.
A new summary supersedes interpretation, not the original source. Archive
configuration versions and flag missing, ambiguous, stale or conflicting sources.
Do not silently replace historical priors or treat an old status as live truth.

## A practical cadence

Start each round by checking the active frontier and identifying what remains
uncertain. Allocate a mixture of deeper work, independent directions and
representation changes. Assign bounded agents with distinct briefs and sources.
Collect committed cards, run the smallest informative tests, review the evidence,
and update the tree and journal.

At a plateau, explicitly audit the framing and context. Ask what would be tried
if the dominant idea had never been proposed. Run a clean-context stream and a
contrarian or neighboring-domain stream. Let them work long enough to construct
real alternatives, then compare their evidence with the archive. Resume depth
where a new intervention creates useful expected value.

The controller should always be able to explain what is being tried, why it was
allocated, what observation would change the next decision, and where the raw
artifacts are stored. The result is an extendable research process with a complete
memory and a deliberately small, revisable working context.
