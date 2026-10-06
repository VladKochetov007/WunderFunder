# Independent bandit continuation

Contract opened 2026-10-06 19:51:12 UTC. Objective mode: competition. This is the
authoritative resource and allocation contract for this continuation; older
morning cutoffs and smaller/local-versus-cloud caps are historical evidence.

The user authorized experiments, Kaggle GPU/TPU quota, diverse coding agents,
and meaningful commits/pushes. The first allocation window is 24 hours. It is a
review checkpoint, not a hard stop: research continues until the user explicitly
stops it or an actual budget/safety boundary prevents further admission. On an
explicit stop, close new admission and let running results land gracefully.

## Objective and evidence

- Improve the complete deterministic root Challenge 2 submission. The root
  confirmed public best remains 0.6660; the long-term 0.7000 milestone is unmet.
- Start from the heterogeneous teacher/student protocol associated with Maxim's
  user-reported 0.6674 ZIP, submission 26n9pg4y, archived from
  origin/research/creative-iteration-20260930@833aaf54. The remote head was checked
  and still matches. Quant-probes@b7fd5d2f is a separate teacher investigation.
- Independently implement in our trainer. Do not merge or copy the friend's
  implementation. Source notes and exact protocol provenance remain linked in
  the bandit tree and branch audit.
- Primary metric: global_wp_clip2_v1, true scored/prediction mask, pooled clipped
  targets weighted by their original clipped magnitude, averaged over t0/t1.
- Public transfer is verification tier C. Every previously inspected local fold
  remains exposed. Group-excluding fitting is still required; new splits or
  agents do not restore unexposed confirmation. No local-delta public forecast.
- Freeze candidate/control, evaluator, splits, seed pairs, checkpoint selection,
  and serving blend before fitting. Report every attempted fit, including errors.

## Resources and admission

- LOCAL AGGREGATE: 10 logical CPU cores, 20,000,000,000 bytes RAM,
  5,000,000,000 bytes observed VRAM. One local GPU worker. All campaign workers,
  model CLIs, EDA kernels and monitoring helpers share the aggregate accounting.
- Use a verified systemd/cgroup parent with CPUQuota=1000%, MemoryMax=20GB,
  MemoryHigh=18GB and MemorySwapMax=0. Do not reuse stale 8-core/24GiB or
  30GiB-per-job defaults. Check effective parent and child properties.
- Start PyTorch allocator admission at 4.3GB, observed-device ceiling 5GB;
  profile representative forward/backward/export before full fits. Start loader
  workers at zero; stream/memory-map caches and export one teacher at a time.
- Admit only when measured current use plus the profiled job reserve fits the
  18GB working allowance. Preserve laptop headroom and record parent OOM counters.
- Keep new local artifacts within 30GB and at least 25GB disk free. Reference
  existing raw datasets instead of copying them into worktrees.
- KAGGLE has separate headroom. Initial live quota: 19.71 GPU hours and 19.96 TPU
  hours remaining, refresh 2026-10-10 00:00 UTC. Recheck before each cohort and
  after terminal jobs; record actual allocated hardware and provider quota delta.
- Use private vladkochetov07/wunder-funder and verified private source/assets
  datasets. Reserve 20% of usable GPU budget for paired follow-up and serving
  validation, 20% for different mechanism families, 60% for the leading lineage.
  Provider failures and all retries count against budget. Do not rent paid compute.
- TPU starts with the real-device E463 compatibility/numerical/full-shape gate.
  An ERROR with empty logs is not evidence that training ran. Use short preflight
  stages before any production allocation; preserve artifacts at each stage.
- Poll local health about every 5 seconds and cloud status every 5 minutes.
  Persist heartbeat, phase, exit status, RAM/VRAM peaks, OOM counters and quota.
  A transient monitoring timeout is an observation error, not proof a job died.
- At most one infrastructure retry after a terminal, diagnosed provider failure
  with no scientific worker started. OOM or code errors require a revised,
  reviewed attempt and new ID; never blindly restart the same packet.
- Emergency resource violations may stop the offending job. A normal user stop
  drains jobs rather than terminating healthy work. Failed/dormant jobs remain
  in the archive and never automatically reopen old queues.

## First scientific allocation

Begin with fold-specific GRU parents and audited teacher caches. Existing
epoch-3 snapshots are candidates, not admission evidence until their dimensions,
training membership, split hash and producer checkpoint hash are checked.

The leading mechanism is a warm GRU student learning from four causal teacher
families and a training-only family head. Source coefficients sum to 0.8125;
source caches are not renormalized. Our existing multi-teacher loader divides
by teacher count, so supply one explicitly materialized two-column main cache.
Add an independently validated eight-column family cache and auxiliary head,
discarded at serving. Retain existing trainer behavior when the option is off.

Predeclare paired student seeds 3/4, discovery fold 0, equal-weight served pair.
Use our group-excluding epoch-3 parents. Continue for two epochs at lr=0.0001,
chunk=2000, distillation alpha=0.7. The source loops over range(epochs), so ep1
is the second additional epoch: use fixed zero-based checkpoint epoch1 and no
evaluation during fitting. Match batch/update budget across
every comparison; a resource fallback changes both sides as a new recipe.

Initial fixed comparisons:

| ID | Recipe | Main comparison |
| --- | --- | --- |
| C0 | Ordinary warm supervised continuation | Training-budget control |
| C1 | Homogeneous scalar teacher, total coefficient 0.8125 | Distillation control |
| C2 | Heterogeneous scalar teacher, total coefficient 0.8125 | Family diversity versus C1 |
| C3 | C2 plus four-family auxiliary head, coefficient 0.1 | Family representation versus C2 |
| C4 | C3 with normalized main teacher coefficients | Target scale versus C3 |

These are ten paired-seed discovery fits, admitted in small matched blocks after
profiling. No Cartesian hyperparameter sweep. Reuse only provenance-compatible
teachers; regenerate missing families on the declared fitting groups. An
uncertified organizer checkpoint cannot establish an out-of-fit comparison.

The next allocation compares the selected recipe with its matched control on
fold 1 and both paired seeds; this is exposed-data robustness, not fresh external
confirmation. Predeclare at most two additional family-distinct exploration
recipes (each with a matched control) and one deeper child in this first window.
Count all revised/failed recipes. Cap the first window at 24 scientific student
fits; teacher fits/profiles are separately registered and budgeted. Admit fewer
when measured cost requires it; retain the unrun slate in the frontier.

Candidate children: one teacher-family replacement using causal two-stream/event
context; family contrasts/residual-basis transfer; function-preserving widening
with output parity before continuation; a fixed blend with our champion. Each
has its own comparison. Do not sum parent deltas or prune by pessimism alone.

## Agents, literature and bandit memory

Native subagents use gpt-6-luna/max. No Sol or Astra. Builders work in isolated
worktrees; a different critic checks the contract/implementation, and immutable
deterministic checks grade results. The controller owns admission and synthesis.
At most three concurrent subordinate roles and one session per provider profile.

Use OpenCode's verified free Muse Spark/MiMo catalog and Freebuff's GLM option
after a bounded access check. Kimi is eligible if an existing authorized/free
provider becomes available; no new subscription or paid API spend is authorized.
Record actual provider/model, including reported fallback, tool/version, prompt,
context packet, source passage and output. A catalog entry is not a successful
model call. Freebuff currently exposes an interactive CLI; do not invent headless
flags. Keep ideation context read-only and protect grader/budget files.

Each broad round uses at least four distinct creativity lenses, with two random
and two directed literature assignments from the existing Trading library or
primary sources. Save the selection seed and exact source sections. Ask for at
least four distinct, testable ideas including an unconventional one, and insist
on trying small transferable components rather than only a paper's headline.
Do not fabricate a useful connection. Initial proposals stay blind until saved.

New arms may have null numeric beliefs/costs. Start with explicit qualitative
priorities; populate measured cost and reviewed, comparator-specific effects
after experiments. Use the existing knowledge-gradient ranking only for compatible
reviewed beliefs, separately from manual priorities. Unknown public gain stays
unknown. Allocate by mechanism lineage; deepen promising children, preserve
exploration and reproduction reserves, and create explicit interaction nodes.

After two uninformative completed allocations on one recipe, move it out of the
active context, retain all raw evidence, and commission a fresh-context reframe.
New representation or contradictory evidence can reopen it. An infrastructure
failure is resource-censored evidence, not a scientific rejection.

## Validation, artifacts and handoff

Before a full fit verify cache row/group alignment, training membership, masks,
original-label weighting, clipping order, coefficient sums, finite values,
disabled-option parity, and checkpoint round-trip/serving-head removal. Test
monitoring timeout recovery, abnormal worker exit, aggregate admission and
graceful stop with cheap processes before trusting an unattended queue.

Evaluate the fixed served pair and predeclared matched controls. Log per-seed
deltas and chain-bootstrap uncertainty separately. A positive exploratory result
earns additional allocation; it is not a significance or public-transfer claim.
Freeze final candidates before independent implementation/serving checks.

Use marimo-pair for human-readable EDA; save standalone plots in each lineage's
plots/ directory and link them from notes. Useful first plots: teacher residual
correlations, disagreement versus error, per-chain deltas, temporal/class slices,
learning curves, and memory/runtime traces. Do not let reactive EDA trigger fits.

Append hypothesis, experiment and bandit journal records; preserve old bytes,
failed attempts and raw logs; rebuild the index after meaningful results. Keep a
compact live frontier and immutable detailed artifacts. Commit/push meaningful
steps: contract/admission, trainer and tests, sealed experiment results, candidate
package. Preserve friend branches and unrelated work.

Final submission candidates require fixed members/weights, Python3.11 offline
streaming/ABI checks, ZIP<=20MB, one-vCPU/16GB/60min compatibility and delivered-ZIP
hash binding. Public upload remains the existing human gate. Report actual public
results only when supplied/verified, never inferred from local improvements.
