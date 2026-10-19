# Research objective

Maximize long-term expected competition performance of a valid, deterministic submission. The active objective is top-1 and public score above 0.7000. Pursue this as an achievable research target; current results and missing forecasts do not cap the search or justify pruning a promising family. Report public-score expectations only when supported by appropriate evidence. Useful incremental gains, stronger teachers, complementary models, new representations and better measurement all count as progress when their evidence supports that claim.

Select and form submissions from our research and valid local evaluation evidence. Public leaderboard feedback confirms transfer and is not a prerequisite for packaging a locally supported candidate. Explain any omission of a strong locally evaluated run with a concrete technical or evidence reason. See [ADR0030](adr/0030-local-evidence-selects-submissions.md).

Do not optimize for the easiest short-term task, number of ideas, commits, agent agreement or infrastructure completeness. Prefer a bounded experiment that resolves a useful uncertainty or produces a competitive model. Necessary correctness and resource checks protect this objective; unrelated checks must not become prerequisites for every arm.

Preserve exploration alongside deepening a promising lineage. An individually weaker model can still supply useful residuals, diversity, auxiliary targets or a better combination. Measure its contribution to the declared combination. A negative result closes the tested recipe at that evidence level, not every descendant or interaction.

Requirements:

- Future idea screens initially budget4–6 total new fits with explicit matched cells, reused controls and teacher/student costs; no mandatory30-per-arm replication matrix. See [ADR0043](adr/0043-fast-screens-and-transformer-teacher-scale.md). Large Transformer teachers and GRU transfer are the latest strategic priority; scale remains an empirical choice rather than a memory-filling goal.
- Reuse official team code and weights, including teachers and initializers, under [ADR0035](adr/0035-official-team-code-and-weight-sharing.md). Record source and fitting ancestry; shared assets do not count as independent replications.
- Use permitted supplied data and declared training membership. Keep final all-data training separate from held-out evaluation claims.
- Preserve official metric and submission requirements. Training-target clipping is an experimental choice; official scoring clipping stays fixed.
- Separate competition rules from submission engineering limits. Data/weight provenance and official evaluation rules remain binding. During research, model size, architecture, context, precision, ensemble size and inference cost may exceed the final submission envelope; test whether the resulting signal can become a compliant student or blend.
- State evaluation exposure and uncertainty. Local deltas are not public-score forecasts; different agents or repartitioning do not create fresh data.
- Preserve all original notes, source snapshots, failed attempts, plots and results. New summaries must link evidence, not replace it.
- Honor current user resource, provider and model constraints. ADR0033 records the latest explicit Sol 6.1/xhigh exception for analysis roles in the complete-blend task; implementation and independent artifact review remain Luna/max, with at most three subordinate roles total. No new paid authority or Astra task is added. Record actual external model access and fallbacks honestly.

Research is authorized to run autonomously within the recorded allocation. Diagnose failures and prepare corrected attempts without repeatedly asking the user for routine implementation choices. An explicit stop closes new admission and drains healthy jobs; an explicit resume supersedes that stop while preserving its record.
