# Research objective

Maximize long-term expected competition performance of a valid, deterministic submission. The user's ambition is to win, with 0.7000 public score as a long-term milestone. This is an objective, not a forecast. Useful incremental gains, stronger teachers, complementary models, new representations and better measurement all count as progress when their evidence supports that claim.

Do not optimize for the easiest short-term task, number of ideas, commits, agent agreement or infrastructure completeness. Prefer a bounded experiment that resolves a useful uncertainty or produces a competitive model. Necessary correctness and resource checks protect this objective; unrelated checks must not become prerequisites for every arm.

Preserve exploration alongside deepening a promising lineage. An individually weaker model can still supply useful residuals, diversity, auxiliary targets or a better combination. Measure its contribution to the declared combination. A negative result closes the tested recipe at that evidence level, not every descendant or interaction.

Requirements:

- Train our model weights independently. The user confirms hosts approved the team merge; authorized recipe/code inspection does not imply reusing teammate weights.
- Use permitted supplied data and declared training membership. Keep final all-data training separate from held-out evaluation claims.
- Preserve official metric and submission requirements. Training-target clipping is an experimental choice; official scoring clipping stays fixed.
- State evaluation exposure and uncertainty. Local deltas are not public-score forecasts; different agents or repartitioning do not create fresh data.
- Preserve all original notes, source snapshots, failed attempts, plots and results. New summaries must link evidence, not replace it.
- Honor current user resource, provider and model constraints. Native subagents currently use gpt-6-luna with max reasoning; no Sol or Astra. Record actual external model access and fallbacks honestly.

Research is authorized to run autonomously within the recorded allocation. Diagnose failures and prepare corrected attempts without repeatedly asking the user for routine implementation choices. An explicit stop closes new admission and drains healthy jobs; an explicit resume supersedes that stop while preserving its record.
