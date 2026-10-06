# Independent teacher research: recent-branch correction and agent brief

> Refresh update: the user fetched the remote successfully. `origin/quant-probes`
> is now verified at `b7fd5d2fff8dbf6645f34de9835c4f634caa8273`.
> The access failures described below are historical. Read the
> [post-fetch audit](../campaign_2026-10-06/branch_teacher_audit/post_fetch_b7fd5d2f/OBSERVATIONS.md)
> before using the brief. The completed attention recipe was roughly tied with
> GRU256 on one fold; no large public Transformer gain is documented in that note.

The user clarified that the friend's relevant work was during approximately the
last two days. Current cached refs do not expose a teacher branch with October4–6
commits: `origin/quant-probes` ends September30 and `origin/archive/2026-10-01`
ends October1. The earlier audit remains evidence about those cached revisions,
not an identification of the friend's latest result. The user identified `quant-probes` as the probable branch and requested a
refresh. `git fetch origin quant-probes` failed because `.git/FETCH_HEAD` is
read-only; a read-only remote check failed DNS resolution for `github.com`, and
the web fallback failed. The latest commit and actual result remain unverified. No friend code has been merged or copied.

## Brief to give the next agent

Improve the current deployable Challenge2 endpoint, confirmed public best0.6660.
Treat the friend's reported teacher gain as motivation; first identify its recipe
and comparator from his actual recent branch, then reproduce the claimed effect.
Separate teacher quality, transfer quality and final student/ensemble quality.
Do not infer a public improvement from a stronger teacher alone.

Work independently on the `arm:causal-transformer-teacher` lineage and its children.
Develop your own implementation from an explicit specification. Keep the existing
student endpoint as a matched control. Test progressively:

1. Capacity and context: compare a causal attention teacher with a GRU teacher
   under declared comparable training budgets, and separately allow the teacher
   extra offline capacity. Record which comparison answers which question.
2. Transfer: compare equal student architectures trained from the two teacher
   sources, with matched seeds, labels, splits, objective and inference recipe.
   Cross-fit teacher predictions by continuous sequence chain; prior inspected
   folds remain exposed regardless of cross-fitting.
3. Beyond ordinary distillation: test teacher residual targets, confidence-weighted
   or regime-specific supervision, temporal-state transfer, and original-label
   reanchoring. Change one factor first, then explicit interaction tests.
4. Privileged teaching: explore future/origin-target information only in offline
   teachers; the served student remains causal. This is a separate mechanism arm.
5. Complementarity: export aligned predictions for the common endpoint and inspect
   error covariance, quiet/class3 behavior and context dependence against our
   champion. A higher standalone proxy need not produce the better blend.
6. Blends: compare champion, independent student, fixed-weight mixture, and a
   separately justified causal gate. Select blend weights on the declared exposed
   selection slice and evaluate the frozen blend on the remaining declared slices.
   Record all inspected choices. Do not add component gains; measure the complete
   serving endpoint, runtime, memory and quantization together.

Before each predictive test, save an arm card with parent, precise change,
mechanism, cheapest discriminating experiment, falsifier, comparator, serving
endpoint, metric, exposure, paired seeds and measured resource profile. Preserve
failed attempts, prediction artifacts, notebook/EDA figures and raw outputs.
Append experiment records and decision events, then rebuild the tree.

Use a genuinely different library source or creativity lens for a blind ideation
round. Save the task prompt and source passages before sharing proposals. The
builder, critic and reproducer should have distinct roles and documented context;
rotate models/tools later only within the user's explicit selections. Current
allowed setting remains Luna/max. Agent agreement is not experimental evidence.

Current aggregate limits:5CPU cores,3decimalGBVRAM,40GBRAM. No new local
experiments after October6 10:00Kyiv. New experiments require Kaggle capacity,
resource admission and an independent packet check. This brief launches nothing.

## Comparing the two research streams

Keep our lineages and the friend's result provenance separate. Share committed
arm cards and verified evidence after independent exploration. Maintain a table of
matched baseline, standalone student, blend, paired uncertainty, serving cost and
actual public result. A proposed0.7000 milestone is not a predicted score.
The competition is between tested endpoints; independent failures remain useful
branches and can motivate different children.

## New committed child protocols after the refresh

- `arm:teacher-convergence-context`: change convergence and event/context
  construction separately rather than repeat the same four-epoch attention run.
- `arm:learnable-privileged-transfer`: project privileged teacher information
  onto causal inputs and transfer learnable residuals/moments, with cross-fitting.
- `arm:complementarity-gated-blend`: target complete-endpoint complementarity;
  compare a frozen mixture against a separately selected causal gate.

These remain untested proposals with unknown expected gains and costs. The
branch's single-recipe negative conclusion is preserved as evidence, not adopted
as a veto on these children. Do not treat oracle future-label recalibration or
0.71328 validation as a public-score result.

## Leading direction after inspecting the other refreshed branch

The stronger recorded public result belongs to heterogeneous teacher compression
into a GRU256 student pair on the creative-iteration branch, not the completed
Transformer recipe. Use the [heterogeneous-compression brief](HETEROGENEOUS_AGENT_BRIEF.md)
as the first builder task; retain attention and privileged-transfer arms as separate
children, not discarded families. External recorded public0.6674 does not automatically
change our own confirmed best0.6660.
