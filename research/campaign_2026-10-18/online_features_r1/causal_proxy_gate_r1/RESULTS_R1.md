---
kind: result
status: inconclusive
created_utc: 2026-10-18T01:32:00Z
updated_utc: 2026-10-18T01:32:00Z
hypothesis_ids: [H-ONLINE-DELAYED-PROXY-20261018-R1]
experiment_ids: [E-ONLINE-CAUSAL-PROXY-GATE-20261018-R1]
evidence_paths:
  - recipe.json
  - grade_r1.json
  - execution_receipt_r1.json
  - RECIPE_AUDIT_R1.md
---

# Causal proxy-gated mixture R1 results

## Result

On the frozen 32 source-qualified, previously exposed F0 components, the
feature-only lagged-label proxy did not add a measurable improvement to the
fixed 75/25 TL3/R3 blend. The best preregistered gate, half-life 8,192, scored
0.6782911 versus 0.6782431 for the fixed blend: paired delta +0.00004795 WP,
conditional component-bootstrap 95% interval [−0.00003098, +0.00012853]. The
2,048-step gate gained +0.00002945 [−0.00004214, +0.00011054], and the 256-step
gate changed score by −0.00004379 [−0.00020917, +0.00013771]. These intervals
condition on one fitted proxy and fixed cached experts; they omit seed, fold,
model-selection, ancestry and public-transfer uncertainty.

TL3 was the stronger member on this matched panel. It scored 0.6806213;
R3 scored 0.6665986; the fixed blend scored 0.6782431. The best gate remained
0.0023302 below TL3, with a conditional component interval [−0.00349015,
−0.00141226]. Thus this test does not support paying the complexity of this
proxy gate, and it gives no evidence that actual target feedback is available.

The online weight did move, so the gate was not stuck at its 25% R3 prior. For
h8192, the median across sequence-level median weights was 0.2435 for target
0 and 0.2367 for target 1; per-sequence extrema spanned 0.1825–0.4880 and
0.1724–0.5000 respectively. This shows that the pseudo-error state changed
mixture weights, but the endpoint barely changed relative to fixed 75/25. It
does not establish whether the pseudo-loss ranks actual expert error well.

## Validity and resource evidence

The prediction builder read no targets or masks. The independent grader
reconstructed all three gate recurrences from the frozen proxy and expert
caches before opening the validation targets/mask; all fixed and gated
predictions matched within the declared tolerance. It then scored rows >=99
using unchanged `global_wp_clip2_v1` and the official scoring mask over 62,018
selected rows. The 2,000-draw paired component bootstrap resampled the 32
exposed components. Its tail fractions are descriptive bootstrap summaries,
not conventional two-sided significance tests.

The 32 components are disjoint from proxy fitting but have repeatedly been
exposed to earlier model/feature selection. TL3 training, teacher and
initializer ancestry remains unknown. The result is a fixed-prediction
development screen, not clean-CV or public-transfer evidence. The actual
target horizon and maturity are still undocumented; the gate uses a learned
pseudo-label for y[t−1], not verified test-time feedback.

The one-sequence profile took 5.02 s and peaked at 36,503,552 bytes. The
32-sequence label-blind build took 80.48 s, peaked at 71,839,744 bytes, and
wrote a 30 MB prediction cache. Independent grading took 15.10 s and peaked at
333,680,640 bytes. Every attempt completed successfully. Each task-local
worker cgroup had zero OOM, OOM-kill, `max` and `high` event deltas. During the
full-build window, the parent-hierarchical `max` counter increased by 7,090;
the parent OOM/OOM-kill counters did not change. This parent-level counter is
not isolated to the gate worker. No GPU was used. Raw prediction payloads
remain in `/home/vlad/development/wunder/research_tmp`.

## Decision

Do not open a new B300 draft for this particular gate: its three timescales
barely change the fixed mixture, and the overall blend loses to TL3 on this
panel. No numerical practical-gain threshold was preregistered, so this is an
opportunity-cost decision for the exact recipe, not a formal falsification.
This narrows the linear-proxy / cached-expert gate treatment; it does not
reject online learning, learned causal state features, Kalman-style features,
student-side adaptation, or mechanisms that use genuine matured feedback if
the evaluation interface permits it. Keep the already-published causal-state
teacher/student draft PR #17 separate; it is not validated by this screen.

The next useful work is to resolve the observable callback and target
maturity contract, then evaluate a learned causal feature adapter at the
trained recurrent endpoint and its matched GRU student. Avoid optimizing this
exposed panel's pseudo-label gate constants: its small conditional deltas do
not justify more local gate tuning.

## Source records

- [Frozen recipe](recipe.json)
- [Recipe/code audit](RECIPE_AUDIT_R1.md)
- [Official grade](grade_r1.json)
- [Execution receipt](execution_receipt_r1.json)
- [Independent evidence review](../../../unified/reviews/CAUSAL_PROXY_GATE_EVIDENCE_REVIEW_20261018_R1.md)
- [Causal lagged-label proxy](../causal_label_proxy_r1/README.md)
- [Online-feature research frontier](../FRONTIER.md)
