---
kind: method
status: proposed
created_utc: 2026-10-14T01:16:01Z
updated_utc: 2026-10-14T01:16:01Z
hypothesis_ids: [H-STUDENT-LADDER-20261014, H-STATE-INIT-20261014]
experiment_ids: []
evidence_paths: [THEORY_ALGEBRA_CHECK_R1.json, recipes/README.md]
---

# Distillation capacity, useful diversity and cheap recurrent memory

## Question or claim

Which conclusions can the first architecture screen support, and which mechanisms can improve its successors while competition CPU probes wait? This supplements the [runtime and initializer theory](THEORY_AND_DISCRIMINATING_TESTS_R1.md). Derivations below are ours; literature observations are explicitly attributed. No new fitted-quality result is established.

## Mechanism and method

### A larger teacher does not settle the student architecture

The first screen fixes one clean teacher cohort and 14,920 updates. Its result will measure each architecture's usefulness under that finite recipe. It will not measure its best achievable accuracy. A wide model can lose because optimization is slower, or win because its optimization is easier, even at equal exposure. Training loss, teacher imitation error and held-out official score are different observables.

Stanton et al. report that students can fail to match teachers despite sufficient representational capacity, identify optimization difficulties, and find that closer imitation need not improve generalization. This is evidence against using imitation loss as the sole selection criterion, not a theorem about our regression task. [Does Knowledge Distillation Really Work?, NeurIPS 2021, abstract](https://arxiv.org/abs/2106.05945v2).

Mirzadeh et al. find that a larger teacher can yield a worse distilled student in their image-classification experiments and propose an intermediate teacher assistant. This motivates checking student benefit when improving our teacher cohort; it does not establish a universal size threshold or justify discarding stronger teachers. [Improved Knowledge Distillation via Teacher Assistant, AAAI 2020](https://arxiv.org/abs/1902.03393).

For the current actual trainer, each chunk uses clipped-label magnitude weights and target `0.1*clip(y,2) + 0.9*clip(T,2)`, where stored `T=float32(0.8125*mean3)`. The teacher's stored amplitude therefore affects target clipping and fit dynamics. Identical metadata labels cannot substitute for this arithmetic. Preserve this target in the first screen; teacher amplitude or assistant targets are separate matched arms.

Existing task evidence also argues against treating teacher score as a student ceiling: the [October12 clean teacher grade](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_local_equal_trio_grade_r1/F0_EQUAL_TEACHER_TRIO_GRADE_R1.json) gives the equal H384 trio .680902 on exposedF0, while a separately trained historical oldz4 student scores .683142. Their recipes and inference precisions differ, so this is neither a matched causal estimate nor a new winner. Keep the strong historical run visible when interpreting the new fresh-weight screen.

Next evidence: retain a fixed terminal checkpoint, record finite update counts and learning curves, and advance an uncertain candidate if its observed improvement trajectory warrants more exposure. Longer fits use a new matched schedule, rather than calling an extra tail of the one-epoch OneCycle schedule a faithful continuation. When a stronger teacher version is available, redistill the two leading architectures with unchanged student recipes. Only then can teacher-by-student interactions be estimated.

### Why an individually weaker student can improve an ensemble

For one target, let all means/covariances below use the fixed official weights `w=abs(clip(y,2))` over scored rows. Set `p_e=(1-e)p+e q`. Away from prediction clipping boundaries, let `P=clip(p,2)` and `d=(q-p)*1[abs(p)<2]`. Differentiating weighted correlation gives

`d rho(y,clip(p_e,2))/de |0 = [Cov(y,d) - Cov(y,P)*Cov(P,d)/Var(P)] / sqrt(Var(y)*Var(P))`.

The new member helps locally when its change predicts the residual left after projecting the label onto the current endpoint. Its individual score does not determine this sign. For the competition's mean of two target correlations, take the mean of the two derivatives with their own weights. At exactly ±2 use a directional derivative or direct finite differences; if variance is zero this expression does not apply. Finite mixtures can cross clipping boundaries, so the derivative cannot replace grading the whole endpoint.

The [synthetic algebra check](THEORY_ALGEBRA_CHECK_R1.json), implemented separately from the official scorer in [verify_theory_r1.py](verify_theory_r1.py), compares this derivative with the official float32 scorer's central difference. With 4,000 prediction coordinates, including 385 saturated values, the absolute difference is 3.68e-8. This validates the local calculation on that fixture, not an ensemble gain. Use it later as a diagnostic for predeclared small mixtures, with fixed coefficients, paired component uncertainty and distinct fitting seeds. Do not use public CPU probe scores to tune weights.

### minGRU has a different memory mechanism

The minGRU paper removes hidden-state dependencies from the gate and candidate. Its appendix distinguishes a signed linear candidate from a positive candidate for log-space scan. More complex benchmarks add convolution and MLP blocks; Appendix D's inference comparison uses batches8–64 and includes context processing. Those benchmark timings do not measure our single-row CPU callback. [Feng et al., 2024, §3.1 and Appendices B–D](https://arxiv.org/html/2410.01201v3).

For one layer with the same externally supplied input in two trajectories, write `h_t=a_t*h_(t-1)+(1-a_t)*v_t`, coordinatewise. The initial-state difference is exactly `delta_h_t=(product_s a_s)*delta_h_0` in real arithmetic because neither coefficient depends on that layer's previous state. A 20,000-step float64 synthetic check agrees with this product formula to2.64e-16. This is a more restrictive mechanism than the full GRU Jacobian described in the earlier note. It is not a bound for all layers together: perturbing a lower layer changes upper-layer inputs and gates.

With constant retention, an e-folding time of tau rows requires `a=exp(-1/tau)`. For tau=100,1000,10000, the write-gate logits are approximately−4.60,−6.91,−9.21. A gate near0.5 has an e-folding time of only1.44rows. These calculations motivate measuring learned gate timescales, and perhaps a separately declared timescale-initialization arm; they do not justify assuming the task needs10,000-row memory.

Removing tanh does not by itself make this input-conditioned recurrence grow exponentially. For finite fixed inputs each coordinate stays in the convex hull of its initial state and candidate values. There is no universal unit bound on those candidates. This differs from a ReLU-candidate GRU whose candidate still depends on previous state. Quantization and floating-point error require full-sequence checks in either case.

One implementation trap matters for the planned zero-state control: Appendix B's parallel code applies `log_g(h_0)`, whereas its sequential code accepts a state directly. Treating the same numeric zero as both arguments would initialize the scan at `g(0)=0.5` but the sequential callback at0. A replication must declare whether the input is a state or a preactivation; an actual zero state needs an explicit zero contribution in the scan (for example log-state−infinity). Check nonzero states, zero states, chunk boundaries and reset against a direct sequential reference before fitting. The paper's pseudocode is not permission to change our reset semantics silently.

## Evidence and interpretation

Literature motivates bounded alternatives; our algebraic checks establish only the stated synthetic identities. The first eight fits remain ordinary zero-state GRUs. No teacher-assistant, minGRU, new mixture or timescale arm has been trained. CPU timing probes, fitting and theory can proceed independently.

## Limits and validity

All existing folds remain exposed. Complete teacher/propensity/normalization ancestry and frozen versioning remain necessary. A single seed's learning curve cannot separate architecture from fitting variation. Native quantization can alter both trajectories and mixture complementarity, so finalists must be graded as served.

## Decision and next step

Complete the first matched ladder. Prepare independent-width allocation support, retain minGRU as a bounded successor, and compare teacher improvement by resulting student benefit. Use the derivative only to explain measured fixed mixtures. Record a separate hypothesis/experiment before spending on assistant or timescale branches; no extra fit is admitted by this note alone.

## Source records

[Full plan](PLAN.md), [frozen student recipe](recipes/README.md), [official metric implementation](../../../src/wnn/gmetric.py), [actual distilled loss](../../../scripts/train_rnn.py), primary literature linked above, and the reproducible [algebra receipt](THEORY_ALGEBRA_CHECK_R1.json). Sources were opened on2026-10-14; Stanton/Mirzadeh claims here are limited to their abstracts, while minGRU equations and implementation appendix were inspected directly.
