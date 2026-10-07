# ADR-0008: Explore model scale beyond the serving envelope

Status: Accepted. Date: 2026-10-07.

## Context

The user set top-1 and public score above 0.700 as the research objective and explicitly rejected pessimistic pruning based on current results. Current submissions are materially below that target. Existing work has concentrated on recurrent students and a submission-shaped compute/deployment budget, while the current hypothesis set also includes distinct architecture and scale directions.

Submission limits can prematurely hide useful teacher signals. A larger or slower independently trained model may improve a compact student through distillation or may contribute to a permitted blend. Training and serving budgets answer different questions.

## Decision

Treat top-1 and >0.700 as active, achievable long-term research objectives. Explore model family, width/depth, context, precision, training duration and ensemble scale beyond final package/runtime limits when the declared compute budget allows. Prefer small-to-large ladders and identify how each rung changes the model or resource use. For each non-deployable candidate, test a concrete distillation, compression or blend path before dismissing its value.

Competition rules remain binding during every phase: use permitted data and pretrained sources, train our weights independently, do not initialize or teach from teammate weights, and preserve official metric/evaluation semantics. No scaling experiment may use prohibited data or weights. Report measured development evidence and public uncertainty separately; a target is not a forecast.

Current user allocation sets local GPU/TPU work through 10:00 Europe/Kyiv on 2026-10-07; after the cutoff, GPU/TPU experiments use Kaggle only. The durable session-specific allocation is [CONTINUATION_WINDOW_20261007.json](../../research/campaign_2026-10-06/bandit_continuation/runtime/CONTINUATION_WINDOW_20261007.json).

## Consequences

Hypothesis generation must include architectures and scaling laws, not only modifications to the current GRU. A research teacher need not itself fit the submission envelope. Each scale rung needs measured resource use, a falsifier and an explicit student/serving handoff. The best compact submission remains the comparison point, and public ranking claims still require actual submissions/results.

## Revisit when

Competition rules change, measured resource/score evidence changes which scale ladder is useful, or the user updates the target or compute allocation.
