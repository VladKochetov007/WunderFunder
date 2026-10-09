# Combinations and a sampled experiment queue

This records the user's October9 questions and the controller's proposed implementation direction. It does not launch a sweep or change frozen experiments. Existing trainer CLI and finite background runner should be reused; a general sampled-manifest generator is still proposed, not delivered by this note.

## First factorial comparison

| Cell | Optimizer | Loss | Purpose |
|---|---|---|---|
| A | AdamW | Weighted MSE | Matched usual-loss control |
| B | AdamW | Weighted Huber, delta1 | Loss effect and complementary errors |
| C | Muon on recurrent gates, AdamW elsewhere | Weighted MSE | Optimizer effect |
| D | Muon on recurrent gates, AdamW elsewhere | Weighted Huber, delta1 | Optimizer × loss interaction |

D01 supplies A/B and D02 supplies C/D. The initial Muon arm has explicit FP32 orthogonalization, recurrent gate scope, equal group peak LR0.001 and matched OneCycle LR/beta trajectory. It is a narrow equal-LR contrast, not a claim that either optimizer is optimally tuned. A separately identified learning-rate follow-up may be needed. Keep fold, complete-component membership, initialization, order, normalization, architecture, optimization budget and official evaluation fixed within each contrast. New folds require new clean teachers/initializers where the ancestry demands them.

Train each member once and save aligned predictions. Evaluate the four individual models; six pairs; same-optimizer/different-loss pairs; same-loss/different-optimizer pairs; all-four average; then constrained blend weights and component-CV stacking. Four nonnegative blend weights on a 0.1 grid produce 286 allocations, including endpoints. Those are prediction computations, not 286 GPU fits. Continuous weights have infinitely many possibilities. For many members use bounded greedy inclusion/coordinate search or regularized fitting instead of enumerating all subsets.

Assess conditional contribution to a fixed strong reference, not only standalone score or raw prediction correlation. Save residual correlations, weighted error moments, component-level deltas and performance by causal regime/target. Use sufficient statistics only where they exactly preserve the chosen clipping semantics; score finalists from actual predictions with `global_wp_clip2_v1`, original label weights and final clipping. An apparently weaker model may still be a useful teacher or blend member.

Stacking weights, model selection and gating must be trained inside the declared meta-training components and evaluated on other components. Preserve all previous exposure. A second fold provides robustness evidence but does not erase earlier selection. Keep simple averages and regularized convex weights as controls for any learned nonlinear stack.

## Additional dimensions already retained in the matrix

| Dimension | Candidate contrasts | Important distinction |
|---|---|---|
| Architecture | GRU width ladder, LSTM, causal TCN, slow/fast state, attention, low-rank recurrence | Large teachers may exceed serving limits; declare the compression path |
| Activation/readout | GELU, fixed mixtures, target-specific small heads | Exact activation placement and initializer must be declared |
| Loss/targets | WMSE, Huber deltas, correlation-oriented loss, raw/clipped teacher families, target-family specialism | Official evaluation clipping never changes |
| Samples | Whole-component bootstrap, input-regime strata, OOF-difficulty selection | No arbitrary row shuffle or broken causal history; teacher ancestry stays disjoint |
| Schedule | OneCycle, cosine, WSD, plateau/noise policies, optimizer switching | Equal steps/epochs and schedule horizons; switches are separate arms from blending |
| Distillation | Disagreement-selected teachers, family targets, many independently initialized students | Diverse useful teachers, not merely a larger count |
| Second stage | Static weights, ridge/NNLS, small causal gate, residual boosting, metric-gradient corrections | OOF base predictions and component-CV for all fitted meta-targets |
| Serving | Compression, small additional trunk, native inference improvements, quantization | Evaluate the complete served endpoint and causal reset/runtime |

Boosting is different from averaging independent models: fit a successor to OOF residuals or a correctly derived training-metric gradient, use shrinkage, and compare against an equal-cost extra independent model. A large model trained to fit its own training residuals can overfit; teacher/meta ancestry and component boundaries remain binding. Optimizer switching is also distinct from blending Muon and AdamW prediction endpoints. Do not average independently trained neural weights without an explicitly audited alignment method.

## Simple CLI and durable queue design

Keep a JSONL manifest with one complete, immutable recipe per row. Fields include experiment ID/parents, source and data hashes, fold/component membership, model/width/activation, optimizer and parameter groups, loss/target treatment, learning-rate schedule, seed, update budget, precision, output policy and resource/deadline ceilings. The existing trainer CLI executes each row; the existing guarded finite queue supplies exclusive device admission, heartbeat, terminal receipts and checkpoint recovery.

Proposed sampler: deterministic seed20261009; sample without replacement from compatible parameter combinations; preserve matched baseline pairs and coverage of different mechanisms. Stratify the first batch across optimization, representation, target and combination ideas. Do not use unconstrained random flags that form invalid models. Queue deduplication uses a canonical recipe hash, including ancestry and fidelity, rather than only a human run name.

Use one parsed cache per worker/cohort. Map immutable arrays once; distinguish OS page-cache reuse from actual parsed-object reuse. Estimate and cap the entire admitted cohort from measured loading/training/export times. Defer unsupported/OOM configurations with execution evidence and a named repair; they are not scientific failures. A finite queue survives Codex exit, but does not invent new recipes, spend, or expand quotas after its admitted work is exhausted.

Suggested next batch: finish the four optimizer/loss cells, measure their blends, then sample a bounded set of distinct architecture/schedule/sample contrasts. Preserve an exploration reserve. Promote promising or informative uncertain candidates to another fold/full scale; neither a noisy small-screen loss nor a confidence interval crossing zero is a universal kill rule.

## Saved outputs

Keep final/fixed-epoch weights and provenance; one rolling exact-resume checkpoint; dense FP32 predictions with source-qualified row/group IDs, mask and hashes; optional scored-row Parquet with ZSTD compression for analysis. Record per-update unscaled gradient norms, clipping, skipped steps, LR/beta and timing while training. Final weights cannot reconstruct the historical gradient trace. Dense evaluation predictions for 468 sequences ×20,000 rows ×2 targets occupy74,880,000 data bytes per model; full-valid predictions are larger. Retention/admission must include teacher caches and all-data export separately.
