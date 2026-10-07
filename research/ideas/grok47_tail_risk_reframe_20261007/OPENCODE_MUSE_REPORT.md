# Eight experiment cards — tail-risk reframe

> **Controller correction (2026-10-07):** The ideation prompt and the original report incorrectly called the official metric weighted squared error. The official metric is `global_wp_clip2_v1`: globally pooled weighted Pearson correlation, after clipping labels and predictions to `[-2,2]`, using `|clipped y|` weights and the fixed scored-row mask, with the two channel correlations averaged. Weighted MSE can be a training surrogate, but it is not the official metric. Read `OPENCODE_MUSE_REVIEW_20261007.md` for the independent review and which proposals need reinterpretation. The original raw model response remains preserved in `OPENCODE_MUSE_OUTPUT.jsonl`.

Objective fixed: official global clipped weighted Pearson, with predictions/labels clipped to `[-2,2]`, `|clipped y|` weights, fixed scored-row mask, and two target channels averaged. Tail / group metrics are diagnostic or separately proposed regularizers only.

Rule constraints — fixed: supplied competition data only, no prohibited external data/modalities unless rules explicitly permit, all weights trained independently, no teammate-checkpoint init or teacher. Submission-engineering constraints — relaxable during training: size, CPU-only, latency, quantization, context / precision / ensemble cost. Large models allowed as independently-trained teachers or research-only instruments with distillation path.

Paper basis: Holland & Haress (2021) CVaR under potentially heavy tails: CVaR estimator under finite-variance + local quantile regularity, then robust selection among SGD candidates, regression demo. No claim it transfers automatically; no claim our score/gradients are heavy-tailed.

---

## Area 1 — Paper transfers

### Card 1: Robust CVaR selection over independent SGD candidates
- **Mechanism family:** Robust validation / model selection, not new architecture.
- **Source basis and analogy:** Direct transfer of paper's two-stage logic: many noisy SGD trajectories -> robust validation selector under finite-variance only. Analogy: grouped time-series validation means are noisy and group-skewed; naive mean-pick may select lucky seed.
- **Exact intervention:**
  1. Train K independent runs, same data/split, varied seeds + LR schedule only, same architecture.
  2. Score each on held-out groups with per-row official weighted squared error.
  3. Select by robust tail-aware statistic: truncated / median-of-means CVaR estimate at fixed tail fraction, plus report official mean. Official mean remains promotion criterion.
  - Ladder: Rung-1: K=5, one tail fraction, one split. Rung-2: K=12-15, 2 tail fractions, 2 group-splits. Rung-3: add stochastic-weight snapshot candidates per run.
- **Training-data/split:** Same supplied train; group-disjoint validation, ideally temporal + group split. No new data. All candidates independently trained.
- **Predicted observable:** Selection ranking differs from mean-ranking; selected model shows lower seed-to-seed validation variance and no worse official mean vs mean-picked; worst-group weighted error diagnoses improvement.
- **Cheapest falsifier:** Local CPU: compute per-sample validation losses from existing checkpoints; test thin tails / quantile irregularity and rank agreement between mean and robust CVaR. If ranks agree and tails are Gaussian-like, stop.
- **Main risk:** Overfitting validation tail; leakage if groups temporally overlap; compute cost of K runs. Mitigate by frozen splits, fixed mask/weights/clip.
- **Resource class:** Rung-1 local CPU/GPU before cutoff; Rungs 2-3 Kaggle GPU.
- **Output artifact:** Selection script + candidate score table with official mean and robust tail stats, selected checkpoint ID.
- **Next-rung gate:** Only if Rung-1 shows rank disagreement and robust pick replicates on second split without degrading official mean, go to larger K.

### Card 2: Tail diagnostics + quantile-aware truncated optimization
- **Mechanism family:** Loss/gradient robustness: diagnostics then truncated-gradient training.
- **Source basis and analogy:** Transfer of paper's assumptions: finite variance + local quantile regularity needed for CVaR estimation. Analogy: check whether our per-sample gradients and magnitude-weighted errors violate light-tail assumptions before adding CVaR regularization.
- **Exact intervention:**
  1. Profile per-row weighted errors, per-step gradient norms, quantile function slope around candidate tail levels, split by magnitude bin and group.
  2. If heavy: train with adaptive per-sample gradient truncation / robust mean gradient aggregation + small CVaR-penalty auxiliary loss, official loss still primary.
  - Ladder: Rung-1: diagnostics only on one trained model. Rung-2: 2 truncation thresholds x 2 CVaR-penalty weights, short training. Rung-3: full training of best setting.
- **Training-data/split:** Supplied data; same group-disjoint split as Card 1 for comparability. No auxiliary modalities.
- **Predicted observable:** Diagnostic plots: tail decay, quantile slope, gradient-norm histogram; if intervention runs, reduced gradient-norm variance and reduced extreme-error frequency with official mean non-worsened.
- **Cheapest falsifier:** Rung-1 diagnostics on CPU from saved losses/gradients. If tails thin and quantile smooth, do not run Rung-2.
- **Main risk:** Mis-tuning truncation biases learning toward easy rows; CVaR penalty silently replaces official objective if weighted too high. Keep penalty small, monitor official metric.
- **Resource class:** Rung-1 local CPU before cutoff; Rungs 2-3 Kaggle GPU.
- **Output artifact:** Diagnostic notebook/table + truncation config if justified.
- **Next-rung gate:** Proceed only if diagnostics show finite-variance but heavy upper tail / unstable quantiles, and short Rung-2 runs show stability gain without official-mean regression.

## Area 2 — Architecture / family

### Card 3: Multi-resolution causal conv + recurrent state hybrid
- **Mechanism family:** Hybrid local-convolution + gated recurrent state encoder, two-channel head.
- **Source basis and analogy:** Generic sequence fit, not paper-dependent: short motifs need fine local filters; grouped long histories need compressive state. Multi-resolution avoids forcing one timescale.
- **Exact intervention:** Causal dilated conv branches at 2 resolutions + recurrent state path with gating, fused before per-step two-channel regression head. Causal only, respects temporal order and scored-row mask. Clipping and magnitude weights applied in loss exactly as official score.
  - Ladder: Rung-1: small width, short context, single resolution vs dual. Rung-2: add state path, double width. Rung-3: longer context + deeper fusion.
- **Training-data/split:** Supplied sequences; group-disjoint splits; ablation must use identical splits/mask/clip.
- **Predicted observable:** Resolution ablation gap: dual-resolution reduces errors on fast-varying vs slow-drift segments respectively; state path matters only on long groups. Observable via official validation mean sliced by sequence length, not invented gains.
- **Cheapest falsifier:** Rung-1 small-context conv-only vs dual-resolution short run on one split. If no sliced difference, stop.
- **Main risk:** Extra branches overfit small groups; longer context leaks across group boundaries if grouping mishandled. Enforce group-aware batching.
- **Resource class:** Rung-1 local GPU if available else Kaggle GPU; Rungs 2-3 Kaggle GPU.
- **Output artifact:** Config + weights trained independently + ablation table on official metric.
- **Next-rung gate:** Only if Rung-1 shows complementary resolution behavior and Rung-2 state path helps long groups without hurting short, go longer context.

### Card 4: Scaling-law experiment across model and context scale
- **Mechanism family:** Empirical scaling analysis; architecture-agnostic.
- **Source basis and analogy:** Standard scaling methodology: fit official validation error vs params / FLOPs / context length to decide whether to scale or stop. Needed because top-1 / >0.700 push cannot rely on small-gain extrapolation.
- **Exact intervention:** Fix data, splits, loss (official weighted MSE with clip/mask), train grid varying width/depth/state-size and context length at fixed compute budgets.
  - Ladder: Rung-1: 3 model sizes at 1 context, short training. Rung-2: 3x3 model x context, full schedule. Rung-3: 1-2 larger points to test power-law extrapolation + data-scale axis via subsampling.
- **Training-data/split:** Supplied data only; subsampling must preserve group structure; separate held-out scaling-validation groups untouched until final check. All points independently trained.
- **Predicted observable:** Log-log linear trend of official validation error vs compute/params/context; crossover point where context scaling beats parameter scaling; variance bands across seeds.
- **Cheapest falsifier:** Rung-1 three-point line. If flat or dominated by noise / split variance, do not scale further.
- **Main risk:** Overfitting scaling split; confusing optimization under-training with capacity limit. Control with fixed steps-to-convergence and LR tuning per scale.
- **Resource class:** Kaggle GPU/TPU after cutoff; design Rung-1 to fit quota.
- **Output artifact:** Scaling plot + fitted exponents + recommendation to scale or stop.
- **Next-rung gate:** Proceed only if Rung-1 shows monotonic predictable gain with overlapping seed intervals separated, and extrapolation predicts meaningful headroom within quota.

### Card 5: Latent-state probabilistic sequence teacher
- **Mechanism family:** Latent-state / probabilistic head: predicts per-step distribution over two channels, plus deep ensemble variant.
- **Source basis and analogy:** Time-series with magnitude-weighted scoring benefits from uncertainty-aware training; probabilistic teacher can capture heteroscedasticity across groups/channels. Used as teacher, not necessarily submission.
- **Exact intervention:** Encoder as in Card 3/attention variant but head outputs mean + scale per channel, trained with proper likelihood + official weighted MSE jointly, official metric still selection criterion. Ensemble 3-5 diverse seeds/architectures for teacher posterior.
  - Ladder: Rung-1: single deterministic vs probabilistic head, same encoder. Rung-2: 3-member ensemble teacher. Rung-3: 5-8 member diverse-architecture ensemble.
- **Training-data/split:** Supplied data; group-disjoint splits; calibration checked on held-out groups. No teammate weights.
- **Predicted observable:** Better calibration on large-magnitude rows and channel-wise error correlation captured; ensemble disagreement correlates with weighted error; official mean non-worsened.
- **Cheapest falsifier:** Rung-1 head swap short run; check calibration and sliced errors. If no heteroscedasticity signal, stop.
- **Main risk:** Likelihood over-emphasizes tail in ways that diverge from clipped weighted MSE; ensemble cost explodes. Keep official metric primary, cap ensemble for student path.
- **Resource class:** Rung-1 local GPU / Kaggle GPU; Rungs 2-3 Kaggle GPU/TPU.
- **Output artifact:** Probabilistic teacher checkpoint(s) + calibration report, usable for Cards 6-8 distillation.
- **Next-rung gate:** Only if probabilistic head improves high-magnitude sliced error or ensemble disagreement is predictive of error, expand ensemble.

## Area 3 — Relax deployment constraints in training

### Card 6: Oversize long-context full-precision teacher -> admissible student distillation
- **Mechanism family:** Knowledge distillation under relaxed engineering limits.
- **Source basis and analogy:** Engineering relaxation, not rule relaxation: train teacher that violates size/CPU/latency/precision/packaging but respects data/rules/causality, then distill into small deployable student. All weights independent.
- **Exact intervention:** Teacher: largest affordable encoder, longest permissible causal past context, full precision, possibly ensemble-averaged logits. Student: submission-compliant size/context. Distillation loss = official weighted MSE on hard labels + match to teacher means on scored rows, both clipped. No future-peeking, no external data, no teammate weights.
  - Ladder: Rung-1: 1 teacher, 1 student size, 2 distillation weights. Rung-2: longer-context teacher + larger teacher. Rung-3: ensemble teacher to single student.
- **Training-data/split:** Same supplied splits; teacher and student trained on train portion only; distillation uses train predictions, selection on held-out groups via official metric.
- **Predicted observable:** Student-teacher agreement rises; student official validation approaches teacher faster than same student from scratch on same steps; context-ablation shows teacher advantage from history length.
- **Cheapest falsifier:** Train small teacher surrogate locally and test whether longer context or width helps at all on one split. If no teacher headroom, skip large-teacher run.
- **Main risk:** Teacher overfits groups and student inherits bias; distillation cost exceeds quota; accidental temporal leakage if context windows cross groups. Enforce causal group-aware windowing.
- **Resource class:** Rung-1 design local if possible; full teacher Rungs 2-3 Kaggle GPU/TPU.
- **Output artifact:** Non-deployable teacher checkpoint + deployable student checkpoint + distillation recipe.
- **Next-rung gate:** Only if Rung-1 teacher beats deployable baseline on official validation and student closes part of gap vs from-scratch control, scale teacher.

### Card 7: Research-only non-deployable oracle for headroom and hard-slice mining
- **Mechanism family:** Research instrument, explicitly not for submission.
- **Source basis and analogy:** Use unconstrained compute to bound achievable error and find systematic failure modes, then convert findings into cheap student fixes. Clear reason to keep research-only: bidirectional / full-sequence / test-time-compute ensemble violates causality/latency and cannot be distilled directly without leakage.
- **Exact intervention:** Build oracle allowed full-sequence bidirectional context and large test-time ensemble / iterative refinement, evaluated with official clip/weights/mask for comparability but flagged as inadmissible due to future-use and cost. Mine largest residual slices by group, magnitude bin, channel.
  - Ladder: Rung-1: bidirectional small model vs causal control on validation only. Rung-2: add large ensemble + longer optimization. No Rung-3 deployment; output is analysis.
- **Training-data/split:** Train on train groups only; oracle inference restricted to validation for analysis; never train on validation; never use future at submission time. Documents rule vs engineering boundary.
- **Predicted observable:** Gap between oracle and causal deployable model by slice; list of hard groups/channels where future context helps most vs where it does not.
- **Cheapest falsifier:** Rung-1 bidirectional vs causal gap on one split. If gap negligible, abandon.
- **Main risk:** Leakage confusion: oracle gains misread as deployable gains; validation over-analysis. Mitigate by explicit inadmissible label and requiring causal student replication of any fix.
- **Resource class:** Kaggle GPU/TPU after cutoff; Rung-1 minimal.
- **Output artifact:** Headroom report + hard-slice list + candidate student fixes, no submission checkpoint claimed.
- **Next-rung gate:** Expand oracle only if Rung-1 shows concentrated slice gap suggesting actionable causal feature; otherwise stop and redirect quota to Cards 3-6.

## Area 4 — Weaker model as complement

### Card 8: Frozen weak-model residual / disagreement as signal
- **Mechanism family:** Residual learning + disagreement gating using weaker/older independently-trained model.
- **Source basis and analogy:** Weak model discarded by standalone official mean may still be complementary if errors are uncorrelated across groups/channels. Use as frozen residual baseline, representation, or disagreement weight rather than init/teacher.
- **Exact intervention:** Freeze weak model (independently trained, not teammate). Train new model to predict residual = label minus weak prediction under official weights/clip/mask, or concatenate weak output/disagreement as auxiliary feature with gating. Compare to from-scratch control same capacity.
  - Ladder: Rung-1: linear residual probe + disagreement-error correlation check. Rung-2: full residual network vs feature-augmented network. Rung-3: gated ensemble of weak + new model with learned per-group weights.
- **Training-data/split:** Weak model frozen from prior independent training; new model trained on train groups; residual fit and gating validated on group-disjoint held-out; no refitting weak on validation.
- **Predicted observable:** Residual learnability: variance of residual less than label variance on slices where weak is good; disagreement predicts new-model error; gated combo non-worse on official mean with slice wins.
- **Cheapest falsifier:** Rung-1 correlation: error correlation between weak and candidate, and residual predictability probe on CPU. If errors highly correlated or residual is noise, stop.
- **Main risk:** Stacking overfits validation via gating; leakage if weak saw validation groups. Verify weak training split disjointness.
- **Resource class:** Rung-1 local CPU/GPU before cutoff; Rungs 2-3 Kaggle GPU.
- **Output artifact:** Residual/feature-augmented checkpoint + correlation report + gating weights.
- **Next-rung gate:** Only if Rung-1 shows low error correlation or predictable residual structure, train full Rung-2; only if Rung-2 beats from-scratch control on official validation, try gated combo.
