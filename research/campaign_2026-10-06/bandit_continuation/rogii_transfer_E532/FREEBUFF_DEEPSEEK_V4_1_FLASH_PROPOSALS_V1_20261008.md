# Directed-Source Transfer Proposal — Wunder Connectome Challenge2
### Mechanism cards derived from the ROGII 1st-place writeup (Ruby, "1st Place Solution")

## Standing conventions (apply to every card)

**Endpoint controls (non-negotiable):** final output is two deterministic scalar targets per row; the ±2 clip and label-magnitude weighting are applied only when computing the global weighted-Pearson score, never silently inside the model unless a separate validation shows it helps; every input feature at row *t* is computed from rows ≤ *t*; sequence-state reset is honored exactly; no normalization statistic (mean/std/percentile/bin edge) is ever computed over future rows; particle/state updates are deterministic under a fixed seed. Training clipping is kept independent from evaluation clipping (the source explicitly separates these — treat it as a validity rule, not a trick).

**Anonymity rule:** the 112 channels are anonymous. No card assigns semantic meaning (no calling a channel "GR", "TVT", "depth", or a geological invariant). "Anchor", "offset", "view", "reference channel" are operational names for constructs we build, not claims about what a channel is.

**Privileged-teacher ancestry:** any training-only teacher that uses future rows must (a) see only training-fold data, (b) have its smoothing/fitting ancestry documented, (c) deliver a soft *target*, never an input feature, at inference, and (d) be fold-isolated so its outputs never enter a student evaluated on its own fitting fold.

**Cost tiers (no timing claims):** T0 = runs on local 6 CPU / 20 GB; T1 = single Kaggle GPU session; T2 = bounded Vast worker or multi-session; T3 = multi-worker campaign.

---

## CARD M1 — Causal anchor-relative offset distribution head

**ID:** M1-offset-dist-head

**Source hook:** "Task Formulation and Loss" — *"cross-entropy loss as the main objective"* / target is an *"exponentially smoothed probability distribution"*.

**Structural analogy.** The source reframes a continuous localization target as classification over a discretized grid anchored at the last visible value. Generic equivalent: reframe each of the two regression targets as a distribution over a discretized *offset* grid centered on a strictly causal anchor (last observed value, or a causal EMA of recent rows). The target is a kernel-smoothed bump (Gaussian or Laplace of chosen width) centered at the realized offset, normalized over the grid; the head emits grid logits per row; the scalar prediction is the expectation over the grid, and a Huber term aligns that expectation with the true offset. This is reframing + loss, not a new backbone.

**Intervention / components.**
- Causal anchor *a_t* computed from rows ≤ *t* only (carry-over last value or causal EMA; document the choice).
- Fixed offset grid *G* = {g_1 … g_K} at a stated resolution; logits from the existing causal backbone (GRU teacher/student, causal TCN, event-feature GRU) with a K-wide head per target.
- Soft target: normalized kernel bump around realized offset (train only). Smoothing width as a hyperparameter; do not reuse the source's "exponential" form literally, just the smoothed-soft-target principle.
- Endpoint: prediction = Σ p_k · g_k, plus optional residual correction from the same backbone; then the ordinary ±2 / magnitude-weighted scoring applies.

**Assumptions.** Offsets from a causal anchor are more stationary than raw target levels, so a bounded grid covers them; the anchor tracks slowly enough that a fixed grid suffices; the kernel-smoothed soft target reduces overconfidence better than one-hot.

**Causal leakage / validity risks.** Anchor must not leak future rows (easy to get wrong with a running mean that peeks). Grid resolution/range chosen on training folds only. The soft target is a *label transform*, so using realized offsets is legitimate, but it must never become an input at inference. Do not compute any per-sequence offset statistics across the full sequence.

**Predicted observable.** Better-calibrated tails and reduced variance on heavy-magnitude regimes; distribution entropy or top-bin mass should correlate with prediction error (a usable uncertainty signal); the Huber-on-expectation term should reduce the grid-mean's shrinkage toward the center.

**Cheapest discriminating test + matched control.** Take one existing causal model and add a K-wide distribution head with smoothed soft targets, keeping width/params/folds/seeds matched to a plain two-output Huber head. Compare global weighted Pearson on the fixed mask. Control = identical head with one-hot (unsmoothed) targets, to separate "distribution head" from "target smoothing."

**Falsifier (scoped).** If, across the same folds and seeds, the smoothed-soft-target CE head does not exceed the plain two-output head on the fixed-mask weighted Pearson *and* does not improve tail/magnitude-regime error, retire this specific soft-target-CE recipe for both targets — the anchor-offset reframing may still be tested with a plain head.

**Cost tier:** T0–T1.

**Parent / interaction links.** Sits on top of all current families (recurrent teacher/student, causal TCN, event-feature GRU); interacts with M2 (penalty on the same distribution), M8 (per-regime weights), M7 (augmentation changes the offset distribution the head must fit).

**Level-2 successors.** Ordinal/soft-rank head; CRPS or pinball ensemble over the grid; label-magnitude-weighted CE term so the magnitude weighting of the metric is mirrored in the loss; two-target joint distribution head with a shared grid.

---

## CARD M2 — Expected-gap penalty (unconventional small-component transfer)

**ID:** M2-gap-penalty

**Source hook:** "Task Formulation and Loss" — *"GR penalty loss"*, described as a *"small additional improvement."*

**Structural analogy.** A tiny auxiliary term that penalizes probability mass placed far from the anchor: `mean(probability × grid_offset_magnitude)`. In the source it grounds the alignment map against a large jump. Generically it is a soft, differentiable prior that discourages the distribution head from buying mass in the tail to chase outliers, without hard-clipping. It is the deliberate opposite of the usual "let the model learn everything" instinct, and it is exactly the kind of minor loss term that gets dismissed as irrelevant.

**Intervention / components.** Add `λ · Σ p_k · |g_k|` to the M1 objective (or, if M1 is not adopted, apply it to any offset parameterization). Tune λ on training folds only. Keep it small; verify it does not flatten genuine heavy tails.

**Assumptions.** The metric's label-magnitude weighting already tolerates large true magnitudes, so the penalty must be small enough to preserve them; tail mass is mostly noise-driven overconfidence.

**Causal leakage / validity risks.** None structural — it is a function of the predicted distribution and a fixed grid. Main risk is semantic drift: using it to force small outputs would conflict with label-magnitude weighting; monitor per-magnitude-bin calibration, not just the aggregate.

**Predicted observable.** Reduced average predicted magnitude conditional on true magnitude being small; smaller variance of predicted offsets; no loss of tail recall. The negative result from the source ("fancy loss weighting strategies" gave no consistent gain) is a warning that λ must be validated, not assumed.

**Cheapest discriminating test + matched control.** On the M1 head, sweep a small ladder of λ values by training-fold score; matched control = λ = 0 with identical seeds. Look for improvement in weighted Pearson *and* in per-magnitude-bin error, not aggregate alone.

**Falsifier (scoped).** If no λ in the swept ladder improves fold weighted Pearson or magnitude-bin calibration over λ = 0, and flat prediction variance does not help the metric, drop this penalty term specifically (the M1 distribution head may remain intact).

**Cost tier:** T0.

**Parent / interaction links.** Child of M1; interacts with M8 (regime-specific λ) and M7 (injected rare jumps make λ selection nontrivial).

**Level-2 successors.** Asymmetric penalty (penalize only overconfident small-offset mass on known high-magnitude rows); expected absolute-offset regularizer as a monotone variant; anchor-relative entropy bonus.

---

## CARD M3 — Causal multi-scale surrogate for the symmetric U-Net + privileged bidirectional teacher

**ID:** M3-causal-unet-teacher

**Source hook:** "Model" — *"standard 2D U-Net with a ConvNeXt backbone"*; *"average pooling + interpolation performs better"*; plus *"Replacing LayerNorm … with BatchNorm"* / BF16 NaN note.

**Structural analogy.** The source's backbone is spatially symmetric (future and past both visible); that is unusable causally as-is. Its *components* are transferable: causal => per-scale streaming encoder-decoder, fixed average-pool downsampling and interpolation upsampling instead of learned resampling, and normalization choice treated as a stability hyperparameter. The symmetric teacher, if used, is a training-only privileged distilled target.

**Intervention / components.**
- Causal replacement: dilated causal convolutions or a stateful streaming network with the same multi-scale topology (downsample via fixed average pooling, upsample via interpolation, residual decoder blocks). No look-ahead in any layer.
- Privileged bidirectional teacher (optional): a non-causal, multi-scale net trained on training folds only, producing per-row soft targets; the causal student distills those targets. Ancestry audited per the standing rule; teacher outputs never feed student inputs.
- Micro-transfer details: prefer fixed average-pool/interp over learned resampling (test as an A/B); treat normalization choice (batch-statistic variants) as a stability lever; keep training in the low-precision mode needed to make batch-style normalization stable.

**Assumptions.** Multi-scale context helps where the causal receptive field currently saturates; distillation from a future-aware teacher transfers real signal rather than encoding the teacher's future leakage; fixed resampling is enough for this data scale.

**Causal leakage / validity risks.** The highest in this proposal: a bidirectional teacher can leak future *features* into the student's training signal only as a target — but if the student also learns to imitate the teacher's internal states or its future-dependent features, that is leakage. Audit the distillation loss to reference only teacher output distributions per row, fold-isolated. Also ensure no learned downsampling that aggregates across a row boundary in a non-causal order.

**Predicted observable.** Causal multi-scale net matches or beats the current best causal student alone; privileged-teacher distillation adds a gain that vanishes if the teacher is trained on the student's eval fold (the falsifier's tell).

**Cheapest discriminating test + matched control.** Build the causal multi-scale net at a small width and compare against a plain causal GRU/TCN of similar parameter count on the same folds. Control A = same net with learned resampling; Control B = same net trained without the privileged teacher.

**Falsifier (scoped).** If the causal multi-scale net with fixed pool/interp and no teacher does not beat the matched causal baseline, retire the multi-scale-architecture recipe for the current width; if it wins only when the teacher is trained in-fold, the teacher route is invalid, not the architecture.

**Cost tier:** T1 (student), T2 (privileged teacher).

**Parent / interaction links.** Hosts M1/M2 heads; consumes M7 augmentations; distilled teacher variant of the existing heterogeneous recurrent teacher family.

**Level-2 successors.** Multi-scale belief-state fusion with M4; causal version of channel attention over the 112 features; scale-recurrent streaming with fixed state budget for CPU inference; teacher-ensemble distillation.

---

## CARD M4 — Causal particle belief state with rare jumps, smoothing as a privileged teacher

**ID:** M4-particle-belief

**Source hook:** "Particle Filter" — *"Allowing low-probability large jumps"*, *"Updating particles in bins"*, *"FFBSi smoothing"*, *"Blending diverse cfg profiles."*

**Structural analogy.** Maintain an explicit belief over the offset state as a weighted particle set updated row-by-row, with a heavy-tailed proposal so rare jumps remain reachable, observation likelihood from a small learned model over the 112 anonymous channels, and bin-level updates for efficiency. Everything is naturally causal and resets with the sequence. FFBSi smoothing uses future rows and therefore becomes a training-only privileged target generator (distill the smoothed posterior into the causal filter's likelihood model).

**Intervention / components.**
- State = offset from the causal anchor (shared with M1's grid for consistency).
- Transition with a base diffusion plus a low-probability jump component (mix of two kernels, jump rate tuned on training folds).
- Likelihood from a lightweight causal model of the anonymous channels — no hand-coded channel semantics.
- Bin-level updates (aggregate particles) for cost control.
- Deterministic seeding and deterministic resampling given inputs.
- Training-only: FFBSi / smoothed posterior used to fit the likelihood and to produce a soft target; never available at inference.
- Endpoint: filter posterior mean and its spread as (a) prediction candidates and (b) input channels to the neural student (the source's "particle probability heatmap" and "|anchor − PF estimate|" analog).

**Assumptions.** A low-dimensional offset state with a Markov transition is a decent model of the target's local dynamics; the learned likelihood is informative enough that particles concentrate; rare heavy-tail jumps are real and reachable.

**Causal leakage / validity risks.** Smoother must be training-only and fold-isolated; any bin edge computed over the whole sequence is a leak; deterministic resampling is required so the scored inference is reproducible. Injecting the particle posterior as an input channel risks shortcut learning (the source hit this and deliberately corrupted the channel — see M7).

**Predicted observable.** Particle ESS, jump frequency, and posterior-spread-vs-error correlation; the filter as a standalone component should be materially informative on the fixed mask; adding the posterior channel should help only if shortcut corruption is in place.

**Cheapest discriminating test + matched control.** Implement a scalar filter for one target with a simple likelihood (e.g., linear probe of the 112 channels), compare standalone weighted Pearson against a causal EMA baseline on identical folds. Control A = jump component disabled; Control B = smoothed (teacher) posterior rather than the causal filter, to confirm the smoother is genuinely better and therefore worth distilling.

**Falsifier (scoped).** If the jump-enabled filter fails to beat both the no-jump filter and the causal EMA baseline on the fixed mask across repeated seeds, retire the rare-jump transition for that target; the filter may still be kept as a diversity component for M8 rather than a primary mechanism.

**Cost tier:** T0 (scalar filter), T2 (FFBSi teacher + distillation).

**Parent / interaction links.** Feeds M1 (grid alignment), M5 (agreement channels), M8 (diversity); consumes M7 (channel corruption, jump injection); related to the "online feature dynamics" and "block memory" families.

**Level-2 successors.** Rao-Blackwellized multi-target filter; learned transition (recurrent jump model); per-regime jump rates; filter-state reset anomaly detection; filter-guided curriculum for the neural student.

---

## CARD M5 — Cross-view calibration and interaction/agreement channels

**ID:** M5-crossview-calibration

**Source hook:** "Features → Typewell" — *"Typewell GR is calibrated using the visible region"*; "Typewell × Horizontal interactions" — paired difference channels.

**Structural analogy.** Build a *reference view* of the sequence (a causally-smoothed or binned rendering of the anonymous channels) and a *raw view*, then (a) recalibrate each raw channel against the reference by causal binning + interpolation + blending, and (b) construct paired-difference channels between the two views and versus the current prediction. The source's typewell/horizontal pairing is the special case of "two aligned renderings of the same underlying signal."

**Intervention / components.**
- Causal reference view: bin *t*-past rows, interpolate back to the row axis, blend with raw values using a blend weight tuned on training folds. Never use future rows in the bins.
- Per-channel calibration coefficients learned on training folds only.
- Paired |raw − reference| channels and |reference − current_prediction| channels for both targets; also accumulate vs differential variants (the source notes accumulated information wins when an informative auxiliary channel is present).
- Endpoint: channels feed the causal student; the interaction channels themselves are not outputs.

**Assumptions.** A causal binned rendering is a useful denoised reference; absolute scale carries less information than shape/difference (motivating affine-invariant transforms, M7); prediction-agreement channels add information beyond raw features.

**Causal leakage / validity risks.** The reference view is the main leak risk if bins or interpolation touch future rows; blending weights and calibration coefficients must be fit per fold. Prediction-agreement channels must use only the current-row prediction, not a look-ahead.

**Predicted observable.** Calibrated channels reduce scale-driven variance; difference/agreement channels should carry information visible as feature-importance shifts and improved fold score; if the interaction channels are the only gain, scale has become the shortcut.

**Cheapest discriminating test + matched control.** Add calibrated reference and paired-difference channels to one causal model, same folds/seeds. Control A = raw channels plus affine-normalized raw channels; Control B = difference channels only (no calibration), to separate calibration from interaction.

**Falsifier (scoped).** If calibration-plus-interaction channels do not beat the affine-normalized raw baseline on fold score, retire the causal-bin calibration recipe; keep raw difference channels only if Control B alone shows a gain.

**Cost tier:** T0.

**Parent / interaction links.** Pairs with M4 (reference view from filter posterior), M6 (neighbor-context channels), M7 (affine invariance), M1 (grid from calibrated offset).

**Level-2 successors.** Learned causal calibration (small recurrent calibration head); multi-reference consensus channels; per-channel reliability weights.

---

## CARD M6 — Local-linear context with a quality gate and CV-vs-metric decision discipline

**ID:** M6-neighbor-gate

**Source hook:** "XY Neighbor Information" — *"S = TVT + Z + C"*, *"ΔTVT = aΔX + bΔY − ΔZ"*, and *"trust the local CV."*

**Structural analogy.** If a coordinate/index structure exists across the supplied sequences, model the target as locally linear over a neighborhood: estimate local slopes by weighted least squares and extrapolate; otherwise fall back to a learned causal model. The transferable and arguably more valuable part is the *discipline*: compute neighborhood-quality statistics, route to a fallback above a training-percentile threshold, and maintain a documented divergence policy between local validation and the official fixed-mask metric instead of blindly trusting either.

**Intervention / components.**
- Declare a pseudo-coordinate space honestly: either provided indices, or a low-dimensional projection of a subset of the 112 anonymous channels. Do not assume it exists.
- Weighted-least-squares local-linear fit with regularization and singular-case handling; produce a context estimate.
- Quality statistics (neighbor distance spread, weight/prefix balance, query-to-center distance ratio, directional consistency); gate to fallback above a training-fold percentile.
- Decision rule: when local CV and the fixed-mask metric disagree, log both, characterize the disagreement by quality statistic, and choose a fallback channel explicitly rather than assuming one is right.

**Assumptions.** If coordinates exist, the target is locally linear over them; labels are consistent enough; the quality statistics actually separate reliable from unreliable neighborhoods.

**Causal leakage / validity risks.** Neighborhoods must contain only rows available at inference (no future neighbors); coordinate projections fit on training folds only; the percentile threshold is a training-fold statistic.

**Predicted observable.** A context estimate that is informative where quality is good and that degrades gracefully; agreement/disagreement between the context estimate and the learned model on high-magnitude rows; quality statistics that predict when the context estimate misleads.

**Cheapest discriminating test + matched control.** First a *structure test*: does any coordinate projection exist where local-linear context beats a causal EMA/mean baseline on fold data? Control A = same fit with shuffled coordinates (placebo); Control B = gate always off.

**Falsifier (scoped).** If the local-linear context fails the placebo test (shuffled coordinates do as well) or the quality gate does not reduce bad-neighborhood error, retire the coordinates-based recipe entirely. Do not extend the falsifier to "context is useless" — a learned context without coordinates may still be viable.

**Cost tier:** T0.

**Parent / interaction links.** Feeds M5 interaction channels and M8 regime gating; competes/complements the causal heuristic window feature family.

**Level-2 successors.** Learned quality-statistic classifier for gating; per-neighborhood uncertainty; context ensemble with M4.

---

## CARD M7 — Counterfactual trajectory augmentation, invariant transforms, and anti-shortcut corruption

**ID:** M7-invariant-augment

**Source hook:** "Data Augmentation" — *"Z-shift and GR transform"*, reverse path, channel masking; and *"PF rotate/shift (… to reduce shortcut learning)"*.

**Structural analogy.** Two mechanisms: (1) *counterfactual resampling of the target path* while holding an invariant combination fixed, regenerating the observable features through a matched view (block bootstrap of target differences, plus injected rare jumps); (2) *affine-invariant channel transforms* so the model uses shape rather than absolute scale, plus deliberate corruption of auxiliary channels to block shortcut learning.

**Intervention / components.**
- Resample target paths via block bootstrap of observed increments while preserving a chosen invariant; regenerate the matched feature view; inject rare jumps as offsets.
- Affine transform applied consistently across channels derived from the same source (so the transform is coherent); jitter.
- Reverse-traversal augmentation with a retained visible prefix (a legitimate causal analogue of the source's reverse path if the prefix rule is enforced).
- Sequential and 2D channel masking; tail cropping; residual-shift noise.
- **Anti-shortcut:** periodically rotate/shift the auxiliary belief channels (M4) and neighbor channels (M6) so the student cannot overdepend on a single convenient estimator.

**Assumptions.** The chosen invariant is stable across sequences; block-bootstrap trajectories are realistic enough; the auxiliary channels are shortcuts that hurt generalization when uncorrupted.

**Causal leakage / validity risks.** Regenerating features must use the same causal rule as inference (a reverse-path augmentation is the obvious place to accidentally introduce look-ahead). Injected jumps change the label distribution and must be validated. Do not let synthetic data dominate real data (the source found joint training beat a two-stage simulation-first approach).

**Predicted observable.** Reduced feature-importance concentration on auxiliary channels; improved fold score under affine transforms of the channels; robustness to tail cropping; the model should retain performance when auxiliary channels are ablated.

**Cheapest discriminating test + matched control.** Train two identical models with and without the anti-shortcut corruption schedule; ablate auxiliary channels at inference in both. Control A = affine transform only; Control B = corruption only.

**Falsifier (scoped).** If the anti-shortcut corruption schedule does not reduce the fold gap between auxiliary-channel-present and auxiliary-channel-ablated inference, retire that specific corruption recipe; the affine and counterfactual augmentations are tested separately.

**Cost tier:** T0–T1.

**Parent / interaction links.** Applies to M1/M3/M4/M6; interacts with M8 (diversity).

**Level-2 successors.** Learned augmentation policy; invariant-consistency loss; curriculum over jump rate; matched-view regeneration for M5 reference channels.

---

## CARD M8 — Regime-conditioned diversity ensemble with conditional quality gates

**ID:** M8-regime-ensemble

**Source hook:** "Ensemble" — *"For wells without reliable XY-neighbor information (…)"*; per-feature weights differ by regime.

**Structural analogy.** Partition inference by an observable online regime statistic and use different ensemble weights per regime, rather than one global blend. Include feature-family diversity on purpose (default / belief / context / interaction) and quality gates that route away from components whose inputs are unreliable in the current sequence.

**Intervention / components.**
- Causal regime statistic from training-fold percentiles (never sequence-global).
- Per-regime weights fit on training folds; deterministic blend.
- Component portfolio: plain causal (default), belief-state (M4), context (M6), interaction (M5) — deliberately heterogeneous.
- Quality gate routes to non-context components above the M6 percentile threshold.
- Weak standalone components retained only if they add diversity; validate by leave-one-component-out on fold score.

**Assumptions.** Regimes are detectable online; diversity between component families is real; per-regime weighting beats a single global blend.

**Causal leakage / validity risks.** Regime thresholds and weights are training-fold statistics; the regime statistic must be computable at row *t*. The source's headline caution — local CV improving while the official metric degraded — is directly relevant: report both and make the selection rule explicit (M6 discipline).

**Predicted observable.** Leave-one-component-out deltas showing genuinely diverse contributions; regime-specific gains over a global blend; no single component dominating all regimes.

**Cheapest discriminating test + matched control.** Compare per-regime blending against a single global blend with the same components and seeds. Control = weight the belief-state component at zero in all regimes.

**Falsifier (scoped).** If per-regime blending does not beat the global blend on fold score, and leave-one-component-out shows no component with a non-trivial positive contribution, retire the regime-conditioned weighting for this component set (diversity may still pay at the model level rather than the weight level).

**Cost tier:** T0 (blend), T1–T2 (retraining components).

**Parent / interaction links.** Terminal integator for M1–M7; enforces the endpoint controls.

**Level-2 successors.** Stacked meta-blender with belief uncertainty as an input; online regime adaptation; uncertainty-gated abstention to a robust fallback.

---

## Top-3 ordering (diverse, information- and long-term-usefulness-justified)

No score priors are attached; ranking reflects what each mechanism teaches and how much of the pipeline it can improve.

1. **M1 — Causal anchor-relative offset distribution head.** Highest leverage-to-risk ratio: it changes the endpoint form for *every* existing family at T0–T1 cost, its falsifier is clean and scoped, and it produces a reusable uncertainty signal. It also creates the grid that M2 and M4 consume. Information gain is broad because a negative result is still diagnostic about target stationarity.

2. **M4 — Causal particle belief state with rare jumps.** Directly exploits the causal row-by-row / reset structure that defines this challenge, and it is the mechanism most likely to add *orthogonal* information rather than incremental backbone capacity. It yields interpretable diagnostics (ESS, jump rate, spread–error correlation) and a natural training-only-smoothed privileged teacher, matching the "teachers may be compressed later" affordance. Cost is higher, so it is second rather than first.

3. **M6 — Local-linear context with a quality gate and decision discipline.** Ranked third for a different reason: it is the cheapest test of a structural assumption we do not currently know holds. Its placebo falsifier resolves quickly, and even a negative result is valuable (it closes a direction cleanly). The quality-gate and CV-vs-metric decision discipline transfers to every other card. If the structure test passes, it also feeds M5 and M8 with orthogonal signal.

*Near-tie:* **M3** (causal multi-scale net + privileged teacher) is the more foundational choice if distillation infrastructure is the priority; it sits just below M6 because its components overlap the existing causal GRU/TCN work more than M6's structural probe does.

---

## Integration table

| ID | Mechanism class | Endpoint role | Tier | Key dependency | Primary risk | Scoped falsifier |
|----|-----------------|---------------|------|----------------|--------------|------------------|
| M1 | Reframing + loss | Produces per-target offset distribution → scalar via expectation | T0–T1 | Causal anchor definition | Non-stationary offsets; leaky anchor | Soft-target CE head ≤ plain 2-output head on fixed mask + tails |
| M2 | Small auxiliary loss | Regularizes M1 distribution toward anchor | T0 | M1 head | Over-flattening true heavy tails | No λ in ladder beats λ=0 on fold score and magnitude bins |
| M3 | Architecture (causal replacement) | Causal multi-scale student; privileged bidirectional teacher | T1–T2 | Fold-isolated teacher ancestry | Future leakage via teacher internals | Causal net alone ≤ matched causal baseline |
| M4 | Belief state | Filter posterior as prediction candidate + input channel | T0–T2 | Causal likelihood model | Smoother leakage; shortcut dependence | Jump filter ≤ no-jump filter and EMA baseline across seeds |
| M5 | Feature channels (calibration) | Calibrated reference + agreement/difference channels | T0 | Causal binned reference | Bins/interp touching future rows | Calibration+interaction ≤ affine-normalized raw baseline |
| M6 | Structural probe + gating | Local-linear context estimate + fallback gate | T0 | Existence of a coordinate space | No real coordinates; leaky neighborhoods | Shuffled-coordinate placebo matches real fit |
| M7 | Augmentation + invariance | Training distribution shaping; anti-shortcut corruption | T0–T1 | Invariant definition | Look-ahead in regeneration; sim-dominance | Corruption doesn't shrink aux-present vs aux-ablated gap |
| M8 | Aggregation | Per-regime weighted blend of heterogeneous components | T0–T2 | Online regime statistic | Training-fold-statistic leakage | Per-regime ≤ global blend; no component with positive LOO delta |

**Unconventional small-component transfer (called out per brief):** M2 — the expected-gap penalty is a genuinely minor, easy-to-dismiss loss term reported as a "small additional improvement," and it is exactly the kind of low-cost, high-information probe worth extracting from an otherwise geology-specific solution. Paired with it, two further micro-details are extracted even though their headline context looks irrelevant: (i) the source's observation that *learned resampling underperformed fixed average-pool/interpolation* (folded into M3 as a cheap A/B), and (ii) the practice of *deliberately corrupting an informative auxiliary channel to reduce shortcut learning* (folded into M7 as the anti-shortcut schedule). The source's negative results are treated as usable evidence: Transformer-heavy backbones, larger models/higher resolution, two-stage simulation pretraining, fancy loss weighting, and deleting visually bad-labeled rows all failed to give consistent gains — each becomes a matched control or a default we require new cards to beat rather than a direction to replicate.

## Standing validity checklist (applies to all cards before any promotion)

- Every input at row *t* uses rows ≤ *t*; reset honored; no sequence-global normalization.
- Any future-aware component is training-only, fold-isolated, documented, and delivers targets (not inputs).
- Every node/likelihood/teacher/regime threshold is fold-fit; the fixed scored/prediction mask is never used for selection.
- ±2 clip and magnitude weighting live in the scorer by default; training clipping is validated separately.
- Determinism: fixed seeds, deterministic resampling, deterministic blending.
- Auxiliary-channel dependence is measured by ablation with and without M7's corruption.
- Local-CV vs fixed-mask disagreements are logged per M6's discipline rather than resolved by assumption.
- OOM safety: bounded particle counts, bin-level updates, and small widths by default; scale only behind a validated gain.
