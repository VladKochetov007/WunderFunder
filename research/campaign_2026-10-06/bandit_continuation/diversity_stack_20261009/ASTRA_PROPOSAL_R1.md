# Independent proposal: learn useful disagreement, then compress it

Owner: one-off native Astra ideator, requested route `gpt-6-astra`, reasoning `xhigh`, under ADR0027. Initial independent commitment; no competing new proposals read. Base: `c5d324e04a1d84cc4045e6cdf5239a944ec8a1e9`. This is a research design, not an admitted experiment or supported gain. The accompanying source/cost receipt records access and limits.

My recommendation is to spend the first effort on **conditional signal accounting plus three orthogonal teacher changes**, then fit a deliberately small second level. Start with the predictions already paid for. A different optimizer, activation or seed is useful only when its error correction survives a matched endpoint comparison or becomes learnable by a cheap student. Low output correlation is neither necessary nor sufficient.

The most valuable architectural distinction may be **what history a model retains**, not whether its recurrent cell has a different name. The most valuable stacking distinction may be **where correction is allowed**, not the number of parameters. The most valuable distillation distinction may be **preserving a small complementary component separately from the large common prediction**, not averaging more teachers.

## Basis and limits

The supplied brief and CONTRACT_R1 are authoritative over historical snapshots. Own public best is reported as 0.6660; the teammate's 0.6674 is reported, not independently reproduced here. The H230 seed3 endpoint delta +0.000239671 and exact Z3 delta -0.000252075 on exposed fold0 do not establish a new expected-public best. Top1 and >0.700 remain objectives; none of the proposals below assigns a public forecast.

The official endpoint metric remains `global_wp_clip2_v1`: selected rows, two targets, label/prediction clipping at ±2, weights from the magnitude of the original clipped label, pooled weighted correlation per target. I inspected `src/wnn/gmetric.py`. Training-target transformations, loss weights, calibration and clipping order are separate treatments. Evaluation means/variances must not be replaced by averages of sequence-level scores.

All models, teachers and initializers are our own. Own checkpoints may be reused only with complete ancestry exclusion. No teammate source is copied, no teammate weights are used. All histories are causal, including feature normalizers and gate inputs; reset state on source-qualified sequence changes. No inference-time labels or meaningful sequence-ID assumption. Warmup rows update causal state even when unscored. A sequence's final distribution cannot normalize its early predictions.

The final deterministic Python3.11/one-CPU/16GB/20MB ZIP/60-minute limits come from the brief; no new rules question is being asserted. The roughly 36-minute incumbent measurement is a calibration, not a transferable guarantee. Research teachers can be larger; deployable students and complete-endpoint timings must be tested separately.

All costs below are proposed relative allocations, not authorizations. No fit, rental, upload or provider mutation was performed. The current local ceiling remains five CPUs, 20 decimal GB RAM, one GPU and five decimal GB observed VRAM. Model fits belong on admitted accelerators. Small streaming statistics are analysis; neural gate/head fitting still requires a GPU allocation. Root owns admission and an independent critic/grader owns evaluation.

## The reusable experiment skeleton

Let `E` be the fixed outer evaluated component set, `D` everything permitted outside E, `p0` the frozen full incumbent endpoint and `p_j` candidate predictions. A component includes every linked/source-qualified group, across supplied train and valid. Hash ordered membership and the source-qualified row keys before any join.

There are two different, valid levels of work:

1. **Cheap combination discovery using existing fold0 predictions.** Existing base models that excluded all 468 evaluation groups can be held fixed. Partition their 141 components into three deterministic bins, using a recorded hash seed and component IDs, independent of labels. For each bin, fit the meta-model on the other two bins and predict this bin. Pool all three predictions with the official metric. Use a single predeclared regularization recipe, or tune only inside each meta-training partition. The original base models exclude every meta-training and meta-evaluation group, so this construction requires no new base fits. All 468 groups remain exposed development evidence. Do not fit coefficients on all 468 and report their in-sample score as a gain. Do not use the other 1,405 valid groups merely because a full prediction array contains them: those groups fitted the bases.
2. **OOF teacher/residual training for a deployable recipe.** Freeze E and partition D into K component bins. For each k, train each base/teacher on D minus D_k, including all ancestors, and export predictions on D_k. Train the residual learner or student on these OOF targets from D. For evaluation, bases trained on all D predict E and the fitted second level predicts E. Normalizers, target scalers, teacher-selection coefficients, loss moments and any group-difficulty weights derive only from the applicable fitting partition. A teacher initialized from a model that saw D_k is not OOF for D_k. Likewise a fold0-trained teacher cannot simply stand in for every inner teacher.

The second level may be much cheaper than a full nested rerun, but the missing OOF bases are a real cost. Do the first construction now to decide whether that cost has value. A rotating meta-fit within E does not create targets valid for a student trained on D. An all-data teacher is eligible for final all-data training but supplies no held-out evidence on any of its training groups.

For frozen meta-hyperparameters, the three-bin construction gives a legitimate out-of-component exploratory prediction. Searching its pooled outcomes remains selection exposure. Repartitioning, new seeds and another agent do not erase that exposure. A later second exposed fold checks robustness; only genuinely new/prospective evidence can provide fresh confirmation.

Store one SSD prediction bank with raw FP32 predictions, source-qualified row keys, mask, group/component mapping, exact checkpoint/source/config/split hashes and complete ancestry. Mmap/chunk the arrays. A full valid prediction tensor has 1873×20000×2 FP32 entries: 299,680,000 payload bytes plus its actual format header. Do not load a bank of these into the one-GB support scope. Scored-only compact tables are useful for static analysis, but stateful neural corrections require the complete causal feature histories.

## A diagnostic that is closer to the goal than disagreement

For one target, let `y=clip(y_raw,±2)`, `z=clip(p0,±2)`, `w=abs(y)`, `W=sum(w)`, and weighted means/standard deviations be μy, μz, σy, σz. Write the pooled correlation as ρ. For nondegenerate variances and raw predictions away from the clipping boundary,

`∂ρ/∂p0_i = (w_i/W) 1{|p0_i|<2} [(y_i-μy)/(σy σz) - ρ(z_i-μz)/σz²]`.

This is a derivation for our metric, not a literature claim. For direction `d=p_j-p0`, the inner product with this gradient estimates its infinitesimal conditional value. Inspect it per target and component, alongside actual fixed finite blends. A model can be weaker alone and have a positive derivative; a decorrelated noise model can have a negative derivative. At saturated rows the first derivative can be zero even though a finite inward move helps, so the derivative is a diagnostic, never a universal rejection gate.

All label-dependent moments/weights used to **fit** a residual or gate must come from its fitting partition. Evaluation labels may compute diagnostics and grades only. Derivative selection itself consumes development exposure. Use a small fixed candidate list and report it all. Before implementing A6, the independent critic should check this derivative against finite differences on nonsaturated and saturated fixtures, including nonuniform weights, and verify that target construction and regression do not apply w twice.

## Card A1 — Shrunk static stacking as the first conditional-value instrument

**Family:** constrained combination geometry. **Priority:** first; prediction reuse only.

For target c fit `a_c` on meta-training rows:

`min_{a>=0, sum(a)=1} Σ_i w_ic (y_ic-Σ_j a_jc p_ijc)² / Σ_i w_ic + λ ||a_c-a0_c||²`.

`a0` is the frozen incumbent composition, padded by zero for new models. Fit in raw prediction space and run the official clip-after-combination evaluator; do not preclip each constituent silently. The WMSE objective is a stable surrogate, not an assertion that it exactly optimizes correlation. Start with at most four endpoint-level inputs and strong shrinkage. Rescale the penalty by the measured fitting-set covariance scale so λ has a documented meaning. Fix one λ before the first outer score; a later two-value λ comparison is another trial.

**Controls:** incumbent, fixed 0.05 and 0.15 candidate additions, a shared two-target coefficient vector, and target-specific coefficients. Count all coefficient alternatives in the selection budget. Initially prohibit negative coefficients; signed extrapolation is a separate bounded descendant, not an optimizer default. If calibration is needed, use a single fitting-derived per-target scale shared by candidate and control; arbitrary scale tuning against outer scores is another arm because clipping breaks scale invariance.

**Fit/serve and ancestry:** use the three-bin construction above. Serving evaluates the already-admitted bases and fixed coefficients; no labels or online adaptation. The arithmetic is tiny, but every newly included base has its full inference cost. Candidate/control must use the complete endpoint, not a warm parent alone.

**Prediction:** a complementary model earns stable nonzero weight and improves pooled meta-OOF WP. **Cheapest test:** the existing seed3/seed4 parent/student endpoints and one genuinely different available teacher, subject to identical exclusion. **Falsifier for this recipe:** apparent training gains vanish in meta-OOF predictions, or weights flip wildly between component folds and the fixed small addition is no better. A wide interval is unresolved, not a mechanism kill.

**Reuse/cost:** streaming moments and a very small constrained solve; minutes of bounded analysis are a planning class, not measured timing. Freeze coefficient output plus component contributions. **Descendants:** feed a stable static teacher to A7; test a gate only if conditional diagnostics suggest context-dependent relative errors.

**Source basis:** Breiman's stacked regressions uses cross-validation predictions and nonnegative least-squares coefficients. The simplex constraint, incumbent shrinkage, target-specific metric checks and protocol here are our choices. [Breiman 1996](https://doi.org/10.1023/A:1018046112532).

## Card A2 — Robust-loss teacher without throwing away the high-weight tails

**Family:** objective/target diversity. **Priority:** first new teacher pair.

Use an independently initialized, existing GRU training path with a fixed complete-component subset. Hold architecture, initialization, order, optimizer, schedule, precision, number of updates, normalizers and auxiliary losses identical. Change only the main supervised squared loss to Huber:

`h_δ(r)=0.5r²` if `|r|<=δ`, otherwise `δ(|r|-0.5δ)`; `L_H=Σ_i w_i h_δ(f_i-y_i)/Σ_i w_i`.

Start `δ=1` in the declared clipped-label units and `w=|clip(y_raw,±2)|`, exactly the same weighting in the WMSE control. Both first arms use clipped training labels. This controls whether bounding the *residual influence* helps while retaining the official tail weights. A raw-label Huber arm is a later 2×2 target/loss interaction; it is not hidden in the first Huber change. Preserve the closest-reference distillation branch separately.

**Fit/serve and ancestry:** main labels and any auxiliary teachers come only from the clean fitting closure. Serve the same GRU; the loss has no serving cost. Evaluate teacher-alone and fixed 0.10 addition to p0 before fitting A1. Robustness here means a hypothesis about useful influence, not proof that large labels are outliers.

**Prediction:** residual tails and gradients shrink, while the candidate's correction retains positive alignment with p0's metric gradient; it may lose standalone WP yet contribute. **Cheapest test:** one paired screen with equal complete-history examples and fixed checkpoint. **Falsifier:** Huber only destroys high-weight tail signal and provides neither blend value nor student gain. Report magnitude-binned contributions, using fitting-derived bin boundaries; never substitute those diagnostics for the global grade.

**Cost/reuse:** two small matched fits if no exact WMSE control exists; zero extra deployment operations. Reuse their exports for A1/A5/A7. **Descendants:** δ=0.5 versus 1 only after evidence, raw-target treatment, or a short robust first phase followed by the original loss under a matched schedule control. Activation×Huber is a later explicit interaction, not part of the first fit.

**Source basis:** Friedman derives stagewise fitting for least-squares and Huber losses. Our weighted recurrent-teacher recipe and its expected complementarity are untested inferences. [Friedman 2001](https://doi.org/10.1214/aos/1013203451).

## Card A3 — Optimizer and activation diversity with a narrow intervention surface

**Family:** optimization-induced representation. **Priority:** concurrent with A2 when a matched control exists.

First compare current audited whole-matrix Muon against AdamW on the same GRU, with fresh paired initialization and identical examples/update count, target and schedule shape. Preserve the actual Muon matrix partition, nonmatrix optimizer, update-scale multiplier and weight-decay convention in the recipe. Identical numerical LR is not a scientifically fair universal control across optimizers: predeclare each audited baseline LR and permit only a small training-partition calibration budget, counted equally. Report both update-matched quality and measured wall cost. No conclusion is licensed if one arm repeatedly skips AMP updates; fix or declare that treatment.

Then, only one activation descendant: replace an existing feed-forward encoder/readout activation with `SiLU(u)=u sigmoid(u)` versus its current activation, preserving GRU sigmoid gates and bounded tanh candidate state. If the matched architecture has no such nonlinear block, add the same block to both controls before changing activation. Replacing every recurrent tanh by an unbounded activation conflates recurrence stability and activation choice and loses fused-kernel comparability.

**Fit/serve and ancestry:** independent clean fitting closure; no teammate parameters. Optimization changes affect only training. The small front/readout activation may affect native serving implementation and must be benchmarked. The initial Muon/AdamW comparison can mix either teacher with p0 or distill them together; choose by conditional contribution.

**Prediction:** different update geometry produces a useful residual direction, with any change in matrix spectrum merely diagnostic. **Cheapest test:** reuse the already planned audited GRU512 AdamW/Muon pair when both genuine fits/exports exist; current profiles are not teacher results. If that cohort remains blocked, execute a matched smaller pair on a warm admitted lane rather than recreating the delivery stack. **Falsifier:** no conditional value under shrinkage and no compressed-student value at matched budget; a noisier trajectory alone is not diversity success.

**Cost/reuse:** optimizer pair costs two fits, or one incremental fit against an exact existing control. Activation costs one additional paired treatment after the winning control, not a 2×2×loss grid. A momentum-SGD arm is a defensible later optimizer control only if its convergence/gradient scale receives a comparable tuning budget; no performance prior is assigned. **Descendants:** A2×A3 only after single-factor evidence; teacher pair→A7.

**Source basis:** the Muon scaling paper identifies weight decay and per-parameter update scale as important. Its LLM efficiency results are not claimed for this recurrent task. [Liu et al. 2025](https://arxiv.org/abs/2502.16982).

## Card A4 — Explicit slow/fast state as a different teacher family

**Family:** history representation. **Priority:** exploration reserve; one distinct history arm before more seeds.

Construct causal memory channels from a small projection `v_t=U x_t` and three fixed time constants, initially τ∈{8,64,512} rows:

`m_t^(τ)=exp(-1/τ)m_(t-1)^(τ)+(1-exp(-1/τ))v_t`;

`h_t=GRU([x_t, v_t-m_t^(8), m_t^(8)-m_t^(64), m_t^(64)-m_t^(512)],h_(t-1))`.

This separates innovation and slow levels without future windows. Initialize memory consistently to zero at every sequence; optional bias correction `m_t/(1-a^(t+1))` is a separate predeclared preprocessing choice, not tuned on E. Keep the base normalization fixed and fitted outside E. Use a small projection, e.g.16 channels, and a width matched control receiving a same-size learned instantaneous projection. Compare equal recurrent width first, then report parameter/FLOP difference rather than pretending exact compute matching.

**Fit/serve and ancestry:** a new clean teacher, no target-derived online memory. All feature histories are processed even if gradients are truncated. TBPTT detachment is not a state reset. Serving maintains 48 memory values and the GRU state; teacher width may be 384/512 after a small screen, with student compression if too slow.

**Prediction:** residual correction depends on predictable slow/fast changes, especially after causal distribution changes, and supplies a signal absent from ordinary short-gradient training. **Cheapest test:** before a new large teacher, feed these deterministic memory channels into the small A6 correction network against an instantaneous-feature correction control. This tests whether the history representation has residual value, not whether a large augmented GRU will win. **Falsifier:** no reproducible correction from these channels and no benefit when context length is increased under matched fit exposure; a failed small screen can still leave a scale-dependent hypothesis unresolved.

**Cost/reuse:** small streaming feature state; a new training/export pass. A later LMU/HiPPO state is a materially different descendant requiring numerical/profile checks. Do not substitute an old weak TCN result for this experiment. Keep the existing TCN/attention predictions as direct blend controls when their ancestry is clean.

**Source basis:** LMU and HiPPO show causal history compression into structured memory states. The three exponential channels above are our simpler engineering hypothesis, not an implementation or replication of their polynomial-memory algorithms. [LMU](https://papers.nips.cc/paper_files/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html), [HiPPO v2](https://arxiv.org/abs/2008.07669v2).

## Card A5 — A small causal gate with the static solution as its default

**Family:** conditional mixture. **Priority:** only after prediction reuse establishes a plausible complementary pair.

For two expert endpoints use `p_t=p0_t+α_t(p1_t-p0_t)` with

`α_t=(1-κ)α0+κ sigmoid(b+βᵀz_t)`, initially `κ=0.25`.

`α0` is the fitting-derived A1 coefficient. Set b to reproduce α0 when β=0, with a documented floor for boundary coefficients. z contains at most eight standardized causal values: expert disagreement, their magnitudes, clipped saturation indicators, an EWMA of feature innovation magnitude, and sequence age truncated at a fixed scale. Compute current/past expert outputs normally. Do not give the first gate 112 raw features, full hidden states, sequence ID, or label-defined regimes. For each target use separate b and β but one fixed input definition. Add ridge shrinkage of β toward zero; no online label-based updates.

**Control:** exact static α0 with the same expert executions. Train with weighted squared loss first, grade official WP; a metric-objective gate is a separate treatment. Compare also one shared gate across targets. Fit on meta-training components under the common protocol; precompute causal z independently of labels. No gate may infer “this model won on this evaluation group” from a group lookup.

**Prediction:** predictable regions of relative competence improve meta-OOF WP without extreme rapidly fluctuating coefficients. **Cheapest test:** one linear-logistic gate on existing predictions and scalar causal features; if causal features are absent, begin with current-output features only, and record that narrower mechanism. **Falsifier:** constant gate matches it outside the meta-fit components, or coefficient changes follow fold/source identity rather than transferable current/past features.

**Cost/reuse:** tiny gate arithmetic, but both recurrent experts normally still run each row. Hard routing does **not** save recurrent compute if the skipped expert needs a continuously updated hidden state; do not advertise mixture sparsity as free speed. Static/learned mixtures that are too expensive become A7 teachers. **Descendants:** shallow16-unit MLP or a smooth gate with an EWMA state, only after the logistic gate; no tree-depth sweep. Shared trunk heads may give a cheaper gate but reduce diversity and need their own comparison.

**Source basis:** adaptive mixtures of experts learn which experts handle which cases. Frozen experts, causal inputs and shrinkage to a static endpoint are our adaptations. [Jacobs et al. 1991, author PDF](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf).

## Card A6 — One-stage causal residual boosting, metric gradient versus squared error

**Family:** functional correction. **Priority:** first second-level neural fit; do not build a deep boosting cascade.

Freeze the full base endpoint, not just one parent. Fit a small correction `r_θ(x_<=t,p0_<=t)` and serve `p=p0+ν r_θ`, with ν∈{0,0.05,0.15} predeclared. Start a32-unit feed-forward head on current features plus A4 memory channels; a GRU64 correction is a later memory-capacity descendant. Zero-initialize the final linear layer, retain fixed nonzero upstream initialization and regularize correction norm. This means the initial endpoint exactly matches the control.

Compare two target definitions using the same architecture/order/budget:

* **Squared-error target:** `r_MSE=y-p0`, with w-weighted regression.
* **Correlation target:** `r_corr=1{|p0|<2} [(σz/σy)(y-μy)-ρ(z-μz)]`, using pooled moments computed only on the correction fitting partition. Fit it with w-weighted regression. Up to a common positive factor this is the correlation gradient in the w-weighted function-space geometry; multiplying this target by w again would double-count the weighting. Store the moments and target version.

The MSE residual includes calibration corrections; correlation gradient projects away a local scale direction and focuses on angular error. This comparison directly tests which residual the official endpoint can use. At clipped rows the exact gradient vanishes. If saturation is appreciable, register a separate finite inward-correction target `y-z` on saturated rows, rather than silently replacing the derivative or using a straight-through estimator. Report all finite ν results; selecting ν requires an inner fit partition or another explicit selection trial.

**Fit/serve and ancestry:** use the three-bin meta procedure for the cheap screen, then the full D-OOF construction if promoting a deployable residual learner. Do not compute residuals from a base that fitted the same component, even when the residual net itself excludes E. Frozen hidden-state inputs are allowed only if extraction is causal and every base ancestor is clean; they do not waive OOF requirements. All moments are global within the fitting set, not per sequence or outer evaluation.

**Prediction:** metric-gradient corrections produce a better small-ν endpoint than MSE corrections when magnitude/calibration error dominates ordinary residuals. **Cheapest test:** two small matched correction heads on the same existing p0, leaving large teacher fits alone. **Falsifier:** no out-of-component gradient alignment or finite-step gain, or only training-residual gains survive. An in-sample residual training success is not evidence. One failed frozen-parent readout in H233 does not test this cross-fitted metric-gradient recipe.

**Cost/reuse:** existing p0 predictions, new small GPU fits and complete causal feature reads; estimated far below another full recurrent teacher only after profiling. Persist targets and correction outputs so ν checks need no reruns. **Descendants:** A4 memory, A2 robust correction, or A7 distillation of p0+νr. Avoid a second residual stage until stage1 has useful endpoint evidence and valid nested targets.

**Source basis:** functional-gradient boosting motivates fitting a loss gradient as a stagewise direction; the clipped weighted-correlation derivative and this recurrent protocol are our derived proposal. [Friedman 2001](https://doi.org/10.1214/aos/1013203451).

## Card A7 — Distill the best complete teacher system into a serving-sized student

**Family:** compression and teacher diversity. **Priority:** follow a useful static/gated/residual teacher, not arbitrary teacher count.

Let `T(x)` be a fixed clean teacher system selected using training-side OOF predictions. Its constituents may exceed submission runtime. Train a student with

`L=(1-λ) L_main(S,y)+λ Σ_i v_i(S_i-T_i)²/Σ_i v_i`, initially λ=0.5,

with v either uniform or the declared label-magnitude training weights, fixed before fitting. Preserve raw teacher amplitude and clipping order in the manifest; the known0.8125 teacher-mass versus normalization treatment is not “cleaned up.” Compare against (i) same student supervised only and (ii) same student distilled from p0 alone. These distinguish ordinary regularization from added teacher diversity. Use the same parent ancestry or fresh initialization for all three; do not give one student a stronger initializer.

**Fit/serve and ancestry:** for clean outer E, teachers and the selected combination fit outside E; the strict D-OOF target path above also guards teacher target memorization in student fit rows. If affordable only teachers trained on D and predicting D are available, label that as an explicit in-sample-distillation arm, never an OOF replication. E remains excluded from every teacher/combination/student ancestor. Full all-data targets are final-production artifacts after recipe choice, not validation.

**Prediction:** a diverse teacher system's conditional signal survives in the compact student beyond p0-only distillation. **Cheapest test:** one matched student pair, heterogeneous teacher versus p0-only teacher, using the same existing student recipe and one selected λ. If no supervised control exists, add it rather than assuming all distillation gain is diversity. **Falsifier:** teacher blend gain disappears when compressed at equal student cost; this can defer this width without refuting larger students or a two-student mix.

**Cost/reuse:** teacher exports amortized across λ/student descendants. Student ladder128→256→384 (adapt to the actual established architecture), research teacher384→512→768 only on admitted hardware. Profile each rung, and spend no large local CPU fits. Final options are one student, two differently targeted students, or incumbent plus one small correction, compared on complete CPU time/size/WP. Teacher standalone rank is not the filter.

**Descendants:** different teachers for the two targets with shared student trunk; heterogeneous two-student fixed pair versus same-teacher two-seed pair; A8 contrast targets. Shared trunk may create negative transfer, so target-specific heads alone are not assumed sufficient.

**Source basis:** ensemble knowledge can be compressed into a single deployable model. The cited work does not establish the effectiveness of our regression targets or weights. [Hinton, Vinyals and Dean 2015](https://arxiv.org/abs/1503.02531).

## Card A8 — Preserve the complementary contrast instead of washing it out

**Family:** contrast decomposition; unconventional transfer. **Priority:** a small A7 descendant when a weak teacher adds measurable conditional signal.

Suppose a large teacher `T1` adds a small stable contribution to anchor `T0`. On clean fitting predictions, fit only an affine projection `a+bT0` to T1, then define `d=T1-a-bT0`. Standardize d by its *fitting-set* weighted RMS s with a numerical floor. Train a shared student trunk with two heads `m_θ` and `d_θ`:

`L=L_main(m_θ,y)+λ0||m_θ-T0||²_v+λd||d_θ-d/s||²_v`.

Serve `S=m_θ+γ s d_θ`, where γ is fixed from an inner clean combination comparison; include γ=0 and a small positive value as controls. Compare to an ordinary student distilled from the same combined teacher with exactly the same trunk and training budget. Add an extra-head control using a redundant anchor target so a gain cannot be attributed merely to more supervision/parameters. Affine orthogonalization is a diagnostic/decomposition, not a guarantee of label relevance. The contrast must predict residual signal on held-out components.

**Fit/serve and ancestry:** calculate projection, s and combination weights only from training-side clean predictions; all source teachers/initializers exclude the evaluated component. At inference the student predicts both heads from causal features; it does not need either teacher. The common and contrast heads share the recurrent state, so inference can be close to one student, subject to native implementation measurement.

**Prediction:** explicitly balanced contrast supervision retains small useful distinctions that mean-target distillation ignores. **Cheapest test:** one ordinary versus contrast-target student pair for a teacher pair already shown useful; do not choose the contrast by highest variance or lowest correlation. **Falsifier:** it learns disagreement but not label-relevant residual, or the same-budget mean-target student matches it. Strongly unstable/near-zero contrast scale makes this particular pair unsuitable, not all factorized targets.

**Cost/reuse:** no new teacher fit, two student treatments and a few extra head weights. Reuse the complete teacher bank. **Descendants:** target-specific contrast, two contrasts only after one works, or an auxiliary contrast head used only in training and dropped at serving with a matched ablation. Do not whiten arbitrary large teacher banks or optimize a poorly conditioned covariance inverse.

**Random-library basis:** reuse the recorded seed20261008 draw, Markowitz1952 page5/offset143 from `research/ideas/claude_20261008_round/passages_selected.json`. The actual selected passage explains that a weighted sum's variance depends on covariances. The transfer is a design lens: inspect conditional directions and regularize combinations. Prediction labels, clipping, systematic bias and target-dependent weights make this unlike portfolio optimization. The contrast-preserving distillation recipe is my inference, not a claim from Markowitz.

## Card A9 — Complete-component perturbation and capped residual emphasis

**Family:** data/sampling diversity. **Priority:** reserve after loss/history, before many seed repeats.

Compare uniform component sampling with a fixed, tempered difficulty weighting. Obtain component difficulty `e_g` from out-of-component p0 residuals on the fitting population, not from p0's own training residuals. Define

`q_g ∝ n_g [ε+(e_g/median(e))^τ]`, with τ=0.5 and the bracket clipped to [0.5,2]; n_g is the group's eligible-row count if the declared baseline is row-weighted.

Sample **complete components/sequences**, process every row causally, and give the control the same number of row updates. Choose and document one of two scientifically different objectives: importance-correct by baseline_probability/q_g for variance-reduction sampling, or intentionally omit correction for a hard-component specialist. The first trial should be the specialist, compared with uniform sampling, because its proposed role is to complement p0. Difficulty weights are frozen offline and never used at inference. Do not borrow E residuals to determine training difficulty on D.

**Alternative cheapest data arm:** a fixed component bag using a recorded subset seed and the same total update budget, evaluated on the unchanged E. This tests coverage-induced diversity; it is not interchangeable with a new evaluation fold. A smaller bag repeated more often changes reuse/overfitting and must be reported. No row shuffling across histories or truncating histories to speed up the sample.

**Prediction:** a bounded hard-component specialist adds to p0 while uniform remains stronger elsewhere. **Cheapest test:** one uniform versus one capped specialist only if clean OOF difficulties already exist; otherwise the fixed bag is cheaper and avoids inventing OOF scores. **Falsifier:** it follows irreducible/noisy components and its conditional value fails outside sampled groups; reduce or defer the weighting recipe, do not infer that all data diversity is harmful.

**Cost/reuse:** one additional teacher fit against an exact control; same serving architecture. OOF production may dominate, so this is not a first-hour prerequisite. **Descendants:** A2 robust loss×difficulty weighting if hard groups appear heavy-tailed, or A7 specialist distillation. The uniform control protects against confusing oversampling with additional training steps.

## Minimal first matrix, with dependencies and honest cost accounting

These are recommended slots for root review, not launched jobs. “Existing” means terminal exports plus row/ancestry checks, not provider RUNNING or a successful profile.

| Order | Mechanism comparison | What can be reused | New fit requirement | First decision it resolves |
|---|---|---|---|---|
| 0 | A1 incumbent vs fixed additions vs shrunk static meta-OOF | Clean fold0 parents, H230/Z3 endpoints; one audited different-family export | No base fit; tiny statistics/coefficients | Which existing signals deserve any further training? |
| 1 | A5 logistic gate vs the exact static mixture | The same prediction bank | Tiny GPU gate fit; current-output-only version needs no feature re-export | Is relative expertise causally predictable? |
| 2 | A2 WMSE vs Huber | Existing trainer and warm parsed data; only an exact control can count as matched | One matched pair or one incremental treatment | Does bounded residual influence supply useful diversity? |
| 3 | A3 AdamW vs Muon | Planned512 cohort if actual fit/export lands; existing profiles only for cost | Matched smaller pair if needed | Is optimizer diversity useful after proper scale controls? |
| 4 | A6 MSE vs correlation-gradient correction | p0 outputs and causal feature stream | Two small matched correction fits | Is p0 missing an easily learned metric-relevant direction? |
| 5 | A4 slow/fast memory vs instantaneous correction | A6 pipeline; fixed history features | One extra correction treatment | Does a new memory representation add conditional signal? |
| 6 | A7 heterogeneous-teacher vs anchor-only student | Best useful teacher exports | Matched student pair, plus missing supervised control | Can the signal enter the CPU frontier? |
| 7 | A8 contrast-target vs mean-target student | Same teacher pair and student control | One extra student treatment plus redundant-head control as needed | Does compression erase complementary signal? |

Do slots0/1 while independent teacher comparisons run; slots4/5 need not wait for512/768 models. Activation and sampling are explicit reserve cards, not mandatory grid axes. A1/A5/A6 are different mechanisms but repeated coefficients/seeds/checkpoints are not separate “ideas per hour.” Count actual completed comparisons with terminal predictions and official grades.

For a small-data teacher screen, choose complete connected components by a fixed hash until at least256 sequences are included (record the actual overshoot); preserve all20,000 rows and original causal state. Use two complete passes as the initial budget, the same ordered examples and fixed terminal checkpoint in paired arms. Keep the entire fixed outer evaluation population. If this does not fit the measured lane envelope, reduce complete-component membership identically, not evaluation membership or history. Width/optimization fidelity is recorded; a screen can prioritize a large/history-dependent arm without serving as a universal kill gate.

Set a root-reviewed per-screen wall envelope after actual device profiling, with total time = verification/staging + data parse + fit + export + collection + grade. No claim is made that256-sequence/two-pass screens achieve5–6/hour. Full evaluation export may dominate. Warm cohorts should verify immutable data once, maintain mmap/parsed views and export only declared checkpoints. Physical SSD reads, page-cache reuse and in-process reuse are separate measurements. Do not copy37GB inputs to a paid worker before transfer feasibility is demonstrated.

After initial orthogonal evidence, allocate roughly half of useful new GPU time to the best one or two conditional contributors, a quarter to memory/activation/data reserve, and a quarter to scale/second-fold/student/serving checks. This is a proposed allocation policy, not a budget amendment. Keep existing admitted jobs intact and update the split when actual costs/results change. A positive imprecise effect can earn a targeted follow-up; do not spend the full24 hours generating variants without complete graded artifacts.

## Specific existing artifacts and reuse restrictions

* `reports/Z3_EXACT_FOLD0_INDEPENDENT_GRADE_R1.json` binds four full predictions and clean membership: candidate core `f0_z3_combo_s4_student_v2_core_identity_v2@ep1`, prediction SHA `d045c767343d7872a6eaa8ddf7ecf3402982a3d9a7df3cb10510dae4bd09e9d3`; candidate head `f0_z3_combo_s4_student_hft_v3@ep0`, SHA `462326e735d298661abd631d995ced444962de28f8316436ceb8454cd7463c5e`; control `c20261003_combo_s4_f0@ep3`, SHA `f1a3b07072a62dce620dc0c90e9568093367d0aed1afa4e9ec550866f94df12b`; control head `f0_z3_combo_s4_parent_hft_v2@ep0`, SHA `ffd97736e444fdb457e8af8e433860e59d5b0c2e5430ace86d6e464445271d84`. Reuse the declared0.5/0.5 endpoints as initial columns, rather than exposing all constituents as many free coefficients. Locate physical arrays via their producer receipts and verify hashes before analysis; I did not load them.
* `reports/H230_FOLD0_ENDPOINT_GRADE_20261008_R1.json` and `reports/H230_FOLD0_ENDPOINT_MEMBERSHIP_20261008_R2.json` identify the seed3 endpoint for the same proposed bank, conditional on the exact row/membership join. This task inspected their filenames, not their full raw prediction closure; do not treat that candidate as independently verified by this proposal.
* H272 S4 parent/hidden/multiplicative and matched GCorr predictions can diagnose representation-head diversity. The current state reports the ordinary GCorr endpoint ahead of those tested readouts. Do not rerun a static identical blend and count it as a new idea; A5/A6 require a genuinely different learned/cross-fit mechanism.
* `numerical_embedding_E540/E540_FIXED_PAIR_RESULT_20261008_R1.md` already reports linear0.6361859352, equal blend0.6349392976 and PLE0.6254095463. Reuse as a low-priority diagnostic only. These are one-epoch/subset/GRU128 predictions, not comparable full-scale optimization controls. A weaker embedding teacher is not automatically useful, nor does this one recipe refute the family.
* Current-state entries identify H232 scratchGRU384, H218 block attention and raw TCN exports. Their old fixed blend losses are counterevidence for those recipes. Admit to the reusable bank only after their producer receipts establish the same source-qualified exclusion and causal inference policy. Do not claim new usefulness without actual conditional evidence.
* H230 seed99 all-data parent/train/valid exports can support final-production targets after selection. They are **ineligible** for held-out fold0 stack/residual grades because all-data fitting includes the evaluation groups. The same applies to all-data seed9/99 students.
* Capacity512/768 profiles establish device timing/updates, not trained teacher quality. They cannot be used as missing OOF models. Extra exports from a clean existing checkpoint may be a legitimate cheap job; retraining a successful checkpoint after a consumer error is not the default.

## CPU frontier and promotion

Measure complete deterministic replay in the final native/Python interface on one CPU, including feature updates, recurrent state, coefficient/gate work, reset handling, package loading and output conversion. Use the same rows/order/thread settings for candidates and incumbent. Record cold initialization and steady state separately, and benchmark multiple long sequences rather than extrapolating a tiny warm loop. Final full-endpoint checks remain necessary; a short profile only predicts whether they are worth running.

Track `(official WP per target, total WP, CPU seconds, peak RAM, ZIP bytes, train/teacher cost)` as a Pareto frontier. A24-minute numerical subtraction from a36-minute historical run does not authorize adding a model. Prefer a practical runtime margin, proposed target≤50 minutes under representative full replay, then inspect hardware variability before claiming the60-minute bound. That margin is engineering advice, not an altered competition rule.

First serving candidates: frozen incumbent coefficients; one existing-trunk readout/gate; one GRU128/256 student from a large heterogeneous teacher; two students with different teacher targets only if measured incremental quality justifies their full cost. Reuse current feature computation and native kernels. Weight quantization/int16 packing is an explicit parity experiment; record official-score drift, saturation and determinism after packaging, not only tensor error. Do not silently quantize teacher outputs to save disk if they define a frozen training target.

For each admitted experiment the actor owes fixed command/source/data/split/ancestry, matched control, terminal checkpoint and predictions, phase costs and a handoff to an independent grader. The grader checks joins, original weight semantics, mask/clipping, candidate/control endpoints and complete component exclusions. Report paired component resampling as evaluation sampling conditional on the fitted models; seed variation and public transfer remain separate. Freeze a promising recipe for another exposed fold and paired training variation when the expected information value warrants it. No self-approval or new spending follows from this document.

The highest-value first deliverable is therefore a **small, ancestry-valid prediction bank plus one frozen meta-cross-fit comparison**, while a robust-loss teacher and an optimizer/history teacher are trained on already admitted warm lanes. It gives an actionable answer quickly, preserves the wider teacher search, and makes the eventual compact-student decision depend on measured conditional signal.
