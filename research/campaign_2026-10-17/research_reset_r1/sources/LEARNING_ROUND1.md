# Independent learning scout — round 1

Date: 2026-10-17
Scope: source scouting and future-facing proposals only. No training, fitting, inference, benchmark execution, submission, or data inspection was performed for this note. Sources are primary papers and author or official implementation repositories, checked on the date above. The machine-readable source register is in [learning_sources.json](learning_sources.json).

## Operating boundary

The challenge provides 112 numeric features per row, 20,000 ordered rows per independent sequence, 99 warm-up rows, and two targets. The callback must process every row, may use only the current and earlier inputs from that sequence, and must reset its state at sequence changes. Sequence IDs identify boundaries but are not features. Competition rules prohibit external data; public pretrained weights are allowed only when they contain no external market data. The eventual submission runs CPU-only, with a 20 MB ZIP and 60-minute total inference limit. The existing official score remains pooled global_wp_clip2_v1.

Every proposal below is for future research if the user later authorizes training. The current contract says training is not authorized. No score gain is asserted. Existing validation folds and prior panels are exposed; they cannot become fresh confirmation by changing seeds or names. A future quality comparison should use the active challenge protocol, paired fold/seed cells, the unchanged scorer, and a prospective external check where available.

## Prior-work crosswalk

I reviewed the October 17 model catalog and 14-draft inventory, the October 17 live-Git/Main audit, and its LIVE_GITHUB_CREATIVE_DELTA and LOCAL_HISTORY records.

- The published draft set already covers GRU width/depth/capacity, function-preserving growth, initial state, parent anchoring, field/group clocks, causal smoothing, grouped Transformer input encoding, a causal Transformer with output autoregression, Transformer-versus-GRU teacher family, teacher cohort expansion, pretrained teacher, trajectory averaging, causal innovation, and TPU parity. This note does not count those directions as new.
- The existing long Transformer/GRU proposal already describes causal Transformer teachers, an auxiliary future-observation loss, direct causal distillation, long context, teacher assistants, and a deployment-sized GRU route. The masked-current-feature pretraining idea below is a separate objective. The causal distillation card is a framing refinement and must be deduplicated against that proposal before any future work.
- Older creative/Main history contains a completed GDN comparison that was below its matched GRU point estimate on its own strict425 panel; its interval crossed zero. It also contains an accumulated-interval-memory arm that did not beat its matched instantaneous-memory control, and prior fixed detach-horizon screens with small or inconclusive effects. Those results are not directly comparable to the current Main challenge panels, but they are valid warnings against generic reruns. I therefore do not propose a vanilla GDN retry; the state-tap and randomized-gradient ideas below require an exact source/recipe crosswalk first.
- Historical heterogeneous GRU distillation had successful local comparisons, while newer Main alpha-zero versus alpha-0.9 continuation evidence favored the inherited KD recipe on one exposed warm-parent trajectory. Neither identifies the best teacher information set or establishes a new causal teacher family. Current teacher outputs and fitted-valid scores must retain their source and exposure labels.
- The public .66832 team snapshot and the short one-seed H512 screen describe different evidence surfaces. Neither is a prior for the effect sizes proposed here.

Status labels mean: **new** = no matching mechanism was found in the reviewed current inventory; **reopened** = an adjacent mechanism was tested or proposed before, so a source/recipe difference must be shown; **refine** = an explicit change to an existing draft; **conditional** = outside the active permission boundary unless its stated gate is resolved.

## Ranked idea cards

### 1. Mamba-3 multi-input/multi-output recurrent teacher — new

**Mechanism.** Encode each 112-feature row as one numeric token and process the complete sequence with a causal selective state-space layer. Compare the paper's single-input/single-output recurrence with its multi-input/multi-output read/write form at matched state size. The MIMO form is the point of interest: several projected feature channels can write to and read from a shared recurrent memory, while complex-valued state updates can represent phase-like temporal structure. Reset every recurrent state at sequence boundaries. Do not add sequence identity, target history, or external series.

**Evidence and assumption.** Mamba-3 reports long-context retrieval and state-tracking results and includes authors' implementation in state-spaces/mamba [S1]. The earlier SSD/Mamba-2 work gives the parallel-training and recurrent-inference connection [S2]. The evidence is language/retrieval evidence; the assumption is that a small row-token model can preserve useful market-stream history better than the current GRU families.

**Prior status.** New in the October 17 drafts; adjacent to the older GDN experiment. A generic SSM result is not a reason to repeat the old GDN recipe.

**Cheapest no-fit diagnostic.** Before model training, audit the actual author implementation and dependency path, then specify a deterministic synthetic stream test: future-row perturbations must leave earlier outputs unchanged, state must reset exactly, and whole-sequence versus stepwise outputs must agree within a frozen tolerance. This diagnostic has not been run.

**Falsifier.** Under a matched supervised recipe, neither the causal teacher endpoint nor its frozen-GRU student improves the complete official endpoint or its predeclared fixed mixture over the matched GRU/Transformer control across paired cells.

**Cost and submission route.** High teacher training and kernel-integration cost; linear-in-sequence recurrent inference, but the released research kernels are accelerator-oriented. Treat Mamba-3 as a large teacher. Export causal per-row outputs on permitted fitting data, train an existing CPU-sized GRU student on those outputs plus labels, and qualify the student ZIP separately for time, size, and deterministic reset.

### 2. xLSTM with separate scalar and matrix-memory channels — new

**Mechanism.** Give a numeric row encoder to an xLSTM stack with both scalar-memory LSTM blocks and matrix-memory LSTM blocks. The scalar path can preserve compact gated state; the matrix path can store feature associations and update them with a covariance-style rule. Compare an sLSTM-only, mLSTM-only, and fixed mixed block schedule at equal parameter or compute budget. Use current/past rows only and reset all cell state per sequence.

**Evidence and assumption.** The xLSTM paper specifies exponential gates, stabilization, scalar memory, and a parallelizable matrix-memory cell; NX-AI publishes the authors' code [S4]. The paper's evidence is language modeling, not a two-target continuous prediction task. The transfer assumption is that matrix memory may retain cross-feature relations over a 20,000-row stream that a narrow GRU compresses away.

**Prior status.** New relative to the current 14 drafts; adjacent to ordinary GRU width/depth and minGRU work. The experiment must compare memory update structure, not merely hidden width.

**Cheapest no-fit diagnostic.** Inspect state-size and sequential scan semantics, then run synthetic impulse/association tests for reset and numerical finiteness before binding any real data. No test was run here.

**Falsifier.** At a matched training budget, the mixed xLSTM teacher fails to improve either its causal endpoint or its CPU-GRU student, or the learned matrix memory has no measurable value over the matched sLSTM/GRU controls.

**Cost and submission route.** Medium-high to high GPU training; likely high serving-engineering cost because optimized kernels and matrix state are not the current CPU path. Use it as a research teacher, distill into a current GRU, and retain xLSTM itself only if actual CPU packaging meets the competition envelope.

### 3. Griffin-style local-attention plus recurrent stack — new

**Mechanism.** Alternate short causal local-attention blocks with gated linear recurrent blocks. Local attention reads recent row tokens to capture fast quote/trade interactions; recurrent blocks carry a compact state across the rest of the 20,000-step sequence. A 128- or 256-row attention window is only a starting profile, not a quality assumption. This differs from the existing full causal Transformer proposal: global context comes from recurrence, not a growing attention cache.

**Evidence and assumption.** Griffin defines a hybrid of gated recurrence and local attention; Google's RecurrentGemma repository is based on Griffin [S5]. Its paper reports language-model efficiency and length extrapolation, not competition performance. The assumption is that short local neighborhoods and a recurrent summary complement one another in a feature stream with both fast and slow dynamics.

**Prior status.** New as an architecture family; it must be compared against the already proposed full causal Transformer and existing GRU, not presented as an extension already measured.

**Cheapest no-fit diagnostic.** Build only a shape/state cost estimate and a synthetic parity/reset test for recurrent and local-attention caches; confirm the callback can update all state with current-row information only. Not performed.

**Falsifier.** At matched compute, the hybrid's local-attention branch adds no conditional value over its recurrence-only control, or the final distilled GRU does not improve the fixed served endpoint.

**Cost and submission route.** High teacher training and medium-high implementation cost. The stateful inference path is smaller than full attention but may still be awkward on CPU. Distill the complete causal hybrid's outputs into a GRU and measure the student's quality and final ZIP runtime.

### 4. Dyadic hidden-state taps as an explicit long-memory bank — reopened

**Mechanism.** Keep an ordinary causal GRU transition, but retain compressed snapshots of its hidden state at exponentially spaced ages, for example 64, 512, and 4,096 rows. A small learned gate combines the current hidden state with those stored snapshots for the output and next-state proposal. This makes old state directly addressable rather than hoping the current state preserves every useful detail. Snapshot timestamps are internal offsets only; no sequence ID or future row is used.

**Evidence and assumption.** Titans motivates an explicit long-term memory alongside a short-term attention context [S9], but does not establish this tap design. The assumption is that a few sparse old-state reads help when useful dependencies are much longer than a normal training chunk.

**Prior status.** Reopened at the broad memory level: old creative history's accumulated interval-memory version did not beat its matched instantaneous control, and current work already has initializer/group-clock/GRU-growth proposals. This tap design is different only if it stores direct, non-overlapping compressed state snapshots. Crosswalk the old implementation before spending on it.

**Cheapest no-fit diagnostic.** On synthetic streams, inject distinct signals at the proposed tap ages and verify that only the intended snapshots affect the readout; verify reset and state-size bounds. Do not infer quality from that test. Not run.

**Falsifier.** The tap gates collapse to zero, taps duplicate the current GRU state, or a matched paired comparison shows no improvement in the official endpoint or student blend.

**Cost and submission route.** Medium training cost; low-medium added CPU memory and projection cost if snapshot widths are small. This can itself be a compact student, or accept a larger teacher's causal outputs and then be compared with the plain GRU endpoint. Count snapshot buffers in the actual package.

### 5. Masked current-row reconstruction pretraining — new

**Mechanism.** Before supervised fitting, randomly mask contiguous feature groups of a current row, such as one instrument's price-like, volume-like, or trade fields, and reconstruct only the masked values from the remaining same-row fields plus the causal recurrent state. Use the provided train inputs only; never mask or expose future rows to a prediction at time t. Discard the reconstruction head and fine-tune the same temporal backbone on the two targets. This is a representation objective, not the already proposed future-observation auxiliary head.

**Evidence and assumption.** MOMENT uses masked patch reconstruction for time-series pretraining and releases its research code [S6]. Its published pretraining used a public multi-domain corpus, which is not an allowed source here. The transferable claim is only that raw-series reconstruction is a viable self-supervised objective; whether masked current-feature reconstruction helps this task is unknown.

**Prior status.** New objective; it is distinct from the current grouped-input draft and from the Transformer proposal's future-delta auxiliary loss.

**Cheapest no-fit diagnostic.** Define masking so the target feature cannot leak through a duplicate or normalization path; use a synthetic known-copy test and source-level data lineage audit. A future data-only screen could measure masked-feature predictability without fitting a target model. None was run.

**Falsifier.** The pretraining objective is solved mainly by same-row redundancy, improves only its own reconstruction loss, or does not improve the supervised model/student after matched fitting.

**Cost and submission route.** Extra pretraining pass adds medium-high cost over a large dataset; reconstruction targets need no extra labels or external market data. Pretrain a large teacher or deployment-sized GRU, then fine-tune and, if needed, distill teacher outputs to a CPU student. Retain all train/eval split boundaries in pretraining.

### 6. Randomized long-tail gradient horizon for recurrent training — reopened

**Mechanism.** Keep forward state continuous through each complete sequence, but randomize backpropagation truncation spans over a frozen short/medium/long schedule instead of detaching at one fixed span. A principled ARTBP-style arm uses sampled truncation lengths and the prescribed gradient compensation; a simpler mixed-span arm must be labeled biased and analyzed separately. Hold optimizer-update count, sampled rows, and checkpoint rule fixed. Do not use later labels as features.

**Evidence and assumption.** ARTBP formalizes variable truncation with compensation for unbiased gradient estimates and tests long dependencies on synthetic and language tasks [S12]. This does not prove that a high-variance estimator will work on 20k numeric sequences. The hypothesis is that fixed truncation boundaries under-train rare longer dependencies.

**Prior status.** Reopened. The project already explored fixed detach horizons, including a 500-versus-2,000-row comparison and an older truncated-BPTT proposal; reports were small or inconclusive. This is worth considering only as a genuinely different randomized estimator, not a renamed detach-length sweep.

**Cheapest no-fit diagnostic.** Use a tiny deterministic recurrence with analytically known gradients to verify truncation probabilities, compensation factors, and reset semantics; estimate gradient variance over synthetic sequences. No local implementation or test was run.

**Falsifier.** The estimator has unstable/high-variance updates, fails the analytic gradient check, or the trained endpoint is no better than the matched fixed-horizon control at equal compute.

**Cost and submission route.** Medium-high training implementation cost; ARTBP compensation may increase gradient variance, while long spans increase memory. The trained backbone can be exported directly if small enough or distilled into the CPU GRU. Log effective gradient horizon rather than treating chunk length as a proxy.

### 7. Context-matched causal teacher distillation — refine; noncausal arm is conditional

**Mechanism.** Compare label-only student training against a student that imitates a frozen causal teacher at each same row, using the teacher's own free-running causal state and exactly the inputs available to the student. Freeze teacher targets on fitting rows with source ancestry and precision. Evaluate student quality, not teacher score alone. A bidirectional teacher could be an informative privileged-information comparison only if the rules and current no-oracle-future-input instruction explicitly permit future training inputs to a teacher; the student and every evaluation prediction would remain strictly causal. I consider that branch conditional and not currently eligible.

**Evidence and assumption.** The existing teacher-family and causal Transformer drafts already cover teacher architecture and direct GRU distillation. A 2026 video-distillation paper finds that mismatching teacher and student temporal context can create a causal-supervision mismatch and proposes context-matched teacher supervision [S10]. Video-generation results do not establish numeric prediction gains; the transferable point is to match information sets. The local rules require a causal test-time callback, and the current task instruction forbids oracle future inputs, so do not infer authorization for a bidirectional training teacher from the paper.

**Prior status.** Refine/conditional, not a new general distillation idea. It can survive deduplication only as a precise teacher-context and eligibility test.

**Cheapest no-fit diagnostic.** Audit the existing teacher target producer for causal masks, free-running versus teacher-forced state, boundary reset, row alignment, and fitting-only lineage. A deterministic future-perturbation test must not change a causal teacher output at earlier rows. Not performed.

**Falsifier.** Causal student KD fails to beat the matched label-only or current-teacher student; if a future-aware teacher is later approved, its distilled student must be compared on exactly the same causal evaluation inputs and must justify the added leakage/eligibility complexity.

**Cost and submission route.** Causal teacher export and student fit are medium-high; optional noncausal teacher adds a full teacher and a rule audit. Distill only per-row numeric outputs and deploy a causal GRU student. Do not deploy a teacher that needs future rows.

### 8. Prefix-only no-gradient adaptation of feature scales — new, rule-compatible candidate

**Mechanism.** Maintain per-feature or per-semantic-group causal running location and scale for the current sequence. Normalize the current row using statistics from earlier rows, then update the statistics after producing the current prediction. Initialize/reset all accumulators at each sequence; keep static affine parameters learned from fitting data. This is streaming normalization, not full-sequence instance normalization.

**Evidence and assumption.** RevIN studies reversible instance normalization against temporal distribution shift and releases the authors' code [S11]. The common RevIN setup uses statistics for an input instance/window; using statistics from rows after t would violate this challenge's causal contract. The assumption is that a prefix-only scale estimate can reduce within-sequence feature drift without suppressing useful level information.

**Prior status.** New mechanism relative to the 14 drafts, though adjacent to causal smoothing and field-age proposals. It must not reuse full-sequence or future-window statistics.

**Cheapest no-fit diagnostic.** On supplied inputs, future-row perturbation should leave already-normalized earlier rows unchanged; verify cold-start behavior through the 99-row warm-up, finite variance floors, and reset at every sequence boundary. No EDA or test was run here.

**Falsifier.** Prefix scales are unstable on warm-up, erase level/scale signals needed for the target, or a paired full endpoint is worse than the static normalization control.

**Cost and submission route.** Low implementation and CPU cost; no second network pass. Apply the identical streaming transform during training, teacher export, and deployment. No compression is needed; keep the accumulators and affine parameters in the CPU GRU callback state.

### 9. Feature-only test-time fast memory — conditional but causal

**Mechanism.** Add a small per-sequence fast-weight state that updates online from a self-supervised reconstruction loss on the current feature row or a masked subset of its fields. The update must use only provided current/past features, never targets, score masks, or sequence ID as an input. Reset fast weights and optimizer state on every sequence. A low-rank linear adapter or tiny MLP is the serving candidate; use the ordinary recurrent model for prediction after the feature-only update.

**Evidence and assumption.** TTT layers define hidden state as a small model updated through self-supervised learning on the stream [S8]; Titans studies learned test-time memory with surprise-gated updates [S9]. Their reported scale, hardware, and language/long-context results do not show this would meet the CPU budget or help these labels. Under the local callback/rules, feature-only state updates appear compatible with current/past input and no external data, but no explicit contest ruling on online weight updates was found. Treat that as an eligibility check before any implementation.

**Prior status.** New runtime-state mechanism; related to stateful recurrent models, not an existing training-time test-time adaptation experiment.

**Cheapest no-fit diagnostic.** With synthetic data, verify current-row update order, deterministic repeated calls, update/reset isolation, absence of target/score fields in the adapter interface, and bounded memory/time per row. Do not use real test labels. Not run.

**Falsifier.** The fast learner overfits its reconstruction objective without changing useful causal features, produces order-sensitive nondeterminism, or misses CPU runtime limits; its value must ultimately be shown by the complete endpoint, not reconstruction loss.

**Cost and submission route.** High deployment risk from per-row gradient updates; low-rank state may still add substantial latency over 39.8 million callbacks. Prefer a large causal TTT teacher only if it adds measurable student value; distill its causal outputs into the CPU GRU. A direct adaptive model is eligible only after deterministic runtime qualification.

### 10. Sequence-group robust objective using only supplied training groups — new

**Mechanism.** During supervised fitting, form training-only groups from source-qualified sequence components or from causally defined activity/volatility strata. Optimize the usual row-level objective plus a bounded CVaR/worst-group term, with the group membership used only for the training objective and never passed as an input. Freeze any unsupervised regime boundaries using fitting data alone. Keep the official pooled score unchanged.

**Evidence and assumption.** Group-DRO/CVaR optimization provides algorithms for worst-group risk [S14]. That literature does not choose a meaningful grouping for anonymous order-book features, and equalizing group losses may conflict with the official pooled weighted correlation. The assumption is that a modest robustness term can reduce failures on unusual sequences without sacrificing global WP.

**Prior status.** New objective in the current inventory. Do not confuse it with historical train-versus-valid adversarial validation: a classifier that separates the supplied train and valid sets did not measure hidden-test shift and does not authorize feature or label leakage.

**Cheapest no-fit diagnostic.** Before model changes, compute group sizes and target-weight mass from fitting rows only; show that the chosen groups are stable, non-empty, and do not rely on seq_ix as a predictive feature. Existing train/valid labels remain exposed for evaluation; no diagnostic was run here.

**Falsifier.** Group weights collapse onto tiny groups, the robust term degrades global WP, or apparent gains disappear when scoring with the exact pooled official metric across paired folds/seeds.

**Cost and submission route.** Low-medium optimization overhead but medium analysis burden because the official score is nonlinear and pooled. Train either a teacher or student directly under the robust objective; export its causal outputs and distill if the objective's best model exceeds CPU limits.

### 11. Sharpness-aware updates for a sequence model — new optimizer arm

**Mechanism.** On an otherwise matched GRU or causal Transformer recipe, use a SAM-style update: compute the gradient at current parameters, take the gradient at a small adversarial parameter perturbation, then update from that second gradient. Keep batch, loss, fitting rows, update count, and decay fixed; compare actual sequences and seeds. The goal is lower sensitivity to fitting examples, not a change in architecture.

**Evidence and assumption.** SAMformer reports that sharpness-aware optimization improved its time-series forecasting setup and reduced seed sensitivity on its benchmark collection; the authors release code [S7]. The model/datasets and objective differ from this task, so its reported deltas are not transferable. The assumption is that our large sequence models may overfit the limited number of independent train sequences.

**Prior status.** New optimizer in the reviewed draft inventory. Existing AdamW/Muon histories are not SAM comparisons.

**Cheapest no-fit diagnostic.** Check that the intended implementation keeps forward/backward stochasticity, clipping, optimizer state, and perturbation scale paired; profile a synthetic one-update shape to estimate the added pass before any quality allocation. Not performed.

**Falsifier.** SAM does not improve the official endpoint or paired-seed stability over the same optimizer/update budget, or its extra pass reduces completed scientific endpoints enough to make it a worse allocation.

**Cost and submission route.** Approximately a second gradient pass per update is an engineering expectation, not a measured local factor. It affects training cost only. If a SAM-trained teacher wins, distill into the CPU GRU; if the GRU itself wins, it needs no extra serving state.

### 12. Process-control change-point gating for sequence memory — new cross-domain transfer

**Mechanism.** Borrow online change-point detection from process monitoring. At each row, form a past-prediction innovation for the input features, for example a feature-only one-step reconstruction residual, update a causal run-length posterior, and use that posterior to shorten the effective memory decay or blend to a fresh fast state after a likely regime change. The structural correspondence is explicit: both process-control telemetry and this challenge are ordered multivariate sensor streams with potentially changing latent operating regimes; the algorithm detects a switch from current observation surprise without labels. The detector reads no target, seq_ix value beyond the required boundary reset, or future row.

**Evidence and assumption.** Bayesian Online Changepoint Detection gives a streaming posterior over time since the last change under a piecewise-generative model [S13]. Its exactness depends on the model assumptions; the anonymous feature stream may not have clean iid segments. The assumption that detected feature-distribution changes align with target-predictive changes is unverified.

**Prior status.** New transfer proposal. Historical event-rate and event-state features exist, but they are not a tested change-point-driven state reset; the old train/valid discriminator is not evidence of within-sequence changes.

**Cheapest no-fit diagnostic.** Simulate stable and switched synthetic streams with known change times; measure delay and false resets, then verify causal invariance and state reset. A later descriptive EDA may compare run-length estimates to feature-only autocorrelation/activity changes, but no such analysis was done.

**Falsifier.** The detector fires on ordinary noise, reacts too slowly to changes, resets away useful memory, or fails to improve the final official endpoint over a matched ungated model.

**Cost and submission route.** Low-medium state and CPU overhead for a small detector; medium risk from thresholds/hazard tuning. Train a larger gated teacher if useful, then distill its causal outputs into a GRU with an explicit low-cost state gate. Any detector thresholds must be frozen from fitting data, not selected on hidden labels or public feedback.

## Source register

The structured fields, exact publication dates, primary URLs, author-code URLs, verified claims, and source limits are in [learning_sources.json](learning_sources.json).

- [S1] Mamba-3 (2026 ICLR) and its authors' state-spaces/mamba implementation: complex state updates and MIMO recurrence; reported language retrieval/state-tracking evidence only.
- [S2] Mamba-2 / Structured State Space Duality (2024): recurrent/attention duality and parallel training; language-modeling evidence only.
- [S3] Gated Delta Networks (2025 ICLR): gated forgetting plus targeted delta writes; the old local GDN result was negative at its point estimate, so no generic rerun.
- [S4] xLSTM (2024 NeurIPS) and NX-AI code: scalar and matrix memory variants; language-modeling evidence only.
- [S5] Griffin (2024) and Google DeepMind RecurrentGemma: gated recurrent/local-attention hybrid; language evidence only.
- [S6] MOMENT (2024 ICML) and author code: masked-patch reconstruction pretraining; its public multi-domain pretraining corpus is not allowed for this competition.
- [S7] SAMformer (2024 ICML) and author code: sharpness-aware optimizer applied to multivariate time series; benchmark/objective differ.
- [S8] TTT layers (2024) and official JAX code: self-supervised updates to expressive hidden state; hardware and task differ.
- [S9] Titans (2025 NeurIPS): test-time neural memory; no project-specific serving or quality evidence.
- [S10] Context-Matched Distillation (2026 preprint) and NVIDIA code: teacher/student context matching for causal video generation; numeric forecasting transfer is untested.
- [S11] RevIN (2022 ICLR) and author code: instance normalization against time-series distribution shift; full-window statistics require causal adaptation here.
- [S12] ARTBP (2017): unbiased variable truncation with compensation; variance and applicability need measurement.
- [S13] Bayesian Online Changepoint Detection (2007): exact online run-length inference under its generative assumptions; no learned market-regime result.
- [S14] Group-DRO/CVaR algorithms (2022): robust risk over named groups; suitable groups and metric transfer remain open.
