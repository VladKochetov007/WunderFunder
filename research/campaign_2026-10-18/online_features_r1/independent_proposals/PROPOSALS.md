# Fresh-context scout: online feature and model-transfer hypotheses

Created 2026-10-18 (UTC). This is a source-grounded proposal packet, not an experiment result. The first-proposal receipt records hashes before any competing campaign proposals or current actor path were inspected. I read only the assigned skills, the current online-features contract and data audit, general source indexes, primary sources/code listed in sources.json, and the recorded random library passage. No data payload, predictions, checkpoint, model code, or other agent proposal was read. No fit or implementation was run.

## Contract applied

The callback receives seq_ix, step_in_seq, need_prediction, and 112 current feature values. It sees neither targets nor evaluator masks. It processes every row in sequence order, including warmup steps 0–98; it predicts from step 99 onward; state resets at every sequence boundary; each answer may depend only on the current and earlier inputs from that same sequence. The 2 targets are anonymous future-movement indicators. Keep the official global_wp_clip2_v1 objective and its fixed prediction/scored masks, ±2 clipping, and original clipped-label magnitude weights unchanged in offline training and evaluation. Never use test labels, masks, future rows, sequence IDs as ordering signals, or state carried from a different sequence.

The audit identifies one known component-clean F0 reference, but its evaluation panel is repeatedly exposed. All four folds have prior exposure; any follow-up needs clean source-qualified fit, teacher, initializer and component ancestry and must be described as exposed robustness evidence. The eventual scientific comparison follows the project’s paired fold/seed protocol; a cheap screen is only a screen. Current local limits are 2 aggregate CPU cores, one GPU worker with 5 GB observed VRAM and 5 GB task RAM, physical scratch at /home/vlad/development/wunder/research_tmp, no new paid compute and no B300 execution in this assignment. Final serving remains 1 CPU, 60 minutes, 20 MB. Larger teacher work is a proposal/handoff only.

The participant reports supplied in the task are unverified leads, not evidence of Challenge 2 score or transfer. The six cards below are distinct mechanism families, not a requirement to run all six in one wave. The first two are intentionally cheap, input-only mechanisms; the last two ask whether training-time representation/temporal priors transfer to a compact recurrent endpoint.

## P1 — Warmup-fingerprint gate over fixed experts
**Lens:** factor the case space; change the search unit from a whole model to a sequence-level routing rule. **Family:** conditional ensemble allocation.

**Source component and analogy.** Jacobs et al.’s supervised mixture of local experts uses an input-dependent gate to allocate cases among specialist predictors. The historical experiment learns subdivisions on vowel-recognition labels; it does not establish a time-series or competition gain. The randomly selected VPIN passage below is a weak analogy for conservative allocation under uncertain specialists, not evidence about these anonymous targets or the official symmetric clipped-error objective.

**Intervention.** If at least two eligible source-qualified endpoints have cached predictions, fit a tiny sequence-level gate from summaries of only warmup rows 0–98 (for example, robust center/scale and a few channel-family summaries) to a small set of expert weights. Train its signal only from training-sequence out-of-fold expert predictions and training labels/masks scored by the official metric. Use strong shrinkage toward the predeclared fixed blend and constrain weights to a simplex. At test, compute the gate once from those 99 observable rows, hold its weights fixed for that sequence, and reset at the next seq_ix. The gate must never read that test sequence’s target errors or evaluator mask.

**Assumptions and prediction.** There are stable, observable input-profile differences between sequences and the existing experts have different conditional error profiles. If so, the held-out component score of the gated blend should exceed its matched fixed blend while gate weights vary reproducibly with warmup fingerprints. The gate can instead overfit noisy per-sequence losses; shrinkage is central to the hypothesis.

**Cheapest informative test.** On cached OOF predictions from eligible fit-only experts, fit only a low-capacity ridge/softmax gate, then score it against the same fixed blend on an outer component-excluded panel. Report pooled official WP and fold/seed exposure; use per-sequence expert errors only as evaluator diagnostics. If there are not multiple eligible experts with clean OOF predictions, do not fabricate this comparison.

**Falsifier.** The gate does not beat the matched fixed blend across clean paired cells, or its expert weights are unstable across folds/seeds and do not predict conditional expert usefulness. One noisy exposed panel is a screen, not universal evidence against conditional ensembles.

**Cost and artifact.** Very low inference cost: one small gate call per sequence after warmup, with no per-row gate overhead; likely a few KB plus gate-training metadata. Preserve the gate-fit sequence manifest, OOF prediction ancestry, frozen coefficients, fixed-blend comparator and evaluator receipt.

## P2 — Input-only innovation clock (unconventional cheap heuristic)
**Lens:** instrument hidden regime changes; change time from ordinal position to causal surprise. **Family:** online feature state / change detection.

**Source component and analogy.** Adams and MacKay’s Bayesian online change-point detector maintains a posterior over current run length from sequential observation likelihoods and a hazard model. Its feedback is the observed input stream, not task labels. Here only the run-length/surprise idea transfers; there is no basis for inferring market events or an exchange clock from the anonymous channels.

**Intervention.** Add one deterministic, target-free scalar state to the causal feature stream. Maintain reset-on-sequence diagonal fast/slow predictors for the 112 inputs; at row t compute a bounded standardized innovation summary from x_t versus statistics learned from rows before t, update a cheap CUSUM or small truncated run-length score, and append that scalar to the predictor input. Use no full covariance inverse and no target-dependent online update. Fit the supervised endpoint with this same causal statistic and official training objective. This tests a transition/surprise clock, not another fixed set of rolling w3/w13/w21 means.

**Assumptions and prediction.** Some errors may arise when the observed feature process changes faster than the static recurrent state can track. If the innovation clock helps, held-out official WP should improve in a predeclared late/transition subset without degrading pooled WP, and the scalar should be causal and reset correctly.

**Cheapest informative test.** First generate the scalar on bounded cached feature streams, check prefix/future-perturbation invariance and reset behavior, and inspect its distribution against step position. Then do one matched small supervised endpoint screen if the trace is nondegenerate. Keep all target/mask access inside the offline fit/evaluator.

**Falsifier.** The feature-only statistic is nearly constant, mainly encodes step position, violates causal/reset invariants, or a matched endpoint shows no stable official-WP benefit. No target-free unsupervised statistic is itself evidence of a quality gain.

**Cost and artifact.** O(112) scalar arithmetic and O(112) state per row, constant-size serving state and one added input; must still be profiled in the actual callback. Save the exact recurrence, numerical bounds, seed/config, invariance receipt and matched endpoint.

## P3 — Cross-fitted residual head on top of a frozen recurrent endpoint
**Lens:** re-encode the prediction as baseline plus a small correction; invert the question from “build another full model” to “is the baseline’s residual predictable?” **Family:** residual correction / conditional bias reduction.

**Source component and analogy.** N-BEATS’ doubly residual stack feeds a backcast residual to later blocks and sums partial forecasts. Its paper trains point forecasts on M3/M4/Tourism-type series; its result does not transfer automatically to multivariate anonymous indicators or this metric. The proposal borrows only staged residual modeling, not the full N-BEATS architecture or its code (the ServiceNow repository is marked CC BY-NC).

**Intervention.** Freeze an eligible base GRU and train a low-capacity two-target correction head on causal current-row and lag-difference features. Train against base errors generated out of fold for the sequences used to fit the head, so the head does not learn the parent’s in-sample optimism. Optimize the final combined prediction with the unchanged official objective; apply clipping only in the official metric path. Keep the head sparse/small enough to fit the package and measured runtime budget.

**Assumptions and prediction.** Baseline residuals have a repeatable feature-conditional component beyond the recurrent representation. Improvement should appear in matched component-excluded pooled WP and survive fold/seed variation; the correction’s contribution should be small and interpretable enough to audit.

**Cheapest informative test.** Use cached out-of-fold base predictions and a ridge correction on a predeclared compact feature set; compare base and base-plus-head on the same outer component panel. Do not compute correction targets from in-sample base predictions or tune features from evaluation labels.

**Falsifier.** The correction’s apparent gain disappears with OOF parent predictions, or the correction adds unstable tail error / misses package or serving limits. In-sample residual fit alone is not a successful screen.

**Cost and artifact.** One small fit and O(K) arithmetic per row for K compact inputs; likely low parameter and package cost but actual callback runtime is mandatory. Preserve OOF prediction lineage, correction features, fit manifest, output clipping path and matched score receipt.

## P4 — Fit-only masked input pretraining for the eventual causal GRU
**Lens:** relax the supervised-label bottleneck; transfer a representation objective from masked time-series modeling. **Family:** self-supervised representation pretraining.

**Source component and analogy.** SimMTM learns from masked input reconstruction and uses relations among multiple masked views to reconstruct time-series structure. Its reported feedback is input reconstruction during pretraining, followed by forecasting/classification fine-tuning and evaluation; it does not report this competition’s weighted indicator metric. The official code is available, but this proposal transfers the objective idea only.

**Intervention.** Pretrain a small causal GRU encoder on permitted fit-group feature rows only. For each supervised endpoint t, expose no rows later than t; mask selected current/past input entries and reconstruct only the masked input values, then discard the reconstruction decoder and fine-tune the same encoder on the official target loss. Do not include evaluation sequences in pretraining, and do not add a retrieval store or cross-sequence test-time memory. This is a causal adaptation of the reconstruction task, not a reproduction of SimMTM’s full multiview implementation.

**Assumptions and prediction.** Input dynamics contain reusable structure that supervised labels alone do not reveal, and the reconstruction-trained encoder retains task-relevant information rather than spending capacity on irrelevant channel detail. Improvement should be visible in a matched student endpoint at comparable supervised updates, not merely in lower reconstruction loss.

**Cheapest informative test.** On one source-qualified fit recipe, compare supervised-only initialization with fit-only causal masked pretraining followed by the same fine-tuning budget. Keep masks, epochs, seed and total update budget declared; then grade on an exposed component panel with an independent evaluator. Treat it as a fidelity/feasibility screen until the prescribed paired-cell comparison is complete.

**Falsifier.** Reconstruction quality improves while official WP does not, or pretraining loses the relevant temporal variation / exceeds fit or memory budget. A cross-domain pretraining gain in the paper is not a local prior.

**Cost and artifact.** Medium-to-high GPU training cost and careful group exclusion; no inference cost if the decoder is discarded, and the final GRU should retain its serving envelope. Preserve pretraining membership, masks, objective, encoder SHA, fine-tune ancestry and separate pretraining/quality results.

## P5 — Chrono gate-bias initialization as a fixed-budget GRU arm
**Lens:** change the time representation before learning. **Family:** recurrent timescale prior.

**Source component and analogy.** Tallec and Ollivier derive gate-bias initialization for LSTMs/GRUs from a distribution of temporal scales; their paper studies sequence prediction and long-delay/warping tasks, not the anonymous Challenge 2 labels. This suggests an initialization ablation where sequence lengths may expose temporal dependencies at varied scales, but the maximum 20,000-step length is not evidence that useful target memory is 20,000 steps.

**Intervention.** On the canonical GRU component, initialize update/retention gate biases to a predeclared spread of timescales, deriving the sign and gate convention from the actual cell semantics. Hold parameter count, data membership, optimizer, update order, clipping, loss and budget fixed. Do not add parameters, features or inference branches.

**Assumptions and prediction.** The current gate initialization may make useful time constants hard to learn under the admitted training horizon. A positive result should improve the official validation trajectory or final WP across matched seeds without changing serving cost; faster early loss alone is not sufficient.

**Cheapest informative test.** Fit one matched pair with identical source-qualified data and seed/update budget, standard versus chrono bias only. Check gate initialization numerically and compare the same evaluator outputs; later promotion requires the paired fold/seed protocol. Community code exists, but no author-maintained reference implementation was verified for this scout; use the paper equations and project cell semantics, not copied code.

**Falsifier.** No stable quality effect after equal training budget, or the selected time-scale range pushes gates into saturation / hurts adaptation. One negative initialization is not a universal rejection of all recurrent time-scale priors.

**Cost and artifact.** One standard-size GRU fit, zero extra inference cost or package parameters. Preserve exact gate-bias vector, model factory/source SHA, initialization checks and paired training/evaluation receipt.

## P6 — Larger causal-window teacher distilled into the competition GRU
**Lens:** transfer capacity, then change the search unit to a serving-sized student. **Family:** heterogeneous teacher-to-student transfer.

**Source component and analogy.** TimeKD provides a time-series teacher/student distillation framework with prediction/feature transfer, but its teacher explicitly uses future ground-truth prompts as privileged input. That part is unavailable here and must not be copied. The only proposed transfer is offline distillation from teacher outputs computed on legal causal input windows into a compact GRU.

**Intervention.** For each labeled fit row t, build a teacher input from x[max(0,t−59):t], ending at t. A larger window Transformer or bidirectional recurrent teacher may be bidirectional within this fully observed trailing window; it must not read x[t+1:] or teacher labels at inference. Train its endpoint predictions with the same official training objective, then stream its predictions (and optionally aligned hidden summaries) as soft targets while training the existing GRU student on hard labels plus a predeclared prediction-distillation term. Keep evaluation data excluded from teacher, student initializer and teacher cache. A student-only fit with matched updates is the control.

**Assumptions and prediction.** A wider model can use cross-channel and 60-step context to produce smoother, more informative training targets, and the GRU can compress that signal without serving teacher cost. The prediction is student WP improvement versus matched student-only training, plus teacher WP as a diagnostic; a strong teacher alone is not the competition deliverable.

**Cheapest informative test.** First verify a causal-window materializer on tiny synthetic prefixes and future-perturbation tests. Then one bounded teacher/student feasibility arm and a matched student-only control, with streaming teacher targets rather than a full 250-million-row cache. Run B300 only under a separately assigned, already-authorized operator; this scout did not access or execute B300. Apply clean ancestry and the full paired comparison before making a recipe claim.

**Falsifier.** Teacher has no clean-component advantage, teacher outputs are too noisy or redundant to help, or the student does not exceed the same-budget control. Any teacher that uses future input rows, test labels, is_scored, or cross-sequence adaptation is invalid regardless of score.

**Cost and artifact.** Highest compute cost; B300 training is an unexecuted future handoff, not admitted by this task. The deployed artifact remains a GRU under 20 MB and must pass the 1-CPU/60-minute callback profile. Preserve teacher architecture, causal window code/spec, source/fit ancestry, teacher checkpoint, streamed target manifest, student objective and measured student runtime.

## Triage of the supplied reports

| Report | ABI-aware assessment |
|---|---|
| Rolling windows w3/w13/w21 | Causal if each window ends at the current row and warmup is processed consistently. They are a low novelty baseline; only test predeclared windows with a matched endpoint and runtime measurement. |
| Highway head / two-stage residual GRU or LSTM boosting | The useful bounded form is P3’s small correction trained on cross-fitted parent predictions. A second full recurrent model is a different, more expensive arm and needs its own matched endpoint. |
| Late-checkpoint SWA | Source-supported as an optimization hypothesis, but only average checkpoints from the same architecture and one connected training trajectory. Do not average unrelated seeds/parents. If a compatible trajectory already exists, a same-run average versus final checkpoint is a cheap screen; availability was not checked. |
| Variance augmentation | Fit-side only. A forecasting paper studies GRATIS, moving-block bootstrap and DTW barycenter augmentation for global RNNs on forecasting datasets; this is not evidence that independent channel scaling/noise preserves anonymous-channel relations here. A future screen should preserve sequence/channel structure, train only on fit groups, and measure the unchanged metric. |
| Chrono initialization | P5 is the concrete matched test; no transfer gain is assumed. |
| Causal last-60 bidirectional teacher | P6 preserves causality by ending every bidirectional teacher window at the current t. A window that contains later test rows is forbidden. |
| Adaptive sequence weights from sequence errors | At test this is unavailable and prohibited. P1 replaces it with a training-fitted gate using OOF training errors and test-time observable warmup features only. |
| Masked autoreconstruction | P4 limits it to fit-group inputs and a causal encoder objective; test sequences never enter pretraining. |
| GRU future-input extractor feeding a Transformer | Future inputs are not available at prediction time. A teacher may use channel interactions over the trailing observed window, but no future input stream. |

## Allocation note

No numerical priors are assigned. The high-information first look is to check whether clean, cached multi-expert OOF predictions exist for P1 and to materialize/verify the one-scalar P2 statistic. P3 is the next low-cost supervised screen if a clean OOF base exists. P4–P6 are higher-cost lineages and should not delay a runnable leading endpoint. All reported fold results remain exposed; no score is predicted from public feedback. The research controller owns allocation, admission, implementation and grading.
