# Luna/max: serving and heterogeneous-compute investigation

Date: 2026-10-09 Kyiv

Role: independent native Luna/max investigation, design only
Scope: first bounded experiments and serving plan after the parsed-sequence cache is reviewed and integrated. No fit, provider push, rental, or compute admission was performed here.

## Decision in one paragraph

The first useful question is whether conditional diversity can buy more endpoint score per CPU minute than adding a second recurrent pass. Start with existing out-of-fold (OOF) predictions where their source, split, and checkpoint bindings are complete: test per-target static shrinkage, then compare a small correlation-gradient residual learner and a causal prefix gate on the same outer folds. In parallel, prepare one warm GPU cohort with a component-preserving 10% fit subset and the complete fixed fold-0/fold-1 evaluations. Its first fresh fits should be orthogonal one-factor comparisons—Huber versus weighted-MSE, Muon versus AdamW, and LSTM versus GRU—with equal updates and one shared-data load. Root must first integrate/review the parsed-cache producer, repair the observed staging/memory failure, and admit the exact packet. Do not infer cloud throughput from local timing or spend on idle Vast capacity.

The stated current endpoint is about 36–37 minutes of the 60-minute CPU budget. That leaves 23–24 nominal minutes, not a safe design budget. Reserve at least 10 minutes for runtime variation and collection overhead; initially allow at most about 13 minutes of added inference on the exact submission interface. A second full recurrent model may exceed that. Shared-state heads, static weights, compact residuals, and prefix-only routing are the first serving candidates because they may add less cost, but their complete endpoint latency must be measured.

## Evidence and boundaries

The inspected live-state header is dated 2026-10-08 21:22 UTC, so it is a historical snapshot at this task's 2026-10-09 start. It records local seed99's faithful fit completed and seed9 as the current fit; smoke4 failed during data staging before a model update; TPUv7 was still queued on the collector's 21:21 poll; and the prior Vast r5 worker had been destroyed after target-cache downloads succeeded but raw-parquet staging missed its deadline. It records public best 0.6660 and no supported new expected-public best. Refresh all state and provider quota before a future admission. These failures are execution evidence, not negative evidence about the model families.

ADR0027 and the independent brief set the scientific boundary: own weights and independently implemented methods; source-qualified sequence/component identity; fixed full evaluation membership (fold-0 468 groups, fold-1 457 in the current plan); complete causal histories in fitting subsets; fixed `global_wp_clip2_v1`; and no inference-time labels, future features, sequence-ID routing, or cross-sequence state. The folds have already been inspected and remain exposed, so results are internal allocation evidence rather than a fresh holdout or a public-score forecast. Preserve both fit-subset membership and full evaluation membership independently.

The official metric uses a fixed mask, clips both label and prediction to ±2, weights by the absolute clipped label, and pools weighted correlation separately for the two targets. Use a separate blend/gate for each target if it wins in outer-fold cross-fit; never change metric clipping or select a target policy using the current sequence's label.

## Nine experiment cards

Each card requires a frozen recipe, source and split identities, exact candidate/control, fit-update count, checkpoint and prediction hashes, and time/memory telemetry. “Falsifier” means a reason not to allocate the next exact recipe; it does not kill a mechanism family by itself.

### 1. Add one small independent model, or spend the same budget on a shared-state readout

**Mechanism.** Compare marginal value of a genuinely independent compact GRU (fresh weights, width 128–192) with a shared GRU state plus target-specific shallow readouts. The latter computes the recurrent state once and adds two small heads; it is a different bias/compute trade from a second recurrent pass. The current trainer emits two targets from a shared model, while `campaign_pair_states.py` and `campaign_pair_head_fit.py` provide related frozen-state tooling, not this exact shared-trunk comparison.

**Control and construction.** Reuse an eligible own GRU prediction only if checkpoint SHA, training population, and OOF split all match. Otherwise fit the matched compact GRU control. The independent candidate is scratch-initialized; the shared-state arm has a new trunk and separately parameterized target heads. Keep feature construction, optimizer, loss, sequence subset, and update budget fixed. Fit on component-complete subsets and score both full fixed evaluation folds.

**Cheapest test / observable.** On valid OOF banks, first measure the conditional gain of the existing independent model; then fit the shared-head candidate only if the OOF bank does not already answer the compute question. Record per-target metric delta per additional measured inference minute, peak RSS, and reset parity. A single-model score alone is not the outcome.

**Falsifier / cost / descendants.** Defer the extra recurrent model if its outer-fold conditional gain is near zero and its measured endpoint cost consumes the 13-minute addition budget; defer shared heads if they lose conditional gain versus the same trunk's ordinary head. A positive shared-state result leads to wider target heads or a small auxiliary head; a positive independent-model result leads to distillation or a sparse serving gate. No teacher weights are used as student initialization.

### 2. Robust target loss: Huber versus weighted-MSE

**Mechanism.** Change only the fit loss from `wmse` to `huber` (`--huber-delta 0.5` as an initial fixed value). This probes whether high-magnitude target rows dominate a squared-error-trained member and whether a robust own teacher contributes useful residual structure under the unchanged official metric.

**Control / test.** Same GRU cell/width/layers, feature mode, optimizer/schedule, seed, complete-sequence membership, and exact optimizer update count. Compare from fresh initialization; no selected checkpoint may have seen its fold's evaluation groups. Reuse the data cache, but not the control weights if they were trained under another loss.

**Observable / falsifier.** Compare fixed-metric OOF contribution, calibration by clipped-target magnitude, and residual covariance against every parent. Stop this exact delta if it is conditionally redundant on both outer folds; keep it eligible as a teacher if it improves a constrained blend or student despite a small standalone change. Cost is one fit beyond its matched control; export OOF predictions and clipped-residual summaries.

### 3. Optimizer diversity: Muon versus AdamW, matched

**Mechanism.** Test whether the optimizer changes the learned temporal representation enough to create useful conditional signal. Compare `--optim muon` and `--optim adamw` with identical model, loss, data, seed, schedule horizon, gradient clipping, and completed-update budget. Keep Muon scope explicit (`rnn` or `hidden`) and report it; do not silently compare different precision or batch regimes.

**Construction / observable.** Fresh independent weights for each arm. The shared cache does not imply shared model state. Persist update attempts/completions, AMP skips, per-epoch train loss, checkpoint SHA, OOF prediction SHA and residual covariance. The outcome is conditional contribution to the blend and inference cost, not the training loss or optimizer label.

**Falsifier / descendants.** If a matched pair shows no useful conditional signal on the two exposed outer folds, do not spend a full-scale confirmation on that exact architecture/optimizer pair; do not generalize to all Muon recipes. A positive residual family may justify combining optimizer diversity with the Huber or history arm, as an explicitly new interaction.

### 4. History-cell diversity: LSTM against GRU

**Mechanism.** Use the existing `--cell lstm` path at a matched hidden width, layers, chunk, batch, feature construction, loss, optimizer and subset. This is a distinct recurrent state/update geometry, not a seed repeat. Do not start with a large block-attention or wide TCN: both add serving cost and prior local TCN/attention evidence is not a reason to repeat them unchanged.

**Control / cheapest test.** Compare against the same trainer's `--cell gru` control with matched completed updates. Screen on the same component-preserving 10% subset while retaining full 468/457 evaluation membership; do not shorten sequence histories or evaluate only the sampled members.

**Observable / falsifier.** Record full metric, per-target conditional value, runtime per optimizer update and full-sequence CPU prediction time. Defer the exact LSTM recipe if no conditional benefit offsets slower serve cost. A useful LSTM may serve as a teacher to a compact GRU student; it does not need to win alone.

### 5. Own-teacher distillation to a serving-sized student

**Mechanism.** Fit a fresh compact student on a blend of original labels and predictions from independently trained own teachers, using the existing `--distill` / validated `--distill-cache-root` path. Sweep one preregistered alpha pair only after a parent candidate has established useful conditional signal. This is prediction-target transfer, not teacher-weight transfer.

**Cross-fit.** For every student evaluation fold, teacher targets for its fitting groups must come from teachers whose own fitting membership excludes those groups. Hash-bind teacher checkpoints, prediction files, source-qualified groups, target-treatment variant and split. Never distill a fold's in-sample teacher output into a student evaluated on that fold.

**Control / falsifier.** Compare the scratch same-size student (`alpha=0`) against a distillation candidate with identical optimizer budget. A student that fails to retain teacher conditional value on outer folds or misses the 50-minute serving envelope is not promoted; it may remain a useful compression control. Save student checkpoint, teacher ancestry graph, full OOF outputs and measured endpoint time. The motivation that teacher knowledge can be compressed into a smaller model is established in Hinton et al.; applicability to this metric/data is an empirical question.

### 6. Per-target, cross-fitted static blending with covariance shrinkage

**Mechanism.** With a clean OOF bank, estimate a separate nonnegative simplex of model weights for each target. Compare equal weights, a constrained optimum and a shrinkage path `w(λ)=(1−λ)w_fit+λw_equal`, where λ is selected inside training folds only. Clip member predictions as the official metric does before constructing weighted moments; clip the final blend at grading time too.

**Cross-fit / falsifier.** For each outer fold, fit weights on other folds' OOF rows and apply them once to the held fold; do not fit weights and claim score on the same labels. Retain weights, covariance matrix, effective member count, optimizer trace, fold split hashes and bootstrap-by-component uncertainty. Reject only a weight fit that is unstable or has no consistent conditional benefit, not a model based on disagreement. `scripts/crossfit_blend.py` is an algorithmic starting point, but its current full `valid_targets.npz` loading is not automatically safe for the new host-memory budget; adapt it to bounded cache/mmap reads before a large run.

**Unconventional transfer.** Markowitz's 1952 portfolio result emphasizes covariance in the variance of weighted sums, not each asset's variance alone. Here prediction-residual covariance is a useful analogy for blend design, not an identity: the official objective is weighted correlation, and coefficients must be learned on cross-fitted predictions. This motivates recording residual covariance and shrinking unstable weights, not importing financial assumptions.

### 7. Correlation-gradient residual learner versus squared-error residual learner

**Mechanism.** Let `p` be a parent's clipped OOF prediction, `y` the clipped fit-only label, `w=|y|`, and weighted means be `μp, μy`. For a target with weighted sums `Sxy, Sxx, Syy`, define `β=Sxy/Sxx`. Away from prediction clipping boundaries, the ascent direction for weighted correlation is proportional to

`g_i = w_i * [y_i − μy − β(p_i − μp)] / sqrt(Sxx*Syy)`.

At clipped prediction boundaries, multiply by the clipping derivative (zero outside the open interval). Train a tiny ridge or shallow tree residual learner on OOF parent predictions plus only permitted current/past feature summaries to predict `g`. Compare it with the same learner trained on weighted squared residual `y−p`, and with no residual learner. This is a proposed objective-alignment test, not a replacement for the fixed scorer.

**Cross-fit / test.** Produce parent OOF predictions by component; train the residual learner only from OOF rows in the meta-training folds, then apply to a distinct outer fold. Tune shrinkage η only inside training folds and score `clip(p+ηr,−2,2)` on the held fold. Keep target dimensions separate. First try this on existing OOF arrays; it needs no new recurrent fit if the necessary clean OOF artifacts already exist.

**Falsifier / descendants.** If both targets/folds show unstable η, no gain over static weights, or a gain that vanishes under component bootstrap, retain the artifacts but do not add the learner to serving. Friedman’s functional-gradient view supports the general mechanism of fitting a predictor to a loss gradient; this particular weighted-correlation gradient is derived from our scorer and must be checked numerically against it. Store gradient/residual covariance, η, mask, membership and all model hashes.

### 8. Causal prefix gate and conditional expert execution

**Mechanism.** A small gate selects/blends independently trained model families from the first 99 observed rows or other explicitly causal current/past summaries. It receives neither sequence ID nor labels. Compare a static per-target blend, a soft gate that computes all experts, and only then a hard prefix route that executes one selected expert. A sequence-level prefix route avoids stale recurrent state: compute the gate from the allowed prefix, then run the selected expert from its zero state on the complete causal history. Do not skip arbitrary recurrent steps while carrying stale hidden state.

**Control / cross-fit.** Train gate targets from fit-only OOF per-expert regrets or cross-fitted model losses. On each outer fold, fit gate on other folds only and evaluate its frozen rule on the held fold. Include a uniform soft mixture and a rate-matched deterministic prefix/hash route as controls. Train/evaluate gate using the same fixed official metric; no route may depend on inference-time `y`, score mask, class derived from future labels, sequence ID, or future feature values.

**Observable / falsifier.** Measure conditional score and actual per-sequence CPU time, not theoretical FLOPs. Hard routing is only useful if its quality loss is smaller than the inference time saved while keeping the whole endpoint ≤50 minutes. If the route collapses to one model, overfits folds, or provides less than the static blend, keep it as diagnostic only. `train_rnn.py --arch gated|specialist` and `predict_run.py --gate-mode soft|hard|confident` are existing paths to inspect; their exact semantics and reset behavior need packet-level validation before use.

### 9. Quantization/compression parity and complete CPU budget

**Mechanism.** Compare the current unmodified FP32 submission against one measured compressed path only after a real CPU kernel/export exists. Candidate choices are a rank-bounded recurrent factorization or true weight-only int8 execution. The present trainer exposes `--recurrent-rank` and `--qat`, but QAT or a quantization receipt alone is not proof that inference executes faster; the current prediction path may force evaluation arithmetic back to FP32. Inspect and test the final callback, package bytes and operator set.

**Parity / falsifier.** With identical weights where possible, stream full sequences through the exact Python 3.11 one-CPU submission callback, reset on every sequence boundary, and compare outputs with the FP32 control on every required row (finite values, max/mean absolute error, exact mask/order). Verify determinism across repeated resets, no future access, package ≤20 MB, process RSS ≤16 GB and the entire endpoint p95 ≤50 minutes in repeated cold-ish and warm-ish runs. Do not accept a speedup that changes the official metric through different clipping or misses predictions.

**Artifacts.** Store original/compressed model hashes, native-library build/compiler flags, numerical parity report, per-stage and whole-endpoint timing, peak RSS, package size, and reset/order test results. A model may remain a larger research teacher if it can be distilled into a smaller serving student; final serving constraints do not cap research teacher size.

## First allocation and next minimum packet

### Cheap work from already-valid predictions

Before any new fit, verify whether an ancestry-clean OOF bank exists for the same source and fixed fold manifests. If it does, prioritize Cards 6, 7, and 8 on that bank; these are distinct conditional-combination tests and can be run without training a new recurrent model. Score each on both existing outer folds with component bootstrap and report completed comparisons/hour, including load, parse, meta-fit and scoring. If the required OOF bank does not exist, do not relabel in-sample or all-data predictions as OOF; schedule the smallest necessary clean-fold export after root review.

### First warm cohort after cache/source repair

1. Root integrates and reviews parsed-cache producer commit `d95d2efc`; do not copy its files into a second implementation. Freeze the cache manifest with source hashes, split, exact selected complete-component membership, all train/valid row identities, labels/masks and full eval membership. The current builder estimated about 14.2 GB for selected parsed arrays; register disk space and reserve cache + outputs + checkpoints before materialization. Measure actual parse count, wall time, physical `read_bytes`, `rchar`, page-cache hit, and map/read time separately. `parse_once` is an observed receipt, not a schema constant.
2. Build one fixed component-preserving 10% fit subset while retaining full 468/457 evaluation memberships and causal histories. Keep fidelity and fixed optimizer updates identical across arms. This is allocation evidence only, never a replacement for full-scale confirmation.
3. First fit matrix: one fresh matched GRU control and the three one-factor candidates from Cards 2–4 (Huber loss, Muon optimizer, LSTM cell); reuse a clean eligible control only when recipe/ancestry/membership match exactly. If the fixed control cannot be reused, count its fit and use a shared-data cohort. No seed-only variants. Independent candidates use independent weights.
4. Start with one worker per T4 until actual host cgroup `memory.current`, `memory.stat`, per-process PSS, GPU allocation and data-map sharing are observed. The prior smoke4 failure occurred in staging; do not assume two-worker RSS sums or simply loosen a memory gate. Permit both T4s only if measured physical memory leaves the reviewed headroom. Keep worker count at zero and MPS out unless separately profiled.
5. Each arm exports complete fold-0/fold-1 OOF predictions and checkpoint/provenance. A second-level learner is a separate cross-fitted consumer with exact source/membership pins. Persist predictions and residual covariance to SSD before queueing the next cohort; do not rely on provider artifacts as the only copy.

**End-to-end packet contents:** exact runnable archive and entrypoint; independently reviewed source/config/data hashes; private source/cache manifest and version; task-specific complete-component subset manifest plus full evaluation membership; matched recipe JSONs; consumer validation against real producer NPY headers and sidecars; mounted-input path discovery; parse/I/O/physical-memory telemetry; worker phase heartbeat; firm hard-wall and artifact budget; expected artifacts with existence, size and SHA; OOF prediction sidecars; checkpoint and ancestry graph; independent score receipt; serving-time budget; root admission reference. Any source-stage or artifact failure remains a new identified packet and is not a scientific failure.

### Capacity choice by measured throughput

- **Two T4s:** first choice for the fresh heterogeneous matrix only after real two-worker host/cgroup headroom and shared-cache behavior are measured. Two devices can reduce elapsed cohort time only when input staging, CPU loading and memory allow concurrent fits; that condition is not established by device count.
- **TPU:** the existing `scripts/kaggle_tpu_probe.py` checks tiny GRU equations/updates and an actual small repository model path; it is not full-shape, optimizer, checkpoint-reload or data-pipeline parity. Production training stays blocked until exact model forward/loss/gradient/update/save/reload parity and representative full-shape runtime pass. Then compare completed valid idea screens/hour including compile, setup, data movement and export against T4s under equal recipe/update budget. TPU matrix throughput is not evidence that this recurrent workload is faster; the published first-generation TPU study is a hardware-specific inference result, so any training-efficiency claim here would be an unverified inference.
- **Vast:** do not rent while its data stage is the bottleneck. Prior r5 verified 2.246 GB of target-cache downloads but missed its raw-parquet staging deadline. Future rental needs a bounded exact-source transfer measurement, end-to-end staging duration, fit/export estimate, all-in quote and root admission before instance creation. No idle rental and no new spend are authorized by this proposal.
- **Local GPU:** one worker and ≤5 GB observed VRAM remain the limit. This investigation ran no fit and grants no local-GPU admission. Use CPU only for bounded OOF/meta/serving analysis under the global support wrapper.

The throughput target of 5–6 completed distinct comparisons/hour is not yet demonstrated. Count a comparison only when its declared control, valid cross-fit predictions, official score, runtime and artifacts are complete. Startup retries, seeds, profiles and alternate metrics do not count as ideas. Use measured warm/cold fit and cycle times to choose cohort concurrency; keep full-scale confirmation alongside screens rather than inferring it from the 10% subset.

## Exact repository entrypoints inspected

- `scripts/train_rnn.py`: existing `--optim {adamw,muon,sgd}`, `--sched`, `--loss`, `--huber-delta`, `--arch`, `--cell {gru,lstm}`, `--distill`, `--distill-cache-root`, `--recurrent-rank`, `--qat`, fixed checkpoint and split/subset controls.
- `scripts/predict_run.py`: own checkpoint prediction export; has gated modes and a feature-only E540 subset path. Validate exact split/row order and full-sequence reset behavior before new use.
- `scripts/crossfit_blend.py`: existing constrained, fold-cross-fitted static blend algorithm. Its full target NPZ loading needs bounded/mmap adaptation before use in a memory-sensitive cohort.
- `scripts/campaign_pair_states.py`, `campaign_pair_head_fit.py`, `campaign_pair_head_score.py`: existing frozen-backbone state and paired-head experiment tooling; validate their manifest and memory assumptions against the current cache before use.
- `scripts/cr_scout_tree_stacker.py`: historical tree-stacker prototype only, not evidence of a valid cross-fitted second-level result. Reuse the concept after auditing its exact split/objective path.
- `scripts/parsed_sequence_cache.py` and `src/wnn/parsed_sequence_cache.py`: prepared in isolated writer commit `d95d2efc`, not yet assumed integrated in this base. Root integration/review is a dependency; keep parsed-data reuse measurable.
- `scripts/kaggle_gpu_runner.py`, `scripts/kaggle_kernel.py`, `scripts/kaggle_sequence.py`, `scripts/kaggle_tpu_probe.py`: private Kaggle packet/dispatch/probe tooling. Use only after exact source/metadata/hash review and root admission.
- `scripts/warm_queue_worker.py`: finite local queue runner. It does not authorize model fits or change current lane ownership.
- `scripts/vast_campaign.py`: lifecycle tool; it does not authorize create/rent. Any future rental remains root-owned.
- Final submission interface constraints from the brief: deterministic Python 3.11, one CPU, 16 GB RAM, ≤20 MB ZIP, ≤60 minutes. Measure the whole callback including preprocessing/state reset/export, not isolated model forward time.

## Primary sources and source-to-inference notes

- Wolpert, “Stacked generalization” (1992), [Neural Networks / ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608005800231). Source fact: a second-level generalizer uses predictions from base generalizers trained on other parts of the learning set. Inference here: grouped sequence/component OOF predictions are required before fitting our gate/blender.
- Jacobs, Jordan, Nowlan & Hinton, “Adaptive Mixtures of Local Experts” (1991), [author-hosted paper](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf). Source fact: a gating network allocates cases among experts. Inference here: a prefix-only gate is a candidate for this task only if its inputs are causal and it reduces measured endpoint cost without labels.
- Hinton, Vinyals & Dean, “Distilling the Knowledge in a Neural Network” (2015), [arXiv](https://arxiv.org/abs/1503.02531). Source fact: an ensemble can supervise a smaller deployable model. Inference here: own OOF teachers may be distilled to a constrained student; target metric and sequence-level transfer remain untested.
- Friedman, “Greedy Function Approximation: A Gradient Boosting Machine” (2001), [Project Euclid](https://doi.org/10.1214/aos/1013203451). Source fact: stagewise additive fitting can follow functional-loss gradients. Inference here: a learner trained on our derived weighted-correlation gradient is worth comparing with squared residuals, not assumed to improve the scorer.
- Markowitz, “Portfolio Selection” (1952), [Journal of Finance / Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1952.tb01525.x). Source fact: portfolio risk depends on covariance among returns as well as marginal variance. Inference here: residual covariance can guide prediction blends, but the financial objective is not our weighted-correlation metric.
- Jouppi et al., “In-Datacenter Performance Analysis of a Tensor Processing Unit” (ISCA 2017), [arXiv](https://arxiv.org/abs/1704.04760), [DOI](https://doi.org/10.1145/3079856.3080246). Source fact: the reported first-generation device/benchmark concerns a specific inference accelerator and workload. Inference here: it does not establish training throughput for our recurrent code; compare actual, matched device traces only.

Local method/constraints sources: [ADR0027](../../../../specs/adr/0027-diverse-models-and-second-level-learning.md), [independent brief](CONTRACT_R1.md), [experiment throughput plan](../../../../specs/plans/2026-10-08-throughput-scaleout.md), [`global_wp_clip2_v1`](../../../../src/wnn/gmetric.py), trainer and consumer paths listed above. Prior random-library transfer receipt, if used in synthesis: `research/ideas/claude_20261008_round/passages_selected.json`, seed 20261008, Markowitz 1952 p.5.
