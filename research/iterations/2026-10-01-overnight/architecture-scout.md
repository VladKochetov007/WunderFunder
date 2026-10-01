# Independent architecture/training challenge — commitment

Committed by 2026-10-01 00:58:11 UTC, before opening any new overnight actor proposal, source, result or log. (The initial draft's rounded 01:00 timestamp was corrected against the UTC clock before any outcomes.) This scout has read only the overnight contract, historical reports/ledgers, and pre-existing trainer/model sources. No dataset body, prediction cache or checkpoint was opened; no fit, diagnostic or WP calculation was run. The only mutation is this dossier.

**One proposal: train an otherwise unchanged GRU128 parent with sharpness-aware minimization (SAM), then apply the exact ordinary two-epoch V1-like head fine-tune.** Keep inputs, architecture, teacher/auxiliary losses, examples, optimizer update count, learning-rate schedule and serving graph fixed. Perturb the parent weights for a second loss/gradient evaluation on each TBPTT chunk; update AdamW from that gradient after restoring the original weights. Use one fixed radius rho=0.05, no sweep. This is a parent-training intervention, not another head objective or feature block.

Mechanism-family: parameter-neighborhood robustness / noise-resistant representation learning. The hypothesis is that the existing EMA/distillation recipe still selects a representation that is too sensitive to small coordinated parameter changes; making the parent locally robust can alter the causal representation without increasing inference cost. This is a hypothesis, not an explanation established by the previous results. Width/attention saturation does not identify a Bayes ceiling, and successful EMA/distillation is positive prior evidence for a remaining training-side mechanism, not proof SAM will work.

The first implementation must preserve recurrent semantics: both SAM passes start from the same detached incoming hidden state, EMA teacher outputs are computed only once at unperturbed teacher weights, only the first unperturbed student pass supplies the carried hidden state, weights are restored before one AdamW/scheduler/EMA update, and no perturbed state may leak into the next chunk. Preserve the existing one-update-per-chunk algorithm and auxiliary objective exactly. Frozen-head fine-tuning remains the ordinary existing recipe, without SAM.

Primary falsifier: a fixed replacement of a predetermined plain-h128 member at weight 0.25 in the matched V1-like ensemble fails to improve by +0.0003 pooled, with either fold negative, after two paired seeds. A failed screen stops this fixed rho recipe; it does not refute every possible sharpness method. All folds are exposed exploratory evidence.

This is a scoped proposal, not a compute allocation or a forecast-quality claim. The conduct-creative-research skill and its empirical-validation reference were read and applied.

## Why this is my one choice

The relevant positive history is training-side: E-152's old learning curve gained roughly .002–.003 per data doubling; E-153's long weakly regularized training overfit badly; EMA4 eventually added roughly .0035 in E-164; ordinary distillation improved the current wide member, while stronger privileged teachers did not. These observations are compatible with noisy finite-sample fitting and a restricted optimization recipe. They do not show that useful causal information is exhausted. The historical 22-member ensemble ceiling only bounds mixtures of those fitted models.

I considered four different lenses before committing: (1) explicit interval summaries for event memory, (2) random sequence origins, (3) predictive-state pretraining on observable next inputs, and (4) parameter-neighborhood robustness. Interval summaries already have a concrete A1 proposal in the deep architecture dossier; random origins have a weak prior because existing reset and start-position audits show little initial-state deficit; predictive-state pretraining would need new targets and has adjacent negative quote-auxiliary evidence. SAM is a cleaner independent intervention within this time budget, uses the already successful representation, and requires no serving changes. This is not a claim that it is novel research.

The actual blind spot is **what optimization trajectories the existing comparison covers**. In `scripts/train_rnn.py:941–1089`, every model trains each chunk with one ordinary gradient and one AdamW update. EMA is a moving teacher/weight average; it does not explicitly evaluate the loss in an adversarial parameter neighborhood. Frozen head tuning cannot change the recurrent representation at all. A SAM parent changes that representation's training criterion before the successful unchanged head stage.

No SAM/ASAM/sharpness arm was found in the searched historical experiment ledger, hypothesis ledger, journal, or next/deep/EDA dossiers. This is a bounded search statement, not proof no unpublished trial exists.

| Adjacent negative | Why this is a different intervention |
| --- | --- |
| Width, event attention, dual clock, two stream | Those add capacity/memory under the existing optimizer. This keeps the exact graph and changes the parent's two-pass update. |
| Dropout .1, strong weight decay, weight soups | Random unit removal, parameter shrinkage and post-training averaging do not impose the SAM local worst-direction criterion. Those results still lower the prior for another generic regularizer. |
| Privileged/gen2 distillation | SAM uses the same observed labels and same EMA teacher as the paired control, without a new teacher or target interpolation. |
| V-REx | SAM does not equalize sequence risks or differentiate a variance of per-sequence risks. |
| Current aux-off / ensemble-aware head studies | This changes parent representation learning; head objectives, masks, freeze list and two-epoch schedule remain identical to the paired V1-like control. |
| Periodic reset / long TBPTT | No time-domain inputs, gradient horizon, reset semantics or chunk length change. |

## Exact bounded candidate and control

Candidate: ordinary GRU128x2, chunk2000, batch32, four-epoch EMA4 + since-weight parent, train + allowed nonheld validation chains, same propensity/mask treatment, all existing auxiliary coefficients, same AdamW settings and OneCycleLR schedule. Use fixed `rho=.05` during every parent update; no late start, layer restriction, alternative radius, ASAM, clipping change or objective change. Save fixed parent epoch3. Apply the exact two-epoch head-only gcorr stage used by the reconstructed V1-like anchors, freeze `inp,gru`, save fixed head epoch1. Serving remains the current `RNNModel` and native engine.

Control: the identical isolated trainer with SAM disabled, same initial tensors, data order, normalization, precision, update count and seeds. A synthetic `rho=0` two-pass parity test binds the modified loop to the old one; production control can then take the one-pass path. Existing controls may be reused only after their complete provenance and the two-epoch head semantics match—not simply because names contain `ref` or `H`.

Initial screen: seed3, folds0/1, two candidate parents and two controls, each with the identical two-epoch head stage. Before any readout, freeze the actual four V1-like anchor files and replace the seed3 plain-h128 slot only at weight .25. No posthoc mixing weight or chosen slot. Run fixed paired seed4 only if that serving comparison is nonnegative in both folds and pooled lift is at least +.0003. The seed4 robustness comparison replaces its corresponding original seed4 h128 slot at .25, with the other original members fixed. This is a second paired recipe test, not a fresh holdout.

Promotion requires both seed-specific pooled serving deltas positive, all four seed/fold serving deltas nonnegative, and the mean of the two pooled serving deltas at least +.0003. Keep the predetermined seed3 slot as the eventual single-member replacement; do not select the better seed. These four-member ensembles differ only in the paired slot. Report the candidate/control single-model scores descriptively, but do not let those override serving results. A future two-slot SAM ensemble would be a separately registered readout, not an automatically searched blend.

Maximum from-scratch allocation: eight parents + eight head stages = 16 forecasting fits, plus one final parent/head pair if promoted. Each head stage is one run of two epochs. Reusing verified controls can reduce that count. Authorize and count diagnostic/profiling training prefixes under the controller's fit budget rather than disguising them as free tests. Stop after the failed seed3 gate; no radius sweep or architecture fallback belongs to this card.

Score exact global pooled clipped |target|-weighted Pearson over pure-chain `need_prediction AND is_scored` rows. Report whole-chain paired bootstrap intervals conditional on these models, all paired seeds and folds, and their variation separately. The comparison is exposed discovery; it cannot confirm hidden-test transfer. No public query is needed or allowed.

## Minimal implementation path

Use an isolated copy of `scripts/train_rnn.py` owned by an actor, leaving the baseline source intact. Add only `--sam-rho` (default0), extract the current chunk loss into a side-effect-free closure, and replace the optimizer section as follows. The closure must return the exact existing total loss and outgoing student state without updating any recurrence, EMA, moments, schedule or counters.

```text
h_in = detach_state(h)                         # same object/value for both passes
teacher_out, next_teacher_state = EMA_once()   # unperturbed; same targets in both passes
loss1, h_plain = student_loss(theta, h_in, teacher_out)
g1 = autograd.grad(scale(loss1), trainable_parameters)
g1 = each_gradient.float() / current_scale     # do not unscale_(optimizer) here
eps = 0.05 * g1 / sqrt(sum_over_all_parameters(g1 ** 2) + 1e-24)
saved_theta = exact_parameter_copies()
try:
    theta = saved_theta + eps
    loss2, ignored_h_perturbed = student_loss(theta, h_in, teacher_out)
    scale(loss2).backward()                    # first graph already released
finally:
    theta.copy_(saved_theta)                  # exact restore, even on exceptions
unscale_(optimizer)                           # exactly once per optimizer step
clip_global_grad_norm_(1.0)
scaler.step(AdamW); scaler.update()
scheduler.step(); EMA.update(); step_count += 1
h = detach_state(h_plain); h_ema = detach_state(next_teacher_state)
```

The perturbation is over **all trainable parent parameters**, including biases and auxiliary output rows, using the full unchanged loss. That choice is deliberately simple and frozen, although auxiliaries may dominate the direction. It is not SAM of the clipped competition metric or of an entire 20k-row unroll: the incoming state is held fixed, exactly as in the existing TBPTT objective. Existing hidden-state staleness after an optimizer step is preserved, not claimed solved.

Important implementation failures to prevent: using `h_plain` as the second pass's incoming state would consume the same chunk twice; carrying `h_perturbed` would change the temporal model; recomputing/updating EMA between passes would change the objective; calling GradScaler `unscale_` twice before `step` is invalid; double scheduler/EMA updates would halve the effective averaging horizon. Nonfinite first-pass gradients must skip or abort the complete update with a recorded finite-state policy, never create a NaN perturbation. Keep default fp16/autocast behavior paired; silently switching the candidate to bf16 adds a treatment. Restore model tensors exactly with saved copies rather than relying on finite-precision add/subtract cancellation.

The original model has dropout0 and no BatchNorm, so there is no running-stat update to freeze between passes. Do not introduce stochastic augmentation; if any RNG-consuming option is unexpectedly enabled, preserve/restore RNG for the paired passes or reject the configuration.

A concrete actor interface after implementing the isolated wrapper would be `train_sam.py [all fold_job.sh reference flags] --sam-rho 0.05`; the reference call uses `--sam-rho 0`. The second stage calls the unchanged training script with the frozen parent's `@ep3`, `--epochs 2 --freeze inp,gru --loss gcorr --mask-weight --batch 16 --lr 5e-4`, and the matched source/mask/auxiliary/fold flags. Do **not** use unmodified `fold_ft.sh`: it runs one head epoch followed by full-network FT, which is the wrong comparator. Exact checkpoint/input/evaluator hashes must be filled in by the actor before execution.

## Cheapest checks, falsifiers and runtime

First run only synthetic correctness checks: two successive TBPTT chunks with nonzero incoming state, `rho=0` weight/optimizer/EMA/hidden parity against the original loop, explicit no-perturbation-residue assertions, and equality of first-pass state to the carried state. With `rho=.05`, verify a finite perturbation norm and that the restored weights are byte-identical before optimizer update. Check gradient scaling and an intentionally nonfinite gradient path. This is a high-value independent implementation-review surface.

There is no honest train-only scalar sharpness statistic that can guarantee competition improvement, so I do not impose a data-dependent “sharpness must be large” gate. Fixed-SAM failure to pass the actual serving gate is the discriminating experiment. Expected direction, if successful: similar or slightly worse training fit, more stable model response to parameter perturbations, and improved paired held-chain serving WP after the unchanged head stage. Better training loss alone or a more stable weight neighborhood alone is not success.

Relative parent runtime estimate is about **1.6–2.0x** the same one-pass parent: ordinary EMA training already pays one student forward/backward plus one teacher forward; SAM adds one student forward/backward and a small parameter copy. This is a calculation, not a measured A4000 result. Historical dual-clock EMA4 workflows took about14.5 minutes in the deep architecture dossier. A conservative planning bound is **20–35 minutes per SAM h128 parent**, 10–20 per control, plus predictions/head stages and shared-GPU slowdown. Two paired folds for seed3 should fit roughly **60–100 minutes wall** with two slots; the complete two-seed test plus a final fit and implementation review is plausible in **3–4 hours**, provided a 5.5GiB-per-job smoke profile passes. If observed time is >2.2x baseline, or the projected work approaches the contract's 06:00UTC long-training cutoff, stop allocating this branch.

The two autograd graphs are sequentially released, so peak activation memory should stay near the original parent's, with only parameter/gradient copies added. Do not assume that fits the shared A4000: measure and enforce 5.5GiB per job, at most two jobs, and the global24-thread budget. Do not silently lower batch size only for SAM. If a smaller batch is necessary, both arms need the same changed configuration and an explicit amendment before any score.

At inference the parameter shapes, operations, hidden state and ensemble member count are identical. Expected native latency and ZIP size are therefore unchanged apart from weight compressibility; whole-package parity and the usual runtime benchmark remain mandatory. SAM supplies no permission to relax the60-minute limit or governor policy.

## Primary sources and epistemic limits

[Foret et al., SAM (ICLR2021)](https://arxiv.org/abs/2010.01412) defines the local worst-loss optimization and reports generalization and label-noise robustness on its benchmarks. The [authors' implementation](https://github.com/google-research/sam) gives a standard rho=.05 example. These sources justify the method and a single untuned starting radius, not the expected magnitude on anonymized market streams. [Bahri et al. (ACL2022)](https://arxiv.org/abs/2110.08529) supplies evidence that this optimization idea is not confined to vision; it is not evidence about GRU128, this target, or the hidden leaderboard.

The inference from local history is modest: a zero-serving-cost parent-training experiment has a better practical upside/cost tradeoff here than porting another event-memory architecture, while the chance of a null or negative result remains substantial. Existing EMA may already provide all the relevant smoothing, SAM's finite-radius criterion is parameterization-dependent, and weighting rare large labels could make its adversarial direction harmful. No claim of a large score gain or a ranking improvement is justified in advance.

Status: **proposed / not tested**. No sources, data, checkpoints, caches, jobs or existing files were changed. Only this dossier was created and extended; no new overnight actor files were read.
