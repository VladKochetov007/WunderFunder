# Head-loss training: a controlled intervention, with a stronger reserve alternative

The auxiliary-loss ablation remains a reasonable inexpensive experiment, but its original explanation was too narrow. There are three different effects: competition for the shared nonlinear head, coupling through the global gradient clip, and a mismatch between optimizing a member's correlation and its contribution to a raw-prediction ensemble. Only the first is measured by a shared-gradient cosine. The more ambitious reserve is **ensemble-aware head fine-tuning**, which directly addresses the third effect. Neither has been trained or scored in this round.

This dossier applies [conduct-creative-research](../../../.agents/skills/conduct-creative-research/SKILL.md) and its empirical-validation reference under [contract.json](contract.json). The previous scout is preserved. All folds remain exposed; the proposed comparisons are discovery studies. The gradient diagnostic is specified separately in [training-plan.json](training-plan.json), with implementation [training_diagnostic.py](training_diagnostic.py). Root owns execution and records its outcome separately.

## What actually trains

The deployed raw-input path is 112 raw plus 99 causal event features, normalization, Linear→GELU projection, two GRU layers, and a shared head:

\[
z_t=\operatorname{GELU}(A\tilde x_t+a),\quad h_t=\operatorname{GRU}(z_t,h_{t-1}),\quad
q_t=\operatorname{GELU}(B[h_t,z_t]+b),\quad o_t=Wq_t+c.
\]

The first two outputs are the submitted predictions. The remaining 21 outputs are 12 spread bins, 3 branch classes, 4 market classes, and 2 next-change outputs. `src/wnn/nn.py:57` and `scripts/train_rnn.py:769–776` determine this layout. Freezing `inp,gru` leaves **B,b,W,c trainable**; it does not freeze the representation q. For h128 this is 35,863 trainable head parameters, including a 32,768-parameter shared matrix B. Auxiliary gradients can therefore alter the nonlinear features used by the two main outputs even though the recurrent trunk is fixed.

The actual raw logs `beleriand:~/WunderFunder/artifacts/logs/ft128_f0.log` and `fin128_s3h2.log` establish the head-stage recipe: `loss=gcorr`, `mask_weight=true`, auxiliary coefficients .3/.3, AdamW lr5e-4 and wd.01, batch16, chunk2000. **There is no EMA consistency, no EMA evaluation, no propensity weighting and no since-event weighting in this stage.** Those belong to parent training. `model.load_state_dict` loads weights only; each head fit creates fresh AdamW moments (`train_rnn.py:791–817`). Do not describe this as continuing the parent's optimizer or EMA trajectory.

`model.train()` is called even with frozen parameters. That would matter for stochastic trunks, but these particular logs have `dropout=0`, the encoder is linear/GELU, and there is no batch normalization or head dropout. Consequently train/eval modes have the same mathematical forward computation here. This must be asserted, rather than generalized to other architectures. CPU float32 replay is still not bitwise GPU autocast-fp16 parity.

## The objectives are not interchangeable

For one target and one batch/chunk, write \(\bar y=\operatorname{clip}(y,-2,2)\), scored/warm-up mask \(m\), and \(w_i=m_i|\bar y_i|\). With weighted centering and weighted covariance/variances, the implemented main loss is

\[
L_{main}=1-\tfrac12\sum_{k=0}^1\rho_{w_k}(p_k,\bar y_k).
\]

It pools **16 sequences × 2000 rows**, not the entire evaluation dataset. Predictions p are **not clipped in this training loss** (`nn.py:162–175`). Averaging chunk correlation objectives is not algebraically the full-dataset correlation objective: each chunk centers independently and gets an optimizer step irrespective of total metric mass. This is a surrogate, not a scorer implementation defect. The official scorer pools selected rows globally and clips the final ensemble prediction. Keep this surrogate identical in the first auxiliary ablation; changing pooling or prediction clipping simultaneously would confound it.

The auxiliary structure objective sums three cross-entropies with weights m, without the main loss's |y| factor. The next-change objective is MSE on \((\log(1+\Delta rows)-3)/2\) plus sign BCE, weighted by \(m\,1\{\text{a later target-contrast change exists}\}\). Each auxiliary objective uses its own weight-sum denominator. A nominal coefficient .3 is therefore not a statement that 30% of the effective update is auxiliary. Low-|y| rows and late right-censored rows affect the objectives differently (`train_rnn.py:916–922,1012–1031`).

The labels themselves need precise names:

* `sequence_class(y)` reads full-sequence target quantization. This is legal training supervision; the class is not supplied as an input or serving route in the plain model.
* `next_change_labels` uses later **target-contrast** changes, masks the tail without a later change, and can look beyond the current TBPTT chunk because labels are constructed for the full sequence. This is a training target, not a causal input or observed physical quote-arrival intensity.
* The label called spread comes from `outcomes.py:49–55,108–113`: on winning rows it assumes `spread=-loser-2` in tick units. The later quote-law audit shows a variable adverse-move term in the losing side. Thus this auxiliary label need not equal the currently observable spread. It is still a valid outcome-derived classification target; a misleading economic name does not make supervision leakage or a corrupted label. It may regularize future-outcome prediction, or consume head capacity on a less useful transformation.

No leakage is introduced merely because auxiliary labels use the future on training examples. Leakage would arise from using held-out labels to fit, crossing held-out chain boundaries, or feeding these labels into inference. The folds and all label-generation inputs must therefore be frozen and audited. Parent and head fitting exclude the scored fold; the scorer additionally retains pure validation chains without train neighbours. Final public fits intentionally use all validation labels; their old “holdout” log scores are in-sample.

## Why a negative cosine is neither necessary nor sufficient

Ignoring stabilizers, the derivative for one weighted correlation is

\[
\frac{\partial\rho}{\partial p_i}=\frac{w_i}{\sum_jw_j}
\left[\frac{\bar y_i-\mu_y}{\sigma_y\sigma_p}
-\rho\frac{p_i-\mu_p}{\sigma_p^2}\right].
\]

For shared head parameters \(\theta=(B,b)\), let \(g_m=\nabla_\theta L_{main}\), \(g_s=.3\nabla_\theta L_{struct}\), \(g_n=.3\nabla_\theta L_{next}\). A small **plain SGD** update produces

\[
\Delta L_{main}\simeq-\eta\{\|g_m\|^2+g_m^T(g_s+g_n)\}.
\]

A negative dot product reduces immediate progress at that point. It does not establish worse generalization or a worse finite trajectory. Auxiliary regularization can improve later features even when the immediate main loss rises. [Du et al., §6](https://arxiv.org/html/1812.02224v2#S6) explicitly discuss local-versus-global usefulness and adaptive-optimizer limitations; their work motivates measurement, not a mandatory cosine veto.

The real update first concatenates gradients over **all trainable head parameters**, applies one scalar clip \(\kappa=\min(1,1/(\|G\|+10^{-6}))\), then AdamW. Removing auxiliaries deletes both shared-layer gradients and gradients on the 21 private auxiliary output rows. Those private rows can change \(\kappa\) even if their shared-layer interaction is benign. Main-only is therefore a package of objective and optimizer effects, not a clean identification of shared-feature conflict.

Adam uses coordinatewise first/second-moment estimates of clipped gradients. A constant common rescale approximately cancels in \(\hat m/(\sqrt{\hat v}+\epsilon)\); changing scale over time, coordinate directions, epsilons, finite precision and optimizer history does not cancel exactly. AdamW decay is decoupled from the loss gradient. Also OneCycleLR cycles momentum by default as well as learning rate; retain the exact scheduler and installed implementation in both arms. These distinctions follow the [Adam paper](https://arxiv.org/abs/1412.6980), [AdamW paper](https://arxiv.org/abs/1711.05101) and [official OneCycleLR documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.OneCycleLR.html); the experiment must record its actual installed version rather than silently migrate libraries.

The proposed diagnostic reports shared-layer and private-main/private-auxiliary output-row norms separately. It does not reconstruct historical Adam moments or perform an optimizer step. Opposing predictions are useful: persistent shared conflict strengthens the capacity-allocation story; large private auxiliary gradients with similar shared directions strengthen a clipping-coupling story; small auxiliary gradients weaken both. None decides the eventual score.

## A more consequential alternative: train the member for its fixed ensemble

Standalone un-clipped correlation is invariant to \(p\mapsto ap+b\) for positive a. Its gradients have zero sum, and the radial scale direction is flat, apart from numerical stabilizers. In particular, the main output bias has essentially zero data gradient under this objective; AdamW can still shrink it. Auxiliary supervision does not directly calibrate that bias because auxiliary outputs have their own rows. Shared-head changes can change relative predictions, but do not solve the member-scale identification problem by construction.

The submission averages **raw** member predictions and then clips the aggregate. Scaling one member changes that ensemble even when its standalone correlation is unchanged. A member can improve standalone WP while becoming a worse fixed-weight partner; the earlier auxiliary-sign experiment E-244 already illustrates the single-versus-pair distinction, though it does not identify this scale mechanism.

A separate reserve experiment would freeze other three members and optimize

\[
e_i=\tfrac14\{p_\theta(x_i)+\operatorname{stopgrad}\sum_{j\ne\theta}p_j(x_i)\},
\quad L_E=1-\tfrac12\sum_k\rho_{w_k}(e_k,\bar y_k).
\]

Then \(\nabla_\theta L_E=\sum_i\tfrac14(\partial L_E/\partial e_i)\nabla_\theta p_i\). The loss gradient is computed at the ensemble's centered mean, covariance and variance, not the member's. The factor 1/4 alone is not the improvement: the **direction changes because the other predictions are present**. Member scaling now changes its weight relative to fixed partners and is no longer an invariant; a uniform additive offset remains an invariant until clipping matters.

An exact clipped variant inserts \(\operatorname{clip}(e,-2,2)\), giving zero derivative outside the clipping interval almost everywhere. That may align the scorer but can prevent recovery of saturated rows. Do not silently use a straight-through derivative and call it exact. The first ensemble-aware study should freeze whether it keeps the current un-clipped surrogate or uses exact aggregate clipping before fitting; it must not select between them on outcomes. This reserve is not bundled into the auxiliary-off ablation. If it is later allocated, include a cheaper fixed affine control p_k=a_k*p_initial,k+b_k (two parameters per target, four total), fitted under the SAME ensemble-aware objective with fixed partners and schedule/budget. A full-head gain that the affine control matches is calibration, not evidence of improved nonlinear representation. The intervention can still be useful; its explanation must narrow accordingly.

There are further controls: partner predictions must use the same fold exclusions, may use training-side fitted predictions only for training, and must be frozen. In-sample partner errors may be easier to cancel than unseen errors, so this can overfit. Changing two members sequentially also changes the second member's objective; begin with one named slot and fixed partners, not a coordinate-descent sweep. Any score gain must be measured on the exact unchanged ensemble weights and final clipping order.

A cheaper capacity-control reserve freezes `head.0` as well as the trunk and updates only the final affine map. This tests whether readout movement needs nonlinear head adaptation. It is not the same as the existing frozen-GRU head FT. If auxiliary private rows remain trainable, global clipping still couples them to the main outputs. Removing auxiliaries and freezing `head.0` together changes two factors, so it cannot be used to explain the original ablation after the fact.

## What prior experiments actually constrain

| Evidence | Exact scope | Implication here |
|---|---|---|
| E-097/E-102 | Auxiliary-next addition in older parent-training recipes; pooled gains around .001 | Positive prior for keeping parent auxiliaries; does not test removal during head-only FT |
| E-148 | Older true-mask head-only/LP-FT versus parent/full FT; head gain .0009 ± .0006 at exact mask | Supports a readout lever, but wrong generation and objective for the proposed control |
| E-171 | EMA-parent LP-FT objective comparison; gcorr fold deltas +.0005/+.0004, pooled reported +.0003 ± .0003; mixed wMSE+gcorr slightly negative | Keep gcorr fixed; changing objective and auxiliary coefficients together would repeat an ambiguous combination |
| Current journal, head/full FT | h128 head versus full: fold signs mixed, pooled +.0002; full FT erodes width benefit | Freezing trunk is supported as a practical choice, not proof that its representation is universally optimal |
| E-180 | Running wMSE normalization in parent EMA training: essentially zero, poor pair partner | Common scalar loss normalization alone has weak prior; no evidence against removing distinct auxiliary directions |
| E-244 / E-251 | Added sign/quote-path auxiliaries; sign single +.000656 but pair misses .0005 gate; quote-path pair −.001435 | Better auxiliary prediction or a better single is insufficient; serving-mixture contribution is required |
| M01 / I1 / new TFM screen | Reused artifacts, seed heterogeneity, failed teacher-mixture allocation gates | No justification for a large new teacher or a claim of replicated improvement |

Historical numbers above are evidence indexes, not newly computed outcomes. The raw current head logs and source were inspected; older E-entries remain ledger-reported results. Checkpoint/log SHA256 for the current plain parents and one-epoch controls already exists in `../2026-09-30-creative/M01-checkpoint-provenance.json`.

## The next paired experiment must represent V1, not quietly substitute C

The remote `candV1_ft2/pkg/members.json` names `find256_s3h2`, `fin128_s3h2`, `fin128_s4h2`, `finc500b_s3h2`, all .25. C uses the corresponding one-epoch `h` children. `final_ft.sh` initializes the two-epoch fit from the **same parent epoch3**, with a new two-epoch OneCycle schedule. It is not an extra epoch appended to the one-epoch child. The logged first-epoch learning rates differ materially. Therefore the previous one-epoch C-only proposal is downgraded to a historical proxy; it is not an adequate primary promotion comparison for public V1.

The proposed smallest V1-aligned study is:

1. Freeze ordinary h128 parent seeds3/4, folds0/1, parent epoch3. Run **two epochs from each parent** with auxiliary loss scale0 versus1, same initialization/data order, fixed epoch1 endpoint. This is eight head fits, four matched treatment/control pairs. No new parent fits.
2. Build the four remaining fold-specific **two-epoch control anchors**: distilled h256 seed3 and chunk500 seed3, folds0/1. Their losses stay unchanged. This supplies a two-epoch OOF V1-like reference instead of mixing two-epoch candidates into one-epoch C anchors. Source parent identities must be verified from the corresponding logs; do not substitute a similarly named run.
3. Primary serving contrast: replace both ordinary h128 slots in that fixed four-member reference, keep .25 weights and other members unchanged. Report each slot's paired effect separately and both-slot effect. Report standalone paired seed deltas as secondary; seed3/4 are two optimization realizations, not four independent replications merely because each has two folds.
4. Score exact pooled clipped WP on pure chains, folds0/1 and pooled; paired whole-chain bootstrap conditional on predictions; report sample SD of the two paired seed effects separately. No best epoch, scale, seed or mixture search. All evidence remains exploratory.
5. Suggested spending threshold: both-slot pooled gain at least .0003 with positive sign on each fold. Every seed's standalone, single-slot replacement and combined-slot deltas must be reported. A worse standalone seed does not automatically veto a useful fixed-ensemble improvement: the competition objective is the serving ensemble. Seed or slot heterogeneity weakens the robustness claim and motivates replication, rather than silently adding another positivity gate. This is a pragmatic no-added-serving-cost allocation rule, not significance or a claim of public transfer. No submission follows automatically. A failed primary gate closes this specific two-epoch ablation; it does not prove auxiliaries necessary.

This is **12 head fits**, not the earlier four-candidate estimate, because the current comparator lacks two-epoch OOF controls and needs a faithful ensemble. A smaller one-epoch study is possible only if explicitly labeled a cheaper preliminary proxy, not used to claim superiority to V1. The first study modifies only the two plain slots; it does not assume the result transfers to the distilled or chunk500 heads.

Minimal implementation: add `--aux-loss-scale`, default1, multiply the existing structure and next-change loss terms only. Keep `.3/.3` constructor flags so n_out stays23, preserve parameter groups and random draws, and compute the same labels/output graph even at scale0. This keeps zero gradients rather than accidentally changing None-gradient behavior. Keep global clip1, optimizer, weight decay and OneCycle exactly matched. Do not use `fold_ft.sh` unchanged because it also runs an unwanted full-FT stage; invoke the head stage directly with `--epochs 2 --freeze inp,gru --lr 5e-4 --loss gcorr --mask-weight --batch 16` and otherwise replay the logged fold flags, adding `--aux-loss-scale 0|1`.

Required deterministic checks before allocation: scale1 reproduces original loss/gradients on a fixed batch; scale0 equals main-only gradients with identical output dimensions; checkpoint strict-load and exported two-output shape are unchanged; frozen tensors remain byte-identical; exact data-order IDs and schedule are logged. Export already slices `head.2.weight/bias[:2]` (`scripts/package_submission.py:43–44`), and the current C++ packager imports that same export. No serving operator/state/shape is added. Weight-change parity and full ensemble clipping order still need existing packaging verification.

## Cost grounded in the logs

The logger's `seq/s` is cumulative `(batch+1)*batch / elapsed` for the current epoch (`train_rnn.py:1099–1102`). At batch80 of the four ordinary fold head stages it reports 57.3–60.4 seq/s: 1,296 processed sequences in roughly 21.5–22.6 seconds. Evaluation adds about12 seconds. The entire `ft128_f0` script, including head stage, full FT and two prediction passes, ran 09:34:21–09:36:23 UTC:122 seconds. That is not a head-only timing and must not be quoted as one.

For the actual two-epoch final h128 head, batch100 reports59.8/60.9 seq/s: approximately27.0/26.5 seconds through1,616 sequences in each epoch; evaluation is11.0/10.3 seconds. Extrapolating to1,872 processed sequences suggests roughly83 seconds for training plus both evaluations, excluding process startup and prediction. The distilled h256 and chunk500 two-epoch logs run around28 and24 seq/s respectively: approximately133 and156 seconds for two training epochs on all valid rows, before evaluations/startup. These are historical GPU-path observations under that host/load, not CPU forecasts.

For twelve fold head fits, plan on roughly **15–25 GPU-minutes plus prediction/startup/I/O**, with a separately registered hard budget before launch; uncertainty is material and no GPU allocation exists in this round. Reusing frozen parents is the actual economy. Caching every hidden row would consume many GiB and is not automatically justified; this round's diagnostic keeps only16×4000×256 float32 head inputs, about62.5MiB, within the1GiB new-disk budget and without writing a hidden-state cache.

## Completed bounded diagnostic: local shared conflict is not supported

Root executed the frozen TDIAG1 implementation after commit `9981ba2`; [raw result](training-result.json), [input inventory](training-result.inputs.json) and [resource log](training-run.log) are retained. Execution took **3.96 seconds internally / 6.21 seconds process wall time**, maximum RSS **1,122,292 KiB (about1.07GiB)**, exit0. There were no optimizer updates, GPU jobs or held-out WP queries.

| Training-side window | Main/structure shared cosine | Main/next shared cosine | Main/combined-aux shared cosine | Clip factor current / aux-off |
|---|---:|---:|---:|---:|
| rows0–1999, warm-up excluded | +.1100 | −.0082 | +.0684 | .74576 / .80763 |
| rows2000–3999 | +.3079 | +.1622 | +.2943 | .50862 / .53715 |

The summed auxiliary direction is **positively aligned in both windows**. This fails to support the original generic shared-gradient-conflict story at these inspected points. It neither establishes that auxiliaries are useful nor rules out component, later-training or subgroup conflict. The next-change component is nearly orthogonal/slightly negative in the first window; do not elevate that selected component to a rescue claim.

The diagnostic confirms actual clipping coupling. Main-only full-head norms are1.238/1.862; private auxiliary output rows contribute nonzero norms (.316/.343 from structure and .364/.434 from next-change). Auxiliary gradients on the private main rows are exactly zero, as the computation graph predicts. Thus auxiliaries influence shared features and the global clip, while the two main output rows receive only the main data gradient. The changed clip factors are **not proportional changes of Adam's effective parameter step**; constant gradient rescaling can be absorbed by adaptive moments. No optimizer-response experiment was run.

No scored prediction in either window exceeded |2|. This provides no immediate clipping-tail motive for changing this parent under these batches, but says nothing about later head-FT outputs or aggregate V1 tails. The logged final fits have nonzero >2 rates; those are a different model/data state.

These are **16 class-balanced distinct FT-training chains, one seed3/fold0 parent and two early sequence windows**, all at the start of head fitting. They are not a representative sample of the two-epoch training trajectory, deployed ensemble, natural class proportions or held-out chains. Full20k targets were used solely to construct legitimate training labels; raw trunk inputs stopped at row3999. Float32 CPU execution and the absence of Adam states limit optimizer conclusions. Independent code review by the architecture agent found masks, offsets, shared/private slicing and prefix replay coherent.

## Decision after deeper inspection

Retain the auxiliary-off experiment on low implementation/serving cost, with a weaker mechanism prior after the positive-alignment diagnostic; interpret it as a fixed training intervention rather than an identified conflict remedy. Upgrade its eventual comparator to two-epoch V1-like controls. Use the bounded gradient audit to distinguish plausible mechanisms, not as a score oracle. Keep ensemble-aware head fitting as the technically stronger next alternative if the auxiliary intervention is flat: it addresses an exact standalone-versus-ensemble invariance mismatch, while carrying its own in-sample-partner and clipping risks. Neither reserve is activated by this dossier.
