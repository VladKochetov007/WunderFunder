# Late categorical residual on the fixed strong backbone

Objective mode: competition. Tier B exploratory screen on repeatedly exposed folds, not fresh confirmation. Improve the qualified fixed four-member ensemble under its serving budget. Current GPU OOF WP is 0.7023796676142182; the native package remains unchanged until quality and runtime evidence warrant replacement.

## Question and intervention

Can categorical supervision learn a useful additional readout of the current shallow256 recurrent state? This is not the failed posthoc decoder replacement: the entire current H2 backbone, regression head, auxiliary head, normalization and causal state evolution remain frozen. No full backbone training is authorized by this contract.

Add one 512-to-64-to-K categorical MLP and a scalar residual gain initialized to zero. The initial forecast must equal the saved baseline exactly. With clipped atom c and amplitude a=mean(abs(c)), decode softmax(logits)/a through per-target absolute-weighted clipped moments. The correction is gain * frozen_p3 * (decode - baseline_prediction). Gain is unconstrained and has a nonzero derivative at zero. Serving p3 uses only frozen predicted class logits; retrospective class labels enter only training supervision and descriptive evaluation slices.

Candidate: ordinary gcorr main loss plus 0.5 times class-3 amplitude-weighted categorical CE. Control: identical architecture, initialization, sampling and main loss, CE coefficient zero. The control logits initially get no main-loss gradient at zero gain; this optimization asymmetry is part of categorical supervision, not an implementation mismatch. Control probabilities are not claimed to be calibrated categorical probabilities.

One common fixed dictionary of 64 atoms is derived from original-train chains with no validation member. Read only training target columns, excluding corrupt row group 528 and warmup rows. Rank canonical class-3 tick pairs by total mean absolute clipped-target weight, break ties lexicographically, exclude zero-weight support. No validation target access or support retuning. Unknown support has zero CE weight but retains main loss. Report vocabulary coverage without selecting a new vocabulary from it.

## Fixed trial budget and metric

Four head fits: candidate/control, folds 0 and 1, seed 3. Start at cont_sh256_s3h2_f{fold}/model_ep1.pt. Train on the same chain-disjoint allowed validation training pool as ordinary H2, full causal sequences, batch16, chunk2000, two epochs, AdamW lr0.0005/wd0.01 with the standard OneCycle schedule, fixed final epoch1. No epoch/seed/head-width/loss-weight/gain selection. Two target-free OOF passes, each evaluating both new heads and the frozen baseline from identical hidden states, with full causal context and compact scored-row storage.

Primary comparison: candidate versus incumbent in the fixed ordered float32 quarter ensemble, using the same three frozen anchors. Secondary comparisons: candidate versus CE0 control and control versus incumbent. Also reproduce the original frozen backbone prediction under the new path. Use all 860 pure sequences, 250 chains, 1,614,484 scored rows; target/fold and existing class3 age>=64 diagnostics. Keep final ensemble/target clipping and the frozen B evaluator definition. Paired 2,000 whole-chain draws stratified by fold/class, seed20261001. These intervals condition on fixed fitted models; they do not include training variability, adaptive selection or test shift.

A negative primary estimate with no convincing diagnostic reason closes this exact recipe without a head sweep. A positive result is weighed by size, conditional uncertainty, fold/target consistency and serving cost; it does not require an arbitrary universal score threshold. Any follow-up seed or native implementation gets a separate explicit bounded allocation. No public submissions or GitHub publication. The closed overnight archive and shared training/model source remain immutable.

## Execution limits and falsifiers

One vocabulary build: CPU2, RAM8GiB, at most600s. Fit jobs: GPU0 only after current free-memory check, allocator<=6GiB, observed ownedGPU<=7GiB, six CPU cores, RAM20GiB, each<=900s. Prediction jobs<=300s each. Sequential queue total<=80min. CPU grading2cores/8GiB/180s. Stop and preserve failures; do not silently retry scientific runs. Only terminate owned scopes. New local+SSH storage should remain below1GiB.

Before fits, verify generated decode oracle, zero-gain original-forward and state identity, nonzero gain derivative, paired initialization/order, strict original-model load and exact frozen-state preservation after an update. Stop for chain overlap, target-dependent inference, nonfinite outputs, mismatched source/input identity, resource exhaustion or initialization drift. A vocabulary with insufficient support stops this recipe. Extra dense work is 36,864 MAC/row, approximately2.73% of the current ensemble's dense MAC count, not a measured native runtime. Runtime must be measured before any delivery change.

This one screen has a modest prior: 44 previous overnight head fits and the corrected fixed-weight atom probe were unsuccessful. Its distinct test is categorical supervision of a new residual on the strongest fixed representation, with baseline behavior preserved and an equal-capacity control. The categorical objective and probability interpretation alone are not evidence of improvement.
