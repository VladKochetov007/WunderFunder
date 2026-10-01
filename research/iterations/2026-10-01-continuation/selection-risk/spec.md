# SELRISK01: bounded selection-gradient diagnostic

**Status: implementation and generated-only checking; root has conditionally authorized one bounded diagnostic after checks.** No real data/checkpoints, empirical selector fits, GPU inference or WP queries have yet been run. GPU selection requires the root-authorized fresh resource check; GPU6/7 are excluded. This is one diagnostic; no automatic forecast fit follows. The earlier scout's per-target-normalized formula was a diagnostic variant. Here the loss uses the trainer's **common denominator across both targets**.

## Estimand and exact model

Use only existing `on_ctl128s3_f0/model_ep1.pt` and `on_ctl128s3_f1/model_ep1.pt`, ordinary H2 checkpoints under `artifacts/runs/`. Keep each outer fold separate: their last-layer coordinates differ, so their gradient vectors cannot be pooled. Source path is `train_rnn.make_model` → `build_inputs` → `forward_chunk`, with frozen checkpoint normalization/flags. Assert linear encoder, H128, two GRU layers, no lookahead/class-residual head, dropout0. Use eval/no_grad, FP32 with autocast and TF32 disabled; carry state through all 20,000 rows in chunks2,000, reset only between sequences. A pre-hook on `head.2` captures its 128-dimensional input. This explicitly defined FP32 surrogate is not the archived AMP/native output.

For sampled rows i, let c=clip(y,-2,2), p be the un-clipped first two model outputs, phi=[captured activation,1], a the fixed sampling inclusion weight, and q a selector probability (or true binary M). Define

`L(q)=sum_i,k a_i q_i |c_ik| (p_ik-c_ik)^2 / max(D(q),1e-8)`,

`D(q)=sum_i,k a_i q_i |c_ik|`,

`G(q)[k,:]=2 sum_i a_i q_i |c_ik| (p_ik-c_ik) phi_i / max(D(q),1e-8)`.

This is the final affine layer's gradient of the declared pooled weighted MSE, excluding auxiliaries, since weights and H2's actual correlation objective. It is not the full historical optimizer gradient. The denominator is shared by both target rows. Keep numerator and denominator per chain/panel; pool these before division. `gradient_core.py` implements only these sums and comparisons; generated-only source checks are recorded separately. A generated CPU analytic-versus-autograd check, unequal target masses and all-zero-weight case must precede real execution.

## Fixed sampling and cross-fitting

Before any feature/label read, freeze a metadata-only manifest:16 pure validation chains per class per outer fold0/1 (128 chains total), chosen by ascending SHA256(`SELRISK01|fold|class|chain`). Select one row group per chain by the analogous fixed hash. If any stratum lacks16 chains, stop rather than alter the quota. Verify no chain has a train member or spans folds. Assign within-class hash ranks modulo4 to four equally sized groups. This is exposed validation, never fresh confirmation.

Within each fold rotate four times: audit group j, calibration group(j+1)%4, fit the other two groups. Thus each fit has32 chains, calibration16 and audit16; each chain is audited once. The frozen H2 forecaster excludes the entire corresponding outer fold. No selector/calibrator receives an audit-chain mask during fitting.

Partition rows99…19,999 into256 contiguous near-equal bins. Three separately seeded panels F/A/B each choose one uniform row per bin, independently of features, labels and masks; F is used only for selector fitting/calibration, while A/B are audit replicates; seeds and realized positions belong in the immutable metadata manifest. Use a=bin length. In each sequence perform **one whole-prefix forward pass**, gather only the union of selected rows, and discard all other activations. Each panel estimates the selected sequence's row sums, not a claim about all validation chains. The ratio gradient retains finite-sample bias.

Cache only selected causal X=[normalized211 inputs,128 head activations,2 predictions] (341 floats), targets, mask, row positions and weights. The head activation already in X is reused for gradients. At most98,304 sampled records: about128MiB for X plus small metadata/targets, not a full-sequence hidden cache. Read raw inputs/targets/is_scored from `SequenceFile.read`; masks/targets go into separate arrays and cannot enter `build_inputs` or X. No existing prediction cache is needed.

## Three selectors, fixed capacity and calibration

For every rotation fit three `HistGradientBoostingClassifier`s, with max_iter64, learning_rate.1, max_depth3, max_leaf_nodes7, min_samples_leaf64, l2_regularization10, max_bins32, early_stopping=False and fixed random_state. X-only uses341 causal columns. XY adds `[c0,c1,abs(c0),abs(c1),c0*c1,abs(c0-c1)]`. SHUFFLED has exactly XY's columns/capacity but receives a deterministic whole sampled Y-stream derangement across chains of the same class **within each role** (fit/calibration/audit separately); both target columns and panel order move together. F never supplies an audit-gradient row; A/B never fit or calibrate a selector. No M or need_prediction feature is allowed. Label-derived class metadata is used only to select/stratify/derange chains, never as a selector input.

Fit on panelF fit chains with a normalized to mean1. Separately calibrate each selector's scalar decision_function by the same two-parameter logistic calibration (C=1, lbfgs, tol1e-8, max_iter200) on panelF calibration chains, again with a/mean(a). Clip all calibrated probabilities to[1e-4,1-1e-4]. Freeze library version before allocation; no tuning, early stopping, alternate capacity, old propensity-cache substitution or AUC gate. Total24 selector fits +24 calibrations; zero forecast fits. Underfitting qX can produce a positive result without establishing true outcome-dependent selection, so interpret this as a finite-model weighting diagnostic.

## Comparison, uncertainty and decision

After assembling out-of-chain q values, compute G separately for each outer fold and each panel. Primary error is E_q=G(q)-G(M). Report E norms, true-mask gradient norm, both target-row contributions, denominator/weight mass and effective sample size. Compare qXY to qX and SHUFFLED using the cross-panel discrepancy difference `E_control,A·E_control,B - E_XY,A·E_XY,B`. Also show the ordinary squared-error difference in each panel. The separate F panel keeps both audit row draws out of selector fitting. The cross-product reduces the positive noise contribution from independently sampled audit rows; finite ratio bias and shared-chain uncertainty remain.

Use1,000 paired chain bootstrap resamples within each class/fold, fixed seed, resampling both panels together and recomputing numerator/denominator ratios. Selectors remain fixed: intervals are conditional on fitted selectors, not their refitting variability. Report all four cross-fit rotations and class contributions; do not turn a convenient subgroup into the primary result. No automatic threshold or both-fold veto is applied. Report contrast sizes, conditional intervals, panel noise and class/rotation heterogeneity continuously. Root interprets the evidence as unsupported, inconclusive or directionally promising and makes any spending decision; the program does not assign an empirical PASS or authorize follow-up. This is a finite-model loss-gradient surrogate, not identification of outcome-dependent missingness. Even a pass says nothing by itself about hidden-mask transport or WP gain.

## Bound and implementation path

Two source-separated stages: (1) `extract_selected.py` reads the fixed manifest and two immutable H2 checkpoints, processes one sequence at a time, writes the small selected-row cache; (2) `fit_selectors.py` consumes that cache and `gradient_core.py`, emits one fixed report plus chain sufficient stats. Both orchestration scripts and the fixed metadata/role helpers are implemented. A single root allocation binds versions, sources, inputs and CPU/GPU lanes; it authorizes both bounded stages. Bind source/checkpoint/dataset/chain-metadata hashes before execution and verify after. Reuse an existing verified full-file identity only if its stat/inode guard is still valid; otherwise account for a new full hash in the cap, do not silently omit identity verification.

Authorized hard caps: extraction GPU7 only after a fresh<=256MiB/no-other-compute-process check, allocator fraction<=.30 (and<=5400MiB), <=5.5GiB GPU,2CPU/4GiB host,600s; selectors/calibration/bootstrap2CPU/4GiB,300s. Each stage uses timeout TERM at600/300s and KILL after5s in its owned UUID scope, no retry and no fallback precision or smaller sample after failure. CPU34/36 are reserved away from student and critic lanes after a fresh idle sample. The small termination grace is reported, not subtracted from wall time. Expected2–6min extraction plus1–3min CPU is an **unmeasured estimate**, not a guarantee. No stream prefetcher or entire-dataset tensor allocation; max one20k×211 feature tensor and one2k forward chunk. If metadata, identity hashing, generated checks or GPU availability make this non-cheap, defer it until the student finishes.

E-103's +.0022 mask AUC and E-170's null sharper causal propensity make the prior weak. E-186 fit coarse causal residual bins and directly changed forecast outputs, losing WP. This diagnostic instead tests label-aware training weights against a fixed predictor's exact declared loss gradient, with no output correction or WP query. It materially differs from E-186 but does not erase its warning that apparent scored/unscored structure need not transfer.

Resource reassignment before empirical execution: GPU3 was occupied by a foreign process. Root assigned GPU7 only after an explicit student-completion notice and the minGRU synthetic smoke release; fresh idle checks and all budgets are unchanged. This is a prospective allocation change, with zero SEL empirical attempts so far.
