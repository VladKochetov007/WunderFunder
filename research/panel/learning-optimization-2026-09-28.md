# Learning and optimization round — 2026-09-28

## Exact public-pair merge result — 2026-09-29

The two-fold merged-model result initially appeared positive against the
`ab82_ft_since4s3 + ab82_ft_base4s4` research reference (+0.001105 pooled
exact-mask WP). That pair is different from the user's public 0.6622 package.
I re-scored the same 736 fold-0/1 development sequences against the exact
public pair (`ab67_final_since_s4_gcorr + ab67_final_since_s3_gcorr`) and
scanned merged weights 0.025–0.50. Every weight was negative on both folds,
exact mask, and lambda-0.02 stress. At weight 0.025 the exact delta was
-0.000094 (chain-bootstrap 95% CI [-0.000144, -0.000049]); at 1/3 it was
-0.002126 (95% CI [-0.002835, -0.001484]). The merge is rejected for the
public package. No final fit or submission bundle was made. See E-269,
`research/merge_2026-09-29/analysis_public_pair_f0_f1.json`, and the plot
`artifacts/plots/merged_public_pair_weight_scan_f0_f1.png`.

## Update — 2026-09-29

The user's latest portal report sets `final_since2_gclp.zip` at 0.6622 as the
public incumbent. `final_ema6_shared.zip` has already been scored at 0.6612 and
`final_mixed_ema6_shared.zip` at 0.6616, so neither is a fresh submission
candidate. The screenshot says 1/5 submissions used today; its date boundary
is not independently verified. See E-267. Do not upload either scored package
again.

The earlier EMA-8 run is no longer active and has no completed comparable
checkpoint/score: the last recovered progress was epoch 3, batch 200, with an
epoch-2 checkpoint; why it stopped is unknown (E-268). The teammate/ours merge
in `research/merge_2026-09-29/` completed both folds with a GRU 160x2, first
differences, 2,000-row resets, corrected pooled Global WP loss, cross-fitted
propensity weights, and our causal since weighting. It improved the research
since/base pair but failed against the exact public pair; see the update above
and `CONTRACT.md`.

## Agent status

Four archived agent threads returned final reports before the session boundary: three idea scouts and one skeptic/diagnostic report. The old agent handles are no longer available to the current agent-control service (`not_found`), and there are no separate scout worker processes in the OS process list. Their final messages were recovered from the local Codex session archive. No new scouts were started.

The independent ideas with the most useful mechanisms were:

1. Factor class-3 prediction into first-move side and conditional continuation/tier.
2. Jointly model event mark and waiting time with a competing-risk objective.
3. Predict future observable event features as a training-only auxiliary.
4. Pool causal event innovations or vary TBPTT boundaries to change the retained state/gradient path.
5. Test depletion/refill and quote-survival signals before fitting a new model.

These are proposals, not demonstrated gains. The tier and future quote-path training heads have already missed paired gates; future ideas must specify a causal input signal that passed a separate residual-information screen before another full fit. The marked-event literature motivates a representation for asynchronous events, but does not establish score improvement here ([Wang et al., 2017](https://arxiv.org/abs/1701.03918)).

## Width-192 arm

The running `ab81_h192_f0` fit completed at fixed epoch 5. On the fold-0 validation set it scored 0.673467, versus 0.677524 for the matched h128 EMA-6 model. Replacing the second h128 seed in the equal-weight pair with h192 changed exact-mask global WP by −0.00139 ± 0.00130 on 432 pure-chain sequences. The λ=.02 and .05 deltas were −0.00112 and −0.00080. The model fails the fold-0 continuation gate; no fold-1 run is warranted. Results are in E-262 and `research/artifacts/ensemble_scan_h192_f0.json`.

## Visualization archive

All canonical plots previously under `artifacts/viz/`, `research/artifacts/`, and `research/visual/`, plus the three active notebook plots, are available under `artifacts/plots/`. The notebook saves new learning-curve, fine-tune, and leaderboard-calibration figures there and refreshes `research/artifacts/plots_manifest.md` after each save. The archive currently has 45 PNG figures plus a vector PDF export for the new MLP screen. The learning-curve plot now includes h192.

## Active work

The older q56/q50 queue completed the MLP, LR/batch, and EMA-8 status checks. The EMA-8 process is stopped without a final checkpoint or score (E-268). The merge analysis is complete and no research GPU job is active.


## Submission candidates

The official Connectome timeline lists a final submission deadline of November 15, 2026, and the FAQ allows five submissions per day; current account usage is not visible here ([timeline](https://wundernn.io/connectome/docs/timeline), [FAQ](https://wundernn.io/connectome/docs/faq)). The live board snapshot at 2026-09-28T21:18:14Z has 3,725 participants and a top score of 0.667720; the rounded 0.6622 reference is approximately rank 21. See `research/artifacts/leaderboard_snapshot_2026-09-29.json` (E-266). The upload proposal and candidate ZIP bundle are at `artifacts/connectome_submission_proposals_2026-09-28.tar.gz`, with rationale and SHA-256 values under `artifacts/submission_proposals_2026-09-28/`. Upload the nested ZIP, not the tarball.

`final_e6x2_base25` has the strongest calibrated public-score forecast from E-261 (mean +0.00107 over its five best proxy variants), but its three-member package changed to one member after 40,000 rows in the capped verification run. `final_since2_all` similarly changed from two members to one. Both require a deterministic runtime-safe repack before upload. The prior `final_ema6_shared.zip` package passed local scorer-image timing at 43.3 minutes and prediction correlations above 0.99996, but it has since received a public score of 0.6612, below the 0.6622 incumbent (E-267). It is not a new submission candidate. The old proposal tarball's EMA6 recommendation has been superseded; the teammate/ours merge failed against the exact public pair and has no upload candidate (E-269).

The MLP arm completed at fixed epoch 5 with broad fold-0 WP 0.676427 (matched h128 comparator 0.677525). In the predeclared equal-weight pair screen on 432 pure-chain sequences, replacing the second EMA-6 member changed exact-mask WP by +0.000152 ± 0.001150; softened-mask deltas were +0.000155 ± 0.001072 and +0.000158 ± 0.000989. The train-mix exact-mask delta was -0.000267, with a class-3 regression. It misses the +0.0005 pair gate and is rejected before fold 1 (E-264). The plot is saved at `artifacts/plots/mlp_pair_screen.png` with a vector PDF alongside it.

The LR/batch arm (lr=2e-3, batch=64) failed at its first batch with CUDA OOM under the 0.9 per-process memory cap; this is a resource failure, not a model result (E-265). The old EMA-8 arm later stopped without a final checkpoint or comparable score (E-268). The active merge fit remains under the hard project ceiling in `src/wnn/resources.py`; do not raise it. The batch-64 hypothesis remains untested.
