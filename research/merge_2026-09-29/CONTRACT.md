# Merge experiment contract — 2026-09-29

## Question

Does the teammate's compact raw-plus-first-difference GRU with periodic state
resets gain when paired with our causal recency weighting and cross-fitted
scoring-mask propensity?

## Experiment

- **Candidate:** GRU 160 x 2, projection 64, raw plus first differences,
  reset state every 2,000 rows, pooled Global WP loss, EMA, propensity-weighted
  scoring mask, and our `since_i0px` loss multiplier with tau 32 and floor 0.25.
- **Train split:** all usable training sequences plus validation development
  sequences outside chain fold 0. Corrupt train row group 528 and the reserved
  validation holdout are excluded.
- **Evaluation split:** chain fold 0 within validation development only.
- **Reference pair:** equal mean of `ab82_ft_since4s3_f0` and
  `ab82_ft_base4s4_f0`, each evaluated on fold 0.
- **Candidate pair:** equal mean of `ab82_ft_since4s3_f0` and the merged model.
- **Primary metric:** current official Global Weighted Pearson, with clipped
  targets and predictions on the true validation scoring mask.
- **Secondary checks:** lambda 0.02 unscored-row stress, pair bootstrap by
  sequence chain, per-class pooled score, model runtime, and memory.

## Promotion gate

Run fold 1 with the same fixed recipe only if fold 0 replacement improves
exact-mask global WP by at least +0.0005, lambda 0.02 is positive, and no
market class loses more than 0.002. Consider final fitting only if the pooled
fold 0/1 result is positive with chain-bootstrap support and scorer runtime
stays below 60 minutes.

### Fold-0 outcome and secondary path — 2026-09-29

Replacing the reference base member with the merged model scored only
+0.000005 at the exact mask, so that primary replacement path stops. The
predeclared three-member mean (reference pair plus merged model) scored
+0.000591 exact-mask WP and +0.000736 at lambda 0.02. Its class deltas were
+0.000376, -0.000251, -0.000584, and +0.001959 for classes 0–3. This clears
the same point-estimate and class-safety screens, so one fold-1 fit is authorized
for this secondary three-member blend only. The fold-0 chain-bootstrap 95%
interval includes zero; promotion still requires a positive pooled two-fold
result with chain-bootstrap support.

## Resource and integrity limits

- CPU quota: at most 80% of logical CPUs.
- Host RAM: systemd memory cap 30 GiB (below the user's 80% ceiling).
- VRAM: PyTorch allocator cap 90% of the 6 GiB device.
- No access to the reserved validation holdout in training, model selection,
  plots, or candidate comparisons.
- No public submission is made by this experiment.

## Prior evidence

- The teammate's `distil160` report gives 0.6888 on its 192-sequence selected
  validation slice. That slice was used for epoch selection, so it is not an
  independent comparison for this experiment.
- Our E-182 supports the causal `since` weight on two OOF folds; E-258 supports
  structurally different recurrent members and the since-based reference pair.
- Our E-102/E-116 found no broad gain from distilling a larger teacher alone.
  This experiment therefore tests the representation and recency weighting,
  not a new distillation-only run.

## Metric implementation audit

The teammate source constructs `GlobalPearsonLoss` under `--global-loss` but
does not set `--seq-loss`, despite the help text saying global loss implies it.
The experiment copy sets it explicitly whenever global loss is requested and
logs both global and legacy per-sequence diagnostics. Official Connectome docs
currently specify pooled Global WP over all selected rows.

### Exact public-incumbent pair follow-up — 2026-09-29

The original fold comparison used `ab82_ft_since4s3 + ab82_ft_base4s4`, which
is not the pair in the user's public 0.6622 submission. Before final fitting,
`analyze_public_pair.py` re-evaluated the merged fold predictions against the
exact public pair `ab67_final_since_s4_gcorr + ab67_final_since_s3_gcorr` on the
same 736 development sequences in folds 0–1. It scanned merged-model weights
from 0.025 through 0.50, including the equal-three-member weight 1/3.

Every tested weight reduced exact-mask and lambda-0.02 Global WP on both
folds. At the smallest tested weight (0.025), the pooled exact-mask delta was
-0.000094 (chain-bootstrap 95% CI [-0.000144, -0.000049]); lambda-0.02 was
-0.000063 (95% CI [-0.000109, -0.000018]). At weight 1/3, the exact-mask
delta was -0.002126 (95% CI [-0.002835, -0.001484]); lambda-0.02 was
-0.001860 (95% CI [-0.002536, -0.001211]). All four class deltas were
negative even at weight 0.025. Thus the earlier +0.001105 pooled result is
specific to the weaker since/base reference pair and does not transfer to the
public incumbent pair. Reject this merge for the current package, do not run a
final fit, and do not build a submission bundle. The weight-scan plot is in
`artifacts/plots/merged_public_pair_weight_scan_f0_f1.png` and `.pdf`;
machine-readable results are in `analysis_public_pair_f0_f1.json`.
