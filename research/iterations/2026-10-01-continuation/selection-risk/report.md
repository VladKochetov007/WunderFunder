# SELRISK01 outcome — 2026-10-01

**Conclusion: mixed and inconclusive.** Adding contemporaneous outcome features to the finite selector did not consistently improve the sampled final-head loss-gradient approximation. This diagnostic gives no convincing broad support for spending a forecast fit on outcome-conditioned weighting; it does not establish that such a mechanism is impossible. Root retains the spending decision.

The frozen comparison used two existing out-of-fold h128 H2 models, 128 pure chains (16 per class per fold), one whole 20,000-row causal prefix per chain, and three independently sampled 256-row panels. Each chain was audited once; fit/calibration chains and audit chains were disjoint. qX received 341 causal input/state/prediction features; qXY added six clipped-outcome features. The shuffled-Y selector had the same feature count and learner capacity as qXY and used within-role/class deranged target streams. No selector received the scoring mask as an input.

For each target, the final-layer numerator is `2 Σ a q |clip(y)| (p−clip(y)) [h,1]`. Both target numerators use the same denominator `Σ_rows,targets a q |clip(y)|`. Numerators and denominators are summed across chains before division. The quantity below is `e_control,A · e_control,B − e_XY,A · e_XY,B`, where each error compares its gradient with the actual-mask gradient. Positive values favor qXY. Fold coordinates belong to different models and are not pooled.

| Fold | qXY versus qX | Conditional 95% CI | qXY versus shuffled Y | Conditional 95% CI |
|---|---:|---|---:|---|
| 0 | +0.01820 | [-0.22386, +0.08217] | +0.02274 | [-0.20082, +0.09109] |
| 1 | -0.00704 | [-0.03532, +0.01412] | -0.00727 | [-0.04049, +0.01404] |

The panel-specific squared-error reductions versus qX were −0.00965 / +0.05684 for fold 0 (A/B) and +0.00206 / −0.01845 for fold 1. Thus the point estimates do not provide a stable improvement across independently sampled rows. The class-3 subset contrast was positive in both folds; classes 0, 1 and 2 were negative in both. These subsets use their own denominators, are not additive contributions, and do not justify selecting a class-specific recipe from this exposed outcome.

All 24 selector fits and 24 calibration fits completed with the frozen arguments and no captured warnings. The conditional bootstrap resampled whole chains within class, using the same draws for all selectors and both panels. Its intervals condition on already-fitted selectors; they omit refitting, seed, model-selection and domain uncertainty. Panel ratios retain finite-sample bias. The fixed class-balanced sample is not an estimate of an unknown test mixture. This is a finite-model weighted-MSE gradient surrogate, not WP, a training intervention, or a diagnosis of true MNAR.

Execution: extraction 13.19s, analysis 128.25s; 2,560,000 causal prefix rows and 98,304 sampled records. Sampled own GPU peak 282 MiB; extraction maximum allocated/reserved 61.94/84 MiB. Host peak 1.420 GiB for extraction and 0.319 GiB for analysis. Both stages exited 0 inside the 600/300-second budgets, actual cgroups enforced two CPUs/4 GiB/swap0, and analysis had an empty CUDA device list. No forecast fit or WP query occurred.

Resources released: extraction 2026-10-01T13:07:05.312536+00:00; CPU analysis 2026-10-01T13:09:14.028119+00:00. The launcher recorded each worker absent and each scope inactive/dead. A later redundant SSH check timed out; it does not alter those already-recorded release checks. No retry occurred.

Identity coverage: small metadata/checkpoint files and all registered sources were hashed before/after extraction. The valid parquet used a previously verified full SHA plus the current inventory inode/size/mtime and a fresh 64 KiB tail hash; no fresh full-file hash was claimed. Sample-cache hashes were checked before/after analysis. X/Y/M caches remain in the remote execution directory; the report, chain sufficient statistics, manifest and receipts are also copied locally.

Bound evidence:

- Main report: `execution/report.json`, SHA256 `6fd7ec9fd552e726909f2af3d2fccc0085525ca05e6b7d1442f1248e7ab8e389`.
- Chain sufficient statistics: `execution/chain-sufficient.npz`, SHA256 `54fca776ad556bea7afe8fd98063b31980249c2faad392031e1024bbd3dc7aa8`.
- Extraction receipt: `execution/extraction.json`, SHA256 `ebff166bf9f339334264c8c93ef76304ab441732fbb4621f691608484820693e`.
- Launcher receipt: `launch-result.json`, SHA256 `ee0e5dd52cf928d47c98d61d5abf62d28bf1433ce03a2faade75c164f987d9c1`.
- Frozen bindings: `bindings.json`, SHA256 `0b6b65c9cd81e906f790d79540ded2481f57b82251af6ef70763af7ed0b750c3`.
- Allocation: `allocation.json`, SHA256 `3df07ad20960664ed91ac205cd2cb68ca79cc5dc81c08598e4c80cb153236440`.
