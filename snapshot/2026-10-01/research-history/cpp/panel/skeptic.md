# Skeptic panel: validation, domain shift, decision statistics (2026-09-26)

VERDICT on "H-valid-like: test is closer to valid than train, so E-144 was 3x the fold estimate": **KILL**.
The 3x claim compares the LB against the wrong pair and the wrong cell. The same-pair out-of-sample
estimate (+0.0011 +- 0.0006 at lambda 0) lies 0.4 sigma from the LB (+0.0014). A narrower claim is
supported instead: **the test scoring mask follows the same rule as valid's.** Replace the stress
panel with a fixed full-valid cell at lambda 0.02 (below). No training was run. All numbers come from
cached per-sequence sums, chain bootstrap (100-300 draws), and `scripts/limited` at 3G/10% CPU.

Method. `scratchpad/build.py` makes one pass over 16 cached `pred_*_valid.npy` (mmap, 20 sequences
per chunk) and produces per-sequence six-sum stats for singles and blends. They match
`cache/seqsums_*` exactly (max abs diff 0). Other scripts: `calib.py` (LB-pair cells), `mech.py`
(E-144 mechanisms), `panels.py` (tercile and weighted panels, leave-one-out), `extra.py`, `prox.py`
(chain links of holdout to dev). All live in the session scratchpad
(`/tmp/claude-1000/-home-vlad-development-wunder-wnn-connectome-starterpack/2bc856f8-1a9a-46aa-919e-5dff974b6961/scratchpad/`),
and helpers are imported from `scripts/decide.py`. The rarity linker is the validation scout's `scratchpad/chains.npz`.
OOS analog of each LB pair (every model scored only on sequences it never trained on):
P1 base-ab18 = same models, full valid. P2 final_tv-base = ab26_tv_oof (train+valid 4-fold OOF) - base.
P3 blend-final_tv = 0.5 tv_oof + 0.5 base vs tv_oof. P4 s4b35-blend = 0.65 ab29_mask_fk + 0.35 base vs
0.5 ab26_tv_fk + 0.5 base on own held-out chain fold k in {0,1}. P5 ft-s4 = same with ab39_ftm_fk vs ab29_mask_fk.
LB-noise SE of each pair for a 1,873-sequence test (analog SE scaled by sqrt(n/1873)):
P1 .0017, P2 .0016, P3 .0006, P4 .0003, P5 .0004. This assumes the public LB scores the full test set
(runtime suggests 31-37M rows). If it scores only a subset, these SEs are larger.

## 1. Q1: what explains E-144, and is the test closer to valid?

| hypothesis | prediction | test (cached data) | result | status |
|---|---|---|---|---|
| H-cell/noise: the LB gain is ordinary; "3x" comes from comparing against the stress lambda 0.05 cell and a different holdout pair | same-pair OOS estimate within ~1 sigma of LB | P5 analog in every cell | lambda0 +0.0011+-.0006 (z +0.4); lambda.02 +0.0008 (z +0.8); stress lambda.05 +0.0001 (z +1.5). The exact LB pair on holdout (ab40_b65 - ab33_b65) is **+0.0012+-.0004**. The "+0.0009" in E-142 was vs the 0.70 blend | **supported** |
| H-streams: test chunks sit next to valid chunks in the same streams (the fine-tune then leaks like a chain neighbour) | (a) valid chains more fragmented than train; (b) fine-tune gain larger on holdout sequences with a dev neighbour | (a) successor rate per class: L1 linker `cache/chains.npz` train .74/.66/.79/.72 vs valid .79/.76/.75/.56; rarity linker train .74/.61/.69/.90 vs valid .75/.63/.56/.90. (b) holdout linked (n=287) +0.0014+-.0004 vs unlinked (n=81) +0.0007+-.0011; difference +0.0006+-.0010 | (a) no chunk-level interleaving: a 50/50 valid/test chunk split would roughly halve valid's link rates. The only local excess break is in one class, and the two linkers disagree on which (c3 L1, c2 rarity: ~40-80 breaks). (b) null. **not supported** |
| H-domain: the fine-tune gain comes from the valid domain, so it grows with valid-likeness | fine-tune gain rises with adversarial p(valid) | fine-tune gain by within-class adv tercile (train-like / mid / valid-typical), blend .65, lambda0 | K-fold ftm-mask +.0019/+.0019/**-.0007** (+-.0009-.0010); E-124 pair +.0021/+.0021/+.0001; holdout +.0018/+.0010/+.0008 | **refuted** (inverse sign, ~2 sigma, replicated 3x) |
| H-calib: the fine-tune fixes per-class scale or level | cross-fitted per-class affine recalibration of the pre-fine-tune model reproduces the gain | weighted-LS per class/target, chain-half cross-fit, 20 reps | slopes 0.91-1.01 before and after; affine gain is negative for every model (-0.0003..-0.0008). The gain is +0.0012 within-sequence corr (K-fold), +0.0008 within / +0.0022 between (holdout) | **refuted** |
| H-mask-rule (new): the fine-tune specialises on the scored-row selection, and the test mask obeys the same rule | gain on scored rows, loss on unscored rows; the LB matches lambda ~ 0 | split by row type | scored/unscored-only/all rows: K-fold +.0011/-.0039/-.0031; E-124 +.0016/-.0023/-.0016; holdout +.0009/-.0024/-.0019. LB +0.0014 implies an effective lambda <= ~0.03 for this lever | **supported** |

Level check. LB minus holdout (lambda0) per submission: base -0.024, ab18 **-0.035** (train-only),
ab21 -0.033, ab21_b50 -0.030, ab33_b65 -0.032, ab40_b65 -0.032. Valid-trained models show no home
advantage over the train-only GRU. The level is weak evidence anyway (component-level SE 0.038).

Answer to Q1. The public test behaves like **the valid population at a mild mask softening
(lambda ~0.02-0.05)**. It does not behave like the "most valid-typical" sequences, it does not look
like train, and we found no evidence that it shares streams with valid. E-144 needs no domain story.
Two things are new: the mask rule transfers, and the fine-tune gain sits in train-like and mid valid
sequences, which the stress panel drops by construction. Side facts: the adversarial score is
saturated (valid OOF p(valid) median 0.999, within-class 1/3 cut 0.996, only 1.4-4% of valid below
0.5), so the stress tercile cut ranks inside a 0.996-1.0 band. The valid-data dividend
(tv_oof - e2 = +0.0014+-.0004) is U-shaped over terciles (+.0013/+.0004/+.0023). That lever
responds to domain; the fine-tune does not.

## 2. Q2: proxy calibration and replacement

Five LB pairs against offline cells (OOS analogs; chi2 = sum z^2 with z = (LB - off)/hypot(SE_off, SE_LB);
chi2 with 5 df has a 95% point of 11.07):

| cell | P1 (LB +.0015) | P2 (+.0045) | P3 (+.0021) | P4 (+.0017) | P5 (+.0014) | chi2 | MAE | signs |
|---|---|---|---|---|---|---|---|---|
| full, lambda 0 | -.0030 | +.0086 | +.0011 | +.0009 | +.0011 | 11.3 | .0021 | 4/5 |
| **full, lambda .02** | -.0015 | +.0076 | +.0019 | +.0010 | +.0008 | **6.7** | .0015 | 4/5 |
| full, lambda .03 | -.0008 | +.0072 | +.0023 | +.0010 | +.0007 | 5.9 | .0013 | 4/5 |
| full, lambda .05 | +.0004 | +.0065 | +.0029 | +.0010 | +.0005 | 6.3 | .0011 | 5/5 |
| full, lambda .10 | +.0026 | +.0052 | +.0041 | +.0010 | +.0001 | 13.9 | .0011 | 5/5 |
| stress, lambda .05 (current) | +.0016 | +.0052 | +.0031 | +.0009 | +.0001 | 7.6 | .0008 | 5/5 |
| valid-likeness ramp, lambda .03 | +.0004 | +.0060 | +.0025 | +.0009 | +.0002 | 7.2 | .0010 | 5/5 |
| inverse ramp, lambda .05 | -.0007 | +.0078 | +.0027 | +.0011 | +.0010 | 3.6 | .0014 | 4/5 |
| LOO: pick best cell on 4 pairs, score 5th | | | | | | **12.1** | | |

Per tercile (lambda 0/.05), the reason no panel wins. P1: train-like -.0062/-.0023, mid
-.0019/+.0015, valid-typical -.0012/+.0018. P5: +.0019/+.0013, +.0019/+.0011, -.0007/-.0011.
Each lever lives in a different tercile. The stress panel gets P1 right by dropping train-like
sequences, and that same choice erases P5. Its 5/5 sign record includes P5 at +0.0001+-0.0007, which is
a coin flip.

Conclusions. (1) The 5 LB points separate lambda, not panels. At matched lambda, stress and full valid
tie (7.6 vs 6.3 at .05; 6.5 vs 6.7 at .02). A valid-likeness-weighted panel is worse than full
at every lambda. Plain full valid at lambda 0 is only marginally rejected (p ~0.05). (2) Choosing a
panel from the LB overfits: leave-one-out selection scores chi2 12.1, worse than any fixed cell with
lambda <= .05. (3) The systematic error left over beyond sampling noise (method of moments) is
tau = 0.0018 at full lambda0, **0.0006 at full lambda .02**, 0.0003 at full .05, and 0.0004 at stress .05.
(4) The choice of analog matters as much as the choice of cell. For P4 the train-only analog of E-140
gives +.0022/+.0025/+.0028 (lambda 0/.02/.05) and the K-fold true-mask analog gives
+.0009/+.0010/+.0010, with the LB at +.0017. Budget +-0.0007 for analog error whenever a final fit
has no clean OOS twin.

Recommendation ("fine-tune-aware" = do not drop the sequences where mask-selection levers act):
primary cell = **full valid, OOS models, lambda 0.02, fixed in advance**. Guard cells = lambda 0 and
stress lambda .05. Reject if either guard shows Delta < -1 SE. Always report the scored /
unscored-only split and dDelta/dlambda, because a lever with a steep negative slope is a
mask-specialisation bet. The 368-sequence holdout is too small and leaky for calibration: 78% of it has
a direct dev neighbour, and P1 on it reads -0.0096+-0.0064. Use it only for final-fit pairs.

Required offline Delta for P(public LB Delta > 0) >= 0.8 (0.9), with sigma_tot^2 = SE_off^2 + SE_LB^2 +
0.0006^2. If the candidate is the best of about 5, add the winner's-curse term 1.19 SE_off:

| pair type | sigma_tot | 80% | 90% | best-of-5 at 80% |
|---|---|---|---|---|
| GRU vs GRU, full-valid OOS (SE .0004) | .0008 | **.0007** | .0011 | .0012 |
| GRU vs GRU, 2 chain folds (SE .0006) | .0009 | .0008 | .0012 | .0015 |
| blend vs single (P3-like) | .0010 | .0008 | .0013 | .0014 |
| our GRU vs baseline family (SE .0016) | .0023 | .0020 | .0030 | .0039 |

## 3. Q3: how to spend training data (what can be validated offline)

- **Fine-tune on true masks: longer, or on all 1,873 valid sequences including the 368 holdout.**
  This is the one lever with LB-confirmed transfer, and its mechanism (mask rule) does not depend on
  domain proximity. Offline design: the K-fold learning curve on held-out chain folds 0/1. Fine-tune
  on 2 vs 3 of the remaining folds to get the data slope, and run 1 vs 2 epochs and lr 2e-4 vs 1e-4.
  All arms start from ab29_mask_fk, about 6 short runs. "All valid" cannot be validated directly: no
  OOS remains, and the holdout readout is lost. Holdout use is the user's call under
  `research/contract.yaml`. It is already weak evidence (leaky, n=368).
- **Oversample valid (true masks) in the main fit.** This can be validated with the same K-fold
  (valid x3, folds 0/1). Compare after the fine-tune, because the fine-tune may absorb it. The
  dividend pattern (valid-typical +.0023) says more valid data is a domain lever, so the expected
  value is modest.
- **Domain importance weights on train (adversarial p/(1-p)).** The effective sample size collapses.
  With a within-class AUC of 0.989-0.993 the binormal separation is d ~3.3, so ESS/n = exp(-d^2) ~1e-5
  (about 0.2 of 10.6k sequences). Tempering w^alpha keeps ESS/n at 5-7% (alpha .5), 33-39% (alpha .3),
  or 61-66% (alpha .2), and at that point the weights barely move the loss. Neither the fine-tune gain
  nor the P1/P5 LB responses are monotone in valid-likeness. Offline validation would be possible (K-fold)
  but is not worth a run.
- **Filter out train sequences unlike valid.** With AUC 0.99 almost all of train is "unlike valid", so
  filtering discards most of the data and inherits the IW objection. Kill.
- **K-fold fine-tune ensembles.** Averaging 4 GRUs breaks the latency budget (36 min now, +6-8 min per
  member). A weight soup from the shared ab33 origin with shared normalization costs no inference time,
  but no clean OOS exists for it. Validate a 2-seed soup on a held-out fold. Expect +0.0001-0.0003,
  since GRU seed diversity is ~0 (E-111).

## 4. Q4: using LB submissions as evidence without probing

1. Pre-register every upload. Log the predicted Delta (primary cell), sigma_tot, and P(LB>0) before
   uploading, and the z-score after. Refit tau by method of moments every 2-3 pairs. Never re-select
   panels, lambda, or blend weights from the LB (the LOO result above shows this overfits).
2. Make one-lever pairs against an LB-scored parent, as P5 was. Multi-lever jumps (P4) cannot be
   attributed.
3. Combine by precision weighting, not by taking the LB at face value. For P5: offline .0008 +-
   hypot(.0005, .0006) and LB .0014 +- .0004 give a posterior of **+0.0013 +- 0.00036**. With private SE
   ~.0004, P(private Delta > 0) ~0.99.
4. Adopt a new incumbent only if the posterior P(Delta > 0) >= 0.9. Among near-ties (< 0.0008 apart)
   the LB picks noise: the best of 5 at SE .0004 is inflated by ~.0005.
5. The private LB is "a separate, hidden dataset" (docs/rules.md). What the public LB told us (mask
   rule matches; test ~ valid at lambda .02-.05) is unverified for private. For the final picks, use
   the best posterior plus one hedge that is non-negative against it at lambda .05, i.e. less
   mask-specialised or with a higher baseline share.

## 5. Ranked decisions

| # | decision | cost | required effect to adopt | falsifier |
|---|---|---|---|---|
| 1 | Replace the stress proxy with the fixed full-valid OOS cell at lambda .02 plus two guard cells and the sigma_tot thresholds of section 2; retire H-valid-like as a planning premise | 0 | n/a | Over the next >= 4 LB pairs, chi2 of this cell > 9.5 (chi2 with 4 df at 95%), or stress lambda .05 beats it by Delta chi2 >= 4 |
| 2 | Tune the true-mask fine-tune schedule on the K-fold learning curve, then fine-tune the final on all valid (user's call on holdout) | ~6 short GPU runs | 2 epochs or lower lr >= +0.0007 at lambda .02, same sign on folds 0 and 1, lambda .05 >= -1 SE; data slope (3 folds minus 2 folds) > 0 | slope <= 0, or the longer fine-tune loses at lambda .05 by > 1 SE (over-specialisation) |
| 3 | Valid true-mask oversampling in the main fit, judged after the fine-tune | 4 K-fold fits | >= +0.0008 at lambda .02 over ab29+ft, both folds | gain after fine-tune < +0.0004 |
| 4 | Weight soup of K-fold fine-tune models (zero inference cost) | 2 extra fine-tunes | >= +0.0005 at lambda .02 on the held-out fold | 2-seed soup - single < +0.0002 |
| 5 | Kill domain importance weighting, train filtering, and valid-likeness-weighted panels | 0 | n/a | A tempered IW (alpha .3) K-fold arm >= +0.0008 at lambda .02 on both folds, or the next 4 LB pairs favour the valid-likeness ramp over full by Delta chi2 >= 4 |

Caveats. Five LB points constrain one knob (lambda) at most. Every tau and chi2 here assumes the
public LB scores a full ~1,873-sequence test. K-fold analogs cover only chain folds 0/1 (925
sequences). The tercile patterns are ~2 sigma each. The fine-tune/valid-typical anti-correlation
replicates 3x, but the E-124 and ftm comparisons share folds. Scoring density does not explain the
pattern (within-class Spearman of adversarial rank vs scored fraction -0.13..+0.23).
