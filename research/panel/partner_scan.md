# Partner scan: which second GRU complements the EMA-4 member (2026-09-27)

Question (follow-up to E-173): with a 2-member runtime budget, which partner B works best next to the
EMA-4 member A, and what property makes it complementary? Method: exact pooled WP (lam 0.02, clipped
|y| weights, members clipped to +-2) of any linear combination, computed from per-sequence weighted Gram
matrices of [1, y, A, baseline*0.25, partners] on held-out chain folds 0/1 (`decide._chains()`), with a
paired chain bootstrap (200 draws, weights fixed at the in-sample choice). A = ab62_base_s3_f{k}. Fold models
count only on their own fold; train-only models count on both. Excluded: copies of A (ab56/ab58_ema_e4),
ab26_tv_oof (the same as ab26_tv_f{k}), ab26_tv_f2/f3, and all finals (in-sample). That leaves 61 partners on
fold 0 and 56 on fold 1 (25 train-only), plus baseline*0.25 as a reference row. Reproduces E-170 (A lam0 0.6761 / 0.6623) and E-173/q23e swap2 (A+s4 50/50 within 5e-5). CPU only.
Scripts are in the session scratchpad: `pscan_gram.py`, `pscan_eval.py`, `pscan_more.py`, `pscan_prod.py`.

## 0. Verdict
- **Best partner: an independent EMA-4 run with the same recipe and a different seed** (ab62_base_s4). 50/50
  with A beats the production blend 0.65 A + 0.35 base by **+0.0012 pooled** (f0 +0.0013, f1 +0.0011,
  pooled sd 0.0005, 2.4 sd; lam 0: +0.0013 / +0.0014). With the weight chosen on the other fold: +0.0011.
- Runner-up: EMA-4 with prop_v2 weights (ab62_p2, same seed as A): +0.0010 (f0 +0.0013, f1 +0.0007). It is
  as complementary as the seed change (partial corr 0.044 vs 0.050) and adds nothing beyond it.
- On the production-like first member (ab63_gcorr, the fold analog of the E-172 gcorr LP-FT final), a raw
  independent EMA-4 partner at 50/50 beats the E-172 candidate 0.75 X + 0.25 base by **+0.0010 / +0.0013**
  (s4) and +0.0010 / +0.0009 (p2).
- The property that matters is **full strength plus an independent from-scratch SGD trajectory**. A partner
  helps only if WP_B > WP_A * corr(A, B). At corr ~0.995 that leaves a margin of ~0.003 WP, so every 2-epoch,
  train-only, GBM or FT-of-2-ep model is now useless or harmful. Residual corr (y - p) is a poor diagnostic:
  on its own it explains 7% of the gain variance.
- FT children of the first member do **not** make second members: a sibling LP-FT (ab63_ctrl) next to ab63_gcorr
  has partial corr ~0 and loses -0.0006 / -0.0012 against the 0.75/0.25 candidate.
- 3 members (A, B, base) add only +0.0001-0.0003 over the best pair (weights ~0.45/0.45/0.10). The baseline's slot
  is worth far less once a second full-strength GRU is present.

## 1. Anchor A, top partners and references (deltas vs 0.65 A + 0.35 base, lam 0.02; +-: chain bootstrap sd)
WP(A) f0 0.65576 / f1 0.64109; production f0 0.65533 / f1 0.64133; best A+base (w 0.85/0.80) +0.0007 / +0.0003.
rc = |y|-weighted residual corr (y - p), scored rows; pc = prediction corr; part = partial corr of y with B given A.

| partner (family) | fold | WP_B - WP_A | rc | pc | part | w* | best pair | 50/50 | 3-member (wA/wB) |
|---|---|---|---|---|---|---|---|---|---|
| ab62_base_s4 (seed, EMA-4) | 0 | -0.0002 | .9955 | .9949 | .043 | .53 | +.00129+-.00089 | +.00129 | +.00139 (.48/.43) |
|  | 1 | +0.0009 | .9960 | .9951 | .057 | .38 | +.00117+-.00046 | +.00111 | +.00141 (.30/.55) |
| ab62_p2 (propensity, EMA-4) | 0 | -0.0001 | .9958 | .9952 | .044 | .52 | +.00128+-.00128 | +.00128 | +.00140 (.48/.43) |
|  | 1 | -0.0000 | .9958 | .9949 | .044 | .51 | +.00069+-.00055 | +.00069 | +.00099 (.43/.43) |
| ab63_gcorr (LP-FT child of A) | 0 | +0.0000 | .9952 | .9959 | .050 | .51 | +.00123+-.00139 | +.00123 | +.00150 (.38/.48) |
|  | 1 | -0.0006 | .9932 | .9953 | .043 | .60 | +.00037+-.00068 | +.00033 | +.00092 (.40/.38) |
| ab63_ctrl/ftema/wgc (wMSE LP-FT children) | 0/1 | -0.0004/-0.0010 | .9966 | .996 | .038 | .57/.64 | +.0009/+.0002 | +.0009/+.0001 | +.0012/+.0008 |
| ab61_conv_gru (encoder, 2-ep) | 0 | -0.0022 | .9947 | .9939 | .027 | .74 | +.00073+-.00091 | +.00047 | +.00088 (.70/.20) |
| ab61_mlp_gru (encoder, 2-ep) | 0 | -0.0028 | .9947 | .9940 | .018 | .80 | +.00059+-.00106 | +.00016 | +.00077 (.75/.13) |
| ab48_basehead (baseline-core FT) | 0/1 | -0.015/-0.012 | .963 | .965 | .024 | .80/.77 | +.0007/+.0005 | -.0013/-.0008 | +.0008/+.0005 |
| ab62_td5 (TD targets, EMA-4) | 0/1 | -0.0025/-0.0033 | .993 | .993 | .017 | .77/.80 | +.0007/-.0001 | +.0003/-.0007 | +.0008/+.0003 |
| ab53_p2 (prop_v2, 2-ep) | 0/1 | -0.0025/-0.0031 | .995 | .994 | .020 | .78/.80 | +.0007/-.0001 | +.0003/-.0006 | +.0008/+.0004 |
| baseline*0.25 (reference) | 0/1 | -0.018/-0.014 | .945 | .953 | .026 | .80/.79 | +.0007/+.0003 | -.0016/-.0011 | = best A+base |
| ab46_ref (2-ep; ab56_ema EMA-2 similar) | 0/1 | -0.004/-0.004 | .996 | .995 | <= 0 | .80 | +.0003/-.0004 | -.0004/-.0011 | = best A+base |
| ab39_ftm (old production parent) | 0/1 | -0.006/-0.007 | .992 | .991 | .012 | .80 | +.0003/-.0003 | -.0008/-.0016 | +.0007/+.0003 |
| ab30_eps_aux (best train-only GRU) | 0/1 | -0.006/-0.005 | .995 | .994 | -.011 | .80 | +.0001/-.0006 | -.0012/-.0017 | = best A+base |
| gbm_member / ab12 plain (pooled f0+f1) | both | -0.048/-0.075 | .95/.91 | .93/.88 | ~0 | .80 | -.0013/-.0017 | -.011/-.014 | = best A+base |

All other rows (FT-of-2-ep: ab37/39/41/43/44/45/55 incl. soups and WiSE; 2-ep: ab26/29/54; LSTMs; the other
train-only models) are <= +0.0007 on f0 and <= 0 on f1, with best-pair w stuck at the 0.80 cap. Their
"gain" is just A reweighted up. Pooled over both folds, only 8 partners beat the best-weighted A+base pair
(+0.0005 over production): s4, p2, the four ab63 children, ab48 and ab49.

## 2. Production-like anchor: X = ab63_gcorr (gcorr LP-FT of the EMA-4 parent), refs 0.65 / 0.75 X + base
X alone f0 0.65579 / f1 0.64046; 0.75X+0.25b f0 0.65640 / f1 0.64164 (the E-172 candidate's fold analog).

| partner | fold | w* | best pair vs 0.75/0.25 | 50/50 vs 0.75/0.25 | 3-member vs 0.75/0.25 |
|---|---|---|---|---|---|
| ab62_base_s4 (independent EMA-4 run) | 0 / 1 | .50 / .37 | +.00102+-.00066 / +.00139+-.00081 | +.00102 / +.00128 | +.00113 / +.00164 |
| ab62_p2 (independent run, prop_v2) | 0 / 1 | .50 / .44 | +.00099+-.00077 / +.00088+-.00067 | +.00099 / +.00086 | +.00112 / +.00121 |
| ab62_base_s3 (X's own raw parent) | 0 / 1 | .49 / .40 | +.00016 / +.00006 | +.00016 / +.00002 | +.00043 / +.00061 |
| ab61_conv_gru (2-ep conv) | 0 | .64 | +.00026+-.00039 | +.00012 | +.00046 |
| ab63_ctrl (sibling LP-FT, same parent) | 0 / 1 | .80 | -.00061 / -.00119 | -.00069 / -.00126 | +.00003 / +.00004 |
| ab48_basehead | 0 / 1 | .79 / .69 | +.00009 / +.00019 | -.00150 / -.00053 | +.00009 / +.00019 |

Pair gain over X alone: s4 +0.0016 / +0.0026, p2 +0.0016 / +0.0021, raw parent +0.0008 / +0.0012, sibling FT 0.

## 3. What makes a partner complementary
- **Regression** over 119 partner-fold rows, gain = WP(best pair) - WP(A):
  gain = 0.00007 + 0.028 (1 - rc) + 0.063 (WP_B - WP_A), R^2 0.64. WP_B - WP_A alone: R^2 0.39;
  1 - rc alone: R^2 0.07. Near-equal partners (|dWP| < 0.005, n = 28): 0.00059 + 0.048 (1 - rc) + 0.24 dWP,
  R^2 0.93. Residual corr is dominated by irreducible label noise (0.99+ for every GRU). It also falls as a
  partner gets weaker, so it confounds diversity with weakness.
- **Exact two-predictor identity**: R^2(A,B) = rA^2 + (1 - rA^2) part^2, where
  part = (rB - rA c) / sqrt((1 - rA^2)(1 - c^2)) and c = pred corr. From pooled lam-0 moments it predicts the
  unconstrained best-pair gain at corr 0.957, slope 0.95. So part > 0 only if **rB > rA c**. For GRUs
  (c ~0.994) that means WP_B within ~0.003-0.004 of A. The baseline (c 0.95) is allowed a 0.03 deficit, which
  is why it was the only useful partner when the GRUs were weaker (E-115) and why it stops mattering now.
- **Family table** (mean over fold rows; d = best pair vs production):

| family | n | mean d / max d | mean gain vs A | mean part | mean WP_B - WP_A |
|---|---|---|---|---|---|
| seed-only change, EMA-4 (s4) | 2 | +.00123 / +.00129 | +.00114 | .050 | +.0003 |
| propensity change, EMA-4 (p2) | 2 | +.00098 / +.00128 | +.00089 | .044 | -.0001 |
| fine-tuned from A (ab63 LP-FT, 4 objectives) | 8 | +.00063 / +.00123 | +.00054 | .040 | -.0006 |
| baseline-core FT (ab48/49) | 4 | +.00056 / +.00072 | +.00047 | .029 | -.0132 |
| encoder/cell change, 2-ep (ab61) | 5 | +.00043 / +.00073 | +.00000 | .012 | -.0050 |
| objective change, EMA-4 (TD targets) | 2 | +.00026 / +.00065 | +.00017 | .017 | -.0029 |
| fine-tuned from 2-ep parents | 32 | +.00004 / +.00066 | -.00006 | .013 | -.0073 |
| 2-epoch / EMA-2 (same recipe, shorter) | 12 | -.00010 / +.00066 | -.00019 | -.002 | -.0050 |
| train-only (25 models) | 50 | -.00036 / +.00065 | -.00045 | -.003 | -.0136 |

- **One change at a time, seed 3 fixed** (2-ep anchor ab46_ref; partial corr f0 / f1, pair gain over anchor):
  conv encoder .070 (+.0019), prop_v2 .069 / .055 (+.0018 / +.0013), mlp encoder .064 (+.0017),
  LSTM .036, EMA teacher .028 / .030 (+.0003 / +.0004), dropout 0.1 .005 / .021 (pc 0.999, +0.0000 / +.0002).
  FT seeds from one parent (ab55_ft_s4/s5 vs ab39_ftm): pc 0.999, part -.012 to +.020, gain ~0.
  Diversity comes from a different learned representation (a new trajectory from scratch, or different
  loss weights / encoder). Dropout, FT seeds and sibling FTs don't provide it. EMA-4 shrinks the p2 partial
  from .062 to .044 (x0.7) but keeps members at full strength, and at this level strength decides.

## 4. Recommendation: second final member (first = seed-4 EMA-4 + gcorr LP-FT, E-172)
Common flags (FIN recipe of q23e): `--train-source train+valid --eval-source holdout --full-valid --loss wmse
--aux-struct 0.3 --aux-next 0.3 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 --hidden 128 --epochs 4
--ema-teacher 1.0 --ema-decay 0.999 --eval-ema --fast`. Blend 0.5/0.5, no baseline. Fold-to-LB transfer has
been 0.5-0.8x (E-163, E-165).
1. **Seed change only**: `--propensity prop_v1 --seed 3` (any seed != 4). This is `ab65_final_ema4_s3`, already
   training in q23e. Expected fold gain +0.0012 +- 0.0005 vs 0.65/0.35 production. Against the E-172 0.75/0.25
   gclp candidate: +0.0010 / +0.0013. LB ~+0.0006-0.0010. Measured on both folds; the lowest-risk option.
2. **Conv encoder + new seed**: `--propensity prop_v1 --enc conv --conv-k 16 --seed 5`. Best one-change
   diversity at 2 epochs (partial .070) and the strongest 2-ep single (+0.0018, E-168). Projected fold gain
   +0.0010..+0.0019: seed-level complementarity plus about half of any single-model edge that survives EMA.
   **Not measured at EMA-4**: gate on `ab62_conv_f{0,1}` (queued in q23e) giving part >= 0.045 and WP >= A.
   Needs the native conv port merged (E-175, +5-7% latency).
3. **prop_v2 weights + new seed**: `--propensity prop_v2_devb --seed 5`. Measured +0.0010 (f0 +0.0013 /
   f1 +0.0007) with the same seed as A, and +0.0010 / +0.0009 vs the 0.75/0.25 candidate. It is a hedge with
   the same value as option 1, not an improvement on it. No EMA-4 final with prop_v2_devb exists yet (q19c's
ab60_final_ema4_p2 was never trained).
Do not use as the second member: FT children of the first member (sibling LP-FT = 0), any 2-epoch or
train-only model, TD targets, WiSE extrapolations, the GBM, or baseline-core FTs (+0.0001 / +0.0002 vs 0.75/0.25).

## 5. Open checks and caveats
- Whether to LP-FT the second member too. Only raw partners were measured next to the FT'd anchor. Before
  submitting ab65_final_ema4_s3_gclp, run the ab63 gcorr LP-FT on ab62_base_s4_f{0,1} and rerun
  `pscan_gram.py` / `pscan_prod.py` (about 3 CPU-minutes per fold).
- Best-w figures are in-sample. The cross-fold w changes s4 by -0.0001, and 50/50 is within 0.0001 of the best.
- The bootstrap sd is for deltas vs production, which also moves the baseline weight 0.35 -> 0. Against the
  best-weighted A+base, the s4 pair gain is +0.0006 / +0.0008.
- The seed comparison is n = 1 per fold (seed 4 vs 3). The q23e swap2 check agrees (50/50 vs A+base 0.80:
  +0.00065 / +0.00081).
- soup4 was treated as train-only (E-119 train-arm soup). base_rnn is a near-duplicate of the baseline (same
  WP to 4 decimals on both folds).
