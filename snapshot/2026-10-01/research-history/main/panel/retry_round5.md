# Retry round 5: refuted or shelved ideas that could flip in the EMA-4 / 2-GRU regime (2026-09-27 05:45)

Role: scout (conduct-creative-research, tier B). I trained nothing. The CPU probes ran under `scripts/limited` (10% CPU,
2 threads, 6G). Scripts are in the session scratchpad `retry5/`: `prescreen.py` reads the partner-scan Gram files
`pscan_gram_f{0,1}.npz` and gives the exact pooled WP of any per-sequence linear combination, with a 200-draw chain
bootstrap. `cross.py` does the cross-seed diversity checks. `memgap.py` measures the in-sample vs OOS gap. Units: pooled
WP on held-out chain folds 0/1 at λ .02 unless stated. A = ab62_base_s3 (EMA-4 seed 3), s4 = ab62_base_s4.
Fold to LB transfer is 0.5-0.8x (0.65 used). EMA-4 fold run ≈ 13-14 GPU-min, LP-FT (2 stages + predict) ≈ 5 GPU-min.

## 0. What the new regime changes (drives every card)
- **The pair absorbs variance levers.** Levers that work by averaging or smoothing one model lose most of their value once a
  second full-strength GRU sits in the package (O1/O5, E-177). Measured this round: WiSE, class-routed LP-FT and the
  children soup all pass on a single model and give ≤ +0.0001 inside the pair (§1). Levers that can still flip are
  (a) **strength** levers, which both members inherit, and (b) **structural diversity at equal strength** for member 2.
- **Diversity is additive and has a price: the pair's averaging gain ≈ 0.19·(1 − c)** (c = prediction corr on scored
  rows). The ratio stays between 0.18 and 0.20 across 12 pairs at 2 epochs, and it is 0.186 for the EMA-4 seed pair. So
  pair(A, B) − A ≈ 0.19·(1 − c) + ΔWP/2. A seed change gives 1 − c = .0055 (2-ep) / .0050 (EMA-4).
- **Which changes carry structural diversity** (§1 P4) is the new finding. Online norm, chunk 500 and the raw-only view
  each add +.0029..+.0037 to 1 − c on top of a seed change, and that extra survives a seed change. Conv, p2, class
  weights, eps, distillation and dropout add ≈ 0. That makes E-038/E-118 (online norm) and E-123 (chunk 500) the
  strongest flip candidates: both were refuted as accuracy levers at about equal strength, which is exactly the
  profile a partner needs.

## 1. Offline pre-screens run this round (CPU, cached OOS fold predictions)
| id | probe | result (f0 / f1, pooled ± chain-bootstrap sd) | verdict |
|---|---|---|---|
| P1 | output-space WiSE A + α(C − A), C = gcorr LP-FT child (ab63_gcorr) | single: α .5 − child **+0.00098±0.00040** (+.00076/+.00120), and +0.0007 over the parent. In the pair 0.5 s4 + 0.5 WiSE: α .5 **−0.00009±0.00021**, α .75 +0.00001±0.00010. ctrl child: same pattern | closed for the 2-GRU package; alive only for a 1-member hedge |
| P2 | class routing: gcorr child on classes 0-2, parent on class 3 | single +0.00057±0.00034 vs parent (+.00097/+.00018). In the pair **+0.00010±0.00033** (+.00014/+.00007). Class-3 FT costs −0.0002 in the pair and classes 0-2 FT gives +0.0007 | closed (≤ +0.0001) |
| P3 | output average of the 4 LP-FT children; FT objective next to raw s4 | avg(4) − gcorr **−0.00025±0.00017**. pair(s4, child) − pair(s4, raw A): **gcorr +0.00064±0.00042** (+.00081/+.00047), ctrl/wgc/ftema +0.0004 | soup closed; gcorr is the best member FT → R2 |
| P4 | one-change diversity at 2 epochs (same seed), then cross-seed | 1 − c: seed .0055; **online norm .0092 (s3 or s4 anchor: structural +.0037)**; **chunk 500: +.0029 over seed**; raw-only +.0033 (but −0.0013 WP); eps .03 +.0012; mlp +.0006; conv −.0003 (none); p2 +.0006 at 2-ep → ≈ 0 at EMA-4 (.0049); class-w, eps .10, distill, aux-k ≤ 0; **dropout .0008 (keeps the trajectory)**. ON~ch500 1 − c .0130 ≈ additive | R1: the partner should differ by view and chunking, not by recipe details |
| P5 | per-target pair weights (s3, s4) | best w_s3 .65/.40 (f0 t0/t1), .40/.35 (f1). Gain over 50/50 ≤ +0.00008, direction inconsistent | closed |
| P6 | memorization gap of EMA-4 teachers (critic E6a; gate for q26 distill) | in-sample − OOS: single +0.0065 / +0.0070; pair average **+0.0064±0.0013 / +0.0070±0.0010** (> 0.005 gate). Memorized label share γ = cov(in − oos, y − oos)/var(y − oos) = **0.011** (corr .15-.18, sd 6-8% of the residual) | formal gate fails, but the ξ = 1 target is ~99% teacher function (see §3) |
| P7 | EMA-4 learning curves, 8 runs (s3/s4/p2 × f0/f1, two finals) | epoch 2→3 gain +0.0015..+0.0032 (mean +0.0021). Epoch 3→4 **+0.0004..+0.0011 (mean +0.0007), 8/8 still rising**. Tail ratio ~0.33 | feeds R3 |

## 2. Ranked retry cards (E[ΔLB] = P(pass) × 0.65 × fold gain; GPU cost = screen + adoption)
| # | card (retry of) | P | LB if pass | GPU-min | E[ΔLB]/GPU-h | offline pre-screen |
|---|---|---|---|---|---|---|
| R2 | member-2 LP-FT objective decision (E-171/E-178 open check) | 0.5 | +0.0002 | 22 | 2.7e-4 | yes, P3 done (+0.0006 gcorr in pair) |
| R1 | view-diverse partner: online norm + chunk 500 (E-038/E-118, E-123) | 0.35 | +0.0005 | 38 + 18 | 1.9e-4 | yes, P4 done; readout by exact moments |
| R5 | pooled gcorr in the main fit as member 2 (E-071 corr-mix) | 0.2 | +0.0003 | 28 + 15 | 8e-5 | partly (P3); readout by exact moments |
| R4 | valid oversampling `--valid-repeat 2` (skeptic #3, shelved) | 0.25 | +0.0003 | 30 + 30 | 7e-5 | no |
| R3 | EMA-6 × wd 0.1 (E-083, E-153, E-167) | 0.3 | +0.0004 | 80 + 40 | 6e-5 (screen alone 1.8e-4) | curves only (P7) |
| R6 | heuristics / event-clock view partner (E-088/089, E-056) | 0.1 | +0.0004 | 5 predict + native ports | ~2e-5 | needs a GPU predict of ab22_heur_e2 / ab14_dual |
| R7 | WiSE parent ↔ gcorr child (E-147/E-158) | — | 0 in pair | 0 | — | P1: closed for 2-GRU |
| R8 | class-routed / class-3-anchored LP-FT (E-109/E-127, gc #5) | — | ≤ +0.0001 | — | — | P2: closed |
| R9 | dropout / zoneout / raw-only as partner diversity (E-156, E-120/125) | — | < seed partner | — | — | P4: closed |
| R10 | FT-children soup, per-member or per-target blend weights (E-156, O1) | — | ≤ 0 | — | — | P3/P5: closed |

Base recipe: EMA = `--train-source train+valid --n-valid-folds 4 --eval-source vfold --full-valid --loss wmse --aux-struct 0.3
--aux-next 0.3 --propensity prop_v1 --prop-power 2 --prop-eps 0.03 --valid-true-mask 0.03 --hidden 128 --epochs 4
--ema-teacher 1.0 --ema-decay 0.999 --eval-ema --fast` plus `--valid-fold k` (q26 `run1`). Partner readout for R1/R5:
`pscan_gram.py 0|1` picks up new `pred_*_f{k}` files automatically (~3 CPU-min per fold); then `retry5/prescreen.py
onechange` or `cross.py` with the new names. Partner accept rule: pair(A, B) − pair(A, s4) ≥ +0.0005 pooled, both folds
> 0, λ0 > 0.

**R2. Which LP-FT objective on member 1 (s4), next to the gcorr-LP s3 member.** Mechanism: P3 shows a gcorr LP-FT child
is the best pair member (+0.0006 over its raw parent; wMSE children +0.0004). The two built packages differ only here:
final_2gru_mix (wMSE LP-FT s4 + gcorr s3; holdout λ.02 +0.0009) and final_2gru_gclp (gcorr on both; +0.0006, holdout SE
~0.001). Why it can change now: on the 2-ep parents only wMSE FT existed. Test (q24 recipe): `FT="--train-source valid
--n-valid-folds 4 --eval-source vfold --full-valid --mask-weight --aux-struct 0.3 --aux-next 0.3 --hidden 128 --seed 3
--epochs 1 --batch 16 --fast"`. For k in 0 1 and L in {gcorr, wmse}: head `--name ab68_{L}4_f{k}h --init
ab62_base_s4_f{k} --valid-fold k $FT --loss L --freeze inp,gru --lr 5e-4`, then full `--name ab68_{L}4_f{k} --init
ab68_{L}4_f{k}h ... --lr 2e-4`. Readout: exact pair WP on the grid {s4 raw, wmse4, gcorr4} × {s3 raw, ab63_gcorr}.
Switch the upload only at Δ ≥ +0.0003 pooled with both folds > 0. **Falsifier**: every FT'd s4 pair within ±0.0002 of
(s4 raw, gcorr3). Then upload the cheaper mix package and stop tuning FT. Cost 22 GPU-min, with no new code.

**R1. View-diverse second member: online norm + chunk 500 (+ new seed).** Mechanism: `--online-norm` appends causal
per-sequence z-scores of the 112 raw columns. At 2 epochs it moves 1 − c from .0055 (seed) to .0092 at equal strength
(ΔWP +0.0008 vs ab20; E-118 +0.0003±0.0006), and the extra diversity survives a seed change (s3~ON .9908 = s4~ON .9908).
Chunk 500 adds another +.0029 (s4~ch500 .9898 vs s4~ab30 .9927). ON~ch500 at 1 − c .0130 shows the two add. Why it can
flip now: both were judged as accuracy levers (≈ 0). Under the partner rule, equal strength plus lower c is the payoff:
forecast pair(A, B) − pair(A, s4) = 0.19·(1 − c_B − .0050) + ΔWP/2 ≈ **+0.0005 (EMA shrinks the structural part by
half) .. +0.0011 (no shrink)**. Test: first read q26's ab66_ch500 (§3 has its forecast and a confound). Then run
`$EMA --valid-fold k --seed 5 --online-norm --chunk 500 --ema-decay 0.99975` (ab67_onch_f{0,1}). Add the arm
`--online-norm --seed 5` (ab67_on) if ch500 fails. **Falsifier**: EMA-4 1 − c(A, B) ≤ .0060 (structure erased by EMA)
or pair Δ ≤ 0 on either fold. Cost ~38 GPU-min for 2 folds (ON ≈ +15% compute; chunk 500 up to +30%) + 18 for a final.
Serving: port the Welford z-scores into native (`online_zscore_torch` on the full stream, streaming-safe; ~40 lines).
Parity corr ≥ 0.99997. Budget +15-19% member latency (input projection 211 → 323 columns), ≈ +8% for the package.

**R5. Pooled gcorr in the main fit, as member 2** (retry of E-071). E-071 refuted a *per-sequence* corr mix. The LB metric
is pooled (E-161), and the pooled loss wins in LP-FT (E-171) and makes the best pair child (P3). Why it can flip now: the
earlier test used the wrong metric definition, and an objective change is a candidate source of structural diversity.
Flags `$EMA --valid-fold k --seed 5 --loss wmse_gcorr --gcorr-weight 0.3` (ab70_wgc). Risk: wgc at weight 1.0 in LP-FT
was −0.0002 vs ctrl, and the batch (32 seq × 2000 rows) pools a noisy metric estimate. Accept via the partner rule, or
single Δ ≥ +0.0010 vs mean(s3, s4). **Falsifier**: 1 − c(A, B) ≤ .0055 and single Δ ≤ 0. 28 GPU-min.

**R4. Valid true-mask oversampling in the EMA-4 main fit** (skeptic #3, never run). Mechanism: LP-FT on EMA parents now
adds ≈ 0 (ctrl −0.0007 at λ.02, E-171), so the true-mask signal (+0.0006..10 when first added, E-098) can only enter
through the main fit. EMA's averaging guards against the over-specialisation that sank larger FT doses (E-145). Flags
`$EMA --valid-fold k --seed 3 --valid-repeat 2` (ab71_vrep2) vs s3/s4 (2-seed ref: STACK +0.0010). Report the
train-unlike tercile and stress λ .05. **Falsifier**: pooled Δ < +0.0005 or λ .05 < −1 SE. Upside: the one valid-like
lever to reach the LB transferred at ~3x its fold size (E-144). 30 GPU-min + 2 finals.

**R3. EMA-6 with wd 0.1.** Mechanism: 8/8 EMA-4 curves still rise at epoch 4 (P7), so the anneal ends early. E-153 shows
wd .01 collapsing after mid-schedule (−0.027) while wd .1 held 0.672-0.676 for 8 epochs, so wd .1 guards the longer
schedule. It is a strength lever, so it survives the pair. Flags (ab69): `$EMA --valid-fold k --seed 3 --epochs 6 --wd
0.1`, and the control `--epochs 6` (4 runs × ~20 min). OneCycle stretches (total = epochs × steps). The checkpoint is the
best epoch (critic F1): judge on history's last epoch and require argmax = last. Forecast +0.0000..+0.0010
(median +0.0004). **Falsifier**: < +0.0005 pooled, or a peak before epoch 6 by > 0.0003. Screen with the wd .1 arm alone
(40 GPU-min). Adopting it means refitting both finals (~2 × 20 min).

**R6. Heuristic or event-clock view as member 2** (E-088/089, E-056). Same logic as R1, but the serving cost is large (the
native engine has neither the heuristic library nor the dual arch). Gate with no training: GPU-predict `ab22_heur_e2`
(vs ab20_gru128_e2) and `ab14_dual_h128_ev64_pow2` (vs ab9_pow2_h128, cached), then run `cross.py`. Pursue only if
1 − c ≥ .0120 at ΔWP ≥ −0.001, which would beat ON by enough to pay for the port.

**Closed by pre-screen (do not spend GPU).** R7 WiSE: +0.0010 on one model but −0.0001 in the pair (P1). Keep it only
as the S2 single-member hedge: `scripts/interp_runs.py ab60_final_ema4_p1 ab64_final_ema4_gclp 0.5 ab72_wise05` plus a
predict. R8 class-routed LP-FT: +0.0001±0.0003 in the pair (P2); class 3 gains nothing from FT in any setting. R9
dropout gives no trajectory diversity (same-seed 1 − c .0008). Raw-only is diverse but −0.0013 weaker: pair ≈ +0.0008 <
seed partner +0.0011. R10 children soup −0.00025 (P3). Per-target / per-member weights ≤ +0.00008 (P5). 50/50 stays.

## 3. Forecasts for runs already queued (pre-registered here, before they land)
- **q26 ab66_ch500 (chunk 500, seed 5).** Confound: the EMA update runs once per optimizer step, and chunk 500 makes 4x the
  steps. At decay 0.999 the averaging horizon falls from 0.27 to 0.07 epoch, and the teacher lag shrinks with it.
  Forecast at matched strength: 1 − c(A, ch500) ≈ .0075, pair Δ vs pair(A, s4) = **+0.0005 ± 0.0007**. If ΔWP(ch500 −
  s3) < −0.001, rerun with `--ema-decay 0.99975` before calling it refuted.
- **q23e ab62_conv.** Conv adds no structural diversity (same-seed 1 − c .0052 ≈ seed), so as a partner it is worth
  seed + ΔWP/2 and nothing more. Its case rests on strength alone (critic: CLAIM ≥ +0.0018).
- **q26 ab66_distil (ξ 1.0).** P6 fails the critic's 0.005 gate (gap .0064/.0070). The memorized part is γ = 1.1% of the
  label residual, though, so the target is essentially the pair function and no cross-fitting is needed. Forecast:
  student − s3 = 0..+0.0005, and 1 − c(student, s3 or s4) ≤ .003. A student is worth a slot only if pair(student, X) beats
  pair(s3, X) for a diverse X (e.g., R1's B). Otherwise it just duplicates the pair it came from.
- **q23e ab62_since / ab62_lnorm.** Both are strength levers and would survive the pair. lnorm f0 has landed at 0.6759 vs
  0.6761 (λ0), which looks flat.

## 4. Order
R2 now (22 GPU-min, decides the upload). R1 right after q26's ch500 read (start the native ON port on CPU in parallel).
R3's wd .1 arm and R4 as one ~70 GPU-min batch. R5 if R1 shows that structure survives EMA. R6 only if R1 passes and a
third view is wanted. Ledger rows for the controller: P1-P7 → one E-id ("retry pre-screens"); R-card outcomes → E-ids
as they land.
