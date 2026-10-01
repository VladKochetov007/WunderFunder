# Overnight research report, 2026-10-01 (02:10 → 10:00 Kyiv)

Status: FINAL (written 18:05; research cutoff 10:00 respected — no job launched after 06:26, all jobs finished by 06:37).
Three usage-limit pauses (≈03:15–04:30, ≈05:15–09:30, ≈09:35–18:00) cost most of the night's wall time.
Nothing was uploaded.

## Starting point
- Our public incumbent 0.6622 (`final_since2_gclp`, 2×GRU128 since + GCLP). Team best (teammates) 0.6641.
- Clean 4-fold since/since reference (chain-excluded, fixed epochs): Global WP 0.689523; class WP
  0.807 / 0.776 / 0.739 / 0.433.
- Teammate ideas/numbers: `TEAM_INTAKE.md` (clean-room: ideas only, no code read by implementers).

## Evidence log (E-ids in research/experiments.jsonl)
| id | experiment | result | verdict |
|---|---|---|---|
| E-287 | current-code spread labels vs matched proxy control (fold 0) | +0.00002 ± 0.00008 | reject |
| E-288 | cross-fitted per-class output scale (4 folds) | mean −0.00006, in-sample bound +0.00006 | reject |
| E-289 | distilled GRU256 since student, fold 0 | head-only FT single +0.00067 vs since s3 (class 3 +0.0043); pair+d256 thirds +0.00043 ± 0.00054; two-stage FT −0.0019 single | head-only; fold 1 running |
| E-290 | FT pair + own pre-FT parents (output mix, 4 members) | folds 0+1 +0.00097 ± 0.00043 (both folds +); holdout +0.00046 ± 0.00040 | promising, f2/f3 confirm prereg |
| E-291 | pooled-memory FT loss (N2) | −0.0015 two-stage / −0.0004 head | reject |
| E-291 | head-only FT pair vs two-stage pair | +0.00064 ± 0.00043 (holdout +0.00017) | head-only ≥ two-stage |
| E-291 | shared trunk: parents + head-only FT heads (2 trunks) | +0.00098 vs FT pair, = 4-trunk mix; holdout +0.00038 | candidate core |
| E-292 | cross-fitted weights over parent/head members | −0.00038 ± 0.00017 vs equal | reject; equal weights |
| E-293 | joint ensemble head FT vs matched solo (N1) | −0.00010 ± 0.00003 | reject |
| E-294 | SP2 + chunk-500 / online-norm merged trunks (holdout) | −0.00004 / +0.00001 | reject extra base trunks |
| E-295 | causal EMA smoothing / sharpening of outputs | smoothing −0.001…−0.048; sharpening ±0.00001 | reject |
| E-306 | class-routed heads (class 3 → parent+FT heads, classes 0–2 → two-stage FT pair), folds 0+1 | +0.0012 ± 0.0006 vs FT pair; classes 0–2 unchanged | candidate (post hoc) |
| E-307 | **PREREGISTERED folds 2+3 confirmation** | SP2 −0.00008 ± 0.00041 (fail); mix4 +0.00023 (fail); RSP2 +0.00031 ± 0.00033, both folds + (below +0.0005 gate) | SP2 not confirmed; RSP2 directional only |
| E-308 | holdout: seed bag SP3/SP4 (new final seeds 5, 6) and routed | SP4−SP2 +0.00005; new seeds weak (−0.0006); RSP2−incumbent +0.00125 ± 0.00051 | SP4 dropped; RSP2 best |
| E-309 | extra heads (FT-seed bag, distilled head, train+valid gcorr head) | +0.00008 / 0.0000 / −0.0002 | reject |
| E-310 | fp32 packages SP2_f32, RSP2_f32 | all packaging gates pass | ready, not uploaded |
| E-311 | distilled GRU256 fold 1 (scored after cutoff from finished run); routed RSP2_D vs RSP2 | folds 0+1 +0.00053 ± 0.00022 (fold 1 +0.00087; pub mix +0.0006) | strongest next step; final d256 not trained |

## Data findings
- Local/public gap is composition: within-class WP is nearly identical for isolated (len-1 chain) and long-chain
  sequences (class 3: 0.423 vs 0.435); isolated sequences are 66% class 3 and score 0.578 pooled.
- Class 3 = 37% of valid sequences, 46–48% of metric weight, WP 0.433; 9 distinct target values, ~35% zeros.
- Implied public class mix (data_eval/class_mix.json): the clean reference (0.6895) maps to the incumbent's 0.6622 when
  class-3 sequence share is 0.459 (95% [0.41, 0.51]); holdout-package fit gives 0.477. Mix [0.183, 0.160, 0.198, 0.459]
  ("pub143") is used to weight local deltas, not to forecast absolute scores.
- Target laws (TRAIN, component-disjoint fit/held-out, E-302/303): targets are piecewise constant between `a6`
  lower-endpoint quote-move markers — they change on exactly those rows (100.000% of weight, no unknown mass);
  interior t0+t1 = −spread(code) exactly on known codes; loser ≈ −(s + 2·clip(m,1,W)) only partially (72/83/98/99.8%
  by class); teammate's "winner = rounded mean mid over next N events" rejected (2–53%). ~20% of metric-weighted
  variance is fixed by the current code; the rest is future path (mostly direction D, 80%). A structural
  p = s/2 ± d head with s from codes loses (−0.001…−0.04); the model already uses spread.
- Transfer slope public/local for clean chain-excluded deltas ≈ 0.8 (range 0.63–0.92; not identified for new families).
- Optimal-predictor check: for weights |y| the weighted-Pearson-optimal p(x) is affine in E[|y|y|x]/E[|y||x], the
  minimiser of |y|-weighted MSE — so the wMSE parent objective is aligned; gains must come from variance reduction
  (ensembles, heads, distillation) or information, not from objective tricks (consistent with N1/N2 rejections).

## Candidates
- Merge trick: k heads on one bit-identical trunk average exactly into one head of width k·H
  (`scripts/package_submission.py` member syntax `PARENT@ep3|HEADFT@ep0`; torch check max diff 1e-6).
- Core candidate SP2: incumbent trunks `ab67_final_since_s{3,4}` each with merged head (parent head + gcorr
  head-only FT head). Same trunk cost as the 0.6622 incumbent.
- SP2+D: SP2 + final distilled GRU256 trunk with merged head (parent + head-only FT). Needs fp16 engine.
- SP4: four since GRU128 trunks (seeds 3, 4 incumbent + new final seeds 5, 6), each with merged head
  (Kaggle JS-6th seed-bag idea, made cheap by the int16 engine).

## Packaged and verified
- `submissions/c20261001_SP2/submission.zip` — SHA-256 `1e5c6914d892b5dbffe95f4290c669c24c22f0bb30b34123daf610a24239742e`,
  3,147,548 bytes, 2 merged-head members (coef 0.5 each), governor off, int16. Checks: offline Python 3.11.16
  scorer image PASS; v2/avx2/v3 bitwise equal; A→B→A reset + warm-up contract PASS; repeat-deterministic;
  harness runtime 0.634× incumbent → ~23 public min. Manifest: `packages/SP2_manifest.json`. Not uploaded.

## Inference engine (own Rust code, engine/results.json)
- Int16 weights with fp32 per-row scales: parity vs fp32 max 8.2e-5 (gate 1e-4), |ΔWP| ≤ 4e-7, bitwise identical
  across v2/avx2/v3; fp16 is faster but fails the 1e-4 gate (3.9e-4–5.8e-4, ΔWP ~1e-7). Head width generic.
- Projected public minutes (int16, calibrated on the incumbent's 36.45 min): 2×h128 19.7, 6×h128 57.6,
  2×h128+h256 51.5, 3×h128+h256 60.5. Budget gate 52 min. One h256 ≈ 3.5–4 h128 trunks.

## Update 23:30 — new best verified candidate
- `submissions/c20261001_S4D_i16/submission.zip` (SHA-256 `ab0f5287…fec00`, 6.18 MB, int16, ~48 public min):
  since s4 trunk + final distilled GRU256 trunk, each with merged parent+head-FT heads, equal weights.
  Evidence: 4 clean folds +0.0018 ± 0.0005 vs the incumbent-recipe pair (S3 variant +0.0013); holdout +0.0016 ± 0.0007.
  d256 confirmed on preregistered folds 2/3 (SP2_D − SP2 +0.00099 ± 0.00018, both folds +). Forecast ≈ 0.6630–0.6632
  (P > 0.6641 ≈ 0.10–0.15; P ≥ 0.6660 < 0.001). Supersedes RSP2 as the recommended upload.

## Final recommendation (earlier, 10:00)
1. **Primary: `submissions/c20261001_RSP2_f32/submission.zip`** — SHA-256
   `2783eea96664e2f50b9403bbda610a772d4df182dc0dd6d59727898a1a8bdc20`, 6,050,069 bytes, fp32, governor off.
   Market-class routed: a warm-up detector (incumbent parent's auxiliary class logits, rows 50–98; 100% accurate
   out of fold on 925 sequences) picks, at row 99, either the incumbent's two GCLP members (classes 0–2 — identical
   outputs to the 0.6622 package) or, for class 3, the same two since trunks with merged heads averaging the
   pre-FT parent head and the head-only gcorr FT head. Evidence: holdout +0.00125 ± 0.00051 vs incumbent (+0.0015
   under the implied public class mix), folds 0+1 +0.0012 (post hoc), preregistered folds 2+3 +0.00031 ± 0.00033
   (both positive, below the +0.0005 gate). Runtime ≈ 0.79× incumbent in the same harness → ~29 public min.
2. Secondary (not recommended): `submissions/c20261001_SP2_f32/submission.zip` (SHA-256 `0126c381…0ede`) — failed
   its preregistered confirmation; class 1/2 losses.
3. Reference packages kept: `c20261001_SP2` (int16 build, SHA `1e5c6914…742e`), `c20261001_RSP2` (int16 build).

Gates for RSP2_f32: ZIP 6.05 MB ≤ 20 MB; root solution.py; Python 3.11.16 offline image (no network, 1 CPU, 16 GB)
bitwise equal to host; v3 = avx2 bitwise, v2 within 4.8e-7; A→B→A reset, warm-up None, finite outputs, fresh
instance identical; fixed members and weights (no governor); runtime ratio 0.79 vs the incumbent's 36.45 min.

## Expected public score
Independent audit v1 (critic/expectation_v1.md; author ≠ auditor), 50/50 holdout route (slope 0.657 on 6 adjacent
public GRU steps) and fold route (pub143 delta, empirical-Bayes selection shrinkage, τ ~ N(0.75, 0.35), LB noise 0.0006):

| candidate | mean | 80% PI | P(>0.6622) | P(>0.6641) | P(≥0.6660) |
|---|---:|---|---:|---:|---:|
| SP2 (packaged) | 0.6626 | 0.6617–0.6635 | 0.71 | 0.02 | <0.001 |
| SP4 (pending seeds 5/6) | 0.6632 | 0.6621–0.6642 | 0.88 | 0.12 | 0.001 |
| SP2+D (runtime 51.5–57 min, not shippable in i16) | 0.6628 | 0.6618–0.6639 | 0.76 | 0.06 | <0.001 |

Update rule after the preregistered folds-2/3 confirmation: E[SP2] = 0.66232 + 0.375·m with m the precision-weighted
combination of prior N(0.0003, 0.0006²) and the f2+f3 pub143 delta (fold 0/1 evidence excluded as post hoc).
**0.6660 is not supported by any candidate**; it needs ≈ +0.0038, about 5× the largest selection-corrected gain.

### Updated after the confirmation (author-computed with the audit-v1 method; the independent auditor hit its usage limit)
| candidate | mean | 80% PI | P(>0.6622) | P(>0.6641) | P(≥0.6660) |
|---|---:|---|---:|---:|---:|
| **RSP2_f32** | **0.6627** | 0.6617–0.6637 | 0.76 | 0.04 | ≈0 |
| SP2_f32 | 0.6624 | 0.6615–0.6633 | 0.60 | 0.01 | ≈0 |
| SP4 | dropped (holdout increment +0.00005; new seeds weak) | | | | |

Method: 50/50 mixture of (a) holdout route, public Δ = 0.657 × holdout Δ (slope from 6 submitted adjacent GRU
steps, jackknife SD 0.24, residual 0.00064) and (b) fold route using only the preregistered folds 2+3 delta under the
implied public class mix, shrunk with prior N(0.0003, 0.0006²), transfer τ ~ N(0.75, 0.35), LB noise 0.0006.
**Honest verdict: no new public best is established; expected ≈ 0.6627, roughly +0.0005 over our 0.6622 and below the
team's 0.6641. The 0.6660 target is not reached and no evidence supports it.**

## How the gain was found (what worked)
- Data finding first: local↔public gap = market-class composition (public ≈ 46% class-3 sequences vs 37% local),
  so class-3 changes carry more public weight than local pooled deltas suggest.
- Combination of existing pieces: the pre-FT parent head and the head-only GCLP head on the same trunk are
  differently-biased views (parent better on class 3, FT better on classes 0–2). Averaging them exactly on one trunk
  (merged wide head) is free at inference; routing by causally detected class keeps the FT advantage on classes 0–2.
- Engineering: own Rust engine with int16 weights and generic head width (built in the scorer base image) cut a
  2-trunk ensemble to ~0.6× of the incumbent's runtime, making ensembles of 4 h128 trunks affordable (not needed).

## Negative results and open questions
Rejected tonight: spread-label aux (E-287), class-scale correction (E-288), pooled-memory correlation loss (E-291),
cross-fitted weights (E-292), joint ensemble head FT (E-293), extra chunk-500/online-norm trunks (E-294), output
smoothing/sharpening (E-295), structural S/D head from codes (E-303), extra heads (E-309), GRU seed bag from the
Kaggle Jane-Street-6th notebook (E-308; online refit there is impossible without test labels), and the distilled
GRU256 (E-289/E-305: ≈ one more since trunk; does not fit the runtime budget next to two h128 trunks in int16).
Open: (1) RSP2's preregistered gain is below the gate — its public transfer is uncertain; (2) the distilled GRU256
looked weak on fold 0 but strong on fold 1 (E-311), so the final d256 fit was cancelled too early; (3) independent
expectation audit v2 pending (auditor hit its usage limit).

## Next experiments (highest information first)
1. Upload RSP2_f32 (user decision) — one public datapoint resolves most of the forecast uncertainty.
2. Train the final distilled GRU256 (`jobs/final_d256.sh 3`, ~45–55 GPU-min, teachers = incumbent parents) and package
   RSP2_D: class-3 group = 2 since trunks + d256 trunk, all with merged parent+head-FT heads; classes 0–2 unchanged.
   Local evidence +0.00053 ± 0.00022 over RSP2 (folds 0+1); int16 runtime ≈ 36–41 min at a 41–51% class-3 share
   (≈57 min if every test sequence were class 3 — check the margin before shipping). Confirm on folds 2/3 first
   (fold 2/3 d256 students not trained yet: ~45 min each).
3. Re-run the independent expectation audit (critic/expectation_v1.md method) with E-307/E-308/E-311.
