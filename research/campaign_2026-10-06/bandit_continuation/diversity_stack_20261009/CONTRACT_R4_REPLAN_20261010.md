# Contract R4: replan after the two external reviews and the axiom audit (2026-10-10 evening)

Objective mode: competition. Utility: expected public score (held-out), mean-based selection (user policy), public decides. Tier B (exposed folds, noisy single-fold evidence, public confirmation). Baseline: public 0.6664 (Z5 lineage); local fold-1 best student 0.66923 (H230 ep3), control 0.6660 (3 runs). Leaderboard: 1st 0.6717, teammate best 0.6674.

## 1. Audit of our own axioms (what I said, what the evidence shows)
| My earlier claim | Verdict | Evidence |
|---|---|---|
| Slow/fast clock probe passes Grok's falsifier (R2 0.26) | **WRONG** | `scripts/slow_fast_residual_probe.py` used the fast increment on the SAME row as the slow write. Excluding it (`cf[u-1]-cf[p]`) gives median R2 0.006 (0.010 for the 13% block). The 0.26 was contemporaneous redundancy, as the unified review said. |
| Native kernel is tuned; speed gains <=10-20%; E416 says int8 does not help | **WRONG** | E525 (matched, one host): own Rust i16 64.8 us/row, own q8x 51.4 (-21%), C++ dynamic int8 with VNNI 19.3 (3.4x). Teammate's 2-member pair served in 24m57s. My "733 us/row for h384" was never reconciled with E420's 99 us (`original_w384_q16x`). |
| Single-member ZIPs are the timing-safe choice | **INCOMPLETE** | The public best is a head-finished tensor merge in ONE member (Z5: 0.5/0.5 merge of student core and head-FT; trunk identical, 5.2 MB). Teammate's 0.6674 is a 2-member pair at 25 min. |
| Lookahead ladder implies the causal task is forecast-limited | **OVERREACH** | It shows future rows help, not how much the past still holds (trained models are not optimal functions). Privileged-teacher branches stay open; no ceiling claim. |
| EMA does not help | **CONFOUNDED** | Older arms used `--ema-teacher 1.0 --eval-ema` (consistency loss) and a different 8-epoch trajectory. Our new `ema6_mu3` / `ema4_mu3` use pure `--eval-ema`; `raw6_mu3` queued as the same-trajectory raw control. |
| h384 teachers do not transfer to the student (0.66836 vs 0.66938) | **NOT A MATCHED TEST** | `dsq_t384` used the 4-epoch `vt_f1_gru384_s4/s5` teachers; the current 3-epoch h384 seeds (`combo_scratch_h384ep3s4/s5_onecycle_f1`, singles 0.66931/0.66816, average 0.67020) were never distilled. Matched pair queued: `vt_f1_dsq_t256n` vs `vt_f1_dsq_t384n` (3-epoch h256 vs h384 seeds 4,5, same parent, cache recipe). |
| Metric pools both targets | **WRONG WORDING, correct code** | `gmetric.py` averages per-target correlations (verified with the critic's example). Only my CI helper pooled; fixed. |
| HPO ranking | **DEFECT CONFIRMED** | fold-0 offset was re-estimated as trials arrived (0.0131 -> 0.0120, up to +0.00055 advantage); fixed offsets now default. Zero-warmup pruning also contradicts the protocol (documented; not changed because HPO is at trial 56/60 and a test encodes it). |
| Family-head blend helps (critic) | Tested: **no** (main head best; every blend -0.0003..-0.0019). |
| Lattice / discrete levels | Retracted earlier (tolerance artifact). |

Unverified review claims left open: E413 numbers (+0.00096, CI [-0.00008,+0.00206]), E420 99 us/row, E462 ABI repair, E535 hold result. They are used only to choose cheap replications.

## 2. What the evidence says is the realistic path (stack of small, measured gains; no single lever is >0.003)
| Lever | Measured basis | Expected public gain | Needs |
|---|---|---|---|
| Head finishing of the new best student (head-only gcorr FT 1 epoch, 0.5/0.5 head merge) | public seed 3: parent 0.6641 -> head 0.6654 -> merge 0.6655; local +0.0013 | +0.001 | local fit queued (items 0870/0880) then merge + score |
| h384 at serving instead of h256 | width +0.0012 on four folds; fold-1 h384 average of two 0.67020 | +0.001-0.002 | speed (VNNI int8) and a h384 student chain |
| 2-member ensemble | greedy banked fold-1 (selection-biased, subsample): best single 0.6711 -> 2 members 0.6733 -> 3 members 0.6734 (saturates) | +0.002 | speed: 2 members must fit ~91 us/row total |
| Matched h384-teacher transfer | untested (queued) | 0 to +0.002 | Vast jobs queued |
| Flow/volume history features | flow_v2, vol28 arms | 0 to +0.002 | queued, native port if positive |
| Clean EMA / long epochs / dropout / Muon-all | ladder + arms running | 0 to +0.002 | running |
Stack of the first three at their measured values is about +0.004-0.005 (0.670-0.671 public) if effects add, which they need not. The VNNI int8 serving path is therefore a score lever, not only a speed project.

## 3. Work in flight under this contract
- Serving: a builder agent is writing our own AVX-512 VNNI int8 GRU path in a worktree (independent implementation; deliverable: us/row for 1-4 members and h384, official-metric quality delta). Gate: int8 WP loss < 0.0005 on 40+ fold-1 sequences.
- Finishing: head finishing recipe digest `Z5_AND_HEAD_FINISHING_DIGEST.md`; local queue 0870/0880 fits and exports the finished fold-1 student.
- Teacher-source test: `w12_t256n`, `w12_t384n` pipeline jobs on Vast B.
- Representations/regimes: flow_v2, vol28 (+ size-matched `volume28_current` control), prop power 1/0, running loss norm, dropout/gated/EMA arms, long-epoch ladder (Vast), Kaggle 2xT4 (stable lr + SWA, thinning), 12-epoch long student (local).
- Open design: out-of-fold distillation pilot (in-sample teacher targets may transfer memorisation).

## 4. Falsifiers and rules
Head finishing: finished student must not score below the raw student on fold 1 (mean). VNNI int8: any WP loss above 0.0005 or non-determinism blocks packaging. Teacher transfer: t384n must beat t256n by >0.0005 mean for h384 teachers to be adopted. Selection by mean; report paired uncertainty; never add unrelated gains into a forecast. Do not run ad-hoc RAM-heavy jobs beside a queued fit.
