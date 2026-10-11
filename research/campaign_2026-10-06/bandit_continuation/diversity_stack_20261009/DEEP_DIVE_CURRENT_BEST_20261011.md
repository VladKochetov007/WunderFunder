# Deep dive: current best, what made each gain, what did not (2026-10-11 12:30 EEST)

Audience: the next agent and the user. Everything here is traceable to a file in this repository (links at the end of each section). Fold-1 numbers are `global_wp_clip2_v1` on the 457 fold-1 validation sequences (component-disjoint), one-run noise sd ~0.001-0.002, seed sd ~0.001-0.002 at the standard 3-epoch recipe and larger (0.002-0.005) at 4+ epochs. **Fold 1 is historically exposed: it supports mechanism claims, not a forecast of the public score. Never add a local delta to a public score.**

## 1. Where we are

| Item | Value | Source |
|---|---|---|
| Public best (browser-verified by the user) | **0.6668** = `4_z1c_z4_pair_q8vnni.zip` (two all-data H230 students, int8 Q8Vnni engine, 27 min portal) | [PUBLIC_SCORES_20261011.md](PUBLIC_SCORES_20261011.md) |
| Previous public best | 0.6664 (z5 seed9 single, i16) | same |
| Public single q8 vs i16 | 0.6649 vs 0.6649 -> int8 Q8Vnni loses nothing | same |
| Leaderboard top (user report) | ~0.6717 | user |
| Best fold-1 **servable single** (h256) | **muon-all x prop0 x 4 epochs: s3 0.66976, s4 0.67195 (mean 0.6709, n=2)**; 3-epoch version mean 0.6690 (n=2) | [LOOP_LOG_20261010.md](LOOP_LOG_20261010.md) pass 120/126 |
| Best fold-1 distilled student (h256) | **0.67103** (alpha 0.9, 6 ep, 3 muon-all h384 teachers); alpha 1.0: 0.67068; alpha 0.7 6 ep: 0.66991 | pass 120/125 |
| Best fold-1 teacher recipe (h384) | muon-all h384 3 ep: mean 0.6696 (n=6, sd 0.0014) | pass 109 |
| Ensemble ceilings (fold 1) | 3 muon-all h384 seeds = 0.67254; independent 3 seeds 0.66977 | [distillation_deep_dive_20261011/CEILING_AND_PLAN.md](distillation_deep_dive_20261011/CEILING_AND_PLAN.md) |
| Public portal speed | q8 single 15.7 min, q8 pair 27 min => ~13 min/model; limit 60 min | PUBLIC_SCORES |

**The honest gap.** The local fold-1 gains of the last day (control 0.6660 -> 0.6709 for a servable single) have **not** been trained on all data, int8-packaged or uploaded. The LB has moved only 0.6664 -> 0.6668 (the pair). Closing that gap is the number-one job (section 8).

## 2. What the best submission is (and why it works)

Public best = pair of two **H230-recipe students** (details: [solution_report/CURRENT_BEST_SOLUTION_EXPLAINED.md](solution_report/CURRENT_BEST_SOLUTION_EXPLAINED.md)):
1. Data: 20,000-row sequences x 112 anonymous columns = order books of 2 instruments (+8 extras); targets t0 = F-ask, t1 = bid-F (F = future common price), piecewise constant labels (mean run 9.5 rows) that step at book events; metric = pooled weighted Pearson, weights |clip(y,2)|, rows >= 99.
2. Model: 211 inputs (112 raw + 99 causal event features) -> Linear+GELU -> 2-layer GRU 256 (cuDNN, TBPTT chunks 1000-2000) -> head [GRU;skip] 512->256->2 (+aux heads).
3. Training: scratch **teacher** (Muon on the recurrent matrices, one-cycle, 3 epochs, weighted MSE with the metric weights, propensity row weights, aux heads) -> **student** warm-started from a teacher, 2-3 epochs AdamW lr 1e-4 on 0.3 labels + 0.7 blended teacher outputs (four cache families) -> **all supplied data** (train+valid) for the final.
4. Serving: own Rust int8 VNNI engine (Q8Vnni; AVX-512 VNNI/AVX2/portable bit-identical), [native/INT8_VNNI_REPORT.md](../../../../native/INT8_VNNI_REPORT.md), scorer = AMD EPYC 9224 ([SCORER_CPU_SPEC_20261010.md](SCORER_CPU_SPEC_20261010.md)).
5. Why a pair wins: members share 96-99.9% of residual; the blend recovers the small non-shared slice (+0.0019 public over z1c, +0.0006 over the better member z4).

## 3. What made a gain (fold 1, with seeds)

| Lever | Evidence | Verdict |
|---|---|---|
| Loss weighting: `--prop-power 0` (no propensity down-weighting) | 5 seeds mean ~0.6678 vs ctrl 0.6660 (+0.0018), fold-0 +0.0035 | **Real, reproducible** teacher-level gain |
| Label-weight floor / class weight (`lwfloor`) | prop0+lwf 4 seeds mean .6680 vs prop0 .6678 | Null on top of prop0 |
| **Muon on all 2-D weights** (`--muon-scope all`) instead of rnn-only | h384: mean 0.6696 (n=6) vs plain h384 0.6685; h256: ~0.6674 vs 0.6660 | **Real**, +0.0012-0.0014 (first 3-seed estimate +0.0022 regressed to the mean) |
| muon-all x prop0 | h256: 3 ep .6690 (n=2), **4 ep .6709 (n=2)**; at h384 does NOT stack (.6702 vs .6696) | **Real at h256**; best servable single |
| 4th epoch | h256 muon-all e3 .6674 (n=5) -> e4 .6680 (n=2, sd large) ; muon-all+prop0 e3 .6690 -> e4 .6709 | Weak/positive at h256; **e5+ collapses** (h256 e5 .6639, e6 .6478; h384 e4 .6652/.6704, e5 .6465) |
| Seed ensembling (same best recipe) | 3 seeds: +0.0013 to +0.0023 | Real, scales with member quality |
| Diverse pair of equally strong members | earlier rounds ~+0.003 | Real, but **weaker partners dilute a strong member** (muon-all + dualtower .66978 < .67021) |
| Distillation | student 0.66861 (no teacher) -> 0.66938 (weak teachers) -> 0.66986 (3 muon-all teachers) -> **0.67103 (alpha .9 e6)** | Teacher signal +0.0005..0.0024; plateaus at single-teacher level (~0.670), 0.0025 under the teacher ensemble |
| int8 Q8Vnni serving | public 0.6649 = 0.6649 | Lossless, 2x faster |

## 4. What did NOT work (do not repeat without a new mechanism)

* **Longer scratch training** (small LR + 20-30 epochs): AdamW 24 ep final 0.610 (peak ep11 0.6646), Muon-all 24 ep 0.578, Muon-rnn 24 ep 0.573; 5-6 epoch teachers collapse. **Warp augmentation (6 ep: 0.622/0.623) and dropout 0.2 (5 ep: 0.657) do not rescue it.** Longer student runs without a teacher: 0.653. With teacher targets a 6-epoch student does not overfit but gains nothing over 2 epochs.
* **HPO** (user stopped it): hpo52 h384 0.6686; all HPO arms dropped.
* **Architectures** (v8): clockbank384 .6636, dualtower .6652, dssm384 .6614, clockbank+prop0 .6647 - all below GRU; **next-row pretraining** (.6650/.6653/.6655), **highway head** (.6665), **anti-curriculum** (.6628), **time-warp at 3 ep** (.6659), **muon-hidden** (.6664): null/negative.
* **Data size / wd**: 79% data -0.0004; all-data wd/N rescale .6660 (null); batch 16/64 with lr rules .6647/.6658 (null).
* **GBM windowed teacher** (LightGBM) 0.618; blend -0.0003.
* **Adaptive blending**, **post-hoc EMA / momentum / segment-mean** of predictions: all hurt (EDA).

## 5. Diagnostics that generate ideas (the "why")

* [eda_residuals_20261011/RESULTS.md](eda_residuals_20261011/RESULTS.md): class 3 = 50% of weight and 69% of residual SS (local corr 0.44 vs 0.74-0.81) with a ceiling shared by 40 bank members; 94% of residual is a per-run level offset (labels step only at book events; oracle run-average +0.031); label leads features by ~1 row; members share 96-99.9% residual; t0/t1 difference mode under-dispersed.
* [distillation_deep_dive_20261011/CACHE_ONLY_DIAGNOSTICS.md](distillation_deep_dive_20261011/CACHE_ONLY_DIAGNOSTICS.md): of the ensemble gain, **82% is in the difference channel (p0-p1)** and **66% in the per-run level**; plain MSE on (t0,t1) spends half its weight on the sum channel where nothing is to gain. This motivated the v10 KD-loss flags (queued as `w19_kd_*`).
* [distillation_deep_dive_20261011/LITERATURE.md](distillation_deep_dive_20261011/LITERATURE.md): Beyer et al. (long schedules + same augmented view, fixed targets are bad), Stanton et al. (optimization not capacity), Cho & Hariharan (stronger teachers can be worse teachers), DEGU, DIST (Pearson loss), LUPI (privileged lookahead teacher).

## 6. In-flight experiments and logical ETA (snapshot 11:55 local; verify with the dashboards)

| Direction | Jobs | State | Decision rule |
|---|---|---|---|
| **All-data finals of the servable lead recipe** | `fin_mu256p0e4_s3/s4/s5` (Vast B, front of queue) | queued; first log must be checked (flags `--valid-fold -1 --train-source train+valid --include-heldout-valid`, no split manifest) | feeds packaging; ETA fits ~1.2 h each => ~14:00-15:30 |
| Seeds of that recipe on fold 1 | Vast `muonall_prop0_e4_s3/s4` done, Kaggle v6 (`kg_mu256_prop0_s5/s6/s7`, `kg_mu256_prop0_e4_s5`), local `muonall256prop0` s5 / `prop0lwf` s3 | running | need n>=4 for the claim "e4 > e3" |
| KD-loss variants (diff-channel / Pearson / run-level) | `w19_kd_ch052`, `kd_ch02`, `kd_pearson`, `kd_msepear`, `kd_run05` (alpha .9, 2 ep; control `dm_a9_e2` 0.67018) | queued, front of queue | win only if >= +0.002 or reproduced with a 2nd seed |
| Student chains | `w17_dm_a95_e6`, `a9_e10` (running), `w18_dm_pm_a9_e6` (parent = muon-all h256), `w18_dm_ba_a9_e6` (born-again from the 0.67103 student), `w15b_dm_h384_t3` | running/queued | does parent quality / iteration beat 0.6710? |
| OOD / augmented-path distillation (v9) | `w14_ood_only/splice/spmix/swn/swn25` | queued | main test of "train on teacher outputs off the train rows" (user's idea); vs dsq_t256 0.66938 / dm_t3 0.66986 |
| Teacher-count test | `muonall_h384_e3_s8/s9/s10` (+ s3-5, kg s6/s7) | queued | does a 6-teacher set beat 3 for students? |
| Epoch ladder replicates | `muonall_h384_e4_s4` done (0.67037), local h256 e5/e6 done | closing | claim: optimum ~3 (h256 maybe 4) |

**ETA of "all current ideas logically tested": ~3-5 h of GPU for the queue above (Vast B 7 slots at ~1.2 h per fit under sharing, 14 jobs pending; Kaggle v6 ~2 h; local ~2 h) => by ~16:00-17:00 local the KD-loss, OOD, student-chain and seed questions are answered. Submission candidates (all-data finals -> int8 -> gates) ~15:00-18:00 if the all-data flags work on the first try. This is an estimate under 7-way GPU sharing; check `controller.py dash`.** After that the open (not-started) ideas are: privileged-lookahead teacher R^2 check (H-371), residual/class-3 objective (H-361), event-change aux head probe (H-362), multi-lag aux (H-363), (u,v) loss (H-365), masked pretraining, residual-stage GRU, GBM on hidden state.

## 7. How ideas are generated and how context is kept (our "research style")

1. **Surprise-mining first**: residual EDA with marimo/creative-research lenses ("mine surprises", "instrument", "factor") on cached predictions -> top-5 surprises -> hypotheses with a *cheapest falsifier* each.
2. **Cache-only diagnostics before GPU**: blend and swap probes on banked predictions (`scripts/blend_score_remote.py`, `scripts/distill_ceiling_diag.py`) cost minutes and decide which GPU fits are worth running (e.g. the channel/run-level split).
3. **Independent ideators, public task + facts file**: Grok-4.7 (`--model grok-4.7 --permission-mode plan`), OpenCode free models (nemotron, step-5, longcat, muse), and a literature sub-agent with WebSearch; each gets only `DISTILL_FACTS.md`-style public facts, writes distinct methods with falsifiers; we keep the raw outputs under `ideation/`. Treat their citations as unverified.
4. **Implementer sub-agents** (isolated git worktree, CPU tests, flag-off bit-identity vs previous source, README with deploy list) -> merge -> deploy by stage dir + atomic `mv` -> queue arms. Researcher (me) never edits a frozen source in place.
5. **Every result is written the same hour**: loop log (chronological), idea ledger (A-F sections with status), `hypotheses.jsonl` (append-only, `H-xxx`, parents, status, evidence), crash post-mortem (every incident with root cause + rule), commit + push each step. Retractions are written explicitly (e.g. the e4 export bug).
6. **Parallel branches, bandit-style**: never stop a direction because one seed is low; close it by a recorded falsifier. Prefer different mechanisms over seeds; use seeds only to confirm a candidate (n>=3).
7. **Keep every lane busy**: Vast B (`backlog/IDEA_BACKLOG.json` -> `controller.py feed`), Kaggle (`kaggle_long_20261010/build_long_packet.py` ARMS -> dataset version -> kernel push), local GPU (`queue_local_experiment.py`). A queue is never allowed to run empty.

## 8. Next logical steps toward the LB (do in this order)

1. Verify `fin_mu256p0e4_s*` started (log head, no argparse error); if the all-data flag combination fails, fix the job file (compare with `runtime/attempts/h230-all-data-student-gru256-seed3-v1-20261008.json` command) and requeue.
2. Pull the three final checkpoints (9 MB each) with `controller.py`'s scp helper; build Q8Vnni singles and a 2-3 member ensemble with `packaging_q8vnni_20261010/build_upload3.py` style scripts and run the gates (`run_gates_upload3.sh`: CRC, ABI, reset/determinism, Python 3.11 offline, timing) - see [PUBLIC_SCORES_20261011.md](PUBLIC_SCORES_20261011.md) for the portal-time budget (~13 min/model; stay <= 3 models).
3. Before proposing an upload, measure the pair/triple gain on fold-1 sibling runs (e.g. `muonall_prop0_e4_s3 + s4`) with `blend_score_remote.py`.
4. Give the user a ranked list of ZIPs (expected direction, uncertainty, portal minutes). **The user uploads; never claim a public score.**
5. Student chain on the best h256 recipe as parent + best teacher ensemble (`w18_dm_pm_a9_e6` tells whether it pays).
