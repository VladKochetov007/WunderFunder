# Session report 2026-10-02 (04:00–10:10 Kyiv)

Goal «honestly expected public ≥ 0.6660»: **not reached**. Best deliverable forecast 0.6644 (80% 0.6633–0.6655,
P(≥ 0.666) ≈ 0.03). Public best so far 0.6641 (our distilled GRU256 alone, = team best; leader 0.6677).

## What the leaderboard said (E-324)
D256 alone 0.6641 > R4D (routed since+D256) 0.6638 > S4D (since+D256) 0.6633: adding the since GRU128 members dilutes
the distilled student. The public-like class-mix reweighted 4-fold OOF level predicted both single-family public
scores to 0.0001 (0.6621→0.6622, 0.6641→0.6641); mixes came in 0.0006–0.0009 below it.

## Experiments tonight (GPU queue `jobs/queue4.txt`; scored with the frozen pooled metric)
| id | idea | result (pooled, fold 0 unless noted) | verdict |
|---|---|---|---|
| E-326 | average alpha 0.5 + 0.7 students | −0.0009 vs 0.7 alone | weaker members dilute |
| E-327 | teacher memorisation (cross-fitted teachers?) | no clear in-sample excess | not a lever |
| E-328 | alpha 0.7 vs 0.5, fold 1 | −0.0007 (fold 0 was +0.0023) | weak: 2-fold mean +0.0008 |
| E-329 | alpha 0.85 | −0.0013 vs 0.7 | 0.7 is the top of the curve |
| E-330 | H-060 teacher ensemble (since pair + alpha-0.7 student) | +0.0013 vs 0.7 (mix +0.0014) | promising; fold-1 run `te07_f1` in progress |
| E-331 | H-059 noise-adaptive alpha | −0.0003 vs 0.7 | ties constant alpha |
| E-332 | H-062 weight soup ep2+ep3 | −0.0002 | falsified |
| E-333 | final alpha-0.7 students on holdout | s3 +0.0003, s4 +0.0005 vs the 0.6641 student; pair +0.0004 more | packaged s3, s4 |
| E-334 | DTW / piecewise-constant round (4 agents) | see `dtw/SUMMARY.md` | no DTW use survives |

## DTW / piecewise-constant answer (E-334, `dtw/SUMMARY.md`)
Segments are causally observable (all 6.5 M target changes sit on a6-endpoint rows). The in-segment variation of
predictions is information about the next quote move, not noise: every causal piecewise readout (hold, running mean,
reset EMA, jump-only, stackers) loses on 4/4 folds; the model does not lag. DTW-kNN on event paths, DBA prototypes,
DTW lead-lag i1→i0 and DTW retrieval of per-sequence scale are all null after cross-fitting; class-3 quiet runs are
time-warped copies of each other but target-neutral. Targets aligned to the run end are biased (future event); the
unbiased stopped-bootstrap distillation target (teacher at min(t+h, run end)) is the one follow-up — patch applied to
`scripts/train_rnn.py` (`--distill-boot-h`, default off) and fold-0 screen `c20261002_sbd_f0` queued.

## Infra incidents
- 01:39 campaign GPU cap (40,000 s) killed the fold-1 alpha run near its end; GPU idle until 04:03 (E-325). Cap raised.
- CPU agents + desktop (32 GB) pushed the training scope into memory reclaim (GPU util 27–57%, ~8 seq/s vs 17.5);
  freed 6 GB tmpfs scratch, CPUWeight watcher for training scopes.
- Controller session hit the provider usage limit 04:50–09:00; the GPU queue kept running. The engine agent
  (integer-madd GRU kernel for 2 students in budget) stopped mid-work: its edits in `native/` are uncommitted and
  unbuilt; packaging uses the unchanged `engine/native_build` (sha e8b1f871… = the 0.6641 package's engine).

## Running at 10:10 (left to finish)
`c20261002_te07_f1` (H-060 fold-1 confirmation) then `c20261002_sbd_f0` (stopped bootstrap, fold 0). Score with
`research/campaign_2026-10-02/packages/night_f0.json`-style specs (arms `c20261002_te07_s3_f1` + `c20261002_ft_te07_s3_f1h`;
`c20261002_sbd07_h8_s3_f0` + `c20261002_ft_sbd07_h8_s3_f0h`).

## Next highest-value steps
1. If te07_f1 confirms (≥ +0.0005 on fold 1): final teacher-ensemble student (teachers ab67_final_since_s3/s4 +
   c20261002_final_d256a07_s3, alpha 0.7) → package; expected +0.0005–0.001 over D07.
2. Finish the integer-madd engine (2 GRU256 students within 52 min; pair worth +0.0004 on holdout).
3. Stopped-bootstrap result decides whether target design keeps paying.
