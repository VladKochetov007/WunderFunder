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
| E-330/337 | H-060 teacher ensemble (since pair + alpha-0.7 student) | +0.0013 fold 0, **+0.0010 fold 1** vs 0.7 (2-fold +0.0011, CI +0.0003…+0.0022) | **confirmed**; final student training |
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
- Controller session hit the provider usage limit 04:50–09:00; the GPU queue kept running. The engine agent was
  resumed 09:12 and finished q16x (E-336): one GRU256 291 → 234 kcyc/row, all parity/level/image gates pass, but two
  GRU256 members still cost 464 kcyc ≈ 57 public min (> 52-min gate) — memory-bound; not shipped.
- E-335 (H-063) class-routed parent/FT head weights on one trunk: 4-fold +0.00002, falsified.

## Running at 10:10 (left to finish)
`c20261002_final_te07_s3` (final H-060 student, launched 09:55, ~60 min). Unit `wnn-after-final-te` then scores it on
the holdout (`packages/te_final_holdout_report.json`) and packages it with all gates (`submissions/c20261002_TE07s3_i16`);
log `research/campaign_2026-10-02/packages/after_final_te.log`. The stopped-bootstrap screen was swapped out (E-338).

## Next highest-value steps
1. Check `after_final_te.log`: if TE07s3 ≥ D07s3 on the holdout, upload `submissions/c20261002_TE07s3_i16` first
   (expected ~0.6650). Then a second teacher-ensemble seed, and the stopped bootstrap on top of the ensemble.
2. Finish the integer-madd engine (2 GRU256 students within 52 min; pair worth +0.0004 on holdout).
3. Stopped-bootstrap result decides whether target design keeps paying.
4. Teacher-ensemble follow-ups (cheap, same recipe): add the alpha-0.7 seed-4 student and the TE07 student itself as
   teachers (gen-3 ensemble), and combine with the stopped-bootstrap target; each needs a 2-fold screen before a final.
5. Engine: L2-resident split of the GRU256 weights (est. −15%) to fit two students (holdout pair +0.0004).
