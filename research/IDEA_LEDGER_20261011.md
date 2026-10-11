# Idea ledger 2026-10-11 (everything proposed, tried or not; evidence and files)
Rule (user 2026-10-11): every idea, evidence, literature item and generated file mentioned in session summaries is recorded here with a status and a path. Machine rows: research/hypotheses.jsonl (H-347..H-360). Chronological detail: research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/LOOP_LOG_20261010.md. Index: research/RESEARCH_NOTES_INDEX_20261010.md.
Status: DONE+pos / DONE+null / DONE+neg / RUNNING / QUEUED / PLANNED / NOT-STARTED. Scores = fold-1 WP (control scratch teacher 0.6656-0.6662, one-run sd 0.002, seed sd 0.001) unless noted.

## A. Training-recipe and scaling (teacher level)
| id | idea | status | evidence / files |
|---|---|---|---|
| A1 | prop-power 0 (flat propensity weights) | DONE+pos | 0.66777/0.66717/0.66777 (+0.0015, 3 seeds); fold0 0.68115 vs 0.67763 (+0.0035). LOOP_LOG pass 71,76 |
| A2 | prop0 + label-weight floor .18 + class3 1.19 | RUNNING (replicates vast s4,s5; local s6,s7; kaggle prop0 s6,s7) | 0.66876 (+0.0027) pass 79 |
| A3 | Muon on all matrices at h384 | RUNNING | 0.67021, 0.67150 (+0.0024 vs plain h384); h256 muon-all null (0.6672/0.6661) pass 77,87 |
| A4 | width 320/384/448/512, depth x3/x4 | DONE (mixed) | SCALING_DEEP_DIVE_20261011.md: 384 +0..0.0026 at 3 ep; worse at 4 ep on folds 0,2,3; 512 collapses |
| A5 | epoch ladder 6..15, EMA, SWA, WSD, SGDR | DONE+neg | peak at ep3-4; e12 0.6626, e14 0.6531, e15 0.6605, e18-thin 0.615; kaggle_long_20261010/output_v2 |
| A6 | HPO (Optuna) | STOPPED by user 2026-10-11 | hpo/studies/teacher_v2; t35 replicate 0.6623 vs 0.6688 objective (winner's curse); t52 h256 0.6675/0.6684 |
| A7 | weight soup of fine-tunes (R3-2) | DONE+neg | soup2 s11 0.6601 s12 0.6613 < parent |
| A8 | gated+dropout, dropout, chrono, TCN, LSTM-256, encconv/mlp | DONE null/neg | pass 66,65; DISCORD deep dive table |
| A9 | lr/batch scaling per optimiser (Muon vs AdamW: b16/b64 sqrt vs linear lr) | QUEUED vast scale_b* | LR_AND_DATA_SCALING_LITERATURE_20261011.md |
| A10 | data-size scaling and wd ~ 1/N (Wang&Aitchison), 8400-seq arms, wd .0126/.0085 | RUNNING | data8400_e3_s3 0.66557 (-0.0004 vs ctrl: data volume weak lever, 1 seed); literature note LR_AND_DATA_SCALING_LITERATURE_20261011.md |
| A11 | USER: small lr + 20-30 epochs (Muon/AdamW) | QUEUED (H-351) | previous long runs used lr 4e-4..5e-4 (worse); new arms lr 1.5e-4..2.5e-4 e24 |
## B. Student / distillation
| B1 | student chain with teacher families (dsq_t256/t384/none) | DONE | dsq_t256 0.66938, t384 0.66836, none 0.66861 => teacher signal only +0.0008 over plain continuation |
| B2 | student chain on prop0 teachers (w13_dsq_prop0) | QUEUED (re-shipped after missing distill_pipeline.py on host) | pass 90 |
| B3 | wide student (h384 parent) | PLANNED | never trained; needs muonall_h384 s4 done |
| B4 | USER: scale students with OOD data: random/synthetic input paths + teacher outputs as targets after pretraining (H-350) | PLANNED, implementer assigned | below |
| B5 | privileged 8-row-future teacher (oracle 0.711 vs 0.666) distilled | NOT-STARTED | external_ideation round5 grok #5 |
| B6 | second-stage residual GRU distilled | NOT-STARTED | grok #6, discord 2-stage boosting |
| B7 | head finishing / Z5 merge | DONE neutral on student | pass 68 |
## C. Blending / serving
| C1 | static 0.5/0.5 pair (diverse) | DONE+pos | fold-1 pair 0.6722 (selection-exposed); pair public evidence +0.0005 |
| C2 | adaptive gates, ridge stacker, regime table, early routing | DONE+neg | blend_probe_20261011/RESULTS.md |
| C3 | Rust q8vnni kernel round 2 (26us 1-member, 68us 2-member) | DONE | native/INT8_VNNI_REPORT.md; numba 10x slower scripts/bench_numba_gru.py |
| C4 | candidate ZIPs z1c+z4 pair q8vnni, singles | READY, user uploads | submissions/upload_2026-10-11/ |
| C5 | int4 weights, software prefetch | DONE+neg | kernel report |
| C6 | Zen4 specifics (L2 1MB, 256-bit datapath) | NOTED | SCORER_CPU_SPEC_20261010.md |
## D. External-trick and architecture ideas
| D1 | Discord tricks: highway head, next-row pretrain, frozen extractor, warp aug, anti-curriculum, input-act | QUEUED (v7 flags; requeued after --save-training-state clash) | DISCORD_TRICKS_DEEP_DIVE_20261011.md; v7 README |
| D2 | new archs clockbank / dssm / dualtower (4 ideators converge) | QUEUED (v8) | external_ideation round6/oc_r6_*.md, v8 README |
| D3 | GBM/AutoGluon/GluonTS teacher (H-347) | LightGBM windowed: DONE+neg (0.618 vs 0.6985 subset; blend -0.0003). GBM on hidden state / residual: NOT-STARTED. GluonTS: NOT-STARTED low prior | gbm_teacher_20261011/RESULTS.md |
| D4 | fast-weight bilinear cell, leaky mixture-of-timescales, QGRU, MR-GRC | NOT-STARTED | round6 md (step-5, nemotron, longcat) |
| D5 | MAE/masked-block pretraining on train+valid inputs | NOT-STARTED (v7 pretrain-next is next-row, not masked) | round5 grok #9, muse #6 |
| D6 | time-warp aug, median-variance rescale | QUEUED (d_warp) | v7 |
| D7 | residual EDA (marimo-pair + creative-research lenses) | DONE -> H-361..365 | eda_residuals_20261011/RESULTS.md |
| D8 | class-3 objective (H-361); event-change aux head (H-362, probe first); multi-lag aux (H-363); residual-decorrelated member (H-364); (u,v) loss (H-365) | PROPOSED, none queued | EDA RESULTS |
| D9 | OOD-path online distillation v9 (H-350) | DEPLOYED to Vast B 2026-10-11 (stage+mv), 5 arms queued w14_ood_{splice,swn,spmix,swn25,only} vs dsq_t256 0.66938 | h230_source_v9_ood README |
## E. Infrastructure lessons
E1 local queue treated admission_recheck_failed as permanent failure -> cascaded blocked items (patched scripts/local_queue_runner.py). E2 CPU agents exceeded RAM/CPU admission (rule: agents <=1 core, <=3 GB). E3 Vast: non-atomic deploy raced running jobs; missing distill_pipeline.py on host; jobs with --save-training-state conflict with pretrain flag. E4 watchdog alerts were not read by the session: use scripts/alerts_unread.sh each pass. E5 HPO alerts silenced by .runtime/hpo_disabled.

## F. Distillation deep dive (2026-10-11 05:31+)
| F1 | ceilings: 3-seed muonall ens .67254; student best .66938 | DONE | distillation_deep_dive_20261011/CEILING_AND_PLAN.md |
| F2 | literature survey (Beyer long schedules, Stanton, Cho&Hariharan, DEGU, DIST, LUPI) | DONE | LITERATURE.md |
| F3 | external ideation grok-4.7 + opencode (nemotron/step-5) | DONE (longcat/muse empty) | ideation/ |
| F4 | cache-only diagnostics: ens gain = 82% diff channel, 66% run-level | DONE | CACHE_ONLY_DIAGNOSTICS.md |
| F5 | w15 student arms (muonall teachers, e6, lr2, alpha .9, none_e6, h384) | QUEUED front of Vast B | CEILING_AND_PLAN.md |
| F6 | H-366 diff-channel KD, H-367 run-level KD, H-369 Pearson KD flags | IMPLEMENTATION NEXT | CACHE_ONLY_DIAGNOSTICS.md |
| F7 | H-368 residual KD, H-371 LUPI, H-373 hidden KD | PROPOSED | grok_distill.md |
