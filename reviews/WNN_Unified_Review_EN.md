# WNN Research Campaign: Unified Critical Review

**October 10, 2026.** Public score **0.6664**; target **>0.672**, gap **+0.0056** (archived leader 0.6717). This English review consolidates the earlier detailed review and the follow-up critique. It distinguishes measured results from interpretations and proposed tests.

## Executive verdict

The campaign's biggest weakness is **experimental inference and endpoint discipline**, not the absence of ideas. Narrow experiments have been used to reject broad mechanisms; incomplete submission pipelines have been compared with fully finished ones; and important serving results have been lost between logs and summaries. There is no verified recipe that delivers +0.0056, but the evidence does not establish a causal ceiling at 0.6664.

**Recommended order:** (1) finish new students exactly as the public champion was finished; (2) compare the *actual* new strong h384 teachers with current h256 teachers through matched student pipelines; (3) retest own q8x and h384 native serving; (4) isolate clean EMA and cheap target-specific/family-head blends; (5) test asynchronous history with proper controls; (6) reserve one costly branch for original-target privileged teaching or learned memory.

## 1. Corrections to earlier concerns

The existing system has real strengths: H230 distillation gains about **+0.003** locally over scratch teachers (reported fold-1 paired interval about **+0.0016 to +0.0049**), h384 is positive on four folds (mean **+0.0012**), and the team maintains component-aware splits, seed-noise measurements and native benchmarks.

Source inspection also rules out several supposed bugs. `gmetric.py` **correctly** computes separate target-specific weighted Pearson correlations and averages them; the report's “pooled over rows and both targets” phrasing was inaccurate. Teacher weights summing to **0.8125** are intentional, with a separately normalized target already supported. The auxiliary already predicts future-feature **differences**, not simply future levels. Event clocks, raw-only, attention, PLE, activation mixtures and stacking have already been tested in at least some forms. Do not relaunch old work under new names.

## 2. Conclusions that were too strong

**Lookahead does not prove a causal ceiling.** Fold-1 scores with 0/2/8/32/128/512 future rows were approximately **0.666/0.680/0.711/0.780/0.891/0.960**. This establishes that future information helps, not the best possible score from past information. The R3 contract's future-8 threshold of ~0.75 for privileged teaching is not justified as a predictor of a +0.0056 student gain. A failed auxiliary only rejects that objective and training regime.

**Capacity and optimization were confounded.** A four-epoch h512 run near **0.658** was used against scaling; later three-epoch h512 and h448 reached **0.66722** and **0.66812**. Changing epoch count under OneCycle changes the entire LR trajectory. Changing chunk size also changes update count, loss normalization and gradient horizon. A recipe comparison can select a winner, but cannot isolate its mechanism without controls.

**“EMA failed” did not test pure EMA.** HPO's `avg=ema` passes `--ema-teacher 1.0 --eval-ema`: the first flag adds consistency loss and changes gradients. The bad eight-epoch result also varied optimization. Use `--eval-ema --ema-teacher 0` to compare raw and averaged weights from the **same** trajectory; preserve both checkpoints. The teammate's reported +0.0016 six-epoch EMA gain is a reason to test, not a transferable estimate.

**Frozen probes cannot reject trainable representations.** E415/E418 used about **6,158 scored rows**, 140 components, random compressed frozen features and ridge. They only test that readout. The planned E435 instrument-1 innovation comparison against own-instrument and timing controls was not established as a completed modern end-to-end experiment. All 112 raw columns are already accessible; the opportunity is representing asynchronous history better.

**Persistent labels do not justify fixed predictions.** Target-run age uses realized label boundaries unavailable online. E535's hard output hold reduced score approximately **0.68172 → 0.55817**, with ~83% of scored weight between price events. Even during a constant realized target, new observations change the conditional forecast. A new memory must learn *when* to update; do not revive hard holds.

**Class contributions are not additive.** Class 3 carries substantial difficult metric weight, but global weighted Pearson is not a fixed weighted sum of class correlations. Multiplying a hypothetical class gain by its weight share does not predict overall improvement. Similarly, reconstructed identities tying both targets to one latent future price are exact only in certain branches; prefer soft structure constraints or free sum/difference outputs.

## 3. Confirmed HPO defects

**Changing objective scale.** `hpo/orchestrator.py` re-estimates the fold-0 offset as trials arrive without rewriting older objectives. The offset changed about **0.013100 → 0.011994**, granting later two-fold trials up to **0.000553** artificial advantage. Trial 2 moves tenth to eighth after common-scale recalculation; **the top five do not change**. Rank complete trials by a fixed mean paired delta to controls, or the same two-fold mean. This is a real bug, not evidence of a hidden winner.

**Pruning contradicts the protocol.** T0 was meant to reject catastrophes (~0.005 worse), not decide close differences inside ±0.004. MedianPruner with zero warmup pruned trial 50 after fold 1 at **0.6670755**, above the recorded baseline **0.66642**, before fold 0. This may save compute but is not a scientific rejection. Give plausible new mechanisms two dev folds.

**Select by mean, not CI lower bound.** Paired bootstrap intervals quantify uncertainty and guide replication; they are not automatic vetoes. A positive noisy estimate is not proven, but may be the best decision. Keep seeds, components, exposure and checkpoint ancestry visible; evaluate the official global metric, not hard subgroup thresholds.

## 4. Incomplete comparisons of final solutions

| Candidate | Public | Difference |
|---|---:|---|
| **Z5** | **0.6664** | All-data H230, Muon parent, **head finishing** |
| **P2** | **0.6639** | Fold-1 held out, no equivalent finishing |
| Earlier H230 pair | **0.6653** | Different parent/teacher lineage |
| P1/P3 | **No score** | Timeout, not a negative quality measurement |

An earlier finishing stage improved public **0.6641 → 0.6654**. It does not guarantee the same improvement now, but P2 versus Z5 cannot isolate student quality. Test the chosen student raw, after identical head finishing, and with a small fixed blend. Compare **finished full-data methods** under matched serving conditions. An older pair has a score receipt despite a runtime above 61 minutes; do not conflate all long runs with unscored timeouts.

**The failed own pair had a different structure.** Two students shared an **AdamW/C2000 parent** rather than separate stronger **Muon/C1000 parents**. Two `raw_wmse` sources had `raw_only=false`. This does not prove corrected diversity wins; it means 0.6653 rejects that *particular* lineage. Raw-only was already trained (~**0.66094** under Muon×3), so test its marginal contribution, not its novelty. Hold propensity/since row weights fixed because some raw-only paths alter weighting features.

**The allegedly failed transfer of new h384 quality was not actually tested.** Reported fold-1 students: h256 teachers **0.66938**, h384 teachers **0.66836**, no teacher **0.66861**, mixed **0.66840**. But `dsq_t384` used `vt_f1_gru384_s4/s5`, older **four-epoch-schedule** teachers exported at ep3, **not** the new three-epoch h384 ensemble scoring **0.67020**. Re-run current h256 versus actual current h384 teachers with identical OOF components, student init, data, teacher normalization/clipping, coefficients, schedule and finishing. Also evaluate direct serving if feasible. The previous negative result applies only to its own sources.

**Use existing outputs more intelligently.** The official metric averages separate t0/t1 correlations, so target-specific blend weights are legitimate. Small D02 found a WMSE/Huber blend improving `t1` by **+0.001293** despite losing overall; blending only t1 would give **+0.000647 on that small panel** (350 training sequences, 220 updates), not an established Z5 gain. Test a few predefined target-specific mixtures on the prediction bank. H230 family heads are already trained; if their outputs and main outputs are rows of one final linear layer, a fixed raw-output mixture can be **folded into two final weight rows**, with clipping performed once afterward. No second GRU is required. Broad ridge/MLP/booster stacking already has negative evidence.

## 5. Serving: overlooked opportunities

**E525 measured a real own q8x speedup:**

| Kernel | One model μs/row | Two models μs/row |
|---|---:|---:|
| Own Rust i16 | 64.79 | 136.18 |
| **Own Rust q8x** | **51.38** | **103.75** |
| Reference C++ INT8 VNNI | 19.33 | 49.49 |
| Reference C++ forced AVX2 | 77.86 | 149.09 |

Own q8x is **~21% faster** for one and **~24%** for two. Later summaries citing older E416 as “INT8 does not help” missed E525. ISA, dispatch, precision and kernel matter; “rewrite in C++” does not follow. Maximum reported FP32 deviation: own q8x **0.01726**, reference INT8 **0.07182**; quality was checked on only two exposed sequences. Re-evaluate official score on full available component-clean OOF data.

E462 had i16 **0.699821638** versus q8x **0.699804134** on its own small panel (delta **−0.00001750**), but public scoring failed on `libpython3.13.so.1.0`. A Python 3.11 repair later passed checks and 640,000 rows before interruption. This is an **unfinished technical path**, not proof INT8 harms quality; its panel result does not automatically transfer to Z5.

**h384 timing is inconsistent.** E420 measured **99.064 μs/row** for `original_w384_q16x`, whereas a later report says **733 μs/row**. These could differ in binary, layout, precision, CPU, callback scope or dispatch. Benchmark identical weights on identical CPU and full callback, recording actual ISA/precision and official score. Do not call GRU384 intrinsically unservable from the slower figure alone.

Quantization changes the model: test both predictive quality and the stronger feasible ensemble it permits. High output correlation to FP32 is not enough; requiring nearly identical FP32 predictions may also reject a useful compromise. The 60-minute limit for ~39.4M rows implies **~91.4 μs/row end-to-end**. Local 74 μs/row implies ~48.6 minutes while some portal singles ran in 33–37 minutes: conditions differ. Two similar packages timed out while P2 completed in 34 minutes; portal contention is suspected, **not proven**. Keep a safety margin.

## 6. Undervalued ideas worth testing

**Asynchronous slow/fast history and instrument-1 innovations.** The slow/fast probe reported median R² ~**0.26** for a 22-column slow block, but included the fast increment **on the same row** as the slow update. This demonstrates contemporaneous redundancy, not prediction ahead of the update. The probe used 40 sequences and 8,000 initial rows; column grouping was inferred from the first 40 changes in one sequence. First exclude the simultaneous endpoint increment and remeasure. Then construct causal cumulative fast displacement since last slow update, update age, and residuals against a mapping fitted on training components only. Compare against size-matched own-instrument lag and timing controls. Judge end-to-end score, not R² alone.

**Volume history (E413).** A matched test of 28 history features against equal-size current-input features gave **+0.00096179**, paired interval **[−0.00008,+0.00206]**, on one exposed fold and seed. This is not a proven improvement, but a better reason to replicate than many new ideas. Repeat on the current recipe and another fold, then a second paired seed if promising. E427 volume-gated recurrence has CPU checks but **zero scientific fits**; it is untested, not negative.

**Original-target privileged teacher.** The failed boot8 replaces labels under a target-run condition using a teacher prediction at `t+8`. A different mechanism is a teacher that sees future **inputs** but predicts the original **Y(t)**, generating component-out-of-fold soft targets for a strictly causal student. `privileged.py` and `privileged_crossfit.py` contain preparations, but completed matched fits were not established. Compare with an equal-capacity teacher receiving extra **current** inputs; keep future information entirely out of student inference. This is a reserve research branch, not a promised gain.

**Learned residual memory (E497).** A delta erase/write prototype has 16×16 key/value dimensions and ~1 KB state per sequence, but **zero scientific fits**, incomplete trainer integration and no native export. Test a small residual memory block with explicit causal read/write order and zero-initialized output against size-matched fixed-decay memory. It is different from hard output hold, fixed attention checkpoints and ordinary dual-clock GRU. Profile sequential autograd before investing heavily.

## 7. What should remain deprioritized

| Idea | Evidence / limitation |
|---|---|
| Hard output holding | E535: ~0.68172 → 0.55817 |
| Existing H218 attention checkpoint/blend | Negative for that checkpoint, not a matched modern architecture test |
| Fixed GELU/Leaky output mixture | Full-scale ~0.66490 vs controls ~0.66557 / 0.66675 |
| Ordinary dual-clock GRU | Older and newer negative tests |
| Same PLE recipe | Lost to matched linear control; small one-epoch regime |
| Generic future-feature auxiliary | Already implemented; horizon 2/8 weight 0.3 ~0.66544 vs control mean ~0.6659 |
| Future quote-path event-time auxiliary | E251 already tested 8/24 book events negatively |
| Lattice/inverse-QT reconstruction | Earlier apparent 63-column finding retracted as tolerance artifact |
| Broad ridge/MLP/booster stacking | Meaningful prior negative blend tests |
| Chunk-500 as a confirmed lever | Initial 0.66845 regressed across seeds to ~0.6673 mean; fold-0 flat/negative |
| Stopped-bootstrap h8 | Student ~0.66819 vs ep3 student ~0.66923 |

These results reject specific implementations, not entire architectural families. A rerun needs a concrete changed mechanism, repaired confound or relevant new endpoint. For the existing auxiliary, a narrower experiment is fixed train-only per-channel/horizon scales and controlled aggregate auxiliary gradient weight; current normalization depends on batch/chunk. First collect the already running variants.

## 8. Research execution problems

The logs show HPO reseeding each tick and repeating one configuration six times (roughly twelve fold runs wasted); a queue feeder accepting `ready` while backlog items were `queued`; one export overwriting another model's prediction artifact; and lexicographic queue sorting changing intended order. These incidents were repaired and must not be described as current bugs. Their shared lesson is that artifact identity and task lifecycle need **executable invariants**, not just documentation.

Track each hypothesis through **proposed → implemented → smoke-tested → trained → graded → matched comparison → independent confirmation → finished native endpoint → public result**. Require immutable prediction names, hashes of source/checkpoints/teacher caches, explicit fold membership, seeds, finishing, precision and ZIP provenance. Reject duplicate configurations before scheduling GPU time. Treat a source-only prototype, failed launch and completed controlled experiment as different evidence statuses. For component bootstrap, recompute the **global** correlation after resampling components rather than averaging per-component correlations. Do not repeatedly mine supposedly reserved folds.

## 9. Bounded recovery plan

**Phase A — reconstruct a comparable baseline.** Map the public 0.6664 Z5 to its exact ZIP, parent, teachers, training membership, finishing and precision. Repair stale prediction-bank identities. Recalculate HPO objectives on one scale. Test predefined H230 main/family and target-specific mixtures on identical components. Freeze controls and seed pairs.

**Phase B — finish existing promising endpoints.** Apply Z5-style head finishing to the strongest new student. Compare *actual* current h256 and 3ep h384 teachers under identical student conditions. Evaluate direct teacher ensembles if serving permits. Test own q8x versus production i16 on matched weights, complete callback and official OOF metric; reconcile the h384 99/733 μs discrepancy. Evaluate clean EMA raw/average checkpoints from one trajectory. Test true raw-only input diversity with unchanged row weights and independent strong parents.

**Phase C — one controlled history mechanism.** Either reproduce E413 volume history on the current recipe or test endpoint-excluded slow/fast+i1 innovations with size-matched timing and own-instrument controls. Do not merge every proposed feature into one uninterpretable pack. Prefer a small zero-initialized adapter and a parameter-matched control.

**Phase D — one reserve branch.** Choose original-target privileged teaching **or** learned residual memory based on integration readiness and the cost of one complete causal comparison. Do not automatically run both plus another architecture grid.

**Decision rules:** select by mean official score; report paired uncertainty without using CI lower bound as a universal veto; use matched components, seeds, ancestry, finishing and serving; measure **marginal ensemble contribution** rather than standalone quality or correlation; record timeouts as missing quality; never add gains from unrelated local controls and claim a projected public score.

## 10. Priority table

| Priority | Action | Reason |
|---|---|---|
| **P0** | Finish strongest new student and compare fairly with Z5 | Known finishing value; previous comparison incomplete |
| **P0, CPU parallel** | Reconcile E525 q8x and E420 h384 serving | Could unlock a stronger feasible ensemble |
| **P1** | Transfer actual new strong h384 teachers | Existing negative conclusion used older teacher sources |
| **P1, cheap** | Target-specific and H230 family-head blends | Underused outputs, potentially foldable into final layer |
| **P2** | Clean EMA without consistency loss | Previous negative run was confounded |
| **P2** | E413 replication or causal asynchronous-history adapter | Positive/incomplete evidence and meaningful new representation |
| **P3** | Original-target privileged teacher or learned memory | High uncertainty and integration cost |

**Final judgment:** The largest strategic mistake would be another broad student-LR/dropout/chunk/stacking sweep while these precise missing comparisons remain unresolved. The strongest route is **a properly finished strong student, the correct strong teacher sources, and an honestly measured native serving frontier**, followed by one causal asynchronous-history mechanism. The supplied evidence does not justify declaring the available signal exhausted.

## Source map

`BC` means `research/campaign_2026-10-06/bandit_continuation` within the uploaded archive.

- **S01:** `LOOP_LOG_20261010.md`, especially passes 12–20, 22, 25–30, 33–38, 40–47: results, width/seed studies, public scores, lookahead and process incidents.
- **S02:** `BC/diversity_stack_20261009/solution_report/CURRENT_BEST_SOLUTION_EXPLAINED.md`: model, student chain, HPO, serving, plateau interpretation.
- **S03:** `BC/continuous_execution_E529/h230_source_v6_lead/src/wnn/gmetric.py`: actual target-specific scoring.
- **S04:** `scripts/build_distill_cache.py`, `scripts/distill_pipeline.py`, `BC/vast_teachers_r1/make_refill_jobs.py` and `refill_jobs` JSON: teacher normalization and exact checkpoint lineage.
- **S05:** `BC/diversity_stack_20261009/CONTRACT_R3_TEACHER_CEILING_20261010.md`: R3 falsifiers and Z5/P2 framing.
- **S06:** `hpo/orchestrator.py`, `hpo/store.py`, `hpo/space.py`, `hpo/studies/teacher_v2/events.jsonl`, V6 `scripts/train_rnn.py`: HPO offsets, pruning and EMA.
- **S07:** `hpo/VALIDATION_PROTOCOL.md`, `RESEARCH_NOTES_INDEX_20261010.md`: uncertainty and selection policy.
- **S08:** V6 `scripts/train_rnn.py`: stopped bootstrap and future-feature-difference auxiliary.
- **S09:** `BC/diversity_stack_20261009/OPTIMIZERS_SCHEDULES_STABILITY_R1.md`: full-scale earlier recipe comparisons.
- **S10:** `research/experiments.jsonl` E251/E413/E415/E418/E427/E435/E497; `scripts/feature_screen.py`: probe limitations and incomplete experiments.
- **S11:** `BC/model_data_forensics_E535/E535_H230_OUTPUT_HOLD_REPORT_V1_20261008.md`: output-hold result.
- **S12:** `research/ELEPHANTS_REVIEW_2026-10-03.md`, `...10-04.md`, `...10-05.md`, `SECOND_PAIR_OF_EYES_2026-10-05.md`: historical priorities.
- **S13:** `BC/continuous_execution_E529/H230_PAIR_USER_PUBLIC_RESULT_20261008_R1.json`, `BC/reviews/MAXIM_READY_PAIR_METHOD_COMPARISON_20261008_R2.md`, H273 raw-feature contrast, E457: parent structure, public pair, finishing.
- **S14:** `BC/engine_efficiency_E525/COMPARISON.md` and matched quantization JSON: own/reference kernel measurements.
- **S15:** H218 attention review, E540 PLE result, H042 and E079/E080/E431: negative candidates and target identity caveats.
- **S16:** `research/experiments.jsonl` E409/E416/E420/E462: precision, width timing and ABI repair.
- **S17:** `scripts/slow_fast_residual_probe.py`, loop pass 30: contemporaneous endpoint inclusion and column grouping.
- **S18–S22:** `privileged.py`, `privileged_crossfit.py`, E497 prototype, blend studies and D02 target-specific evidence, as referenced in the archived research notes.

**Evidence convention:** A positive result on an exposed fold is not a proven public improvement. A public timeout is not a quality measurement. An implemented prototype with zero scientific fits is neither validated nor refuted.
