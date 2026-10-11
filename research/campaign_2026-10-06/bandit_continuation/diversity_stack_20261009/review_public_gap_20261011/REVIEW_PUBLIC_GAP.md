# Review: why the "good" fold-1 models did not lift the public score (2026-10-11 13:50 EEST)

Reviewer: previous session owner (now review-only). Scope: the three `submissions/10-11-67/` ZIPs uploaded 12:39-13:10 and their public scores. No new fits were launched; two CPU parity checks and table/statistics work from banked files were done. Everything below is traceable to files named in section 9.

## 1. Verdict in five lines

1. **The research gain partly transferred, the scratch member did not.** The distilled student improved local fold 1 by +0.0017 over the previous fold-1-held-out student and improved public by +0.0013 (0.6639 -> 0.6652). The scratch Muon-all x prop0 x 4-epoch single, locally the best servable model (0.67195), scored 0.6616 public: about 0.005 below what the student-class offset predicts.
2. **It is not serving or quantization.** Shipped-ZIP (Q8Vnni) vs fp32 PyTorch on the gate sequences: correlation 0.99997-1.00000, WP 0.75778 vs 0.75779 (scratch) and 0.76060 vs 0.76080 (student). The ZIPs reproduce the models.
3. **The local headline numbers were never comparable to the public best.** All three ZIPs are *fold-1-held-out, exposed-fold, best-of-N* models; the public best (0.6668) is two *all-data* students. Under the one calibrated relation we have (fold-1-held-out distilled students: public = local - 0.0057, n=2), the pair would be expected at ~0.6676 if it behaved like students; it got 0.6650.
4. **The unexplained part is specific to the scratch recipe** and has three candidate causes that the data cannot yet separate (fold-1 selection optimism; sitting next to an overfit cliff; recipe never validated on another fold). Nothing here shows the *direction* is wrong; it shows our *selection protocol* over-trusts fold 1.
5. **Realistic expectation after this review:** all-data versions of the student recipe should land around 0.666-0.668 public (not 0.670+). Closing the 0.005 gap to the leaderboard top needs a different information source or objective, not more of the same seed/epoch/loss tuning.

## 2. The facts

| Upload (public time) | Kind | Local fold-1 (exposed, held-out training) | Public | Local - public |
|---|---|---|---|---|
| `1_p2_student_distilled_from_scratch_h256_teachers` (10-10) | fold-1-held-out student, dsq_t256 | 0.66938 | 0.6639 | **0.0055** |
| `7_dm_t3_a9_e6_fold1_q8vnni` | fold-1-held-out student, 3 muon-all h384 teachers, alpha 0.9, 6 ep | 0.67103 | 0.6652 | **0.0058** |
| `9_muonall_prop0_e4_s4_fold1_q8vnni` | fold-1-held-out scratch h256, Muon-all, prop0, 4 ep, seed 4 | 0.67195 | 0.6616 | **0.0104** |
| `8_..._pair_fold1_q8vnni` (7 + 9, 50/50) | pair | 0.67329 | 0.6650 | 0.0083 |
| `4_z1c_z4_pair_q8vnni` (10-10, user list) | two **all-data** students | n/a (all data fitted) | 0.6668 | n/a |
| `5_z1c_q8vnni_single` | all-data student | n/a | 0.6649 | n/a |
| earlier z5 seed9, z4 seed99, z3 | all-data/other students | n/a | 0.6664 / 0.6662 / 0.6662 | n/a |
| c20261004 combo parent/head/s4 (i16) | teacher-class combo models | ~0.667-0.668 (bank, c20261003_combo_*) | 0.6641 / 0.6654 / 0.6660 | 0.002-0.003 (mapping approximate) |

Public noise between near-identical models: 0.6639-0.6664 for ten student-class uploads (sd ~0.0008). So 0.6616 is >3 sd below that band: real, not portal noise.

## 3. Checks performed in this review

| Check | Result | File |
|---|---|---|
| Q8Vnni vs fp32 (CPU torch, same 10 gate sequences, official accumulator) | corr(fp32,q8) >= 0.99997 on all 10 sequences; scratch WP 0.75779 vs 0.75778; student 0.76080 vs 0.76060 | `parity_scratch_v3.txt`, `parity_student_v3.txt`, script `parity_check.py` |
| Int8 weight statistics (per-row symmetric int8 relative error) of scratch vs student | scratch head.0 0.0157 vs student 0.0128; everything else within 0.001; no outlier explosion | inline (this file section 3) |
| Cross-fold transfer of recipe ranking, 29 recipes with paired fold-0/fold-1 models | Pearson 0.82, Spearman 0.76; slope 0.93; **residual sd 0.0015**; but the top-6 recipes on fold 1 were ranks 3,2,4,5,1,14 on fold 0 | `prediction_bank/LATEST_CROSSFOLD_F0_F1.json` |
| Public noise among similar uploads | sd ~0.0008 | user paste lists |
| Seed spread of the shipped recipe on fold 1 | muon-all x prop0 e4: s3 0.66976, s4 0.67195; plain muon-all e4: 0.66900 / 0.66707; e3: 0.6674 (n=5, sd ~0.001); muon-all x prop0 e3: 0.66897 / 0.66912 / 0.66928 | `LOOP_LOG_20261010.md`, `ops_continuation_20261011/LOCAL_SCORES_R1.json` |

## 4. Reading the numbers

**4.1 What transferred.** For fold-1-held-out *distilled students* the local->public offset is stable: -0.0055 and -0.0058 (two points, residual 0.0003). The student improvement (+0.0017 local) became +0.0013 public (~75%). Distillation with a stronger teacher set, alpha 0.9 and a longer low-LR schedule is a real, small, transferable improvement.

**4.2 What did not.** The scratch single sits 0.0046 below the student-class offset line (0.67195 - 0.0057 = 0.6663 predicted; 0.6616 observed). The pair behaves like "student, diluted by a weaker member": local blend gain over the student +0.0023, public -0.0002.

**4.3 Why the headline comparison was misleading.** The README of the release correctly says the scores are exposed fold-1 numbers; the *ranking* put the pair first. But the public best (0.6668) is an all-data pair; a fold-1-held-out ZIP also omits 457 validation sequences / 149 components that the all-data models see. From our own pairs: all-data z1c (0.6649) vs fold-1-held-out p2 of the same recipe (0.6639): +0.0010. So even a perfect-transfer fold-1-held-out student would need ~0.6657 public to be on par with the old best single, and the pair ~0.667.

**4.4 Magnitude check of selection optimism.** Fold 1 has been evaluated by >300 banked members and >30 distinct recipe decisions (propensity power, label-weight floor, Muon scope, epoch count 3 vs 4, width). With cross-fold residual sd 0.0015 per comparison, the expected winner's-curse inflation of the best of ~30 recipes is ~+0.003, plus ~+0.001 for picking seed 4 of two. That is 0.004 of the 0.0046 residual - the dominant, parsimonious explanation, but it cannot be proven from one scratch canary.

**4.5 Near-cliff fragility (mechanism, partially measured).** The scratch recipe is trained on the edge of an epoch cliff: plain muon-all h256 e5 0.6639, e6 0.6478; h384 e4 0.6652 (s3) / 0.6704 (s4), e5 0.6465; seed spread at e4 is 0.002 vs 0.001 at e3. A model that is 2-3 sd better on fold 1 but sensitive to epoch/seed is exactly the kind that regresses on a different sequence set. Students (warm-start from a converged parent, 2-6 epochs at lr 1e-4, targets regularised by teachers) do not show a cliff (6 epochs with teacher 0.6699 vs 0.6533 without).

**4.6 Things that are NOT supported.** (a) Quantization/serving bug (disproved above). (b) Leakage: fold-1 ancestry checks passed in the release review; leakage would raise public, not lower it. (c) Portal noise (sd 0.0008 << 0.005). (d) "Pair/ensemble is broken": the pair of two *all-data* students gained +0.0019 public earlier; the failure is the weak/non-transferring partner.

## 5. Ranked hypotheses for the scratch gap, with the test that separates them

| # | Hypothesis | Evidence | Discriminating test (cheapest first) |
|---|---|---|---|
| H1 | Fold-1 selection optimism (recipe x seed) | cross-fold residual sd 0.0015; >30 recipe decisions; best-of-2 seeds | Same recipe, **folds 0, 2, 3, 3 seeds each**: if mean fold gain over matched control is < +0.002, optimism confirmed (needs fold-0/2/3 assets: Kaggle has fold-0 assets; Vast lane was drained, check) |
| H2 | Near-cliff fragility of 4-epoch Muon-all | e5/e6 collapse, seed spread | Public canary with the same recipe at **3 epochs, seed not selected** (e.g. `r3_muonall256prop0_f1_s5_e3` 0.66928); or EMA/SWA of epochs 3-4 on fold 1 and 0 |
| H3 | prop-power 0 interacts with the public scored-mask distribution | prop0 gain on fold 0 (+0.0035) and fold 1 (+0.0018), never public | Public canary of a prop0 student vs matched propensity-weighted student (both fold-1-held-out) |
| H4 | Muon-all scratch models transfer worse than warm-started Adam students regardless of recipe | only this one scratch point | One canary of **plain h256 teacher** (no Muon-all, no prop0) from the same fold-1-held-out family gives the family offset |
| H5 | Fold-1-held-out vs all-data handicap larger for scratch models | z1c vs p2 +0.0010 (students) | Train all-data scratch version; compare to its fold-1-held-out sibling on public |

Rule for upload slots (2 left today): one canary can answer H2+H4 (use a *3-epoch, plain-ish* scratch fold-1 model, not the selected seed). Do **not** spend slots on more fold-1-held-out variants of the selected recipe.

## 6. Consequences for the research protocol (what to change)

1. **Shrink fold-1 deltas** before ranking: use `delta_eff = 0.5 x delta_fold1` for any recipe selected on fold 1 alone; require a second fold before treating a gain as real. The fold-0/1 pair table gives residual sd 0.0015 - anything below ~0.003 is noise for *selection*.
2. **Use students as the transferring path**; they carry a calibrated offset (-0.0057, n=2). Next improvements to evaluate: KD-loss variants (diff-channel, run-level), OOD-path distillation, teacher count - each on folds 0 and 1.
3. **Do not choose epoch counts at a cliff.** Pick e3 for Muon-all scratch recipes or add EMA; seeds averaged, not selected by max.
4. **Never pair a strong member with a member of unknown public transfer.** Equal-weight blends only among members with the same measured transfer class.
5. **Report two numbers for every candidate:** fold-1 mean over n>=3 seeds *and* the same recipe on a second fold. The release README should list "transfer class" (student / scratch / all-data).
6. **For the LB, use all-data versions of the student recipe** (teachers must be all-data teachers; parent all-data parent as in the H230 all-data student command) and build pairs from two such students (the only pair type with a proven public gain: +0.0019).

## 7. Honest expectation for the next uploads

| Candidate | Expected public (judgment, +/- 0.0015) | Why |
|---|---|---|
| all-data version of `dm_t3_a9_e6` student recipe, single | 0.665-0.667 | fold-1-held-out sibling 0.6652 + all-data bonus 0.001 |
| pair of two all-data students of that recipe (different seeds/teacher subsets) | 0.667-0.669 | pair gain +0.0019 on top |
| scratch Muon-all x prop0 (any epochs) fold-1-held-out | 0.661-0.664 | observed 0.6616 |
| anything with a scratch member alongside students | <= student alone | observed pair 0.6650 < student 0.6652 |

Do not read these as forecasts of the leaderboard; they are the transfer model fitted to four public points.

## 8. What the next owner should do, in order

1. Do **not** upload the fold-1-held-out variants of the scratch recipe again.
2. Decide the canary (H2/H4) and use one slot: **fold-1-held-out `muonall256prop0` e3 seed 5 (local 0.66928)** is not the selected seed and not at the cliff. Expected under H1-only: ~0.6635; under "scratch Muon-all family does not transfer": ~0.661.
3. Run **recipe confirmation on other folds** before more packaging: muon-all x prop0 e3 and e4 vs matched controls on folds 0 (and 2/3 if assets exist), 3 seeds each. Kaggle has fold-0 assets (`wnn-c20261004-fold0-assets`); the local queue helper is fold-1 only (extend it or use Kaggle `build_long_packet.py` with fold-0 arms). Vast B was drained by the other agent: confirm with `controller.py dash` whether it is still available; do not rent a new instance without the user.
4. Build **all-data student finals** and a **pair of two** (not three), check pair gain locally on the fold-1 sibling pair, and give the user a ranked list.
5. Keep improving the student path (v10 KD flags, OOD) but evaluate on two folds.

## 9. Files

* This review: `review_public_gap_20261011/REVIEW_PUBLIC_GAP.md`; parity logs `parity_scratch_v3.txt`, `parity_student_v3.txt`; script `parity_check.py` (CPU, 1 core).
* Release being reviewed: `submissions/10-11-67/{README.md,READY.json,gates/,*_build_receipt.json}`.
* Public results of the previous round: `PUBLIC_RESULTS_20261010_R2.json`, `PUBLIC_SCORES_20261011.md`.
* Local scores: `LOOP_LOG_20261010.md` (passes 120-128), `ops_continuation_20261011/LOCAL_SCORES_R1.json`, `RESEARCH_PAIR_SCORES_R1.txt`.
* Cross-fold: `prediction_bank/LATEST_CROSSFOLD_F0_F1.json` (29 paired recipes).
* Deep dive and handoff: `DEEP_DIVE_CURRENT_BEST_20261011.md`, `research/HANDOFF_PROMPT_NEXT_AGENT_20261011.md`.

## 10. Addendum 13:45 - Kaggle v6 seeds sharpen the diagnosis (fold 1, h256 Muon-all x prop0)

| Recipe | Seeds (fold 1) | Mean | sd |
|---|---|---|---|
| 3 epochs | s3 .66897, s4 .66912, s5 .66924 (Kaggle) / .66928 (local), s6 .66934, s7 .66813 | **0.6688** | 0.0004 |
| 4 epochs | s3 .66976, s4 .67195, s5 .67011 | **0.6706** | 0.0011 |

* The same seed on different hardware reproduces to 4e-5 (s5: Kaggle .66924 vs local .66928). **Fold-1 seed noise of this recipe at 3 epochs is tiny (sd 0.0004), and 0.0011 at 4 epochs.** The fold-1 ranking of this recipe against its controls is therefore *precise*.
* Winner's curse from picking seed 4 is only ~ +0.0013 (0.67195 vs the e4 mean 0.6706). It cannot explain a 0.0046 shortfall relative to the student-class offset.
* Consequence: the gap is **systematic, not seed noise** - fold 1 rewards this recipe more than the public set does (recipe-level fold-specific gain, H1), or scratch Muon-all models transfer worse as a family (H4). This raises the value of the two decisive tests in section 8: (a) the same recipe on other folds, (b) one non-selected scratch canary on public. Seed replicates on fold 1 add nothing more.
* Kaggle v6 jobs: 4 finished, fit 2.9-3.5 ks each on T4 (rc 0). Preds in `kaggle_long_20261010/output_v6/` (gitignored).

## 11. Corrections after the owner's response (13:55) - read this before sections 4, 6, 7

The owner's note `PUBLIC_TRANSFER_RESPONSE_20261011.md` correctly rejects several things I stated too strongly. Adopted corrections:

1. **No calibrated local->public offset.** "public = local - 0.0057" rests on two student points; it is a descriptive coincidence, not a rule. Section 7's expected-public table is **withdrawn** as a forecast; keep only the qualitative ordering (distilled student improved vs the previous fold-1-held-out student: .6639 -> .6652; scratch single much weaker; pair no better than its student).
2. **No "3 sigma" significance claim** from dispersion across *different uploaded models*; that is not repeated-measurement portal noise. The .0002 pair-vs-student difference is observed, not significant.
3. **The ~0.004 winner's-curse number is an illustrative independent-Gaussian calculation**, not a measured correction for adaptively chosen, correlated recipes/seeds. Use the *qualitative* statement: ranking a selected endpoint is not ranking a stable recipe mean; the shipped seed (s4) was the best of the first two completed seeds.
4. **"Shrink fold-1 deltas by 0.5" and "ignore < 0.003" are unsupported rules** and are withdrawn. Uncertainty guides bounded allocation; it is not an automatic kill gate (ADR0009).
5. **Leakage direction**: leakage can inflate local validation without improving public; ancestry/clean-fitting receipts are the relevant evidence, not my sentence that leakage "would raise public".
6. **Infrastructure note**: Vast B is active (7 running / 7 pending, $32.37/$70 at the 10:26 UTC receipt), not drained. Live receipts override the infrastructure sentence in section 8.
7. **Parity evidence**: 10 gate sequences only; it makes gross serving drift unlikely but does not exclude a concentrated error elsewhere. The owner's planned complete-fold served-endpoint check (shipped callback, official mask/clip/weights, Q8 vs native FP32) is the right closing test.
8. **Hypothesis ids**: my rows `H-376..H-381` collide with the owner's `H-375..H-381` (different statements). Canonical review ids are **`REV-1..REV-6`** (appended to `research/hypotheses.jsonl`); the colliding rows are historical and should be read together with their `mechanism_family` (validation_protocol / transfer_model / serving / blend).

What stands: serving parity on the gate sequences; the observed public values; the student-vs-previous-student direction; the Kaggle v6 seed table (fold-1 seed sd small for this recipe); and the recommendation for cross-fold confirmation + a served-endpoint full-fold check before more scratch packaging.

## 12. Addendum 14:30 - results found on the live Vast host (scored by the reviewer, read-only) and one ops failure

All fold-1 (exposed), h256 students from the same 3 muon-all h384 teachers, shared cache, warm start from the combo parent unless noted. Control for the 2-epoch KD-loss arms = `dm_a9_e2` 0.67018.

| Arm | WP | vs control / reading |
|---|---|---|
| `kd_ch052` (sum x0.5, diff x2) | 0.67040 | +0.0002, null |
| `kd_ch02` (sum 0, diff 2) | 0.66862 | -0.0016 (dropping the sum channel hurts) |
| `kd_pearson` | 0.67028 | +0.0001, null |
| `kd_msepear` | 0.67029 | +0.0001, null |
| `kd_run05` (run-level 0.5) | 0.67023 | +0.0000, null |
| `dm_a95_e6` | 0.67094 | = a9_e6 0.67103 |
| `dm_a9_e10` | 0.67094 | epochs 6->10 no change |
| `dm_pm_a9_e6` (parent = muon-all h256 e3) | 0.67033 | stronger parent: no gain |
| `dm_ba_a9_e6` (born-again, parent = best student) | 0.67054 | no gain |
| earlier: `dm_a10_e6` .67068, `dm_a9_e6` .67103, `dm_t3` .66986, `dm_a9_e2` .67018 | | |

* **Student plateau = 0.6703-0.6710 whatever the alpha, epochs, parent or iteration.** All of those knobs are now exhausted for the cached-train-row-target student.
* **The cache-only diagnostic (82% of the ensemble gain sits in the difference channel, 66% in the run level) did not translate into a distillation gain.** Matching outputs on train rows cannot transfer it: the teachers have fit those rows (small disagreement), consistent with the memorization argument in the literature note. The untested remaining routes are targets *off* the train rows (OOD/augmented paths, `w14_*`) and a different information source.
* New local datum: muon-all x prop0 e4 seed 7 = 0.66953 (e4 now n=4: .66976, .67195, .67011, .66953, mean 0.6703, sd 0.0010; e3 mean 0.6688, n=6). The e4 gain over e3 is ~+0.0015 on fold 1.
* **OPS FAILURE (owner action needed):** the four OOD-distillation arms `w14_ood_splice/spmix/swn/swn25` finished **rc1** with `OSError: distillation cache needs 13.38 GB; available budget ... is 10.5 GB` (host disk 30 GB free; a cache build needs >= 33.4 GB free). They never trained. `w14_ood_only` (no cache) is the only OOD arm that ran/is running. The OOD route is the main untested mechanism for the user's "train on teacher outputs off the train rows" direction; free >= 5 GB on the Vast host (e.g. host copies of `pred_*_valid.npy` that exist locally with identical byte size) and requeue them. The watchdog alert `vast_b_job_log_unidentified:w14_ood_swn25` is this failure surfacing.

## 13. Addendum 15:40 - two new local scores (fold 1, h256 Muon-all x prop0, seed 3; reviewer-scored)

* `r3_muonall256prop0_f1_s3_h378_e6` (owner's same-prefix branch: the 3-epoch trained state continued to 6 epochs with a small-LR tail) = **0.66891** vs the same-seed 3-epoch endpoint 0.66897 (and e3 mean 0.6688): the extra small-LR epochs from an identical state neither help nor collapse (contrast: from-scratch 6-epoch schedule 0.6478). This is a clean separation of "more epochs from the same state" (null) from "a longer high-LR schedule" (collapse).
* `r3_muonall256prop0lwfrecovery_f1_s3_e3` (+ label-weight floor) = **0.66930** vs 0.66897 same seed: +0.0003, null (consistent with earlier lwfloor-on-prop0 nulls).

## 14. Addendum 16:10 - second same-state tail (reviewer-scored, fold 1, seed 3)

`r3_muonall256prop0_f1_s3_h380_e6` (owner's H380 same-prefix Muon tail to 6 epochs) = **0.66885** vs 0.66897 (same-seed 3-epoch endpoint) and 0.66891 (H378 tail): two different tail constructions, both null (-0.0001). Together with the cliff (scratch 6-epoch 0.6478) this closes "more epochs from the same trained state" for this recipe on fold 1: no gain, no collapse. Local GPU queue is empty at 16:05 (0 pending, GPU 17%) - the owner needs to refill it.

## 15. Addendum 17:30 - first OOD/augmented-path distillation results (fold 1, reviewer-scored from the owner's rescued predictions)

| Arm (v9, shared-cache serial R2, weak combo_scratch teachers, h256 student) | WP | Matched references (same teachers) |
|---|---|---|
| `ood_splice` (r2, homshare) | **0.66897** | dsq_t256 0.66938 ; no-teacher continuation dsq_none 0.66861 |
| `ood_spmix` (splice + mixup_paths) | **0.66893** | same |

* Both land between the no-teacher continuation (+0.0003) and the plain cached-target student (-0.0004): **no measurable gain from teacher outputs on spliced/augmented paths** with this teacher set. Together with section 12 (output-matching KD variants null, student plateau 0.6703-0.6710 for the strong-teacher cache) the picture is consistent: off-train-row targets from teachers that are themselves ~0.666-0.669 models carry no more usable signal than the cached train-row targets.
* Not yet tested: OOD paths with the **strong** (muon-all h384 / 3-seed ensemble) teacher set, and `swn`/`swn25` (noise/warp variants) - the owner's run is serial; `ood_only` result pending. The mechanism is not refuted for strong teachers; it is refuted as a way to beat the 0.6705 plateau with the weak set.

## 16. Addendum 18:30 - FIRST CROSS-FOLD CONFIRMATION RESULT (owner's `H382_F2_E3_GRADE_R1.json`, clean fold-2 fit, exposed development panel)

| Fold 2 (443 sequences), h256, seed 3, 3 epochs | WP |
|---|---|
| **Candidate: Muon scope all, prop-power 0** (`h382_muonall256prop0_f2_s3_e3`) | **0.70281** |
| Contextual control: Muon scope rnn, prop-power 2 (`vt_f2_h256_e3`) | 0.70283 |
| Difference | **-0.00002** |

* On fold 1 the same recipe contrast is about **+0.0028** with tight seeds (muon-all x prop0 e3 mean 0.6688, n=6, sd 0.0004, vs control ~0.6660). On fold 2 it is **zero** (one seed; the owner correctly notes the contrast changes scope *and* propensity power together, so no single-factor claim, no seed mean).
* This is the first direct evidence for the review's leading hypothesis (REV-1): the large fold-1 advantage of the muon-all x prop0 recipe is **largely fold-specific**, consistent with its failure to transfer to public (scratch single 0.6616). It is not yet a refutation of the recipe (n=1 on fold 2; fold-0 and fold-3 and 4-epoch contrasts pending in the owner's H382 queue), and fold 2 is a different, easier panel (absolute 0.7028).
* Read together with section 12 (student plateau, KD variants null) and section 15 (OOD null): **the fold-1 research gains of the last day concentrate in recipes that fold 1 happens to reward**; the part that transferred publicly (distilled student, +0.0013) is the warm-started, teacher-regularised path. The cross-fold program (H382 fold 0/2/3, 3 and 4 epochs) is the right gate before any further scratch-recipe packaging. Per CLAUDE.md (2026-10-11) submissions are now ranked by cross-fold evidence with distinct seeds.
* Open: whether prop-power 0 alone (earlier fold-0 +0.0035) or Muon-all alone carries any cross-fold effect; whether the 4-epoch variant behaves differently on folds 0/2/3.

## 17. Addendum 19:40 - third OOD arm (reviewer-scored from rescued predictions)

`ood_swn` (splice + warp + noise, weak combo_scratch teachers, shared-cache R2) = **0.66882** vs cached-target student `dsq_t256` 0.66938 and no-teacher `dsq_none` 0.66861; the earlier `ood_splice` 0.66897 and `ood_spmix` 0.66893. **Three OOD variants all land within 0.0006 of the no-teacher/cached references**: the OOD-path mechanism adds nothing measurable with this weak teacher set (REV-9 stands; `swn25` and `ood_only` predictions are bound/PASS in the owner's receipts but not yet in the local cache; strong-teacher OOD remains untested).

## 18. Addendum 20:00 - fold-2 4-epoch result and last OOD arm (reviewer-scored; fold-2 scoring reproduces the owner's grade exactly)

| Fold 2 (443 seq), muon-all x prop0, h256, seed 3 | WP |
|---|---|
| 3 epochs (owner grade 0.70281; reviewer re-score) | 0.70281 |
| **4 epochs** | **0.70214** |
| Muon-rnn / prop2 3-epoch control (owner grade) | 0.70283 |

* **On fold 2 four epochs are worse than three (-0.0007) and below the control (-0.0007)**, while on fold 1 four epochs were +0.0015 over three (n=4 vs n=6). Two folds, opposite signs for the 4th epoch; and the muon-all x prop0 recipe gives no fold-2 gain at 3 epochs. The shipped 4-epoch scratch single was therefore doubly fold-1-specific. (n=1 seed on fold 2; folds 0 and 3 pending.)
* `ood_swn25` = 0.66911 (fold 1) - the fourth OOD variant, again within 0.0005 of the no-teacher/cached-target references (0.66861/0.66938).

## 19. Addendum 20:15 - cross-fold table now has four folds (reviewer re-scored every prediction with the repository scorer; fold 2 matched the owner's grade exactly)

All muon-all x prop0 contrasts use seed 3, 3 epochs unless stated. "Control" = Muon-rnn, prop-power 2, 3 epochs, same fold/seed (fold 1 control from the long-standing ctrl runs, ~0.6660).

| Fold | Candidate (muon-all x prop0, e3) | Control | Delta | e4 candidate | e4 - e3 |
|---|---|---|---|---|---|
| 1 | 0.6688 (mean n=6) | ~0.6660 | **+0.0028** | 0.6703 (n=4) | +0.0015 |
| 2 | 0.70281 | 0.70283 | **-0.00002** | 0.70214 | **-0.0007** |
| 3 | 0.70865 | 0.70676 (`vt_f3_h256_e3`) | **+0.0019** | - | - |
| 0 | 0.67948 | n/a locally (muon-all prop2 e3: 0.67901, so prop0 alone +0.0005) | partial | 0.68026 | +0.0008 |

* **Revision of section 16.** The recipe effect is **heterogeneous, not null**: +0.0028 / 0.0000 / +0.0019 on folds 1/2/3 (mean +0.0016 over those three), with fold 1 the largest. Fold 1's value overstates the average by roughly a factor of 1.7 on this sample - consistent with selection on fold 1 but not "purely fold-specific". All single-seed except fold 1; no seed means on folds 0/2/3.
* **Fold 0 partial contrast (Kaggle `h382r3`, muon-all in all arms):** prop0 vs prop2 at e3 = +0.0005; e4 vs e3 = +0.0008. Fold-0 control with Muon-rnn is not in the local cache, so Muon-all-vs-rnn is not separable there.
* **4th epoch:** fold 1 +0.0015, fold 0 +0.0008 (prop0 arm), fold 2 -0.0007: mixed, mean about +0.0005 - not established.
* **Expected consequence for public:** an average recipe gain of ~+0.001-0.002 cannot explain, by itself, a public result 0.005 below the distilled student; the seed-4/e4 pick and fold-1 selection together with scratch-vs-student transfer remain the working explanation (REV-1/REV-5 unchanged), but the recipe is a modest real improvement on three of four folds, not noise.
* New fold-1 datum: `h391` (h384 student, alpha 0.9, 6 epochs, seed 3) = 0.67024 vs the h256 student 0.67103: a wider student does not help (plateau again).

## 20. Addendum 21:10 - fold-3 4-epoch result closes the epoch question (reviewer-scored)

Fold 3 (seed 3, muon-all x prop0, h256): **e4 0.70657** vs e3 **0.70865** (-0.0021) and vs the Muon-rnn/prop2 3-epoch control 0.70676 (-0.0002).

| 4th epoch effect (e4 - e3, seed 3) | fold 1 | fold 0 | fold 2 | fold 3 | mean |
|---|---|---|---|---|---|
| WP delta | +0.0015 (n=4 vs 6) | +0.0008 | -0.0007 | -0.0021 | **-0.0001** |

* Across four folds the 4th epoch is **neutral on average (-0.0001) with sign flips**; it is a fold-1 feature. The shipped `ZIP9` (4 epochs, selected seed) took exactly the setting that only fold 1 rewarded; with 3 epochs the recipe would have been +0.0016 over the control on folds 1-3. Rule for the next owner: **muon-all x prop0 at 3 epochs only**; no 4-epoch variants without a multi-fold reason.

## 21. Addendum 23:20 - review of the new full-data packages (`submissions/10-11-exploit-now_r2/`, ADR0032 packaging-now)

Packages: (A) `h390_full_data_pair_q8` = two all-data H390 students (strong-T3 teacher recipe, seeds 3/4, 50/50); (B) `old_pair_plus_h390_pair_q8` = old all-data z1c + z4 + the two H390 students, 0.25 each (4 members, 16.8 MB). Owner's selector note: `submission_selector_20261011_r1/SELECTION_EDA_R1.md` (fold-1: 3-seed H384 trio 0.67254, H256 + trio 0.67264, +0.0001 with CI spanning zero; no weight tuning).

Reviewer observations (no new fits; from the receipts):

1. **Package A is the right kind of candidate.** It is a *pair of students of the recipe that transferred* (+0.0013 public vs the previous fold-1 student), all-data, equal weights, ~27 s on the 10-sequence callback panel - the same cost class as the public-best pair (27 portal-min). It contains no scratch Muon-all member. Its all-data fit means no held-out grade exists for it; the relevant evidence is the fold-1 sibling (student 0.67103) and the three-fold-1 pair data.
2. **Package B carries a portal-timeout risk.** The callback panel for B took **58.3 s vs 27.0 s for A (2.15x)**. The two earlier public timings were pair = 27 min and single = 15.7 min (~13.5 portal-min per h256 member), and two-model h384 packages timed out at ~62 min under a 60-min limit with ~2x portal variance. A 4-member h256 package projects to ~54-58 portal-min typical: **at or above the limit with no variance margin.** Unless the owner has a measured portal-equivalent estimate well under ~40 min, do not upload B; if it is uploaded, expect a real chance of `SCORE:TIMEOUT` (wasting a daily slot).
3. **Value check for B.** In fold-1 terms the extra pair adds a pair-of-pairs effect of unknown size (z1c/z4 and H390 are different lineages: old H230 students vs strong-T3 students); no same-recipe clean-fold grade exists for the 4-member mixture, and the selector's own result for adding a 2nd family to an ensemble was +0.0001 (CI spans zero). Expected benefit does not justify the timeout risk; a 3-member variant (z1c + two H390) would cost ~40 portal-min.
4. Packaging hygiene: the owner already retracted a mislabeled FP32 packaging and required Q8 gate rows; package A has an FP32-vs-Q8 official-metric attempt receipt on the fold-1 sibling only - the complete-fold served-endpoint comparison remains the open closing check from section 11.

Recommendation (reviewer, non-binding; user decides uploads): **A first.** Consider a 3-member variant instead of B. Ask the owner for a measured portal-minute estimate for B (callback time per row x 39.4M rows x the 0.616 portal/local factor used earlier: 58.3 s per panel vs 27.0 s for A scales A's ~27 portal-min to ~58).

## 22. Addendum 23:35 - owner's precision grade of the H390 pair agrees with the reviewer's parity check

`submission_selector_20261011_r1/q4_precision_h390_r1/PRECISION_GRADE_REPORT_R1.md`: on the same ten 20,000-row groups the corrected Q8 H390 pair scores 0.76337826 vs 0.76344916 for FP32 (**Q8 - FP32 = -0.0000709**); a Q4-rounded variant loses -0.0018 vs Q8 (smaller ZIP but not a native INT4 package or a speed measurement). This is independent confirmation of section 3 (Q8 serving lossless to ~1e-4 WP on gate sequences) for a second model family; Q4 is not supported as a replacement. The complete-fold served-endpoint check remains open but the evidence now spans three packages.

## 23. Addendum 00:40 - portal-time estimates for the FP16-transport packages (`submissions/10-11-exploit-now_r3_fp16transport_q8/`)

Owner's callback panel (loaded, 1 core): existing H388 s4/s5 pair 103-117 us/row; H388 trio 148-149; H388 FP16-transport trio 152-156; H390 student 1/2 + H388 thirds 177-185. ZIPs: trio 13.7 MB (FP16 transport, 45.6% smaller than FP32 25.3 MB, which breaks the 20 MB ceiling), student+trio 15.7 MB.

Calibration from our own public data (`SCORER_CPU_SPEC_20261010.md` / `PUBLIC_SCORES_20261011.md`): portal minutes ~ 0.616 x (QUIET local us/row) x 39.4M rows / 60 s; the earlier public pair measured 113.6 us/row loaded (~68-85 quiet) and took 27 portal-min. Loaded numbers overstate quiet ones by ~1.4-1.6x. Applying the same conversion:

| Package | loaded us/row | ~quiet us/row | est. portal-min typical | at 1.8x portal variance |
|---|---|---|---|---|
| H388 s4/s5 pair | 103-117 | 70-80 | 28-32 | 50-58 |
| H388 trio (FP16 transport) | 152-156 | 100-110 | 40-45 | 72-81 (timeout) |
| H390 half + H388 thirds (4 members) | 177-185 | 115-125 | 47-51 | 85-92 (timeout) |
| H390 pair (A, earlier) | ~ pair-class | ~70-80 | ~27-31 | ~49-56 |

* The h384 members are no longer prohibitively slow (about 50 us/row each vs 733 us/row on the old kernel), but a **three-h384 package is at or beyond the limit once portal variance (observed ~1.8x, with two earlier TIMEOUTs at ~62 min) is included**. The 4-member student+trio is worse. Prior public evidence: pairs are safe (27 min), two-model h384 i16 timed out.
* Fold-1 value of the trio: 0.67254 vs the h256 student 0.67103 (+0.0015, exposed); a package that times out scores nothing. Expected benefit per upload slot is therefore **lower than the H390 student pair at similar quality unless the portal time is confirmed well under ~35 min**. Recommended order unchanged: H390 pair (A) first; use a trio only if the user accepts the timeout risk for one slot, preferably after A's portal duration is known (calibrates the loaded->portal conversion for the current runtime).

## 24. Addendum 01:00 - owner's `TOP_CANDIDATES_R3.md` (2 operational candidates) against the portal-time estimate

1. `12_h388_teacher_trio_all_fp16store_q8vnni.zip` - three own H384 seeds 3/4/5 (all-data), 13.7 MB. Fold-1 sibling (float) 0.67254. **Portal estimate (section 23): ~40-45 min typical, ~72-81 min at 1.8x variance: timeout risk.**
2. `11_h390_student_s3_h388_teacher_s5_all_q8vnni.zip` - half H256 student, half H384 teacher, 12.9 MB, two members (~33-50 us/row quiet each => ~35 portal-min typical, ~63 at 1.8x). Related float prototype 0.67257; the student/teacher mixture has a *closer-in-kind* public precedent (pair 27 min). Lower timeout risk than #12 but not zero.

Reviewer note: both are all-data finals (no held-out score), built from the *cross-fold-weak* recipe components (Muon-all teachers, 3 epochs - acceptable per section 20) and from the student recipe that transferred. The owner's README correctly gives no public forecast. Portal duration of whichever is uploaded first should be recorded to recalibrate the conversion before the second slot.

## 25. Addendum 01:20 - the right baseline was the existing H230 lineage, not the plain control (reviewer-scored fold-0 old-Z4 reconstruction + banked cross-fold table)

Banked cross-fold scores (`prediction_bank/LATEST_CROSSFOLD_F0_F1.json`, same recipe trained for each fold, single models):

| Recipe | Fold 0 | Fold 1 |
|---|---|---|
| `c20261004_combo_head_s3` | 0.6819 | 0.6673 |
| **`h230_s3_onecycle` / `h230_s4_onecycle`** | **0.6817 / 0.6814** | **0.6688 / 0.6691** |
| `c20261003_combo_s3/s4` | 0.6809 / 0.6810 | 0.6674 / 0.6679 |
| **new: muon-all x prop0 e3, seed 3** | **0.6795** (Kaggle `h382r3`) | **0.6688** (n=6) |

* The new owner-run fold-0 old-Z4-parent reconstruction (`oldz4_h230_parent_f0_s3_r3`) scores **0.68282**, its student **0.68314** (+0.0003 over the parent on fold 0, versus a student lift of about +0.003 on fold 1), reproduced with the exported copy (0.68282).
* **Reframing:** the muon-all x prop0 recipe equals the existing H230-onecycle teacher recipe on fold 1 (0.6688 vs 0.6688-0.6691) and is **0.0022-0.0033 below it on fold 0**. Its "+0.0028 on fold 1" was measured against the *plain* 3-epoch control (0.6660), which is not the strongest prior recipe; relative to the lineage that produced the public best, the recipe is neutral on fold 1 and negative on fold 0. This explains why the fold-1-selected scratch single did not beat the H230-lineage uploads on public, without needing selection optimism alone.
* The H230 student lift over its parent is also fold-dependent (+0.003 fold 1, +0.0003 fold 0): much of the "student gain" on fold 1 is fold-1-specific as well. The public student improvement observed (+0.0013 for the newer student vs the previous fold-1 student) therefore has two causes that cannot be separated with the data at hand.
* Consequence for selection: every new recipe must be compared with the **H230 onecycle parent on the same folds**, not with the plain control (ADR0023: match teacher stages and parents). Cross-fold comparisons against `h230_s3/s4_onecycle` are available now from the bank for folds 0 and 1 and need only matched fits on 2/3.
