# [Experiment] Train-only teacher track: PR49 reproduction, distillation into a Muon child, TL3C public results

1. **Status, lineage and owner.** Partly complete; stopped by the user on Oct 29. Continuation
   proposed. Owner: Vlad / Claude Code. Inputs:
   - teacher recipe from PR49 (Semyon), reproduced locally under ADR0044 train-only;
   - TL3C ZIPs packed with Ivan's PR51 packet and the team `wnn_cpp` INT8 runtime.
2. **What is established.**
   - **PR49 teachers reproduce across machines.** Full-valid WP at epoch 3, local vs PR49:

     | Teacher | Local | PR49 |
     |---|---|---|
     | fin256_s3 | 0.68780 | 0.68826 |
     | fin256_s4 | 0.68739 | 0.687136 |
     | fbc1k_s5 | 0.69142 | 0.691355 |

   - **Distillation into the fixed Muon child.** D1 (labels/teacher α 0.9, fbc1k teacher,
     seed 3) reaches **0.68997** at the fixed E5 endpoint. That is +0.00326 over max(parent P
     0.68671, labels-only child C0 0.67562).
     - C0 overfits: it is worse than its own parent.
     - Caveat: D1 stays below its equally deployable teacher fbc1k (0.69142). It is useful only
       as a blend member or as the student stage in a chain, not as a replacement.
   - **TL3C public results** (served by INT8 on the host runtime, uploaded Oct 29):

     | ZIP | Valid | Public | Portal duration |
     |---|---|---|---|
     | blend 0.34 fbc1k_s5 + 0.33 tlead_s5 + 0.33 tlead_s6 | 0.692336 | 0.66355 | 42 min |
     | tiedlead-s5 single | 0.691434 | 0.66246 | 18 min |
     | tiedlead-s5 F32 | — | **timeout** | 62 min |

     Both are below the full-corpus TL3 (0.66832).
3. **Compute advantage.** The track turns PR49 B300/L4 teacher compute into deployable GRU
   students. The open question is whether train-only distillation gains survive the final
   train+valid refit. The public gap above says the refit dominates selection gains of 0.001.
4. **Frozen comparison.** Child recipe = our fixed train-only Muon child (`recipe_args.sh`
   CHILD: 6 epochs, LR 1e-4, batch 16, chunk 2000, aux-next/aux-struct 0.3, since-weight 32,
   Muon rnn scope, strict FP32). Initialized from parent `tt29_parent_s3@ep3`. Teacher
   `tt29rep_fbc1k_s5`. Arms differ only in `--distill-alpha`.
5. **Fitting panel (continuation).**
   - D2 α 0.6, seed 3: **1 fit**. It was interrupted in epoch 1 and needs a rerun.
   - D1 α 0.9, seeds 4 and 5: **2 fits**. The plan gate passed (≥ +0.001), so confirmation is
     eligible; it was never launched.
   - Seeds 4/5 need their own parents P s4/s5: **2 parent fits**, unless the existing local
     parents are reused.
   - Zero-fit: export D1 valid predictions; compute ρ and the blend-law value against the three
     TL3C members and the served blend.
   - Total: 3–5 new fits, all local-GPU sized (≤ 5 GB VRAM).
6. **Validation and uncertainty.** Official metric on full valid (exposed development data).
   Seed-3 single cell so far. Public scores are single observations with ≈ 0.0008 noise.
7. **Leakage and coverage.** Train-only fitting (10,325 sequences). The local split SHA differs
   from `0c4de3a2…` only by embedded receipt mtimes; counts are identical, and the byte
   membership check is pending. Scratch `train_targets.npz` was rebuilt with row group 528
   (undecodable in the pre-Oct-3 download); the shared cache was untouched.
8. **Decision-changing result.**
   - If D1 seeds 4/5 hold ≥ +0.001 over P, and D1 enters the TL3C blend at ρ below break-even,
     a full-corpus refit of the distilled child is worth one submission slot.
   - Otherwise close the child stage and keep the TL3C students as they are.
   - Separately: compare the team's full-corpus refits (PR54 w18 0.66453) with TL3 0.66832 before
     another refit wave. A newer recipe has not yet beaten the old full-corpus TL3 publicly.
9. **How to run.** `research/campaign_2026-10-29/teacher_track_r1/run_child.sh 3 d2 --distill
   tt29rep_fbc1k_s5 --distill-alpha 0.6` (scratch root `research_tmp/chrono_root`). Each run's
   log header must show the distill/alpha/init fields (CONFIG_OK check in `run_finish.sh`).
   Run the thermal guard alongside.
10. **Evidence to return.** Fit logs per epoch, fixed-endpoint grades, D1 prediction manifest
    and blend-law table, and the seed 4/5 paired deltas.
11. **Results and trace.** [Teacher track](../../campaign_2026-10-29/teacher_track_r1/README.md) ·
    [plan](../../campaign_2026-10-29/teacher_track_r1/PLAN_R1.json) ·
    [submissions and public scores](../../campaign_2026-10-29/submissions_pr49_r1/README.md) ·
    [stop receipt](../../campaign_2026-10-29/STOP_R1.json). Source branch
    `experiment/next-cycle-recovery-20261021` at `0ea63b982`. ZIPs and weights are git-ignored,
    held locally. No independent review of this PR.
