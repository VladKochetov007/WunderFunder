# Final-quality execution: PR91 input panel and L4 since-weight studies

As-executed packet for the fresh final-quality wave. The evidence cutoff for the compact native status is 2026-11-12 18:47:28 UTC. Four L4 since32/zero fits had completed; four L4 since8/since128 fits and four B300 PR91 fits were active. A later read-only prediction-export controller was started for the four completed L4 models after the cutoff. The later L4 export launch is separate read-only prediction support. This packet is a local preparation record; it does not select or upload a model.

## 1. Status, lineage and owner

Experiment ID: E-FINAL-QUALITY-EXECUTION-20261112. Related hypotheses: H-FINAL-QUALITY-PR91-BILINEAR-20261112, H-FINAL-QUALITY-SINCE32-20261112, and H-FINAL-QUALITY-SINCE-SCALE-20261112. Status is partial execution with four completed scientific fits, eight active fits at the stated cutoff, and no final candidate selection.

B300 source is PR91 packet commit 190d04be8d722daca598db3efd97728c1c29b32d, archive SHA-256 68244c2501e83790cc225ea0b97a2a2f4661c9acd893a374ad10bfa12958579f. Its pinned canonical training runtime is 8626000a4b965763a70011a0291a891aa6535d7e. L4 source is sequence-GRU runtime commit 390c846cbdacd8e0e06e6079dd3b04312fa6bcc5, source identity 29fb45164667e69a71d4be2c696731d9824884d2d51674f37d2a633bd17d9511. The copy manifest lists exact local copies and hashes.

ROOT owns execution and result acceptance. Closed previous waves and their clocks remain closed. This wave uses the new fixed CLOCK_R1; no clock was reset.

## 2. Why these comparisons were run

The B300 input panel tests whether a bilinear input map improves the fixed raw-z model versus a linear input map at matched seeds and the same endpoint. The mechanism is useful only if the paired final-epoch scores improve without changing the rest of the training contract.

The L4 label-only panel tests causal age weighting at tau32 versus no age weighting, then tests tau8 and tau128 around tau32 with the same two seeds. A gain at tau32 is observed on two exposed validation evaluations; whether a neighboring scale improves remains unmeasured at this cutoff. The failed PR49 distillation proposal is preserved as rejected evidence and is not part of the current scientific fits.

## 3. Compute advantage and measured cost

Four B300 fits run concurrently on all eight owned B300s, two GPUs per fit. The concurrent 50-update profiles passed in 84.2-87.3 seconds with approximately 110 GiB reserved per GPU and exact within-profile model/optimizer/scheduler/RNG reload. The execution-only R2 allocation changes concurrency from the earlier serial dispatcher to four simultaneous cells; configs, seeds and fixed E4 endpoints are unchanged.

The L4 fits use one GPU each. Four new age-scale profiles completed 30 updates each; three took about 3.6 seconds and the GPU6 profile took about 6.85 seconds while a foreign process occupied roughly 18 GiB. No foreign process was signalled. A later read-only B300 resource-observer attachment succeeded on all four fit families; it only added observer journals. It did not change training or checkpoints. Earlier native L4 receipt 168 reported about 135.8 GB of physical free disk. Compact receipt 174 does not measure disk space. The L4 controller stops below 300,000,000 free bytes; the B300 controller has a separate 128 GiB floor.

## 4. Frozen comparisons

Primary split SHA-256 is 0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c: 10,325 official-train sequences for fitting, 282 linked train sequences excluded, and 1,873 official-valid sequences for evaluation. The official valid has 3,528,273 scored rows.

B300 cells are be_input_linear_s206201, be_input_bilinear_s206201, be_input_linear_s206202, and be_input_bilinear_s206202. The endpoint is fixed E4; monitoring does not select checkpoints. Precision is the PR91 bf16_bf16x9 amendment with FP32 master weights and optimizer state; no strict IEEE-FP32 parity is claimed.

The L4 base cells are label_since32_k16_seed203401/203402 and zero_since_control_k16_seed203401/203402. They use the fixed six-epoch endpoint, final epoch index 5. The follow-up cells are age8_k16_seed203401/203402 and age128_k16_seed203401/203402. The age manifest binds each follow-up to a same-seed since32 configuration and declares since_weight as its only changed field. No precision or optimizer recipe change was made to L4 cells.

## 5. Bounded fitting panel

| Panel | Arms and matched seeds | New scientific fits | State at 18:47:28 UTC |
|---|---|---:|---|
| B300 input map | linear vs bilinear, seeds 206201 and 206202 | 4 | 4 active; E2 last completed full-valid evaluation |
| L4 age weighting | zero vs tau32, seeds 203401 and 203402 | 4 | 4 completed at fixed E6 |
| L4 age scale | tau8 and tau128, seeds 203401 and 203402 | 4 | 4 active; tau32 endpoints reused from the preceding panel |

Total newly admitted scientific fits: 12. There are no scientific PR49 KD fits. The since32+KD configs failed the source config guard. KD configs corrected to since_weight zero then failed a separate profile/admission monitor guard; both failures and all frozen configs are retained. Profiles and failed guards are engineering support, not scientific fits. A later four-model L4 export run is prediction support and adds no fit.

## 6. Validation and uncertainty

All quality scores use global_wp_clip2_v1 with the official scored-row mask, all 1,873 valid sequences and 3,528,273 rows. The valid partition is repeatedly exposed development evidence, not an independent holdout. Checkpoint selection is fixed to E4 or E6, and within-fit valid monitoring is report-only.

The B300 epoch-2 observations were linear 0.676833889 and bilinear 0.677728864 at seed 206201 (paired delta +0.000894975), and linear 0.677090715 and bilinear 0.677299168 at seed 206202 (delta +0.000208453). These are interim values, not endpoint promotion evidence.

The completed L4 E6 values were tau32 0.685478984 vs zero 0.683253088 at seed 203401 (delta +0.002225896), and tau32 0.685397405 vs zero 0.681814860 at seed 203402 (delta +0.003582545). These two seed-paired differences are descriptive; they do not establish significance or training-seed robustness. The tau8/tau128 fits had not reached validation at the cutoff.

## 7. Leakage and coverage

All fitting, normalization and checkpoint ancestry is tied to the 10,325 fit sequences. Validation was used only for full-partition evaluation and repeated monitor feedback; it is now exposed. Sequence membership is bound by the frozen chronological split manifest; seq_ix is not used to infer chronology. No row-random split or validation-based fitting was introduced.

The rejected KD route has no scientific fit or accepted teacher-cache contribution in this wave. After candidate choice, the explicitly requested final-fit stage is expected to use all official labels: 10,607 train plus 1,873 valid sequences (12,480 total). That stage is pending and is not part of these fits. Its resulting weights cannot be evaluated as an independent validation result on those same labels; preserve the pre-final validation-stage scores as exposed development evidence.

## 8. Decision-changing result

The wave will support final candidate ranking only after all admitted fits reach their fixed endpoints, the independent prediction exports and official scoring are accepted, and paired seed results are reconciled. Compare each seed and the equal-seed descriptive mean; retain every failure and do not infer significance from two seeds. No numeric improvement threshold or leaderboard gain is declared by this execution packet.

If the tau8/tau128 panel is inconclusive, it narrows only these four settings. Any new KD or other successor requires a separate source-qualified panel and dated allocation amendment. Candidate selection on this exposed valid is development selection; it does not create a fresh holdout.

## 9. How to reproduce or continue

Do not restart or resume the active fits from this note. The exact B300 configs, receiver source and runtime are preserved in sources/pr91/pr91-packet-r1.tar.gz; fit allocation and ownership logic is in controllers/b300_fit_wave_r2.py. The L4 panels and age configs are copied under sources/l4; controllers include l4_wave_r2.py, prepare_age_panel_r1.py, native_age_preflight_start_r1.py, and native_age_admit_fit_r2.py. The original wave clock and amendments are copied under inputs.

B300 output root: /data/epanov_s/tabdata-final-quality-20261112-r1/runs/pr91-input. L4 output root: /data/lc-epanov_s/tabdata-final-quality-20261112-r1/runs. After the 18:47:28Z fit snapshot, controllers/l4_export_completed_r1.py launched four completed-model prediction exports on L4 GPUs 0-3. The 18:52:36Z operator status reported about 295/1873 sequence exports per model; scoring remained pending. This adds no scientific fit. These copies are documentation/provenance, not new launch commands; ROOT owns any continuation and stop decision.

## 10. Evidence to return

Exact source archives, configs, manifests, controllers, clock and amendment files are listed with byte counts and SHA-256 in COPY_MANIFEST_R1.json. The compact 174 native status receipts are copied under evidence; large 168/171 source captures remain at their original artifact paths and are referenced only by SHA-256 and byte count in evidence/TELEMETRY_SUMMARY_R1.json. The exact PR91 progress post and GitHub API receipt are preserved.

The live read-only dashboard and epoch plots remain in the separate local F3 artifact tree. Four fitted B300 resource-observer journals were attached read-only; observer implementation is included. Model weights, raw data, teacher payloads and large status maps are not copied into this packet. Final prediction, score and weights archive hashes remain pending.

## 11. Results and trace

At the cutoff, four L4 fixed-E6 fits were complete and eight fits were active. B300 had positive interim E2 deltas on both matched seeds; final E4 remains pending. L4 tau32 exceeded its matched zero controls on both seeds at E6, while the new age-scale comparison remains in progress. No model is selected and no portal upload is recorded in this packet.

The as-posted PR91 progress update is https://github.com/VladKochetov007/WunderFunder/pull/91#issuecomment-6832684074. Run roots and original receipts are identified in the evidence files and copy manifest. The reviewer should update this record with accepted terminal/score receipts and source/checkpoint hashes after the remaining fits and exports finish. PR94 final-package work is outside this packet and unchanged.
