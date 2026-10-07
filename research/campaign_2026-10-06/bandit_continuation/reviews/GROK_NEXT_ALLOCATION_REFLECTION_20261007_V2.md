# Two-hour live allocation reflection

Started UTC: 2026-10-07T15:47:04+00:00

This is a read-only allocation reflection. I am using the evidence-review and Challenge 2 skills, and I will only read live state, frontier, and the named receipts. No experiments, file edits, admission changes, or subagents.The live reconciliation is newer than the narrative below it. I am checking artifact timestamps and owners so queued or historical wording is not treated as a completed fit.The 18:04 Kyiv reconciliation is already behind newer receipts. I am reading the grade, export, E520 terminal, H230 seal, and the schedule that defines this window.The V9 terminal is an identity abort before any profile. I am checking whether the local GPU is idle and whether the H230 contract still treats the finished exports as running.The 15:04 UTC reconciliation in `specs/current-state.md` and `research/campaign_2026-10-06/bandit_continuation/FRONTIER.md` is behind the receipts. Public best remains **0.6660**. Nothing here is a submission or a public-score forecast. This review changes no allocation.

## 1. What landed since the two-hour timer was set

The reflection was armed at 13:54 UTC (`TREE-20261007-053`) for 18:47 Kyiv. Since then, on exposed fold 0:

The seed-4 scratch GRU384 fit finished at 14:33:53 UTC, exit 0, 5162.2 s, peak observed VRAM 3.994 GB, parent RAM 17.23 GB, scoped RAM recorded at the 6.5 GB cap, OOM delta 0. Receipt: `local_wide_teacher_E521/SEED4_FIT_TERMINAL_RESULT_V1.json`. Valid export v1 was refused on file-cache headroom before a worker ran. Valid export v2 finished at 14:38:51 UTC, exit 0, 101.1 s, VRAM 1.192 GB, parent RAM 12.72 GB (`runtime/attempts/e521-w384-s4-export-v2.json`).

Both fixed-epoch-3 train exports then finished, exit 0, about 556 s each, VRAM 1.192 GB, parent RAM 15.72 GB (seed 3) and 14.96 GB (seed 4), scoped peaks again on the 6.5 GB cap, OOM delta 0. Seed 3 ran 14:50–14:59 UTC; seed 4 ran 14:59–15:09 UTC. Producer receipts: `TRAIN_EXPORT_S3_TERMINAL_PRODUCER_RECEIPT_V1.json` and `TRAIN_EXPORT_S4_TERMINAL_PRODUCER_RECEIPT_V1.json`. `TRAIN_EXPORT_PAIR_SEQUENCE_RECEIPT_V3.json` matches the seed-4 completion heartbeat.

The independent equal-pair grade finished at 15:31:37 UTC, exit 0, 5.9 s, 1 CPU / 1.5 GB, no GPU (`E521_EQUAL_PAIR_GRADE_RECEIPT_V6.json`, report `E521_EQUAL_PAIR_EXCLUDED_FOLD0_REPORT_V6.md`). Official `global_wp_clip2_v1` on 468 groups, 141 components, 808,863 rows:

| Predictor | WP |
|---|---:|
| seed 3 | 0.672810 |
| seed 4 | 0.671273 |
| equal pair | 0.674093 |

Pair minus seed 3 is +0.001283, conditional 95% component interval [−0.001906, +0.004743]. Pair minus seed 4 is +0.002821, interval [−0.000093, +0.005912]. Those intervals condition on the two fitted predictors. They are not a training-seed variance estimate, and a zero crossing does not close the pair. There is still no matched GRU128 control, so this is not a width effect. Seed 3 reproduces the earlier grade. The point estimates are positive and imprecise on exposed data. Keep the pair as a teacher family.

H230 static contract V4 passed at 15:08 UTC with cache still blocked (`H230_INDEPENDENT_STATIC_EXECUTION_REVIEW_V4_20261007_R1.json`). It binds source bundle **V1**. The later actual seal, bundle **V2**, finished at 15:40:47 UTC: eight teachers, sixteen prediction exports, no cache, no fit, no score (`H230_ACTUAL_SOURCE_SEAL_V1.md`). The seal records `ft_since_s4` labeled seed 4 while its training provenance and checkpoint arguments record seed 3. Scoped peak was 1,499,996,160 bytes against a 1,500,000,000-byte cap; `ru_maxrss` was reported as 2,585,739,264 bytes. Both figures are preserved and unresolved.

E520 profile V9 is terminal `ERROR`. Input identity and hashes passed. The run aborted at `cuda_initialized_pre_profile_resource_gate` with `selected physical GPU UUID is missing or mismatched` and `gpu_selected_physical_gpu_uuid: null` (`cloud_preflight/downloads/e520_profile_v9/e520_profile_v9/resource_abort.json`). The poll note attributes a hardcoded V8 UUID on a different T4 (`e520_profile_terminal_v9_poll_receipt.json`). No 20-batch profile, fit, or score. Fit V5 replayed review targets only: `packet_created: false`, `provider_called: false` (`E520_FIT_V5_CONTROLLER_REVIEW_TARGET_REPLAY_RECEIPT_20261007_R1.json`).

E463 v3 was still `QUEUED` at the 15:50:25 UTC poll. That is not a fit. Kaggle quota then: GPU campaign headroom 45,612 s, reserved 0; TPU campaign headroom 7,159 s. Account refresh date is 2026-10-10. H217 remains a CPU long-lag correctness pass. H231 remains an untested event-time proposal under `ideation/tcn_teacher_recipe_grok_20261007/`.

## 2. Compute is idle locally and blocked in the cloud

At this read, the GPU showed 87 MiB and 0% utilization, and no `wnn-bandit-*` unit was running. `systemctl show wnn-campaign.slice` returned an unset memory current and an infinite max in this session, so this check does not confirm the live cgroup ceiling. The recorded cap stays 6 CPU / 20 GB RAM / 5 GB observed VRAM.

The productive path is blocked on two gates. Local H230 cache is not admitted. Cloud E520 never reached the trainer profile.

Narrow repair for E520: one new profile packet that selects the live session UUID inside the process and fails closed if that UUID is missing. Keep the V9 hash and membership path. Do not push fit V5, and do not repush `e520-profile-v9-memory-domains`. Refresh quota at the push. The last observed GPU headroom was about 12.7 hours.

## 3. Next bounded allocation

This is a recommendation only.

**H230, leading.** One independent review of source bundle V2, including the `ft_since_s4` seed-label versus provenance split, the 32-file worktree rehash, and the cgroup versus `ru_maxrss` discrepancy. Then one serial CPU cache admission: 13,372,800,000-byte payload, chunk 2, zero workers, 6.5 GB worker reserve, and at least 40,216,345,600 bytes free. That disk floor was not remeasured here. Profile the real B16 / chunk-2000 shape before either student fit. Contract V2 still says no cache, profile, fit, or grade is authorized (`H230_EXECUTION_CONTRACT_V2.json`). The student pair is GRU256×2, two epochs, fixed epoch 1, compared later with the audited E511 C1 pair on the same exposed fold. That comparison is not a family-head ablation.

**E520, faithful control.** The single UUID-repaired profile above, inside the existing recovery budget. The scratch GRU128 pair stays eligible. V9’s abort is execution evidence, not a negative recipe result. Fit only after that profile shows the pair inside the cap.

**Exploration, cheap and separate.** One CPU diagnostic below. Leave H217 and H231 unfitted. H232 in `H232_CAPACITY_EVENT_FEATURE_INTERACTION_V1.md` is a full GRU384 refit with a null prior; it waits behind the current teacher interaction. E463 stays polled and does not hold the GPU repair.

## 4. Cheap interaction worth running

Stratify the frozen E521 equal-pair residual by a causal event clock. Use the V6 prediction artifacts and a quote-change predicate from current and past inputs only, with no labels in the mask. On the same 808,863 scored rows, predeclare event-age bins and matched row-age bins. Report the frozen pair’s official WP in each bin, and the pair-minus-seed-3 difference in the stalest event bin. Fit no coefficient and no new model. One CPU, 1.5 GB, no GPU.

If event-age and row-age order the residual differently, the next spend is H231’s toy streaming parity (chunk parity, quiet-row holds, prefix invariance, reset). If the bins match, H231 does not earn a teacher fit from this pair. A zero-crossing interval does not decide that. The hardware-diversity critique’s per-target Gram parity for HD-1/HD-5 (`ideation/hardware_diversity_round_20261007/INDEPENDENT_SPEC_CRITIQUE_20261007.md`) is the matching cheap architecture check: synthetic T0/T1 weights that differ, chunked statistics versus a direct solve, still no parquet fit.

## 5. Stale recipe to archive out of active context

Retire the exact E520 profile V9 packet, kernel `vladkochetov07/e520-profile-v9-memory-domains`. It dies on a missing or mismatched physical UUID before any profile batch. Keep the terminal download, the abort JSON, and the poll receipt. The scratch GRU128 scientific recipe stays eligible under a new packet id. H230’s bundle V1 is the stale source map; admission has to use bundle V2 after review.

## 6. Incomplete end-to-end ownership

H230’s actor is `weak_diversity_kaggle_actor` and the reviewer is `tpu_recovery_delivery` (`CONDITIONAL_DELIVERY_ALLOCATION_V1.json`). The V4 pass predates the V2 seal and does not admit cache. Nobody yet owns a completed review of that seal, so the cache, profile, student fit, and grade have no admitted command. The execution contract’s E521 snapshot still says `running` as of 14:29 UTC.

E520’s operator owns the collected V9 failure. Fit V5 has no packet and no provider call. No successor profile with a live UUID is owned.

E521’s fits, exports, and descriptive grade are delivered. The matched-width control, the H230 student endpoint, and any serving package are not. E463’s poller owns a queue slot, not a diagnostic result.


Grok CLI exit status: 0
Completed UTC: 2026-10-07T15:53:35+00:00
