# Online relationship screen — October22 amendment

## Resumed engineering after the user pause

The user explicitly resumes experiments. The strict serial compiled stream passes 35 exactness tests and its [independent code/math review](../../unified/reviews/ONLINE_RELATIONSHIP_NUMBA_FP32_REVIEW_20261022_R1.md). The [new feature profile](FEATURE_PROFILE_COMPILED_RESULT_R1.json) projects 523.40 seconds for one pass and 1,286.24 seconds for three-epoch recomputation, compared with the preserved Python projection of 15,818.88 seconds. Its [execution/accounting receipt](FEATURE_PROFILE_COMPILED_EXECUTION_R1.json) records 9.53 wall seconds, 534,920 KiB peak RSS and zero swaps. Parent inference, complete fitting-only normalization, updates, disk I/O and grading are still omitted; zero scientific fits are admitted.

The [canonical readout fitter receipt](../../unified/reviews/ONLINE_RELATIONSHIP_READOUT_FIT_RECEIPT_20261022_R1.md) records three passing bounded CPU tests and the in-sample parent-cache API. Data/export/profile orchestration and an independent artifact grader are being completed. The earlier deferral and pause below remain historical records; full delivery must still fit 9,000 inclusive worker seconds. [Thermal resume](../../unified/operations/THERMAL_RESUME_20261022_R1.md) records the active CPU-only saved-model export and shared resource constraints.

Status: core/head/profile implemented and verified; entire12-fit panel deferred at the measured cost gate. No new fit admitted and no online quality result.
The user requested both backward lags after accepting the original six-fit plan: **12 new readout fits**, A/B/C × lags1/16 × F0/1001,F1/1002, zero GRU/teacher fits. The complete panel must fit the unchanged2.5-hour inclusive ceiling after the active warm-history panel finishes. Production keeps12hours.

- [Recipe](recipe.json) freezes controls, lag conventions, fitting/evaluation, counts and admission; the late user instruction expands the original panel explicitly.
- [Scientific rationale and progression](../../unified/ideas/ONLINE_RELATIONSHIPS_20261022_R1.md) records interpretation, exposure, in-sample parent residual limitation and negative-result scope.
- [Random source receipt](RANDOM_SOURCE_R1.json) preserves seed, exact passage, source hash and a bounded transfer interpretation; [literature note](../../unified/sources/ONLINE_RELATIONSHIP_SOURCES_20261022_R1.md) distinguishes external claims from this proposal.
- [PR description](PR.md) gives the reviewer contract. A publication URL will be added only after actual creation.
- [Campaign frontier](../../campaign_2026-10-21/next_cycle_r1/FRONTIER.md) owns current worker and allocation state. The original frozen cycle and dynamics receipts remain unchanged.

Actors: online_actor (features/head), online_delivery (streaming data/profile/fit), independent online_grader (coverage/grade/review), all native Luna/max. Root owns source freezing, resource admission and evidence integration. No native endpoint is yet implemented/validated for release, and no submission is authorized.

Runnable commands, exact source commits and verification receipts will be appended after integration. This draft is not a completed trainer or empirical result.

The canonical component sampler produced [F0/1001](fit_f0_10pct_s1001.json):315components/1,118train+118fit-valid sequences, and [F1/1002](fit_f1_10pct_s1002.json):314components/1,071train+172fit-valid sequences. Full evaluation remains468/457valid sequences. These manifests read split/index metadata only, not labels, and all arms/lags within a cell share exact membership. Recipe pins both verified checkpoint/provenance hashes and the unchanged canonical propensity ancestry.

[Implementation review progress](IMPLEMENTATION_REVIEW_PROGRESS_R1.json) preserves the pre-execution normalization correction, passing first-core checks and pending restoration/FP32 fixes. Core implementation is integrated; passing synthetic checks do not admit fits or establish throughput.

[First profile admission](FEATURE_PROFILE_ADMISSION_R1.json) and [failure receipt](FEATURE_PROFILE_FAILURE_R1.json) preserve an800MBsupport-cgroup OOM after9.183seconds, before timing output. No fit ran. Train+valid Parquet footer decoding is a source-based suspect, not an observed phase; no blind retry/highercap. A bounded fitting-valid-only feature profile is under investigation and cannot admit a complete training panel by itself.

[Independent core review](../../unified/reviews/receipts/ONLINE_RELATIONSHIP_CORE_REVIEW_R1.json) binds corrected sourceed09f19b,23passing tests and a separate scalaroracle across bothlags/allarms. It supports causal core/normalizer behavior only; fulltrainer, artifactgrader and nativeendpoint are not implemented by that source.

## Terminal cost decision

**Entire12-fit panel deferred; zero scientific fits.** The [successful R2 profile](FEATURE_PROFILE_RESULT_R2.json), [raw log](feature_profile_r2.log) and [execution/accounting receipt](EXECUTION_RESULT_R2.json) project **15,818.88 seconds (4.394h)** for a single feature-generation pass over fitting and evaluation sequences. This uses the fastest of8 complete fitting-valid sequences per fold and already exceeds the9,000second ceiling before parent inference, readout updates, disk I/O, export or grading. Three-epoch recomputation projects38,860.94seconds. This is an empirical cost estimate for the current one-CPU implementation, not a mathematical lower bound or a negative scientific result. No panel truncation or production-reserve borrowing.

R2 used92.16wallseconds,386,316KiBpeakRSS, zero swaps, exit0; observed cgrouppeak506,081,280bytes. Together with the9.183second R1 failure, charge101.343seconds conservatively to the online envelope. Historical dynamics550.672seconds stays charged; original prior-cycle history remains unchanged. [Exact profiled recipe](PROFILE_RECIPE_R2.json) preserves the bytes hashed in the profile; mutable recipe.json now records deferral.

Implemented: [causal feature and normalizer](../../../src/wnn/online_relationship.py), [residual head](../../../src/wnn/training/online_relationship.py), [bounded profile](../../../src/wnn/training/online_relationship_profile.py), and [CLI](../../../scripts/profile_online_relationship.py). Full fit/export orchestration, artifact grader and native-serving adapter are **not implemented**, because this implementation fails the prerequisite complete-panel cost gate. No release claim follows.

Reproduction (cost-only; use a new create-only output path, shared resource availability and the frozen recipe):

```bash
WNN_MEM_MAX=800000000 WNN_CPU_QUOTA=100% WNN_RUNTIME_MAX=300 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src scripts/limited .venv/bin/python scripts/profile_online_relationship.py --recipe research/experiment_drafts/online-relationships-20261022/PROFILE_RECIPE_R2.json --output /home/vlad/development/wunder/research_tmp/online-relationships-20261022/feature-profile-reproduction.json
```

The profile opens only valid.parquet, reads selected fitting groups with targets=False/mask=False, and streams full20,000-row sequences. Train-source metadata/read memory remains unmeasured here. Independent review and final verification/publication receipts are linked below when complete.

[Root verification](VERIFICATION_R1.json) and [raw test log](root_verification_r1.log): all23focused tests pass on integrated source, Ruffformat/check pass, canonicalstructure67files/zeroerrors,605004KiBpeakRSS/zeroSwap. The independent scalarreview is separate.

[Independent cost review](../../unified/reviews/receipts/ONLINE_RELATIONSHIP_PROFILE_DEFERRAL_REVIEW_R1.json) confirms exact source/fit-only sample membership and arithmetic; it supports cost deferral only. Subsequent userthermalwarning closes allheavyadmission and requests gracefulcurrentfitdrainthenpause; [live authority](../../campaign_2026-10-21/next_cycle_r1/THERMAL_GRACEFUL_PAUSE_R1.json). Draftpublication is prepared locally on experiment/online-relationships-20261022 but **not pushed/created as a PR** before thispause.
