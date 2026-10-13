# Team model comparison — October 13

Root owns the complete model comparison. The user's subsequent [exploratory release instruction](EXPLORATORY_SUBMISSION_AUTHORITY_R1.json) permits one new cross-team ZIP now, even without a local-best claim; comparison jobs continue. [ADR0035](../../../specs/adr/0035-official-team-code-and-weight-sharing.md) permits direct team code/weight reuse; all training and evaluation ancestry remains explicit.

## Artifacts

- [Exploratory package and check receipts](../../../submissions/10-13-team-r1/README.md) bind the new four-member seed6 ZIP to the intended recipe and source archives. The16.32MB artifact is ready in repository artifact storage after independent byte/recipe audit, ELF ABI, isolated Python3.11 import and full-sequence reset checks on V2/V3. Runtime files exactly match our optimized E submission; no quality gain is claimed.
- [Exploratory submission authority and recipe](EXPLORATORY_SUBMISSION_AUTHORITY_R1.json) fixes an equal four-member blend of teammate distilled/FTT seed6 and our Z4/H390 seed3. It changes the release timing for this public probe, not the interpretation of diagnostic validation scores. The [builder recipe](SUBMISSION_RECIPE_R1.json) contains the exact imported member order and coefficients.
- [Exploratory release rationale](EXPLORATORY_SUBMISSION_R1.md) records E-TEAM-XTEAM-S6-RELEASE-20261013-R1, the seed substitution, reported public references, missing CV ancestry and exact-artifact checks. This is a user-requested public probe, not a locally selected champion.
- [Member-bank grader](../../../scripts/grade_team_member_bank.py) validates actual terminal exporter metadata and hashes, then streams the official metric for fourteen members and nine predeclared mixtures, plus prediction/residual correlations. All fourteen exports and the [admitted grade](MEMBER_GRADE_ADMISSION_R1.json) completed, with [terminal progress log](MEMBER_GRADE_R1.log). Six focused metric/header tests pass after correcting pre-read NPY bounds and excluding atime from file-version checks. Root owns terminal collection.
- `CONTRACT.json` fixes the compared original endpoints, official metric, diagnostic scope, resource bounds and promotion limits before scoring.
- `intake.py` creates the bounded source/weight inventory and preserves exact archives in ignored physical storage. `INVENTORY.json` identifies duplicate tensors, architectures, runtime libraries and the five supplied ZIPs plus own C/E.
- `score_archive.py` replays the exact package callback, streams one complete sequence at a time, exports disk-backed FP32 predictions and pooled/per-sequence official moments. It is diagnostic scoring, not training or weight search.
- `public_results.json` records user-transcribed portal results, retaining the pending E upload as a separate event. Archive-name association is not portal-verified hash identity.
- `profiles/` and `scores/` retain exact-command callback checks and evaluated terminal receipts as they land; large predictions remain ignored.
- `team_model_comparison.py` is the live marimo notebook. Its cells read saved evidence and never launch inference or fitting.

Current stage: eight short profiles and all fourteen member exports are complete; [full FP32 diagnostic grading](scores/member_bank_fp32_r1/README.md) also completed. Four original native callbacks completed; four additional original/conversion callbacks continue under the admitted controller. The user's exploratory four-member ZIP is checked and ready; no joint clean-CV result is claimed. The remote source search still sees teammate research commit833aaf54 (October6), without the later fttall training code or clean CV counterparts.

## Interpretation and operational evidence

- [Navigation refresh receipt](NAVIGATION_CHECK_R1.json) records a successful generated-tree build followed by a stale-catalog check while native replay logs were still changing. Unified links are updated directly; this snapshot freshness issue does not invalidate package or metric receipts.

- [Complete FP32 diagnostics](scores/member_bank_fp32_r1/README.md) score fourteen members and nine predeclared mixtures on all1,873 groups. Original xteam_h4 exceeds E and pair_ftt2 in this final-weight comparison; own H390 seed3 alone is stronger still. These are fitted/unknown-overlap observations, not clean-CV selection or public forecasts. The new seed6 ZIP is not one of the original nine mixtures and has no score in this table.

- [Paired runtime benchmark](../../../scripts/benchmark_team_runtimes.py) implements repeated same-weight C++/Rust A-B-B-A callbacks with actual backend/ELF identity, reset reproducibility and separate wall/CPU clocks. Four focused tests pass; actual controlled timing is pending. It requires `.venv-py311-profile/bin/python` for the supplied CPython3.11 C++ extension. The builder's earlier `.venv/bin/python` sample command is invalid on this machine's Python3.13 project environment.

- [Individual member bank](member_bank/README.md), [fixed plan](MEMBER_BANK_PLAN_R1.json), [review and actual profiles](GPU_MEMBER_REVIEW_R1.md), and [admission](MEMBER_BANK_ADMISSION_R1.json) deliver fourteen deduplicated FP32 diagnostic exports serially on the local GPU. Their quantized-original comparisons and fixed-mixture grades remain distinct from clean CV.
- File-cache-only handoffs [R2](CACHE_HANDOFF_R2.json), [R3](CACHE_HANDOFF_R3.json) and [R4](CACHE_HANDOFF_R4.json) restore measured admission margin under the existing GPU lock without changing limits or terminating healthy jobs. [B1 admission](GPU_PROFILE_ADMISSION_R1.json) records CPU redistribution within five cores, and [B16 admission](GPU_PROFILE_ADMISSION_R2.json) uses the measured smaller profile.

- [EPYC runtime reflection](EPYC_RUNTIME_REFLECTION_R1.md) separates actual portal observations, local Intel profiles and target-specific hypotheses. [Independent binary audit](INDEPENDENT_RUNTIME_REVIEW_R1.md) confirms both engines contain VNNI paths without establishing an EPYC winner.
- [Logical Q8 matrix accounting](LOGICAL_Q8_WEIGHT_TRAFFIC_R1.json) derives active unpadded matrix payload from actual member shapes: a regular H256×2 is near the1 MiB L2 boundary before metadata; merged heads exceed it. This is not a measured cache footprint or a C++ runtime claim. It motivates a matched H192/H224/H256 student compression ladder.
- [Teammate methods and evidence](TEAM_METHODS_AND_EVIDENCE_R1.md) pins the accessible branch and distinguishes source-report findings from current missing CV/FTT lineage.
- [Independent source and metric review](INDEPENDENT_REVIEW_R1.md) reproduces actual group hashes and official short-profile scores, and records the full-file identity correction.
- [Profile index](profiles/README.md) and [full replay index](scores/README.md) explain every original/conversion receipt and its diagnostic scope.
- [First intake incident](INTAKE_FAILURE_R1.json) preserves the cross-subvolume hardlink failure and bounded copy correction. [Cache handoff](CACHE_HANDOFF_R1.json) records file-cache-only reclamation with limits unchanged.
- [Full replay admission](FULL_REPLAY_ADMISSION_R1.json) records actual RAM/profile/source bounds; [controller](run_replays.py) owns the complete queue and [controller log](REPLAY_CONTROLLER.log) records starts/ends. Mutable progress is operational evidence, not a completed grade.
- [Preserved profile source](score_archive_profile_r1.py) binds the original short profiles; [full successor](score_archive.py) additionally verifies the entire validation-file SHA before inference. Both remain frozen while consumed.

## Versioned runtime evidence

[Immutable replay snapshot](runtime_snapshot_r1/README.md) preserves all current logs and sequence progress. [Complete original scores](scores/README.md) now include C, E, pair_ftt2 and xteam_h4. Live output remains on physical disk and is ignored by Git; root must collect pending terminal receipts after completion. Worker/controller sources and healthy jobs were not changed. This separates live staging from committed evidence and keeps the working tree clean.
