# Team model comparison — October 13

Root owns the user-authorized complete comparison before new ZIP preparation. [ADR0035](../../../specs/adr/0035-official-team-code-and-weight-sharing.md) permits direct team code/weight reuse; all training and evaluation ancestry remains explicit.

## Artifacts

- `CONTRACT.json` fixes the compared original endpoints, official metric, diagnostic scope, resource bounds and promotion limits before scoring.
- `intake.py` creates the bounded source/weight inventory and preserves exact archives in ignored physical storage. `INVENTORY.json` identifies duplicate tensors, architectures, runtime libraries and the five supplied ZIPs plus own C/E.
- `score_archive.py` replays the exact package callback, streams one complete sequence at a time, exports disk-backed FP32 predictions and pooled/per-sequence official moments. It is diagnostic scoring, not training or weight search.
- `public_results.json` records user-transcribed portal results, retaining the pending E upload as a separate event. Archive-name association is not portal-verified hash identity.
- `profiles/` and `scores/` retain exact-command callback checks and evaluated terminal receipts as they land; large predictions remain ignored.
- `team_model_comparison.py` is the live marimo notebook. Its cells read saved evidence and never launch inference or fitting.

Current stage: eight short profiles complete; four complete original-callback replays are running, four further endpoints queued. Root owns terminal collection and comparative analysis. No joint clean-CV result or new ZIP is claimed. The remote source search still sees teammate research commit833aaf54 (October6), without the later fttall training code or clean CV counterparts.

## Interpretation and operational evidence

- [Paired runtime benchmark](../../../scripts/benchmark_team_runtimes.py) implements repeated same-weight C++/Rust A-B-B-A callbacks with actual backend/ELF identity, reset reproducibility and separate wall/CPU clocks. Four focused tests pass; actual controlled timing is pending. It requires `.venv-py311-profile/bin/python` for the supplied CPython3.11 C++ extension. The builder's earlier `.venv/bin/python` sample command is invalid on this machine's Python3.13 project environment.

- [Individual member bank](member_bank/README.md), [fixed plan](MEMBER_BANK_PLAN_R1.json), [review and actual profiles](GPU_MEMBER_REVIEW_R1.md), and [admission](MEMBER_BANK_ADMISSION_R1.json) deliver fourteen deduplicated FP32 diagnostic exports serially on the local GPU. Their quantized-original comparisons and fixed-mixture grades remain distinct from clean CV.
- File-cache-only handoffs [R2](CACHE_HANDOFF_R2.json), [R3](CACHE_HANDOFF_R3.json) and [R4](CACHE_HANDOFF_R4.json) restore measured admission margin under the existing GPU lock without changing limits or terminating healthy jobs. [B1 admission](GPU_PROFILE_ADMISSION_R1.json) records CPU redistribution within five cores, and [B16 admission](GPU_PROFILE_ADMISSION_R2.json) uses the measured smaller profile.

- [EPYC runtime reflection](EPYC_RUNTIME_REFLECTION_R1.md) separates actual portal observations, local Intel profiles and target-specific hypotheses. [Independent binary audit](INDEPENDENT_RUNTIME_REVIEW_R1.md) confirms both engines contain VNNI paths without establishing an EPYC winner.
- [Teammate methods and evidence](TEAM_METHODS_AND_EVIDENCE_R1.md) pins the accessible branch and distinguishes source-report findings from current missing CV/FTT lineage.
- [Independent source and metric review](INDEPENDENT_REVIEW_R1.md) reproduces actual group hashes and official short-profile scores, and records the full-file identity correction.
- [Profile index](profiles/README.md) and [full replay index](scores/README.md) explain every original/conversion receipt and its diagnostic scope.
- [First intake incident](INTAKE_FAILURE_R1.json) preserves the cross-subvolume hardlink failure and bounded copy correction. [Cache handoff](CACHE_HANDOFF_R1.json) records file-cache-only reclamation with limits unchanged.
- [Full replay admission](FULL_REPLAY_ADMISSION_R1.json) records actual RAM/profile/source bounds; [controller](run_replays.py) owns the complete queue and [controller log](REPLAY_CONTROLLER.log) records starts/ends. Mutable progress is operational evidence, not a completed grade.
- [Preserved profile source](score_archive_profile_r1.py) binds the original short profiles; [full successor](score_archive.py) additionally verifies the entire validation-file SHA before inference. Both remain frozen while consumed.
