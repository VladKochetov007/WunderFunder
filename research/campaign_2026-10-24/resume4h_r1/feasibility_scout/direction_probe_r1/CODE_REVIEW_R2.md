---
kind: review
status: complete
created_utc: 2026-10-24T13:29:04Z
updated_utc: 2026-10-24T13:29:04Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_RECIPE_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_SOURCE_MANIFEST_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/run_direction_probe.py
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_PREFLIGHT_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/EXECUTION_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/EXECUTION_R1.log
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/PRODUCER_CONSUMER_CONTRACT_R2.md
---

# Independent runner contract review — direction probe R2

## Disposition

**The R2 label-free input-preparation path passes source and schema review.** This is not a review of a completed score pass. I did not execute code or open target/prediction values; I independently checked the source, recipe, manifest, fixture receipt and their hashes. The actor's recorded R2 fixture exercised the actual preflight functions against the saved F0/F1 artifacts and stopped before `run_probe` and scoring.

## R1 correction

R1 failed before either source preflight: its runner indexed lowercase `inputs.f0` and `inputs.f1`, while the frozen recipe has uppercase `inputs.F0` and `inputs.F1`. The preserved invocation receipt reports a 0.219-second failure and the log gives `KeyError: 'f0'`. The earlier [R1 review](CODE_REVIEW_R1.md) said no code blocker was found, but it had not exercised or traced `prepare_inputs` against the actual recipe key schema. Its import-path and synthetic-math checks did not establish runner startup; that conclusion was too broad. The R1 recipe, runner, admission, traceback and execution receipt remain preserved unchanged.

## R2 identity and checks

Hashes derived from the current file bytes:

- Recipe: `a275cdd402aa8e22e181922dd3de0e4f1dee06a5a58fe579709c7f557fdc9a29`
- Source manifest: `7265890a44ebbcdd46ac8030a3050ad0fb2a90d2642020db4f514077554859a1`
- Runner: `f303dc5ff359b97d2257ffd2c82ee1f0783604b3cdcbf325c9500348a1027447`
- Label-free preflight receipt: `81df5e2689ef4c5def60aa34bdd545ac28e63d34f10aa94abe6ff4b466b24fc0`

The recipe's source map equals the manifest map, and all 33 pinned source files match their listed hashes. The new input-binding check ties every recipe path/digest pair for the base grades, recipes, source manifests and saved preflights to that map; the compact input-recipe digest record also matches the three F0/F1 recipe pins. The corrected lookup feeds the actual producers: F0 returns plural candidate paths and a flat `evaluation_membership_sha256`; F1 returns a singular candidate path and nested `report.evaluation.row_refs_sha256`. Both complete reports must match the saved preflight JSON exactly. The runner's group/component counts and membership checks match the producer schemas.

The R2 receipt records 468 F0 groups / 141 components and 457 F1 groups / 149 components with the expected source-qualified membership hashes. It verifies the six F0 files (parent plus five unique endpoints) and two F1 files against their frozen SHA-256 values and NPY shape/dtype headers. The scoring code uses the existing `GlobalAccumulator`, official mask (`mask AND row >= 99`), parent/candidate grade keys and the frozen float32 alpha-0.1 operation. Static inspection found those accesses consistent with the pinned grade and metric schemas.

The fixture reports `prediction_values_read: false`, `target_values_read: false`, `score_function_called: false`, and `run_probe_called: false`; it verifies only source-qualified metadata, file hashes/headers and target archive identity/size. R1 remains the only attempted scoring invocation and failed before these checks. No scientific score, derivative, or endpoint-quality conclusion follows from this review.

## Scope limit and next gate

No blocker remains in the corrected producer/consumer input-preparation path. This review does not establish the admitted scoring path's behavior, memory/time, score reproduction, moment output, or any scientific result. Root must issue/bind the R2 admission before any label-reading score pass; the label-free fixture is not that authorization.
