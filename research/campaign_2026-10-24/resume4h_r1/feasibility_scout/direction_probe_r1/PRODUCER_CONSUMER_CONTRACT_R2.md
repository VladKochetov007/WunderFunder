# Direction-probe producer and consumer contract R2

The failed R1 invocation stopped before the first input preflight because the
runner looked up lowercase `inputs.f0` and `inputs.f1`. The frozen recipe uses
`inputs.F0` and `inputs.F1`. R2 changes this lookup, binds every input path/hash
pair to the pinned source map, and adds a preflight-only CLI path that prints
evidence and returns before `run_probe`.

## Recipe fields and preflight calls

| Recipe field | Actual producer call | Frozen producer output consumed by runner |
|---|---|---|
| `inputs.F0.base_recipe_path` and `base_recipe_sha256` | `prepare_f0_grade(Path, expected_sha256)` | `PreparedF0Grade.parent_prediction_path`, `candidate_prediction_paths`, `evaluation_groups`, `components`, `target_cache_path`, and `report` |
| `inputs.F1.fixed_shrink_recipe_path` and `fixed_shrink_recipe_sha256` | `prepare_f1_grade(Path, expected_recipe_sha256)` | `PreparedF1Grade.parent_prediction_path`, singular `candidate_prediction_path`, `evaluation_groups`, `components`, `target_cache_path`, and `report` |
| `inputs.F0.preflight_path` and `preflight_sha256` | Compare exact on-disk JSON with `PreparedF0Grade.report` | F0 report has top-level `evaluation_membership_sha256` |
| `inputs.F1.preflight_path` and `preflight_sha256` | Compare exact on-disk JSON with `PreparedF1Grade.report` | F1 report has `evaluation.row_refs_sha256` |
| `inputs.F0.fixed_shrink_recipe_path` | Read the frozen shrink recipe after its source-map SHA check | `candidate_predictions[name].prediction_path` and `prediction_sha256`; alias identity in `omitted_duplicate.prediction_sha256` |
| `inputs.F0.parent_prediction` | Compare recipe path with `PreparedF0Grade.parent_prediction_path` | F0 parent array is the saved evaluation prediction endpoint |
| `inputs.F1.parent_prediction` and `candidate_prediction` | Compare paths with the corresponding PreparedF1Grade fields | F1 parent and A-lag1 prediction endpoints |
| `inputs.F0.base_grade_path`, `fixed_shrink_grade_path`, and `inputs.F1.fixed_shrink_grade_path` | Read pinned grade JSON | F0 parent under `endpoint_scores.parent.overall`; F0 raw/shrunk endpoints under `baseline_raw_scores` / `shrunken_scores`; F1 endpoints under `endpoints.parent`, `raw_A_lag1`, and `shrink_A_lag1` |

F0's candidate path mapping is plural and keyed by `A_lag1`, `B_lag1`,
`C_lag1`, `B_lag16`, and `C_lag16`; A-lag16 is the frozen payload alias. F1's
candidate path is singular. This matches the actual Prepared dataclasses and
grade JSON schemas. The runner's score path consumes only the fields listed
above and continues to use official metric accumulators and the existing target
materializer.

Before either reused preflight is called, R2 requires the F0/F1 recipe, grade,
source-manifest, and saved-preflight path/hash pairs to agree with the top-level
source map. It also requires the `input_recipe_sha256` summary to equal the
three recipe digests bound in those input records. Large prediction payloads
remain pinned separately by `prediction_sha256` and are checked against their
source metadata and NPY headers.

R2 preflight-only mode invokes both existing source-qualified preflights,
rechecks prediction file hashes and NPY headers, and verifies the target
archive identity. It reports `prediction_values_read=false`,
`target_values_read=false`, and `score_function_called=false`; it does not call
the scoring path.
