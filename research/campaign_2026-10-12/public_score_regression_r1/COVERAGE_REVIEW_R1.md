# Public-score data coverage review, R1

## Roster and outcome handling

The supplied portal view contains 42 of 45 reported entries: 38 `SCORE`, 3
`TIMEOUT`, and 1 `CHECK_ERROR`. The three unshown entries are unknown. The raw
roster is retained verbatim in `raw_portal_roster.tsv`; displayed timestamps
are not assigned a timezone or interpreted as score-availability times. The
scored 61:22 row remains a score. Timeout and check-error scores are missing,
not zero or imputed. Status, public score, runtime, and memory are outcomes and
are excluded from same-upload predictor inputs.

The generated `endpoint_coverage_r1.csv` maps every visible ID to its portal
filename, available package alias/path/hash, member specification and weights,
precision evidence, fitting/exposure statement, local metric, lineage cluster,
and relative evidence paths. `SOURCE_INPUTS_R1.json` binds the source documents
and current package receipts by digest. Missing source facts are left blank or
qualified; no artifact identity is inferred from a filename alone. The
`10tq248b` portal filename remains `submission.zip`, with the separate source
alias `c20261004_combo_s3_i16` recorded.

## Local-feature coverage

Eleven scored rows have some local metric, but none is a verified exact
public-serving/out-of-sample pair suitable for treating all protocols as one
feature:

| Rows | Count | Evidence and limitation |
|---|---:|---|
| 2026-10-11 fold1 student and fixed-pair endpoints | 3 | Producer/build receipts bind weights and float research predictions on 457 exposed/selected fold1 groups. The delivered public artifacts use Q8VNNI; the local values are pre-native-quantization, not exact served Q8 grades. The pair reuses both singles, so these are one dependent endpoint family. |
| 2026-10-10 fold1 package metric | 1 | Upload manifest reports a float fold1 metric and fold1 exclusion, but the consulted manifest does not bind a checkpoint digest or establish precision conversion. |
| Historical 368-group calibration metrics | 5 | The calibration script asserts reserved-group exclusion, but teacher/initializer/checkpoint ancestry is not receipt-verified in this audit. These are source-reported retrospective values under the historical v2-CPU calibration protocol; equivalence to public package precision is unestablished. Four rows share the S4/baseline family; one separate since/EMA family shares dependencies. |
| Historical EMA6 rows | 2 | Conditional v2-CPU reconstruction only: the two EMA6 packages collapse to identical `member0` under that reconstruction, while the user-reported public scores differ. They are proxies for a membership scenario, not exact served-member features, and share the since/EMA dependency group. |

Twenty-seven scored rows have no comparable local metric. Three non-score rows
have a local value only as a diagnostic: two fold1 package metrics for timed-out
targets and one fold1 sibling proxy for the all-data H388 teacher-trio timeout.
That H388 local value is not the target's score: the target's all-data weights
differ from the fold1 sibling and all fitting groups were used. The visible
score set therefore contains four local-evidence strata above, not eleven
interchangeable local validations. The summary JSON preserves these counts.

## Validation and modeling limits

All available local-score rows were visible during prior selection/calibration
or have unresolved precision/member equivalence. The fold1 values are exposed,
not a new holdout. Repartitioning the stored predictions would not make these
weights or scores fresh. The 368-group exclusion statement is a source
assertion, not independent ancestry proof. Public timestamps do not prove when
each score became available, so any chronological replay is retrospective.

Package identity is better documented for recent uploads than for older rows.
Some older rows lack a ZIP digest or complete member receipt in the consulted
sources, and the three portal rows not shown cannot be mapped. These omissions
remain explicit in the CSV and cannot be repaired by modeling. Any next
regression must keep protocol strata separate, group reused-weight families,
fit preprocessing only inside each training split, and compare a fixed model
against same-row mean and recent-history baselines. With only a few dependent
local-feature groups, such estimates are diagnostics; they cannot support a
reliable prospective numerical forecast or a claim that a candidate will reach
0.67/0.70.
