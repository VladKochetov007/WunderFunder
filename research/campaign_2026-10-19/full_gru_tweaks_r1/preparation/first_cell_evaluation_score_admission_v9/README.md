# First-cell score admission V9

V9 is a preserved source-revision step, not a usable admission body. It added a `SCORE_D` directory fsync after `submit_once` returned, before the launch receipt was reread. The [focused inverse proof](ACTUAL_SCORE_ADMISSION_V9_FSYNC_INSERT_PROOF.json) records the single inserted block, byte/AST restoration to V8, and unchanged six scorer payload pins.

The next review found that `submit_once` itself writes `LAUNCH.json` or `LAUNCH_UNRESOLVED.json` on failed or unresolved outcomes and can return or raise before the V9 main-level fsync. That finding is preserved in [the V9 unknown-branch negative](../first_cell_evaluation_score_admission_v10/ACTUAL_SCORE_ADMISSION_V9_UNKNOWN_BRANCH_NEGATIVE_V1.json). V10 moved durability handling to both receipt-write branches.

[The earlier V8 negative](ACTUAL_SCORE_ADMISSION_V8_DURABILITY_NEGATIVE_V1.json) explains the original successful-path durability issue. The V9 source and binding are [REMOTE_SUBMISSION_V9.py](REMOTE_SUBMISSION_V9.py) (187,751 bytes; SHA-256 `039d52fc297ddfe7b537baab905dbb23933df780f2a815d0ed48aa862c7bd662`) and [SCORE_ADMISSION_SOURCE_BINDING_V9.json](SCORE_ADMISSION_SOURCE_BINDING_V9.json) (5,516 bytes; SHA-256 `2798dc1cee7279425fc4d69c9ef5adcedd8736f3a619392abe40fdab4d448e68`). V9 was not submitted; no score unit or model/payload action occurred for this revision.

See the [source and recovery index](../FIRST_CELL_SCORE_SOURCE_AND_RECOVERY_INDEX_20261021.md) for the full V9–V12 path and current status.
