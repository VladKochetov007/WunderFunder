# First-cell score admission V10

V10 corrected V9’s receipt-branch durability gap: it retained the pre-submit directory fsync and fsynced `SCORE_D` after both the `LAUNCH.json` and `LAUNCH_UNRESOLVED.json` create-only writes. [The focused proof](ACTUAL_SCORE_ADMISSION_V10_FSYNC_BRANCHES_PROOF.json) records the exact inverse to V9 and unchanged six scorer payload pins.

The independent [V10 source review](ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V10.json) rejected the body before any B action. It found that source hashing could read a changed file to EOF before applying the aggregate byte limit. The [V10 source-hash negative](../first_cell_evaluation_score_admission_v11/ACTUAL_SCORE_ADMISSION_V10_SOURCE_HASH_NEGATIVE_V1.json) preserves that finding; V11 added manifest-size prevalidation and a bounded read loop. No V10 unit was submitted.

Candidate artifacts: [REMOTE_SUBMISSION_V10.py](REMOTE_SUBMISSION_V10.py) (187,989 bytes; SHA-256 `24787cf037328baabebe19d4eb6ba5c452740d2b5648dbf478544e3ce33cbd53`) and [SCORE_ADMISSION_SOURCE_BINDING_V10.json](SCORE_ADMISSION_SOURCE_BINDING_V10.json) (5,602 bytes; SHA-256 `9767f437d545ee9a6b448a6606be972ea6a7766ea3efebf24cbbb346df56a42e`).

See the [source and recovery index](../FIRST_CELL_SCORE_SOURCE_AND_RECOVERY_INDEX_20261021.md) for the full V9–V12 path and current status.
