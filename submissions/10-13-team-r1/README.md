# New four-model cross-team submission

The user explicitly requests this exploratory public probe even without a local-best result. [Research rationale](../../research/campaign_2026-10-13/team_models_r1/EXPLORATORY_SUBMISSION_R1.md), [authority](../../research/campaign_2026-10-13/team_models_r1/EXPLORATORY_SUBMISSION_AUTHORITY_R1.json), and [frozen recipe](../../research/campaign_2026-10-13/team_models_r1/SUBMISSION_RECIPE_R1.json) identify E-TEAM-XTEAM-S6-RELEASE-20261013-R1.

The package is `kotiki_xteam_s6_z4_h390s3_q8_20261013.zip`, 16,320,815 bytes. SHA256: `4315eaa9bfe985fdaaca4258b3e8da3c4861264115f048bd8419b1cd6375f21a`.

**Publicly scored at 0.6679:** the ZIP is stored in the repository at `artifacts/team_models_20261013/kotiki_xteam_s6_z4_h390s3_q8_20261013/kotiki_xteam_s6_z4_h390s3_q8_20261013.zip`. [Current location receipt](RELEASE_LOCATION_R2.json) records its verified hash and repository-relative path. [Validation release receipt](RELEASE_R1.json) binds the fixed recipe, all checks and byte-identical optimized runtime; its earlier delivery paths are historical. No external upload was made.

| Weight | Model | Coefficient |
|---|---|---:|
| weights_0.npz | Teammate distilled_s6@ep1 | 0.25 |
| weights_1.npz | Teammate fttall_s6@ep3 | 0.25 |
| weights_2.npz | Our Z4 seed99 merged core/HFT | 0.25 |
| weights_3.npz | Our H390 homogeneous seed3@ep5 | 0.25 |

This changes the teammate pair in original xteam_h4 from seed5 to seed6, preserving our two members and the Q8 native serving engine. It is a fixed four-trunk ensemble with no wall-clock governor. Current FTT training ancestry and clean CV counterparts are missing; supplied-validation model diagnostics are not held-out selection evidence. The user reports public score **0.6679**, portal `0mbzevhv`, duration 46m51s and 650MB. [Leaderboard update](../../research/campaign_2026-10-13/team_models_r1/PUBLIC_LB_UPDATE_20261013_R2.md) records the exact rows and interpretation. This is +0.0004 over E, −0.0002 versus original xteam_h4 and −0.0003 versus the reported team best pair_ftt2.

## Checks

- [Independent byte and recipe review](INDEPENDENT_REVIEW_R1.md) passes: intended membership/order, source weights, identical inherited runtime, safe ZIP layout, CRC and size.
- [ABI receipt](ABI_R1.json) passes for v2, AVX2 and v3 libraries: no version-specific libpython linkage; maximum required GLIBC2.34.
- [Callback resource admission](CALLBACK_ADMISSION_R1.json) reserves sequential bounded checks within the existing parent limits. [V3](CALLBACK_V3_R1.json) and [V2](CALLBACK_V2_R1.json) each pass Python3.11 on two complete sequences repeated twice, including warmup, finite predictions and bitwise reset equality. The saved V2/V3 arrays are also bitwise equal. These checks used 20% CPU under concurrent load and are not target-host timing evidence.
- [Isolated-import checker](isolated_import.py) and [passing receipt](ISOLATED_IMPORT_R2.json) check the delivered ZIP with Python3.11 `-I`, keeping the repository off the import path and confirming actual member coefficients and precision. The [first checker failure](ISOLATED_IMPORT_FAILURE_R1.json) and [preserved checker](isolated_import_r1_failed.py) record a relative/absolute path comparison error after successful import; resolving the temporary root fixed the checker, without changing the ZIP.
- [Own optimized inference identity](OWN_OPTIMIZED_RUNTIME_R1.json) directly compares the five runtime files against our existing `E_c75_plus_z5_q8_20261012.zip`: callback, precision and all native libraries match byte-for-byte. [Build manifest](BUILD_MANIFEST_R1.json) records all imported members and archive hashes.

Large ZIP, extracted weights and callback arrays stay in ignored physical storage under `artifacts/team_models_20261013/kotiki_xteam_s6_z4_h390s3_q8_20261013/`. The original xteam_h4 archive is unchanged. The `.log` files preserve callback output; all final receipts bind the same delivered ZIP hash.
