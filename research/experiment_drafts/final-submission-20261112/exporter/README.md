# Full mix package build (2026-11-12)

`build_fullmix_from_archives.py` exports the exact final checkpoints from the pinned PR54 archive and authenticated W18 evidence, then applies the repository's frozen `full_mix` solution. It does not fit models, build a model object, run inference, or produce a score. It binds all official train and valid labels as fit data (10,607 + 1,873 sequences), with zero independent evaluation sequences and independent score claims forbidden.

This R2 packet has an explicit, hash-pinned Source4 history compatibility overlay. The upstream validator treated every `valid_*` key as an evaluation metric, including the historical `valid_rows` counter. In the pinned sequence trainer, that field is the fitting objective-mask row count. The overlay accepts only `valid_rows` with the exact per-epoch fit counters and continues rejecting `valid_loss`, all other `valid_*` fields, and every `train_eval_*` field. The pinned source checkout remains unchanged. Pass both overlay arguments below to opt into that repair; they must be supplied together.

The fixed blend is read from the pinned source: TL3 roles 0.34 / 0.33 / 0.33, then 0.9 TL3 + 0.1 Source4 LSTM. Source checkout commit and both package source file hashes are enforced by the script. Output is create-only, capped at 20,000,000 bytes, and comes with a SHA-bound build receipt.

## Run

Use the available Python 3.11 CPU environment with `torch` and `numpy` installed. Replace `$python311` with that environment's executable. The packager forces BLAS/OpenMP thread counts to one; keep the process within the host's 16 GiB memory and 3,600 second operational ceilings. It only performs CPU checkpoint deserialization and tensor export. It does not qualify the full-mix inference runtime; a fresh full-mix CPU profile and the existing callback checks are still required before upload.

```powershell
$python311 = '<Python 3.11 executable>'
& $python311 'C:\Users\admin\Documents\Codex\artifacts\final-submission-20261112-r2\package-source\build_fullmix_from_archives.py' `
  --project 'C:\Users\admin\Documents\Codex\reviews\WunderFunder-final-submit-20261031' `
  --checkpoint-archive 'C:\Users\admin\Documents\Codex\reviews\tabdata-final-submit-artifacts-20261031\tabdata-pr54-final-checkpoints-r1.zip' `
  --w18-zip 'C:\Users\admin\Documents\Codex\reviews\tabdata-final-submit-artifacts-20261031\w18-ssh-verified-20261103-r1\tabdata-w18-final-20261103-r1.zip' `
  --w18-build-receipt 'C:\Users\admin\Documents\Codex\reviews\tabdata-final-submit-artifacts-20261031\w18-ssh-verified-20261103-r1\build-receipt.json' `
  --w18-evidence-archive 'C:\Users\admin\Documents\Codex\reviews\tabdata-final-submit-artifacts-20261031\tabdata-w18-final-13-role-evidence-20261103-r1.authenticated.zip' `
  --final-fit-membership 'C:\Users\admin\Documents\Codex\reviews\WunderFunder-final-submit-20261031\research\experiment_drafts\final-submit-all-labeled-20261031\receiving\profile-evidence-r2\inputs\final-fit-membership-r1.json' `
  --source-validation-receipt 'C:\Users\admin\Documents\Codex\reviews\WunderFunder-final-submit-20261031\research\experiment_drafts\final-submit-all-labeled-20261031\receiving\profile-evidence-r2\inputs\SOURCE_VALIDATION_R1.json' `
  --compat-helper 'C:\Users\admin\Documents\Codex\artifacts\final-submission-20261112-r2\package-source\source-overlay-r1\scripts\package_final_tl3.py' `
  --compat-repair-receipt 'C:\Users\admin\Documents\Codex\artifacts\final-submission-20261112-r2\package-source\source-overlay-r1\repair-receipt-r1.json' `
  --output-zip 'C:\Users\admin\Documents\Codex\artifacts\final-submission-20261112-r2\build\tabdata-final-fullmix-20261112-r2.zip'
```

The adjacent `tabdata-final-fullmix-20261112-r2_build_receipt.json` is generated automatically. If either output exists, the command refuses to overwrite it. A successful `built_unprofiled` status means the package was constructed and byte-validated; it does not mean the full-mix runtime gate passed.

## Focused overlay regression checks

ROOT requested positive and negative counter cases. They are provided in `source-overlay-r1/test_source4_history_compat.py` and are marked not run in the repair receipt. ROOT should run them in the same Python 3.11 CPU environment before package export:

```powershell
$env:WNN_FINAL_SUBMIT_PROJECT = 'C:\Users\admin\Documents\Codex\reviews\WunderFunder-final-submit-20261031'
& $python311 'C:\Users\admin\Documents\Codex\artifacts\final-submission-20261112-r2\package-source\source-overlay-r1\test_source4_history_compat.py'
```


ROOT subsequently executed all7 focused compatibility tests successfully; see `../../ROOT_COMPAT_TEST_RUN_R1.json`. The original repair receipt records preparation-time status and remains byte-frozen. Successful export is recorded in the R1 root execution receipt; exact model-set receipt is retained beside the R2 ZIP with its package-bound SHA. CPU runtime acceptance remains pending.

`bench_member_core.py` is retained as the exact component diagnostic source; the [engineering index](../engineering/README.md) explains its scope and limitations.
