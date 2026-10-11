# Old-Z4 HFT profile CLI correction (R6)

R5's profile command failed in c337 argument validation before device or data
initialization. The trainer rejects `--save-training-state` together with its
profile early exit. R6 preserves the failed R5 attempt and changes only the
profile run identity plus removal of that one flag. The separately frozen
`HFT_FIT_CONTRACT_R5.json` and its fit argv, continuation-state policy, source,
membership, optimizer, objective, schedule, and 27-batch profile shape are
unchanged.

The R6 profile binds the exact R5 source snapshot, fit contract, actual R6 crop
receipt and producer, and c337 trainer SHA-256. Its profile argv is the fit
argv with a new run name, no `--save-training-state`, and the existing
2-warmup/25-measured-batch/gradient-statistics profile flags. This is a short
resource/update diagnostic, not a fit or score.

The bounded CLI preflight runs the actual c337 `train_rnn.main()` with the
complete R6 argv. It loads the actual cropped initializer and split, then
stops at `configure_torch`, before device setup, output-directory creation or
sequence-data loading. It verifies that parsing and all earlier recipe checks
pass without starting CUDA or a fit:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=700000000 WNN_RUNTIME_MAX=120s WNN_LOADER_WORKERS=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_hft_profile_cli_r6/profile_hft_stage_r6.py \
  --project-root /home/vlad/development/wunder/wnn_connectome_starterpack \
  --validate-cli
```

The actual profile runner accepts no stage-selection argument: without
`--validate-cli`, it requires a root-created
`ROOT_HFT_PROFILE_LAUNCH_R6.json` with schema
`wnn.oldz4-hft-profile-launch.r6`, decision `PASS`, and exact bindings for
`profile_run`, `profile_contract_sha256`, `profile_runner_sha256`,
`fit_contract_sha256`, `core_crop_receipt_sha256`, and
`trainer_source_sha256`. It then executes the exact c337 profile argv from the
contract. The bounded profile command is:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=6800000000 WNN_RUNTIME_MAX=1800s WNN_LOADER_WORKERS=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_hft_profile_cli_r6/profile_hft_stage_r6.py \
  --project-root /home/vlad/development/wunder/wnn_connectome_starterpack
```

R6 profile success does not authorize the HFT fit. The existing R5 fit runner
still checks its original R5 profile-admission receipt; root must bind the
actual R6 profile evidence into the fit-stage release before starting any fit.
The failed R5 profile attempt remains at
`runtime/attempts/oldz4-r5-f0-s3-hft-profile.json`, with its original log at
`runtime/logs/oldz4-r5-f0-s3-hft-profile.log`.
