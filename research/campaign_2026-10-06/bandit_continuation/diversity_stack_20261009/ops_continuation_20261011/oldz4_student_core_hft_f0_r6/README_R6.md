# Old-Z4 F0 core-crop consumer correction (R6)

R5's admitted scientific contract, source snapshot, CP1, and crop primitive are
unchanged. Its actual attempt stopped before derivation because the consumer
looked up `snapshots.core_crop.script_sha256`, while the bound manifest stores
the source digest at
`snapshots.core_crop.files["scripts/make_core_checkpoint.py"]`. R6 fixes both
the pre-write and post-write receipt bindings and emits the same R5-compatible
producer receipt, augmented with R6 attempt/source/admission provenance.

The original failed R5 receipt remains intact and is bound by SHA-256 in the R6
retry contract. The existing crop destination must still be absent. The R6
runner requires a separate root-owned `ROOT_CORE_CROP_ADMISSION_R6.json` bound
to the retry-contract digest and exact consumer, R5 contract, R5 bindings, CP1,
provenance, crop source, failed R5 receipt, and destination. The producer receipt
is write-once at the R5 path so the already frozen R5 HFT consumer can read it.

The actual-producer fixture copies only the consumed R5 files and actual CP1
into a temporary project root, supplies a test-only admission with the same
bindings, runs the complete R6 consumer, then calls the unchanged R5 HFT
receipt reader. It removes the temporary project tree on success and does not
write to the main project artifacts. Its result is
`ACTUAL_CP1_FULL_CONSUMER_FIXTURE_R2.json`. The earlier R1 fixture receipt is
preserved; R2 additionally rechecks the frozen crop source after derivation.

Run the fixture from the project root under the bounded support wrapper:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=1200000000 WNN_RUNTIME_MAX=120s WNN_LOADER_WORKERS=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/vlad/development/wunder/wnn_connectome_starterpack/src \
scripts/limited .venv/bin/python \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_student_core_hft_f0_r6/verify_actual_cp1_consumer_r6.py
```

The fixture is producer/consumer validation, not a fit, profile, grade, or
root release. Actual crop execution remains owned and admitted by the root.
