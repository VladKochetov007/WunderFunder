# Clean F0 T3 student treatment, Kaggle R22

This packet schedules the own fold-0 seed-3 H256/alpha-0.9 treatment student
only. The matched combo-parent control is already admitted locally, so the
cloud packet does not duplicate it. A later paired grade may reuse that local
control only after its CP5, prediction, membership, and source receipts pass
the root's terminal checks.

The recipe is the reviewed R6 treatment p1/p2/p3 continuation, corrected to
consume the validated R2 teacher cache and to export the actual p3 run at CP5.
It fits 10,535 train and 1,405 fit-valid groups and excludes 468 fold-0
evaluation groups. The target is the FP32 mean of the three own H384 T3 teachers
scaled by 0.8125. The complete feature-only valid prediction is expected to be
float32 `[1873, 20000, 2]` (299,680,128 bytes).

`prepare_treatment_kaggle_r22.py` is create-only for local staging. It does not
call Kaggle or start a worker. It refuses to overwrite the short physical
source and asset stage directories. The staged upload roots are:

- `/home/vlad/development/wunder/.wnn-clean-t3-f0-treatment-r22-source`
- `/home/vlad/development/wunder/.wnn-clean-t3-f0-treatment-r22-assets`
- the packet's `kernel/` directory

The source dataset has four data files plus CLI metadata. The flat private
asset dataset has five data files plus its manifest and CLI metadata. The
kernel payload is one `kernel.py` file and metadata. The cache NPYs in the
physical asset stage are same-filesystem hardlinks to the validated MAIN cache;
they are never chmodded or rewritten. Exact uploaded payload hashes are in
`STAGING_RECEIPT_R3.json`. `STAGING_RECEIPT_R2.json` and its `*_INITIAL` copy
preserve the earlier local seal; R3 only refreshed the generated kernel after
style cleanup and did not change the plan or data payload. The failed tmpfs
copy is recorded in `STAGING_ATTEMPT_R1_FAILURE.json`.

Run the offline packet fixture with the MAIN repository's bounded wrapper:

```sh
cd /home/vlad/development/wunder/wnn_connectome_starterpack
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=650000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python \
  /home/vlad/development/wunder/wnn_clean_f0_pair_kaggle_r21_20261012/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261012/clean_t3_f0_student_treatment_kaggle_r22/offline_fixture_r22.py
```

The fixture receipt `OFFLINE_FIXTURE_RECEIPT_R3.json` binds the final kernel
bytes and its test command. It extracts the actual frozen trainer bundle,
builds flat source/asset mounts matching Kaggle's payload layout, checks all
18 external mounts, validates pending root-gate refusal, and confirms that only
the treatment continuation is emitted. It does not import the trainer,
allocate CUDA, score, fit, or contact a provider.

Before any upload or push, root must bind a fresh quota/session receipt and
admission to these exact staged hashes, confirm the private dataset readback,
and issue the R2 root-release receipt. The kernel performs one actual T4
30-batch profile first; its internal guard must pass before any fit begins.
The fit branch remains closed unless the exact root receipt separately sets
`fit_authorized: true`. Runtime is unknown until the cloud profile. The
14,400-second kernel ceiling is a bounded cap, not a speed estimate or finish
promise; this packet does not authorize paid spending.
