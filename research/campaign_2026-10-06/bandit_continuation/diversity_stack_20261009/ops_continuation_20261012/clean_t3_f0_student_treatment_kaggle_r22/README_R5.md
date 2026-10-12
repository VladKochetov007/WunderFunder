# Treatment kernel R5 bounded-quota successor

R5 is a versioned successor to the committed R4 nested-mount packet. It keeps
the R4 kernel ID, metadata, sources, private datasets, pair plan, model recipes,
profile thresholds, stage arguments, output names and checkpoint policy. The
only runtime-policy change is a 10,000-second total kernel wall ceiling. The
300-second finalization reserve is unchanged. The root-release consumer now
requires an explicit two-device quota binding: 10,000 seconds per device,
2 provisioned GPUs, and a 20,000 GPU-second cohort ceiling. This reflects the
conservative Kaggle two-T4 billing envelope even if a worker uses one device.

The kernel refuses a root receipt that does not match all four budget fields.
The pipeline result reports the same wall/device/cohort bounds. The full
profile-and-three-fit/export sequence may not finish inside the shorter wall
cap; no throughput claim is made. Stage outputs are written incrementally, and
the kernel keeps the 300-second finalization reserve for terminal receipts.
Any timeout or partial stage remains a terminal execution result, not a
scientific negative and not permission to retry without a new admission.

R4's mount resolver is byte-identical here: flat and nested Kaggle input layouts
are both supported through bounded exact-owner/slug discovery, unique payload
selection and the unchanged source/hash guards. R4's actual nested consumer and
prior-producer profile-guard fixtures are rerun; the profile fixture is schema
compatibility evidence only, not an R22 treatment-device qualification.

The embedded root release must be independently authored and admitted by root.
For R5 its `quota_budget` must contain exactly:

- `wall_ceiling_seconds`: 10000
- `single_device_gpu_seconds_upper_bound`: 10000
- `provisioned_gpu_count`: 2
- `cohort_gpu_seconds_upper_bound`: 20000
- `paid_spending_authorized`: false

R5 is prepared but unuploaded. Independent review, root release, fresh quota and
resource admission, dataset READY readback, and actual treatment-device profile
remain open. No provider action or model work was run while preparing it.

Bounded offline validation (50% CPU, one thread, 700 MB, 120 seconds):

    WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1
    WNN_MEM_MAX=700000000 WNN_RUNTIME_MAX=120s
    /home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited
    /usr/bin/python3 offline_fixture_r5.py
