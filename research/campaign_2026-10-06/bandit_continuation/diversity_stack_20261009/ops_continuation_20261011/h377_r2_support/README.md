# H377 R2 profile packet support

R1 remains archived with its daemon-rendering failure. R2 preserves its frozen
trainer snapshot, input identities and profile recipe while moving the source
preflight and profile-only output validator into standalone Python files
outside `/root/wnn/lr377_source_r1`. Each queued command is an argv vector with
ordinary paths and flags; it contains no embedded `python -c` program.

The exact local packet is `p04_h377_b64_profile_s3_r2.json`. It has three
ordered steps: validate the pinned source/input payload, run the unchanged
three-step profile recipe, and verify the emitted profile without checkpoint
or prediction artifacts. The packet binds its R1 failure receipt by SHA-256.
Its remote helper paths point at this committed support directory. A controller
must stage these files at those paths before placing the packet in the refill
queue; this commit does not claim that R2 was queued or run.

Run the bounded offline checks from the project root:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=800000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/pytest -q \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/h377_r2_support/test_h377_remote_refill_packet.py
```

The renderer check extracts the current `launch()` shell-expression from the
read-only `vast_teachers_r1/remote_refill.py`, renders the packet with that
expression, runs `bash -n`, then executes a harmless argv fixture through the
same expression to verify argument round-tripping. It does not import or modify
the daemon.
