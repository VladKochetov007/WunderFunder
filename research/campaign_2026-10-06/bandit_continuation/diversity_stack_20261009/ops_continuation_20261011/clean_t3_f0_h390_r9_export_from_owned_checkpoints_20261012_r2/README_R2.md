# R9 R2 flat-file five-export transport from owned checkpoints

R8 was accepted as a new kernel version, but its push output explicitly says the
R6 kernel source was invalid and dropped. This R9 packet removes that dependency.
It mounts the three already-collected own R6 `model_ep2.pt` and `provenance.json`
pairs from a separate private checkpoint dataset, verifies an exact manifest
and each source byte, then reuses the reviewed R7 export worker's same driver,
export specs, inference precision, 8 GB worker reservation, 5/6 GB GPU guards,
serial-worker limit, wall ceiling and terminal collection bound.

The kernel ID/title is
`vladkochetov07/wnn-clean-t3-f0-h384-exp-20261012-r9` /
`wnn-clean-t3-f0-h384-exp-20261012-r9`. The new private source dataset is
`vladkochetov07/wnn-clean-t3-f0-h384-cp2-20261012-r9`. It contains only the
three own epoch-2 checkpoints, their provenance records and the R9 source
manifest. R9 metadata has no `kernel_sources` entry. The R7 logical recipe target
is retained separately inside the root release binding; R9 is the transport
attempt identity.

R9 requests five exports: s3 fit-valid, s4 fit-train/fit-valid, and s5
fit-train/fit-valid. It omits only s3 fit-train because that exact checkpoint's
fit-train export is already assigned to the local export lane. This leaves
4,045,600,000 prediction NPY bytes, preserves the existing 8 GiB collector cap
and 4 GiB free-disk reserve, and avoids duplicating a completed export.

The first R9 preparation (`clean_t3_f0_h390_r9_export_from_owned_checkpoints_20261012_r1`)
was not uploaded. Its six checkpoint files were nested under `experiment/`, but
the existing Kaggle dataset upload path skips directory trees. R2 uses six
unique top-level filenames plus the manifest. The worker's checkpoint lookup
now requires those exact flat names and its manifest validator requires exact
file closure, so the reviewed upload list contains seven regular files with no
subdirectories. This changes input transport only; R7 export specifications,
checkpoint bytes, provenance, selection, precision and resource guards remain
the same.

`build_r9_transport.py` reads immutable R7 inputs from MAIN, copies and hashes
the six local checkpoint/provenance files into a create-only physical staging
directory using flat seed-qualified filenames, builds the exact dataset manifest, and emits a single self-contained
`kernel.py` plus metadata. Its default embedded ROOT release is pending and
refuses before reading `/kaggle/input`; the builder does not create authority or
perform any Kaggle action. A future root-authored admitted R9 release can be
provided to the builder with `--root-release`, producing a new identified
output directory for review.

The stage output is intentionally short and physical, at
`/tmp/wnn-t3-f0-r9-checkpoint-stage-r2/wnn-clean-t3-f0-h384-cp2-20261012-r9`.
Upload staging and provider operations remain root-owned. The committed
`R9_CHECKPOINT_INPUT_MANIFEST_R2.json` is byte-identical to the staged manifest;
the build receipt records all seven top-level upload files, each hash and byte
count, and the stage parent and dataset directory are mode `0700`.

From this isolated worktree, prepare the pending R9 packet (the two output paths
must not already exist):

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
  /usr/bin/python3 \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_h390_r9_export_from_owned_checkpoints_20261012_r2/build_r9_transport.py \
  --main-repo /home/vlad/development/wunder/wnn_connectome_starterpack \
  --output research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_h390_r9_export_from_owned_checkpoints_20261012_r2/kernel \
  --staging-parent /tmp/wnn-t3-f0-r9-checkpoint-stage-r2 \
  --receipt research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_h390_r9_export_from_owned_checkpoints_20261012_r2/R9_BUILD_RECEIPT_R2.json
```

Run the focused offline transport test with the same wrapper and limits. Then
validate the actual local stage as a simulated Kaggle input mount using
`validate_r9_local_mount.py`. These checks cover bytes and worker input
resolution only; they do not establish remote dataset readiness, an admitted
ROOT receipt, Kaggle startup, T4 resource feasibility, exports, or predictions.
