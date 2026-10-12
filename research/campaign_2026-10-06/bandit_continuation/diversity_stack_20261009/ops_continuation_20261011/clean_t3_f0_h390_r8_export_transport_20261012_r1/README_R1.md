# R8 self-contained Kaggle transport

R7's frozen worker body and its two JSON controls are consumed relative to
`__file__`, but Kaggle `kernels_push` uploads only the file named by
`kernel-metadata.json:code_file`. R8 therefore puts the exact R7 worker,
bindings and ROOT release bytes into one standard-library-only `kernel.py`.
After verifying the release, that wrapper creates the three controls with
exclusive creation under `/kaggle/working/.wnn-r8-export-control` and runs the
unchanged worker with its path in that directory. It sends no provider requests.

The recipe binding intentionally retains logical target R7. An admitted ROOT
receipt must additionally set `logical_recipe_target_kernel_id` to the R7
target and `transport_target_kernel_id` to
`vladkochetov07/wnn-clean-t3-f0-h384-exp-20261012-r8`. This explicitly scopes
the new transport kernel without changing the R7 export recipe. The default
generated file embeds the exact pending R7 release and refuses before reading
`/kaggle/input`. A later generator invocation may embed a ROOT-authored admitted
receipt; this builder does not create or authorize one.

The declared Kaggle dataset and source mounts remain exactly those in R7:
`wnn-c20261004-fold0-assets`, `wnn-clean-t3-f0-teachers-20261011-r5`,
`wunder-funder`, and source
`wnn-clean-t3-f0-teachers-20261011-r5-runner`. The worker retains all original
input, checkpoint, host, disk, export and collection guards.

From the isolated worktree root, generate the pending packet with the MAIN
resource wrapper (the output and receipt paths must not already exist):

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=25% WNN_THREADS=1 \
WNN_MEM_MAX=400000000 WNN_RUNTIME_MAX=120s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
  /usr/bin/python3 \
  research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_h390_r8_export_transport_20261012_r1/build_r8_transport.py \
  --repo . \
  --output research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_h390_r8_export_transport_20261012_r1/kernel \
  --receipt research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_h390_r8_export_transport_20261012_r1/R8_BUILD_RECEIPT_R1.json
```

The output directory is create-only. For final release, the root operator must
provide its admitted receipt via `--root-release`; output must be regenerated
in a new empty identified directory, then independently reviewed and checked
against the actual dataset/source readbacks before any push. The packet here is
transport preparation only: no fit, export, provider action or score is
authorized by this commit.
