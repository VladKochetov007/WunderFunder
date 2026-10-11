# R4 terminal collection

This wrapper binds one already-pushed R4 kernel to its frozen packet, root
dispatch receipts, reviewed profile collector, source-qualified fold-zero
membership, and the exact read-only 108-file terminal inventory. The inventory
endpoint does not provide remote sizes or checksums. It showed the profile and
pipeline receipts but no teacher-fit/export result paths, checkpoints, or NPYs.

The current root authorization is for **one terminal ERROR download only**.
It allows at most 1 GiB, requires at least 4 GiB free, and bounds the inner
download to 420 seconds under a 720-second outer runtime. It does not authorize
fits, retries, another provider action, or overwrite. The wrapper requires that
exact root receipt, status snapshot, and inventory receipt before it starts the
reviewed R2 collector. On ERROR it writes a diagnostic receipt and marks
scientific validation as not run; an ERROR-terminal collection is not a model
result.

The root-owned collection directory is fixed by the admission receipt. After
the independent wrapper review and a fresh local memory check, the operator
command is:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=700000000 WNN_RUNTIME_MAX=720s \
/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited \
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python /home/vlad/development/wunder/wnn_r4_terminal_collector_20261012/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_r4_terminal_collection_r1/collect_r4_terminal.py \
--collect \
--status-snapshot /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r4_root_release/ROOT_PROVIDER_POLL_R2.json \
--download-admission /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r4_root_release/ROOT_TERMINAL_DOWNLOAD_ADMISSION_R1.json \
--inventory-receipt /home/vlad/development/wunder/wnn_r4_terminal_collector_20261012/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_r4_terminal_collection_r1/read_only_inventory_attempt_001/R4_READONLY_INVENTORY_RECEIPT_R1.json \
--collection-dir /home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r4_root_release/terminal_collection_r1
```

The command is intentionally not executed in this builder task. Do not launch collection from
the stale 21:43 status snapshot after root replaces it; use the exact
admission-bound snapshot bytes or stop for a new root receipt.

The success-path validator checks all nine expected arrays with mmap headers,
exact byte counts, dtypes, sidecar run/checkpoint/epoch/split/group-order
bindings and a chunked full-array finiteness scan. The failure-path receipt
preserves the complete collector inventory, byte hashes, worker result/log
presence and pipeline/stage summaries while stating that science validation
was not run. Root retains retry and quota decisions.
