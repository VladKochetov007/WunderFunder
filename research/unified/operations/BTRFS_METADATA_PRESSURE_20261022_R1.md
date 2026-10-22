---
kind: incident
status: complete
created_utc: 2026-10-22T19:50:00Z
updated_utc: 2026-10-22T20:00:00Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: [E-ONLINE-RELATIONSHIPS-20261022-R1]
---

# Btrfs metadata pressure during experiment preparation

## Question or claim

Ruff could not persist its cache file (`ENOSPC`) although `/home` reported about 99 GB available. This interrupted static checks, not a scientific fit. New disk-heavy admission was held while the already-running CPU export continued under its existing bounds.

## Evidence and interpretation

`btrfs filesystem usage` reported a 1.25 TiB device, only 1 MiB unallocated, data allocation 1.20 TiB, and metadata DUP use 20.49/21.00 GiB. These observations are consistent with metadata allocation pressure; free data space alone does not establish that Btrfs can satisfy every allocation. The [official Btrfs documentation](https://btrfs.readthedocs.io/en/stable/trouble-index.html) explains this ENOSPC distinction.

Root ran `uv cache prune` without `--force`, under `scripts/limited` at 25% CPU, 300 MB RAM and zero swap. It reported 16,898 unused files removed (2.9 GiB). Separately, the actor removed only its newly created, failed scratch checkout. No model, dataset, checkpoint, existing research tree, installed environment or previous worktree was deleted. Metadata use fell to 20.42 GiB; normal cached Ruff check and format verification then passed. These concurrent cleanups do not identify each operation's individual physical-space contribution.

## Limits and validity

The filesystem still has little unallocated device space. Large-cache admission must check its actual footprint and current write health. No administrator command, balance, swapoff or reboot was performed. The [Btrfs balance documentation](https://btrfs.readthedocs.io/en/latest/btrfs-balance.html) was consulted; no balance was needed for the recovered checks and none is authorized by this receipt.

## Decision and next step

Reuse existing worktrees and run the small online delivery profile after the saved-checkpoint export releases RAM. Do not interpret this incident as a negative model-quality result. [Live thermal operation](THERMAL_RESUME_20261022_R1.md) and [campaign frontier](../../campaign_2026-10-21/next_cycle_r1/FRONTIER.md) retain execution ownership.
