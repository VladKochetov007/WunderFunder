---
kind: review
status: complete
created_utc: 2026-10-17T01:00:13Z
updated_utc: 2026-10-17T01:00:13Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - ../../../specs/adr/0039-two-core-local-budget-and-disk-scratch.md
  - ../../../artifacts/kaggle_assets/wnn-c20261004-fold0-assets/assets-manifest.json
  - ../../../artifacts/kaggle_assets/wnn-c20261004-fold1-assets/assets-manifest.json
  - ../models/huggingface/README.md
  - ../../campaign_2026-10-15/f3_teacher_export_r2/README.md
---

# Storage retention audit — 2026-10-17

## Question or claim

The project tree uses about 700.5 decimal GB, but that does not mean all of it is irreplaceable. The largest local volume is derived prediction/cache data. Kaggle has 113 project datasets, with most of its reported storage concentrated in six objects. Some remote copies can be recreated from local manifests and files; other remote payloads do not yet have a verified local twin. This is a read-only audit and a proposed retention order, not deletion authorization.

## Method

Read-only inventory used `du -x -B1`, `df`, Btrfs extent accounting, file metadata and hard-link counts, the Kaggle CLI's owned-dataset/kernel listings, and local Kaggle asset manifests. Large payloads were not rehashed. The broad local `du` used `nice`, `ionice -c 3`, and a 50% CPU quota. The checkout was clean before the audit.

At the snapshot, `/home` was Btrfs on the 1.369 TB device: 1.369 TB allocated, 1.335 TB used, and 26.5 GB free by both Btrfs and `df` accounting. A Btrfs subvolume listing was denied with `Operation not permitted`, so snapshots and snapshot-retained blocks could not be audited. Btrfs logical/extent columns are not a promise of physical bytes reclaimed; after any approved cleanup, `df` is the authoritative check.

GNU `du` counts a hard-linked inode once per invocation, so a root-wide tree and an isolated subtree can attribute the same bytes differently. For example, the root walk attributed 102.6 GB to `research` and 30.0 GB to `artifacts`, while separate subtree walks reported 116.6 GB and 42.2 GB. Do not add independently measured subtree totals or infer savings from the isolated values.

## Evidence and interpretation

The single parent-tree walk measured 700,546,383,872 bytes under `/home/vlad/development/wunder` (about 700.5 GB / 652.4 GiB). It attributed 550,509,432,832 bytes to the main checkout, 37.2 GB to `worktrees`, and 60.8 GB to `wnn-agent-worktrees`. The checkout's largest root-walk buckets were:

| Path | Measured bytes | Retention interpretation |
|---|---:|---|
| `cache/` | 359,414,992,896 | Largest local cleanup opportunity, but files are experiment outputs and teacher targets, not disposable cache by definition. |
| `research/` | 102,616,530,944 | Campaign records and raw evidence; preserve the histories and prune only individually audited derived payloads. |
| `datasets/` | 34,947,706,880 | Original supplied `train.parquet`, `valid.parquet`, and `valid_mask.parquet`; preserve as canonical source data. |
| `artifacts/` | 29,968,613,376 in the root walk | Runs, checkpoints, upload packages and preservation archives; subtree accounting differs because of hard links. |
| `.venv/` | 7,503,077,376 | Rebuildable environment, but currently used by a live Kaggle output collector. Do not remove while jobs/collectors use it. |
| `.git/` | 4,150,243,328 | Preserve history and object database; no `git gc`, pruning or history rewrite as a space-clearing shortcut. |

The `cache/` root contains 772 validation-prediction paths totalling 203.3 GB, including 88 symlink aliases, and 42 training-prediction paths totalling 66.1 GB, including two symlink aliases. Of these, 684 validation arrays and 40 training arrays are regular files. Many are roughly 300 MB validation arrays or 1.70 GB training arrays. The cache also contains large target and heuristic arrays. These outputs may be reproducible from source data and checkpoints, but a filename or age does not prove the checkpoint, exact recipe, downstream grades, or cached target treatment survives elsewhere. The per-file consumer/ancestry map has not been built, so no prediction array is approved for removal.

The artifact staging folders are poor first cleanup targets. Under `artifacts/hf_model_upload_20261012/stage`, 7,771 paths represent 21.72 GB, but about 21.71 GB of those file sizes have other hard links outside that staging subtree. The HF archive README records a completed private archive with per-file hash/size verification, but removing the stage path would not remove most underlying inodes. Similarly, `artifacts/kaggle_assets/` contains about 12.25 GB of staged paths, nearly all with outside hard links to their canonical run/cache files. Deleting these directories would save little payload space while removing useful manifests and transport evidence.

The worktree registry has 241 entries, 34 of which point to missing `/tmp` paths and are marked prunable. Those missing paths do not occupy project payload space; pruning registry metadata is not a meaningful disk-recovery action. The live `worktrees/` and `wnn-agent-worktrees/` trees are much larger, but no per-worktree dirty-file, untracked-payload, branch-retention, or active-owner audit was performed. Do not remove a worktree based only on age or a merged-looking branch name.

### Kaggle inventory

Kaggle's owned-dataset listing returned 113 project datasets. The listing's `size` fields sum to 42,241,259,927 bytes, of which six objects account for 42,009,806,506 bytes. The other 107 total about 231 MB by that field. This is a ranking indicator, not an exact reclaim estimate: the API's dataset `size` differs materially from the sum of per-file sizes. For example, `wunder-funder` is listed as 33.76 GB while its three file records total 34,947,700,985 bytes; fold-0 and fold-1 assets are listed near 2.8 GB each while their file records each total about 7.24 GB.

| Kaggle dataset | Listing size | File-list evidence / kernel references | Disposition |
|---|---:|---|---|
| `vladkochetov07/wunder-funder` | 33,761,236,515 B | `train.parquet`, `valid.parquet`, and `valid_mask.parquet` remote names and sizes exactly match the local source files; 28 historical kernels reference it. The most recent related runners checked are terminal. | Strong remote deletion candidate only if Kaggle work can pause; local source stays. Re-upload would be required for future Kaggle runs. |
| `vladkochetov07/wnn-c20261004-fold0-assets` | 2,829,715,147 B | 24 remote file records match the local manifest plus file names and sizes; 34 historical kernel references, including completed Oct 15 jobs. | Re-creatable remotely, but retain until active collection and near-term use are closed. |
| `vladkochetov07/wnn-c20261004-fold1-assets` | 2,808,492,601 B | 24 remote file records match the local manifest plus file names and sizes; two kernel references, both from Oct 4. | Clearest first remote cleanup candidate after checking version history and no pending run. |
| `vladkochetov07/wnn-clean-t3-f0-treatment-assets-20261012-r22` | 1,734,543,942 B | Six file records total 1,914,863,477 bytes, including 1.69 GB training predictions, fit-valid predictions, a parent model and provenance; no kernel references. No exact local file-name twin was found in the searched cache/artifact/Oct 13 campaign paths. | Preserve until an independent local copy and ancestry are verified. |
| `vladkochetov07/wnn-strategy-clean-t3-f3-assets-20261013-r1` | 587,707,894 B | Seven file records total 999,957,283 bytes; two latest F3 teacher/export kernels reference it and both report complete. No local mirror was confirmed. | Preserve pending local-copy and output-collection verification. |
| `vladkochetov07/wnn-e419-qat-dualt4-20261004-v4` | 288,110,407 B | Eleven file records total 316,470,916 bytes; no kernel references were returned. | Hold until its local copy and terminal outputs are identified; savings are modest. |

The three recent status checks for the F3 teacher, F3 export and QAT profile runners report `COMPLETE`; the latest student-shape profile runner reports `ERROR`. The Kaggle dependency query is a kernel-reference list, not a complete audit of dataset version history. A separate local Kaggle output collector was running for `wnn-sampled-recipe-cohort-20261009-r12`, writing text outputs under `research/campaign_2026-10-05/checks/monitor_runtime/provider_outputs/`. Leave that process and destination alone until collection is terminal.

## Limits and validity

Kaggle dataset version history was not audited. The exact effect of deleting each dataset object across its versions must be verified before execution. Remote file metadata was compared to local source/manifest names and sizes, not by downloading and rehashing every remote payload. No cache consumer graph, worktree cleanliness scan, Btrfs snapshot inventory, or full local-file duplicate-hash scan was performed. These are deliberate limits: hashing hundreds of GB or sweeping old worktrees would add load and still would not establish scientific dependency by itself.

## Decision and next step

No local or remote data was removed. No Kaggle mutation or large-file hash was run.

Retention priority:

1. **Never bulk-remove:** the canonical local source Parquet files and mask; tracked source/spec/ADR/research ledgers; frozen submission ZIPs and their provenance; selected endpoint/teacher/initializer checkpoints; the predictions, target caches and receipts required to substantiate accepted grades; and the active output collector's destination.
2. **Audit before any deletion:** all `cache/` prediction and target arrays, campaign binary outputs, `artifacts/runs`, and live worktrees. Build a file-to-experiment/grade/parent/rebuild map first; keep the exact source recipe, fitting ancestry and at least one copy of every selected result.
3. **Remote cleanup candidates after explicit approval:** fold-1 Kaggle assets first, then `wunder-funder` if the user accepts re-uploading the source data before the next Kaggle run. Fold-0 assets are also reproducible locally, but have many more and much more recent kernel references. Before acting, recheck all candidate versions, remote consumers, collectors, and local manifest/source presence.
4. **Do not start with upload staging:** local Kaggle and HF asset paths are predominantly hard-linked to canonical local payloads, so deleting staging folders will not reclaim their apparent size. The `/tmp` worktree records are absent paths and likewise do not free payload space.

The largest plausible local recovery is selective retirement of old prediction arrays, but this audit does not identify an approved per-file deletion set. Keep the original competition source data on disk. A follow-up can produce an exact candidate list with expected reclaim ranges only after the dependency and snapshot checks; execution still requires the user's approval.

## Source records

- [Two-core local budget and physical-disk scratch ADR](../../../specs/adr/0039-two-core-local-budget-and-disk-scratch.md).
- [Private HF model archive and final integrity evidence](../models/huggingface/README.md).
- [Local fold-0 Kaggle asset manifest](../../../artifacts/kaggle_assets/wnn-c20261004-fold0-assets/assets-manifest.json) and [fold-1 manifest](../../../artifacts/kaggle_assets/wnn-c20261004-fold1-assets/assets-manifest.json).
- [F3 teacher/export packet and collection status](../../campaign_2026-10-15/f3_teacher_export_r2/README.md).
- Read-only CLI inventories captured during this audit; no credentials or tokens were recorded.
