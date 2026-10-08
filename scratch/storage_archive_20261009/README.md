# Inactive prediction archive

The user authorized moving inactive experiment artifacts to `~/HDD` on October 9, 2026. This batch moves only 82 historical `cache/pred_abN_*` prediction arrays, with N at most 55: 27,368,330,496 bytes. Current campaign data, teacher/student/propensity caches, checkpoints, metadata and submission packages remain on the SSD.

Archive: `/home/vlad/HDD/wnn_connectome_archive/20261009_legacy_ab1_55/`. The original cache filenames become symlinks, preserving historical consumers. The archive contains `plan.json`, `moves.jsonl` with exact SHA-256 checksums, and the original relative paths. Sidecars remain at their original paths.

Each file is copied to a temporary destination, flushed, read back and hashed. After checksum equality and a check for open file descriptors or mappings, the source is renamed into a private handoff directory. Its identity is checked again, and the original path is recreated as a symlink without overwriting a concurrent new file. Only then is the verified SSD copy released. Interrupted handoffs resume from the journal; a conflicting change preserves both copies for review. No evidence is discarded. The archive keeps at least 20 GB of HDD headroom.

The worker is bounded to one CPU and 512 MB RAM with no swap and idle I/O priority under `wnn-campaign.slice`. It acquires the shared support lock per file, allowing cloud preparation to proceed between copies. Its finite runtime is one hour. `PLAN_R1.json` freezes the file list and source identities; `FIXTURE_CHECK_R2.json` records seven real cross-filesystem checks, including interrupted handoffs, symlink rejection and preservation of concurrent writes. R1 is historical.

To restore an array to SSD, copy its HDD target to a temporary regular file beside the original symlink, verify its SHA-256 against `moves.jsonl`, then atomically replace the symlink with that temporary file. Retain the HDD copy until the restored file has been validated. Existing code can also read the archived array directly through its original path, at HDD speed.
