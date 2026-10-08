# Mandatory resource rules for every delegated agent (2026-10-08)

A previous subagent ran an unwrapped `python3` that reached 15.8 GB RSS outside the campaign cgroup, caused a host-wide OOM and killed the whole Claude Code session (HOST_OOM_SESSION_CRASH_20261008_R1.json). These rules are binding:

1. Run EVERY python, kaggle, pytest, ruff or other non-trivial command through the memory-capped wrapper, from the main checkout:
   `WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=3000000000 WNN_RUNTIME_MAX=600s /home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited <command...>`
   Raise WNN_MEM_MAX only up to 4500000000 and WNN_CPU_QUOTA up to 200% when needed. Plain `ls`, `cat`, `head`, `rg`, `git` are fine without it.
2. Never load a whole parquet file. `datasets/train.parquet` (29 GB) and `datasets/valid.parquet` (5.6 GB) have one row group per sequence (20,000 rows): read only needed row groups and columns with pyarrow (`ParquetFile.read_row_group(i, columns=[...])`).
3. Never read multi-GB `.npy` caches fully; use `np.load(..., mmap_mode="r")` and slice.
4. No docker/podman. No GPU use unless your task explicitly says so. Never start long jobs as session background shells; root launches long work as systemd units.
5. Check `cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice/memory.current` before any step expected to use >1 GB; if it exceeds 14e9, wait or ask root.
6. If a command is killed (exit 137) do not retry with a bigger cap blindly: reduce the data you load.
