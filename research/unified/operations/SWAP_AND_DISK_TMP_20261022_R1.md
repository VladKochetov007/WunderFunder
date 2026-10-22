# Swap and disk temporary storage — October22

The user requests inspection and relocation to disk after laptop freezes. This administrative task does not reopen scientific admission. [Inspection/change receipt](../../campaign_2026-10-21/next_cycle_r1/SWAP_TMP_INSPECTION_R1.json) preserves measured facts and the exact helper hash.

## Findings

Only swap is an8GiB `/dev/zram0`, full at inspection: compressed RAM, using about1.79GiB physical RAM according to mm_stat. `/tmp` is tmpfs but uses only about70MiB; `/dev/shm` about218MiB. The larger21GiB system shared-memory total is not fully attributed; visible temporary files do not explain it.

Research scope uses about3.8GiB and zero swap. VSCode scope is14.8GiB, GNOME Shell7.1GiB, Telegram7GiB. Scope totals include descendants/shared memory/cache, not just main-process RSS. No unrelated apps were stopped. The current fit continues at10%CPU; its supervisor remains suspended.

## Applied user setting

Created private0700 SSD directory `/home/vlad/.cache/tmp`; persistent `/home/vlad/.config/environment.d/90-disk-tmp.conf` and the systemd user manager set TMPDIR/TMP/TEMP there. Verified a newly launched user-service process created/deleted a tempfile on that filesystem. Existing processes retain their environment. The research fit already had disk-backed TMPDIR and was preserved.

## Pending administrator action

[Exact archived helper](../../campaign_2026-10-21/next_cycle_r1/enable_disk_swap_tmp.sh), live copy `/home/vlad/development/wunder/research_tmp/enable-disk-swap-tmp.sh`, passes `bash -n`. It creates16GiB SSD swap using the Btrfs swapfile utility in a separate `/var/lib/local-disk-swap` subvolume, applies the SELinux file label, activates priority200 above zram100, and persists fstab with a backup. It masks tmp.mount and installs an empty zram-generator override for **next boot**. No live swapoff/unmount or automatic reboot. Existing zram pages remain until naturally reclaimed or rebooted.

Not executed: `sudo -n -l` requires a password; the agent has no administrator authentication. Run:

```bash
sudo bash /home/vlad/development/wunder/research_tmp/enable-disk-swap-tmp.sh
```

Reboot only after the current fit lands. The privileged helper has not been runtime-tested. Actual disk swap and system `/tmp` relocation remain pending. Boot choices can be reverted by unmasking tmp.mount and removing the empty local zram-generator override. Remove the user environment.d file independently to revert its temp setting; disk swap removal requires adequate memory first.

## Sources

[Official Btrfs swapfile documentation](https://btrfs.readthedocs.io/en/latest/Swapfile.html) and installed help support mkswapfile, dedicated-subvolume isolation and active-swap snapshot restrictions. [systemd environment.d](https://www.freedesktop.org/software/systemd/man/latest/environment.d.html) describes new user-service inheritance; [temporary directory guidance](https://systemd.io/TEMPORARY_DIRECTORIES/) explains TMPDIR and RAM/swap-backed tmpfs. The local vendor zram-generator config explicitly documents disabling defaults through an empty `/etc/systemd/zram-generator.conf`.
