#!/usr/bin/env bash
# Activate SSD swap now; switch /tmp and zram only on the next boot.
set -euo pipefail

if (( EUID != 0 )); then
    printf 'Run this file with sudo.\n' >&2
    exit 1
fi

swap_dir=/var/lib/local-disk-swap
swap_file=$swap_dir/swapfile
[[ $(findmnt -n -o FSTYPE -T /var/lib) == btrfs ]]
[[ ! -s /etc/systemd/zram-generator.conf ]]
if [[ ! -e $swap_dir ]]; then
    available=$(df -B1 --output=avail /var/lib | tail -n 1)
    (( available >= 24 * 1024 * 1024 * 1024 ))
    btrfs subvolume create "$swap_dir"
fi
[[ ! -L $swap_dir && ! -L $swap_file ]]
btrfs subvolume show "$swap_dir" >/dev/null
chmod 700 "$swap_dir"
if [[ ! -e $swap_file ]]; then
    btrfs filesystem mkswapfile --size 16G "$swap_file"
fi
chown root:root "$swap_dir" "$swap_file"
chmod 600 "$swap_file"
btrfs inspect-internal map-swapfile "$swap_file" >/dev/null
if [[ $(getenforce) != Disabled ]]; then
    if [[ $(matchpathcon -n "$swap_file") != *:swapfile_t:* ]]; then
        semanage fcontext -a -t swapfile_t "$swap_file"
    fi
    restorecon "$swap_file"
fi
if ! awk -v p="$swap_file" '$1 == p { found=1 } END { exit !found }' /proc/swaps; then
    swapon --priority 200 "$swap_file"
fi
if ! awk -v p="$swap_file" '$1 == p && $3 == "swap" { found=1 } END { exit !found }' /etc/fstab; then
    cp -n /etc/fstab /etc/fstab.before-local-disk-swap
    printf '\n%s none swap defaults,pri=200 0 0\n' "$swap_file" >> /etc/fstab
fi

# No --now: current temporary files and swap pages remain available until reboot.
systemctl mask tmp.mount
install -m 0644 /dev/null /etc/systemd/zram-generator.conf
printf 'SSD swap enabled. /tmp moves to disk and zram is disabled at next boot.\n'
printf 'Reboot only after the draining fit has finished.\n'
swapon --show
