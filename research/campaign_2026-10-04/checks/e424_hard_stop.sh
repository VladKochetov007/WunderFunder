#!/usr/bin/env bash
set -u

# Stop only this experiment sequence at the agreed 20:00 Kyiv cutoff.
units=$(systemctl --user list-units --all --plain --no-legend 'wnn-c04-e424-*.service' 'wnn-e424-sequence-*.service' | awk '{print $1}')
for unit in $units; do
    systemctl --user kill --signal=TERM --kill-who=all "$unit" 2>/dev/null || true
    systemctl --user stop "$unit" 2>/dev/null || true
done
systemctl --user kill --signal=TERM --kill-who=all wnn-e424-sequence-v2.service 2>/dev/null || true
systemctl --user stop wnn-e424-sequence-v2.service 2>/dev/null || true
