#!/usr/bin/env bash
# One-off cleanup: the historical native grader binds the old exporter bytes.
set -euo pipefail
cd /home/vlad/development/wunder/wnn_connectome_starterpack
native_unit=wnn-next-native-recovery-r3-20261021.service
expected_invocation=676380f555cf4f1daae236b9eeb432a6
for ((attempt=0; attempt<360; attempt++)); do
    observed_invocation=$(systemctl --user show "$native_unit" -p InvocationID --value)
    [[ "$observed_invocation" == "$expected_invocation" ]]
    native_pid=$(systemctl --user show "$native_unit" -p MainPID --value)
    native_state=$(systemctl --user show "$native_unit" -p ActiveState --value)
    if [[ "$native_pid" == 0 && "$native_state" != active && "$native_state" != activating ]]; then
        printf '%s  %s\n' 51468297d2da604d8f3c8236fb7d6722b8b0d57a637eace94ccb1810b03cf4c4 scripts/package_submission.py | sha256sum --check
        git show e48e41f03:scripts/package_submission.py > /home/vlad/development/wunder/research_tmp/next-cycle-exporter-restored.py
        printf '%s  %s\n' 7882ad0a8b06b7eb9fe83968f6c61fbb7ae15863672ed3ae2f3ec7a70833644c /home/vlad/development/wunder/research_tmp/next-cycle-exporter-restored.py | sha256sum --check
        cp /home/vlad/development/wunder/research_tmp/next-cycle-exporter-restored.py scripts/package_submission.py
        printf 'RESTORED_COMMITTED_DYNAMICS_EXPORTER_AFTER_NATIVE_TERMINAL\n'
        exit 0
    fi
    sleep 10
done
printf 'RESTORE_TIMEOUT_NO_FILE_CHANGED\n' >&2
exit 1
