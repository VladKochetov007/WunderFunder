#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D=research/iterations/2026-10-05-wave16
for name in sparse_twostream sparse_twostream_fastbase sparse_twostream_pair sparse_twostream_pair_fastbase; do
 "$HOME/venv311/bin/python" research/iterations/2026-10-05-wave2/audit_submission.py "$D/$name" "$D/package-audit-$name.json" > "$D/audit-$name.log" 2>&1
done
