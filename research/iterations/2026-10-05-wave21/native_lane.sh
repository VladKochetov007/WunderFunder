#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
D="$PWD/research/iterations/2026-10-05-wave21"
while systemctl --user is-active --quiet wunder-wave21-package; do sleep 10; done
for f in submission-audit.json parity-v2.json parity-avx2.json parity-v3.json; do test -f "$D/$f"; done
cd "$D"
"$HOME/ds" project job execute -p bt1dqf9jg0tuhggcrj4k -c native-ds.yaml > datasphere.log 2>&1
