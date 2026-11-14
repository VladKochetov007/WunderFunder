#!/usr/bin/env bash
set -euo pipefail
echo 'HOLD: R1 uses legacy BF16 Newton-Schulz. Do not push or launch; use a root-pinned R2 source only.' >&2
exit 70
