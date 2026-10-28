#!/usr/bin/env bash
set -euo pipefail
export WNN_SEQUENCE_STUDY=source-holdout-transformer-kd-20261028
PACKET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_EXECUTABLE:?Set PYTHON_EXECUTABLE to the pinned receiver venv Python}" "$PACKET_DIR/packet_runtime.py" "$@"
