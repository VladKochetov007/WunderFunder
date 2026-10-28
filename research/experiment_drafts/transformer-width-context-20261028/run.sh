#!/usr/bin/env bash
set -euo pipefail
PACKET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WNN_SEQUENCE_STUDY=transformer-width-context-20261028
exec "${PYTHON_EXECUTABLE:?Set PYTHON_EXECUTABLE to the pinned receiver venv Python}" "$PACKET_DIR/../source-holdout-transformer-kd-20261028/packet_runtime.py" "$@"
