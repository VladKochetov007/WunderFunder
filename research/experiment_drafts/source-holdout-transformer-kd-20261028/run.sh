#!/usr/bin/env bash
set -euo pipefail
PACKET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_EXECUTABLE:?Set PYTHON_EXECUTABLE to the pinned receiver venv Python}" "$PACKET_DIR/packet_runtime.py" "$@"
