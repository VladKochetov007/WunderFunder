#!/usr/bin/env bash
set -uo pipefail
cd -- "$(dirname -- "$0")"
PYTHON=/venv/main/bin/python
mkdir -p -- "$WNN_OUTPUT_DIR"
worker_exit=0
if [[ ! -x "$PYTHON" ]]; then
  worker_exit=126
else
  timeout --signal=TERM --kill-after=10s 450s "$PYTHON" -m pip install --disable-pip-version-check numpy==2.2.6 pyarrow==19.0.1 psutil==7.0.0 > "$WNN_OUTPUT_DIR/dependencies.log" 2>&1
  worker_exit=$?
  if [[ "$worker_exit" -eq 0 ]]; then
    timeout --signal=TERM --kill-after=60s "${WNN_MAX_RUNTIME_SECONDS}s" "$PYTHON" worker.py
    worker_exit=$?
  fi
fi
export WNN_WORKER_EXIT="$worker_exit"
if [[ -x "$PYTHON" ]]; then
  timeout --signal=TERM --kill-after=10s 300s "$PYTHON" terminal.py
  finalize_exit=$?
  if [[ "$finalize_exit" -ne 0 && "$worker_exit" -eq 0 ]]; then
    worker_exit=$finalize_exit
  fi
fi
exit "$worker_exit"
