#!/usr/bin/env bash
set -uo pipefail
cd -- "$(dirname -- "$0")"

PYTHON=/venv/main/bin/python
if [[ -z "${WNN_OUTPUT_DIR:-}" ]]; then
  printf '%s\n' 'WNN_OUTPUT_DIR is required.' >&2
  exit 64
fi
mkdir -p -- "$WNN_OUTPUT_DIR"

worker_exit=0
if [[ ! -x "$PYTHON" ]]; then
  worker_exit=126
  printf '%s\n' 'Pinned image interpreter is unavailable.' > "$WNN_OUTPUT_DIR/result.json"
  printf '%s\n' 'Worker was not launched; no interpreter fallback is permitted.' >&2
else
  timeout --signal=TERM --kill-after=10s 450s "$PYTHON" -m pip install \
    --disable-pip-version-check numpy==2.2.6 pyarrow==19.0.1 psutil==7.0.0 \
    > "$WNN_OUTPUT_DIR/dependencies.log" 2>&1
  setup_exit=$?
  if [[ "$setup_exit" -eq 0 ]]; then
    "$PYTHON" worker.py
    worker_exit=$?
  else
    worker_exit=$setup_exit
    printf '{"status":"failed","phase":"dependency_setup","exit_code":%s}\n' \
      "$setup_exit" > "$WNN_OUTPUT_DIR/result.json"
    cp -- "$WNN_OUTPUT_DIR/dependencies.log" "$WNN_OUTPUT_DIR/setup_failure.log"
  fi
fi

export WNN_WORKER_EXIT="$worker_exit"
timeout --signal=TERM --kill-after=10s 600s "$PYTHON" - <<'PY'
import hashlib
import json
import os
import tarfile
from pathlib import Path

out = Path(os.environ["WNN_OUTPUT_DIR"])
result_path = out / "result.json"
if not result_path.is_file():
    result_path.write_text('{"status":"failed","phase":"worker_no_result"}\n')
result = json.loads(result_path.read_text())
worker_exit = int(os.environ["WNN_WORKER_EXIT"])
result["exit_code"] = worker_exit
result["worker_success"] = worker_exit == 0 and result.get("status") == "delivered_ungraded"
temporary_result = out / "result.tmp"
temporary_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
temporary_result.replace(result_path)

bundle = out / "artifact_bundle.tar"
inner = out / "artifacts.tar"
if inner.is_file():
    inner.replace(bundle)
elif not bundle.is_file():
    with tarfile.open(bundle, "w"):
        pass

expected = json.loads(os.environ.get(
    "WNN_EXPECTED_ARTIFACTS", '["artifact_bundle.tar","result.json","worker.log"]'
))
records = []
for name in expected:
    path = out / name
    if not path.is_file():
        raise FileNotFoundError(f"expected terminal artifact is missing: {name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    records.append({"path": name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
manifest = {
    "attempt_id": os.environ.get("WNN_ATTEMPT_ID"),
    "packet_sha256": os.environ.get("WNN_PACKET_SHA256"),
    "worker_terminal": True,
    "files": records,
}
temporary_manifest = out / "artifact_manifest.tmp"
temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
temporary_manifest.replace(out / "artifact_manifest.json")
PY
package_exit=$?
if [[ "$package_exit" -ne 0 ]]; then
  exit "$package_exit"
fi
exit "$worker_exit"
