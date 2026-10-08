#!/usr/bin/env bash
set -uo pipefail
cd -- "$(dirname -- "$0")"

PYTHON=/venv/main/bin/python
PACKAGER="$PYTHON"
if [[ -z "${WNN_OUTPUT_DIR:-}" ]]; then
  printf '%s\n' 'WNN_OUTPUT_DIR is required.' >&2
  exit 64
fi
mkdir -p -- "$WNN_OUTPUT_DIR"
export WNN_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export WNN_LOADER_WORKERS=3 WNN_GPU_DUTY=1 WNN_GPU_MEMORY_BYTES=18000000000
export WNN_WORKER_RAM_BYTES=28000000000 WNN_WORKER_VRAM_BYTES=19000000000

worker_exit=0
if [[ ! -x "$PYTHON" ]]; then
  worker_exit=126
  printf 'required image interpreter is missing or not executable: %s\n' "$PYTHON" > "$WNN_OUTPUT_DIR/preflight.log"
  printf 'diagnostic command paths: python=%s python3=%s /usr/bin/python3=%s\n' \
    "$(command -v python || true)" "$(command -v python3 || true)" \
    "$(test -x /usr/bin/python3 && printf present || printf absent)" >> "$WNN_OUTPUT_DIR/preflight.log"
  printf '%s\n' 'Worker was not launched; no global python alias fallback is permitted.' > "$WNN_OUTPUT_DIR/worker.log"
  if [[ -x /usr/bin/python3 ]]; then PACKAGER=/usr/bin/python3; fi
else
  timeout --signal=TERM --kill-after=10s 450s "$PYTHON" -m pip install \
    --disable-pip-version-check numpy==2.2.6 pyarrow==19.0.1 psutil==7.0.0 \
    > "$WNN_OUTPUT_DIR/dependencies.log" 2>&1
  setup_exit=$?
  if [[ "$setup_exit" -eq 0 ]]; then
    "$PYTHON" worker.py > "$WNN_OUTPUT_DIR/worker.log" 2>&1
    worker_exit=$?
  else
    worker_exit=$setup_exit
    cp -- "$WNN_OUTPUT_DIR/dependencies.log" "$WNN_OUTPUT_DIR/worker.log"
  fi
fi
export WNN_WORKER_EXIT="$worker_exit"
if [[ ! -x "$PACKAGER" ]]; then
  printf '%s\n' 'No approved explicit Python interpreter is available for terminal packaging.' >&2
  exit 126
fi
timeout --signal=TERM --kill-after=10s 600s "$PACKAGER" - <<'PY'
import hashlib
import json
import os
import tarfile
from pathlib import Path

out = Path(os.environ["WNN_OUTPUT_DIR"])
exit_code = int(os.environ["WNN_WORKER_EXIT"])
result_path = out / "result.json"
if not result_path.exists():
    phase = "image_interpreter_preflight" if exit_code == 126 else "setup_or_worker_crash"
    result_path.write_text(json.dumps({"status": "failed", "phase": phase, "exit_code": exit_code}) + "\n")
result = json.loads(result_path.read_text())
result["exit_code"] = exit_code
result["worker_success"] = exit_code == 0 and result.get("status") in {
    "delivered_ungraded", "partial_delivery_ungraded"
}
temporary = out / "result.tmp"
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
temporary.replace(result_path)

bundle = out / "artifact_bundle.tar"
inner = out / "artifacts.tar"
if inner.is_file():
    inner.replace(bundle)
else:
    with tarfile.open(bundle, "w") as archive:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.name not in {bundle.name, "artifact_manifest.json"}:
                archive.add(path, arcname=path.name, recursive=False)

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
temporary = out / "artifact_manifest.tmp"
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
temporary.replace(out / "artifact_manifest.json")
PY
package_exit=$?
if [[ "$package_exit" -ne 0 ]]; then exit "$package_exit"; fi
exit "$worker_exit"
