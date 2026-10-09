#!/usr/bin/env bash
set -uo pipefail
cd -- "$(dirname -- "$0")"

PYTHON=/venv/main/bin/python
if [[ -z "${WNN_OUTPUT_DIR:-}" ]]; then
  printf '%s\n' 'WNN_OUTPUT_DIR is required.' >&2
  exit 64
fi
IMAGE_PREFLIGHT="$WNN_OUTPUT_DIR/image_preflight.json"
mkdir -p -- "$WNN_OUTPUT_DIR"

worker_exit=0
if [[ ! -x "$PYTHON" ]]; then
  worker_exit=126
  printf '%s\n' 'Pinned image interpreter is unavailable.' > "$WNN_OUTPUT_DIR/result.json"
  printf '%s\n' '{"status":"failed","error":"pinned_interpreter_unavailable"}' > "$IMAGE_PREFLIGHT"
  printf '%s\n' 'Worker was not launched; no interpreter fallback is permitted.' >&2
else
  timeout --signal=TERM --kill-after=10s 180s "$PYTHON" - <<'PY'
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

output = Path(os.environ["WNN_OUTPUT_DIR"])
started = time.perf_counter()
record = {
    "status": "failed",
    "preflight_started_utc": datetime.now(timezone.utc).isoformat(),
    "python_executable": sys.executable,
    "python_version": sys.version,
    "expected_image": None,
    "torch_version": None,
    "cuda_runtime": None,
    "device_count": None,
    "device_name": None,
    "device_capability": None,
    "cuda_bf16_supported": None,
    "cuda_init_elapsed_seconds": None,
    "container_uptime_seconds_at_preflight": None,
    "image_pull_timing": "Not observable inside the worker; use the provider lifecycle receipt.",
}
try:
    specification = json.loads(Path("worker_spec.json").read_text(encoding="utf-8"))
    record["expected_image"] = specification["expected_image"]
    if sys.executable != "/venv/main/bin/python":
        raise RuntimeError("worker interpreter path differs from the frozen image contract")
    import torch

    record["torch_version"] = torch.__version__
    record["cuda_runtime"] = torch.version.cuda
    if torch.__version__ != "2.14.0+cu130" or torch.version.cuda != "13.0":
        raise RuntimeError("pinned image Torch/CUDA runtime version mismatch")
    cuda_started = time.perf_counter()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    bf16 = bool(torch.cuda.is_bf16_supported())
    record.update(
        {
            "device_count": torch.cuda.device_count(),
            "device_name": name,
            "device_capability": capability,
            "cuda_bf16_supported": bf16,
            "total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "cuda_init_elapsed_seconds": time.perf_counter() - cuda_started,
        }
    )
    if "3090" not in name.lower() or capability != [8, 6] or not bf16:
        raise RuntimeError("device does not satisfy the frozen RTX 3090/CUDA 8.6/BF16 gate")
    try:
        record["container_uptime_seconds_at_preflight"] = float(
            Path("/proc/uptime").read_text(encoding="ascii").split()[0]
        )
    except (OSError, ValueError, IndexError):
        pass
    record["status"] = "passed"
except Exception as error:
    record["error"] = f"{type(error).__name__}: {error}"
finally:
    record["preflight_elapsed_seconds"] = time.perf_counter() - started
    record["preflight_finished_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = output / "image_preflight.tmp"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "image_preflight.json")
print(json.dumps(record, sort_keys=True))
sys.exit(0 if record["status"] == "passed" else 78)
PY
  preflight_exit=$?
  if [[ ! -f "$IMAGE_PREFLIGHT" ]]; then
    printf '%s\n' '{"status":"failed","error":"image_preflight_process_did_not_write_receipt"}' > "$IMAGE_PREFLIGHT"
  fi
  if [[ "$preflight_exit" -eq 0 ]]; then
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
  else
    worker_exit=$preflight_exit
    printf '{"status":"failed","phase":"image_preflight","exit_code":%s}\n' \
      "$preflight_exit" > "$WNN_OUTPUT_DIR/result.json"
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
    "WNN_EXPECTED_ARTIFACTS",
    '["artifact_bundle.tar","result.json","worker.log","image_preflight.json"]',
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
