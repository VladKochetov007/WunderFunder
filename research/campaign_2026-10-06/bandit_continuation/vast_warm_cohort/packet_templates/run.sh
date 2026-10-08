#!/usr/bin/env bash
# Warm smoke packet: image preflight, startup validation, one warm queue, bounded finalization.
set -uo pipefail
cd -- "$(dirname -- "$0")"

PYTHON=/venv/main/bin/python
# Relocated Parquet bytes are accepted only through the sealed portable registry.
export WNN_PORTABLE_DATA_MANIFEST="$(pwd)/project/research/campaign_2026-10-04/portable_inputs.json"
if [[ -z "${WNN_OUTPUT_DIR:-}" || -z "${WNN_INPUT_DIR:-}" ]]; then
  printf '%s\n' 'WNN_OUTPUT_DIR and WNN_INPUT_DIR are required.' >&2
  exit 64
fi
IMAGE_PREFLIGHT="$WNN_OUTPUT_DIR/image_preflight.json"
mkdir -p -- "$WNN_OUTPUT_DIR"

worker_exit=0
if [[ ! -x "$PYTHON" ]]; then
  worker_exit=126
  printf '%s\n' '{"status":"failed","error":"pinned_interpreter_unavailable"}' > "$IMAGE_PREFLIGHT"
  printf '%s\n' '{"status":"failed","phase":"image_preflight"}' > "$WNN_OUTPUT_DIR/result.json"
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
        raise RuntimeError("device does not satisfy the RTX 3090/CUDA 8.6/BF16 gate")
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
  if [[ "$preflight_exit" -ne 0 ]]; then
    worker_exit=$preflight_exit
    printf '{"status":"failed","phase":"image_preflight","exit_code":%s}\n' "$preflight_exit" \
      > "$WNN_OUTPUT_DIR/result.json"
  else
    timeout --signal=TERM --kill-after=10s 450s "$PYTHON" -m pip install \
      --disable-pip-version-check numpy==2.2.6 pyarrow==19.0.1 psutil==7.0.0 \
      > "$WNN_OUTPUT_DIR/dependencies.log" 2>&1
    setup_exit=$?
    if [[ "$setup_exit" -ne 0 ]]; then
      worker_exit=$setup_exit
      printf '{"status":"failed","phase":"dependency_setup","exit_code":%s}\n' "$setup_exit" \
        > "$WNN_OUTPUT_DIR/result.json"
      cp -- "$WNN_OUTPUT_DIR/dependencies.log" "$WNN_OUTPUT_DIR/setup_failure.log"
    else
      timeout --signal=TERM --kill-after=10s 1800s "$PYTHON" stage_inputs.py
      stage_exit=$?
      if [[ "$stage_exit" -ne 0 ]]; then
        worker_exit=$stage_exit
        printf '{"status":"failed","phase":"startup_validation","exit_code":%s}\n' "$stage_exit" \
          > "$WNN_OUTPUT_DIR/result.json"
      else
        # The warm queue runs from the project root; steps use project-relative paths.
        cd -- project
        timeout --signal=TERM --kill-after=60s "${WNN_MAX_RUNTIME_SECONDS:-7200}s" \
          "$PYTHON" scripts/warm_queue_worker.py --queue ../queue --idle-exit-seconds 600 \
          > "$WNN_OUTPUT_DIR/queue_worker.log" 2>&1
        worker_exit=$?
        cd -- ..
      fi
    fi
  fi
fi

export WNN_WORKER_EXIT="$worker_exit"
export WNN_PACKET_SHA256="${WNN_PACKET_SHA256:-}"
timeout --signal=TERM --kill-after=10s 600s "$PYTHON" finalize.py
finalize_exit=$?
if [[ "$finalize_exit" -ne 0 && "$worker_exit" -eq 0 ]]; then
  exit "$finalize_exit"
fi
exit "$worker_exit"
