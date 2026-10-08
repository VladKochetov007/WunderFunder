"""Execute one authorized GRU arm on the runner-assigned visible T4."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from typing import cast

ARM_DEFINITIONS = {
    "muon512": {"hidden": 512, "optimizer": "muon"},
    "muon768": {"hidden": 768, "optimizer": "muon"},
}
RUN_NAMES = {
    "muon512": "bd26_gru512_full_s9_f0_muon_seeded_b8_amp4096_capacityv5_fit_20261008",
    "muon768": "bd26_gru768_full_s9_f0_muon_seeded_b8_amp4096_capacityv5_fit_20261008",
}
MINIMUM_FREE_BYTES = 6 * 1024**3
EXPECTED_CAPABILITY = (7, 5)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path.name}")
    return payload


def _assert_assigned_device() -> dict[str, object]:
    """Verify the runner's physical UUID maps to one visible CUDA T4 before fitting."""
    physical_uuid = os.environ.get("WNN_PHYSICAL_GPU_UUID")
    physical_index_text = os.environ.get("WNN_PHYSICAL_GPU_INDEX")
    if not physical_uuid or physical_index_text is None:
        raise ValueError("runner did not bind a physical GPU identity")
    physical_index = int(physical_index_text)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != physical_uuid:
        raise ValueError("CUDA visibility does not match the assigned physical GPU UUID")
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_uuid}",
            "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = list(csv.reader(StringIO(result.stdout), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError("assigned physical GPU query did not return exactly one device")
    index, observed_uuid, name, total_mb, free_mb = (part.strip() for part in rows[0])
    total_bytes = int(total_mb) * 1024**2
    free_bytes = int(free_mb) * 1024**2
    if (
        int(index) != physical_index
        or observed_uuid != physical_uuid
        or "Tesla T4" not in name
        or free_bytes < MINIMUM_FREE_BYTES
    ):
        raise RuntimeError("assigned physical GPU identity or free-memory gate failed")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("arm process does not see exactly its assigned CUDA device")
    properties = torch.cuda.get_device_properties(0)
    capability = (int(properties.major), int(properties.minor))
    if capability != EXPECTED_CAPABILITY or properties.name != name:
        raise RuntimeError("logical CUDA device does not match the measured physical T4")
    cuda_free, cuda_total = torch.cuda.mem_get_info(0)
    if cuda_free <= 0 or cuda_total <= 0:
        raise RuntimeError("assigned CUDA device reports invalid free-memory telemetry")
    receipt = {
        "physical_index": physical_index,
        "physical_uuid": physical_uuid,
        "name": name,
        "compute_capability": list(capability),
        "nvidia_smi_total_bytes": total_bytes,
        "nvidia_smi_free_bytes_before_fit": free_bytes,
        "torch_cuda_total_bytes": int(cuda_total),
        "torch_cuda_free_bytes_before_fit": int(cuda_free),
        "visible_device_count": 1,
    }
    print("GRU_CAPACITY_DEVICE " + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def run(arm_id: str, batch: int, run_name: str) -> dict[str, object]:
    """Validate shared bindings and run one fresh fixed-epoch fit/export."""
    script_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_root))
    import capacity_worker as runtime

    if arm_id not in ARM_DEFINITIONS or batch != 8 or run_name != RUN_NAMES[arm_id]:
        raise ValueError("arm command differs from the reviewed B8 run identity")
    project_root = Path(os.environ["WNN_PROJECT_ROOT"]).resolve(strict=True)
    context_path = Path(os.environ["WNN_FIT_CONTEXT_PATH"])
    context_sha = os.environ["WNN_FIT_CONTEXT_SHA256"]
    if runtime.sha256_file(context_path) != context_sha:
        raise ValueError("fit context changed after runner validation")
    context = _read_json(context_path)
    if context.get("schema") != "wnn-gru-capacity-fit-context-v1":
        raise ValueError("fit context schema is unsupported")
    if context.get("packet_id") != "E529_H232_GRU512_768_MUON_SEED9_FIT_V5_20261008":
        raise ValueError("fit context names a different packet")
    binding_path = Path(os.environ["WNN_INPUT_BINDING_PATH"])
    binding_sha = os.environ["WNN_INPUT_BINDING_SHA256"]
    runtime.validate_shared_input_binding(project_root, binding_path, binding_sha)
    split_path = project_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split_manifest = _read_json(split_path)
    if runtime.sha256_file(split_path) != context.get("split_manifest_sha256"):
        raise ValueError("fit context is detached from the frozen split")
    source_hashes = context.get("source_hashes")
    evaluation_groups = context.get("evaluation_groups")
    if not isinstance(source_hashes, dict) or not isinstance(evaluation_groups, list):
        raise ValueError("fit context source or evaluation bindings are malformed")
    if len(evaluation_groups) != 468 or evaluation_groups != sorted(set(evaluation_groups)):
        raise ValueError("fit context evaluation membership differs from the frozen 468 groups")
    deadline_text = os.environ.get("WNN_ARM_DEADLINE_MONOTONIC")
    if deadline_text is None:
        raise ValueError("runner did not bind an arm-specific deadline")
    deadline = float(deadline_text)
    now = time.monotonic()
    if not math.isfinite(deadline) or deadline <= now + 600:
        raise TimeoutError("arm-specific deadline leaves no fit/export reserve")

    assigned_device = _assert_assigned_device()

    arm = {"id": arm_id, **ARM_DEFINITIONS[arm_id], "primary": True, "run_name": run_name}
    started = now
    result = runtime._fit_arm(
        arm=arm,
        batch=batch,
        split_path=split_path,
        packet_root=script_root,
        project_root=project_root,
        receipt_root=project_root / "capacity_receipts" / arm_id,
        input_identity=Path(os.environ["WNN_INPUT_IDENTITY_RECEIPT"]),
        worker_started=started,
        worker_deadline=deadline,
        split_manifest=split_manifest,
        source_hashes=cast(dict[str, str], source_hashes),
        evaluation_groups=cast(list[int], evaluation_groups),
    )
    result.update(
        {
            "packet_id": context["packet_id"],
            "arm": arm_id,
            "run_name": run_name,
            "batch": batch,
            "fit_started": True,
            "fit_context_sha256": context_sha,
            "shared_input_binding_sha256": binding_sha,
            "worker_seconds": round(time.monotonic() - started, 3),
            "assigned_device": assigned_device,
        }
    )
    terminal_path = project_root / "capacity_receipts" / arm_id / "fit_terminal.json"
    runtime.write_json(terminal_path, result)
    print(
        "GRU_CAPACITY_ARM_TERMINAL "
        + json.dumps({"arm": arm_id, "status": result.get("status")}),
        flush=True,
    )
    if result.get("status") != "complete":
        raise RuntimeError("arm did not satisfy its strict fit and prediction contracts")
    return result


def main() -> None:
    """Parse and execute one sealed arm command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARM_DEFINITIONS), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    run(args.arm, args.batch, args.run_name)


if __name__ == "__main__":
    main()
