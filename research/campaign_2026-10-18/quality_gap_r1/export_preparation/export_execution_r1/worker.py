"""Recheck the prepared contract and invoke the unchanged frozen predictor."""

from __future__ import annotations

import argparse
import importlib
import os
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bindings import (
    B_PYTHON,
    EXECUTION_DIR,
    RESOURCE_ENVELOPE,
    read_json,
    sha256_file,
    worker_preinference_record_path,
    write_json_exclusive,
)
from prepare import build_contract


def _verify_environment(contract: dict[str, object]) -> None:
    """Ensure the child imported the reviewed R2 package ahead of frozen code."""
    project = Path(str(contract["project_root"]))
    overlay = (
        project
        / "research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r2"
    )
    frozen = project / "frozen_source_snapshot"
    expected = {
        "WNN_PROJECT_ROOT": str(project),
        "WNN_PORTABLE_DATA_MANIFEST": str(
            project / "research/campaign_2026-10-04/portable_inputs.json"
        ),
        "WNN_F0_COMPATIBILITY_RECEIPT": str(
            project
            / "research/campaign_2026-10-18/quality_gap_r1/export_preparation/compatibility_overlay_r2/COMPATIBILITY_RECEIPT_R2.json"
        ),
        "WNN_F0_MEMBERSHIP_RECEIPT": str(
            project
            / "research/campaign_2026-10-18/quality_gap_r1/export_preparation/F0_MEMBERSHIP_COMPARISON.json"
        ),
        "PYTHONPATH": f"{overlay / 'src'}:{frozen / 'src'}",
        "WNN_GPU_MEMORY_BYTES": str(RESOURCE_ENVELOPE["allocator_bytes"]),
        "WNN_GPU_MEMORY_RESERVE_BYTES": str(
            RESOURCE_ENVELOPE["gpu_free_memory_safety_reserve_bytes"]
        ),
        "WNN_LOADER_WORKERS": "0",
        "WNN_LOADER_RAM_GB": "1",
        "NVIDIA_TF32_OVERRIDE": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise ValueError(f"worker environment mismatch: {key}")
    if os.environ.get("WNN_GPU_DEVICE") != contract.get("gpu_uuid"):
        raise ValueError("worker did not inherit the admitted GPU UUID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != contract.get("gpu_uuid"):
        raise ValueError("CUDA visibility differs from the admitted GPU UUID")
    if (
        os.environ.get("TMPDIR")
        != "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp"
    ):
        raise ValueError("temporary storage is not the physical B research scratch")
    if os.environ.get("WNN_TMP_ROOT") != os.environ.get("TMPDIR"):
        raise ValueError("WNN_TMP_ROOT and TMPDIR differ")
    if os.environ.get("WNN_THREADS") != "1" or os.environ.get("OMP_NUM_THREADS") != "1":
        raise ValueError("worker thread count is not one")


def _seal_utility_code(contract: dict[str, Any], source_paths: list[Path]) -> None:
    """Check every runtime utility source explicitly bound by the contract."""
    expected = contract["runtime"]["export_utility_code"]
    supplied = {path.name: path for path in source_paths}
    if set(supplied) != set(expected):
        raise ValueError("guarded command utility-source inventory differs from contract")
    for name, identity in expected.items():
        path = supplied[name]
        if path.resolve(strict=True) != (EXECUTION_DIR / name).resolve(strict=True):
            raise ValueError(f"guarded command names unexpected utility source: {name}")
        if path.stat().st_size != identity["bytes"] or sha256_file(path) != identity["sha256"]:
            raise ValueError(f"export utility source changed: {name}")


def _record_tf32_state(contract: dict[str, Any], torch: Any) -> dict[str, Any]:
    """Set strict FP32 backend flags and save their observed pre-inference values."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    observed = {
        "schema": "quality_gap.strong_control_pre_inference_runtime.r1",
        "status": "PRE_INFERENCE_SETTINGS_VERIFIED",
        "created_utc": datetime.now(UTC).isoformat(),
        "attempt_id": contract["attempt_id"],
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wnn_gpu_device": os.environ.get("WNN_GPU_DEVICE"),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "inference_precision": "fp32",
        "allocator_bytes": int(os.environ["WNN_GPU_MEMORY_BYTES"]),
        "loader_ram_gb": os.environ.get("WNN_LOADER_RAM_GB"),
        "loader_workers": os.environ.get("WNN_LOADER_WORKERS"),
        "predictor_sha256": contract["pins"]["predictor_sha256"],
        "preflight_receipt_sha256": contract["preflight_receipt"]["sha256"],
    }
    if (
        observed["cuda_matmul_allow_tf32"]
        or observed["cudnn_allow_tf32"]
        or observed["cudnn_benchmark"]
        or not observed["cudnn_deterministic"]
        or observed["float32_matmul_precision"] != "highest"
        or observed["nvidia_tf32_override"] != "0"
    ):
        raise RuntimeError("could not establish strict FP32 TF32-off settings")
    record_path = worker_preinference_record_path(str(contract["attempt_id"]))
    write_json_exclusive(record_path, observed)
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--sealed-source", type=Path, action="append", default=[])
    args = parser.parse_args()
    contract_path = args.contract.resolve(strict=True)
    if sha256_file(contract_path) != args.contract_sha256:
        raise ValueError("contract digest differs from bandit_resources reservation")
    contract = read_json(contract_path)
    attempt = str(contract["attempt_id"])
    admission_path = Path(contract["root_admission"]["path"])
    actual = build_contract(
        attempt,
        admission_path,
        require_fresh_admission=False,
        check_outputs=True,
    )
    saved = dict(contract)
    saved.pop("contract_path", None)
    if actual != saved:
        raise ValueError("live inputs no longer match the reserved export contract")
    _seal_utility_code(contract, args.sealed_source)
    _verify_environment(contract)
    predictor = Path(contract["execution"]["predictor"])
    if args.predictor.resolve(strict=True) != predictor.resolve(strict=True):
        raise ValueError("predictor path differs from the contract")
    if sha256_file(predictor) != contract["pins"]["predictor_sha256"]:
        raise ValueError("frozen predictor source changed")
    if Path(sys.executable).resolve() != B_PYTHON.resolve(strict=True):
        raise ValueError("worker interpreter differs from the prepared B environment")
    importlib.invalidate_caches()
    sys.path.insert(0, str(predictor.parent))
    torch = importlib.import_module("torch")
    _record_tf32_state(contract, torch)
    sys.argv = [str(predictor), *contract["execution"]["command"]]
    runpy.run_path(str(predictor), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
