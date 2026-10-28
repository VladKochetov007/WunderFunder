from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import torch

from wnn.galam.config import GALAMConfig, galam_config_from_mapping


def configure_cuda(device_name: str) -> torch.device:
    """Require explicit CUDA strict FP32 execution; never fall back to CPU."""
    if os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION", "0") != "0":
        raise ValueError("Unset CUBLAS_EMULATE_SINGLE_PRECISION or set it to 0")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0") != "0":
        raise ValueError("Unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE or set it to 0")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This receiver check requires a CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one explicitly free GPU with CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    return device


def read_config(path: Path) -> GALAMConfig:
    """Parse the frozen GRU72 recipe without accepting another model family."""
    values = json.loads(path.read_text())["model"]["galam"].copy()
    if values.pop("variant", None) != "schematic_gru_full":
        raise ValueError("Only schematic_gru_full is accepted")
    if values.pop("execution_backend", "step") != "step":
        raise ValueError("Supply the preserved step control config")
    config = galam_config_from_mapping(values)
    if (config.width, config.input_features, config.output_features) != (72, 112, 2):
        raise ValueError("Receiver packet is scoped to GRU72, 112 inputs and 2 outputs")
    return config


def digest_file(path: Path) -> str:
    """Hash bytes without importing or executing the artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(config_path: Path, expected_commit: str, device: torch.device) -> dict[str, object]:
    """Bind a clean source tree, immutable configuration and CUDA runtime."""
    root = Path(__file__).resolve().parents[3]
    command = ["git", "-C", str(root)]
    actual = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
    if len(expected_commit) != 40 or actual != expected_commit:
        raise ValueError("Expected full source commit does not match HEAD")
    dirty = subprocess.check_output(
        [*command, "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if dirty:
        raise ValueError("Receiver must use a clean committed checkout")
    paths = sorted((root / "src/wnn/galam").glob("*.py"))
    paths += sorted(Path(__file__).parent.glob("*.py"))
    paths += [root / "src/wnn/training/galam_adapter.py"]
    files = {str(path.relative_to(root)): digest_file(path) for path in paths}
    properties = torch.cuda.get_device_properties(device)
    return {
        "source_commit": actual,
        "source_files": files,
        "source_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "config_sha256": digest_file(config_path),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": properties.name,
        "device_uuid": str(getattr(properties, "uuid", "unavailable")),
        "device_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "precision": "fp32_tf32_off_bf16x9_off",
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "CUBLAS_EMULATE_SINGLE_PRECISION": os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION", "0"),
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0"),
    }


def write_receipt(path: Path, receipt: dict[str, object]) -> None:
    """Create a JSON receipt exclusively; preserve previous evidence on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_output(path: Path) -> None:
    """Reject overwrites and outputs inside the pinned code checkout before GPU work."""
    root = Path(__file__).resolve().parents[3]
    if path.exists() or path.resolve().is_relative_to(root):
        raise ValueError("Receipt output must be new and outside the source checkout")


def validate_acceptance(path: Path, current: dict[str, object], shape: list[int]) -> None:
    """Reject stale, failed, mismatched or incomplete numerical acceptance."""
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != "wnn.galam-block-acceptance.v1":
        raise ValueError("Wrong acceptance receipt schema")
    if receipt.get("status") != "passed" or receipt.get("identity") != current:
        raise ValueError("Acceptance must pass on this exact source/config/runtime/device")
    if receipt.get("representative_shape") != shape:
        raise ValueError("Profile shape was not accepted")
    if receipt.get("suite") != "state-gradient-bn-causal-update-v1":
        raise ValueError("Acceptance suite is incomplete")
    results = receipt.get("results", {})
    if not isinstance(results, dict) or (
        results.get("batch_norm_cases") != 16
        or results.get("causality") != "passed"
        or results.get("reload") != "passed"
        or results.get("diagnostic_observers") != "passed"
    ):
        raise ValueError("Missing successful BN, causality or reload evidence")
    cases = results.get("cases", [])
    if not isinstance(cases, list) or len(cases) != 7:
        raise ValueError("Missing numerical cases")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Malformed numerical case")
        errors = case.get("max_absolute_errors", {})
        if not isinstance(errors, dict) or set(errors) != {
            "output",
            "state",
            "input_gradient",
            "gradient",
            "buffer",
        }:
            raise ValueError("Numerical error evidence is missing")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in errors.values()
        ):
            raise ValueError("Numerical evidence must be finite and nonnegative")
    representative = [
        case
        for case in cases
        if case.get("batch") == shape[0] and case.get("chunks") == [shape[1], shape[1]]
    ]
    if len(representative) != 2 or {case.get("training") for case in representative} != {
        False,
        True,
    }:
        raise ValueError("Missing representative forward and backward checks")
