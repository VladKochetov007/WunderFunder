"""Extend the frozen R3 receiver admission with R4 component checks."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from catalog import normalized_runtime_config

PACKET = Path(__file__).resolve().parent
R3 = PACKET.parent / "large_teacher_r3_muon30"
sys.path.insert(0, str(R3))
import admission as base  # noqa: E402 - reuse the frozen R3 admission implementation

NEW_TESTS = (
    "test_sequence_inner_splits",
    "test_sequence_cache_ensemble",
    "test_high_compute_campaign",
    "test_high_compute_runner",
    "test_sequence_precision",
    "test_sequence_attention_backends",
    "test_sequence_compile",
    "test_sequence_precision_cuda",
)
PRECISION_MODE = "bf16_bf16x9"
# Config and receipt objects are heterogeneous JSON values at this admission boundary.
PRECISION_IDENTITY = {
    "mode": PRECISION_MODE,
    "cublas_emulate_single_precision": "1",
    "parameter_dtype": "float32",
    "gradient_dtype": "float32",
    "optimizer_state_dtype": "float32",
    "activation_cache_dtype": "float32",
    "amp_scaler": False,
    "model_compute": "BF16 autocast plus process-wide cuBLAS single-precision emulation",
    "model_forward": "FP32 tensor interface with BF16 autocast and cuBLAS emulation",
    "loss": "FP32 after prediction and hint casts",
    "optimizer": "process-wide cuBLAS emulation may affect Muon operations",
}


def _require(condition: bool, message: str) -> None:
    """Fail closed when a profile does not match the frozen R4 runtime contract."""
    if not condition:
        raise ValueError(message)


def _check_config_contract(config: dict[str, Any], *, require_eight_ranks: bool = True) -> str:
    """Return the required attention backend for an admitted R4 cell config."""
    model = config.get("model")
    _require(isinstance(model, dict), "R4 profile config lacks a model object")
    _require(config.get("precision_mode") == PRECISION_MODE, "R4 requires bf16_bf16x9")
    _require(config.get("batch_size") == 2, "R4 freezes batch_size=2 on every rank")
    if require_eight_ranks:
        _require(
            config.get("required_world_size") == 8, "R4 production profiles require eight ranks"
        )
    else:
        _require(config.get("required_world_size") in (1, 8), "invalid R4 profile world size")
    architecture = model.get("architecture")
    if architecture == "transformer":
        _require(
            model.get("attention_backend") == "efficient", "R4 teachers require efficient attention"
        )
        _require(model.get("compile_blocks") is True, "R4 teachers require compiled blocks")
        _require(model.get("checkpoint_blocks") is False, "R4 teachers disable block checkpointing")
        return "efficient"
    if architecture == "gru":
        _require(model.get("attention_backend") == "math", "R4 GRU uses the math attention setting")
        _require(model.get("compile_blocks") is False, "R4 GRU must not compile Transformer blocks")
        return "math"
    raise ValueError("R4 supports only Transformer teachers and GRU students")


def _check_runtime_contract(config: dict[str, Any], identity: dict[str, Any]) -> None:
    """Bind explicit precision, attention, and compiler provenance for one profile."""
    backend = _check_config_contract(config, require_eight_ranks=False)
    runtime = identity.get("runtime")
    _require(isinstance(runtime, dict), "profile lacks runtime identity")
    precision = dict(PRECISION_IDENTITY) | {"attention_backend": backend}
    _require(runtime.get("precision") == precision, "profile precision identity differs from R4")
    _require(
        runtime.get("attention_backend") == backend, "profile attention backend differs from R4"
    )
    compiled = config["model"]["architecture"] == "transformer"
    compiler = {
        "enabled": compiled,
        "backend": "inductor" if compiled else None,
        "mode": "default" if compiled else None,
        "fullgraph": False,
        "dynamic": None,
        "recompile_limit": 64 if compiled else None,
    }
    _require(runtime.get("compiler") == compiler, "profile compiler identity differs from R4")
    _require(runtime.get("matmul_precision") == "highest", "R4 requires FP32 matmul controls")
    _require(runtime.get("matmul_tf32") is False, "R4 requires TF32 matmul to remain disabled")
    _require(runtime.get("cudnn_tf32") is False, "R4 requires cuDNN TF32 to remain disabled")
    _require(runtime.get("autocast") is False, "R4 expects autocast disabled outside model forward")
    _require(runtime.get("amp_scaler") is False, "R4 does not use an AMP gradient scaler")
    _require(
        runtime.get("cublas_workspace_config") == ":4096:8",
        "R4 requires CUBLAS_WORKSPACE_CONFIG=:4096:8",
    )


def _check_profile_runtime(path: Path, config: dict[str, Any]) -> None:
    """Validate the exact precision and compiler identity recorded by a profile."""
    profile = base.read(path)
    identity = profile.get("identity")
    _require(isinstance(identity, dict), f"profile lacks an identity: {path}")
    _check_runtime_contract(config, identity)


def accept_checks(directory: Path) -> None:
    """Require execution of every R4 module before extending the R3 receipt."""
    cases = list(ET.parse(directory / "pytest.xml").getroot().iter("testcase"))
    for name in NEW_TESTS:
        if not any(name in case.get("classname", "").split(".") for case in cases):
            raise ValueError(f"R4 receiver checks did not execute {name}")
    base.accept_checks(directory)
    receipt = base.read(directory / "acceptance.json")
    paths = [base.ROOT / "tests" / f"{name}.py" for name in NEW_TESTS]
    paths += [
        base.ROOT / "scripts" / name
        for name in ("prepare_sequence_inner_split.py", "merge_sequence_teacher_caches.py")
    ]
    paths += [p for p in PACKET.rglob("*") if p.suffix in {".py", ".sh", ".json"}]
    paths += [p for p in R3.iterdir() if p.suffix in {".py", ".sh"}]
    receipt["files"].update({str(path.resolve()): base.file_hash(path) for path in paths})
    receipt["extension"] = "wnn.high-compute-checks.v1"
    receipt["limits"] = "Receiver correctness checks only; no fit, quality or resource approval."
    base.write_new(directory / "r4_acceptance.json", receipt)


def main() -> None:
    """Run explicit correctness or per-cell profile admission on the receiver."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    checks = sub.add_parser("checks")
    checks.add_argument("--directory", type=Path, required=True)
    profile = sub.add_parser("profile")
    for name in ("directory", "checks", "config"):
        profile.add_argument(f"--{name}", type=Path, required=True)
    profile.add_argument("--reviewer", required=True)
    profile.add_argument("--host-ram-gib", type=float, required=True)
    args = parser.parse_args()
    if args.action == "checks":
        accept_checks(args.directory)
    else:
        if base.read(args.checks).get("extension") != "wnn.high-compute-checks.v1":
            raise ValueError("R4 requires its complete receiver acceptance receipt")
        raw_config = base.read(args.config)
        backend = _check_config_contract(raw_config)
        normalized_config = normalized_runtime_config(raw_config)
        _require(
            normalized_config["model"].get("attention_backend") == backend,
            "normalized profile config changes the R4 attention backend",
        )
        one_config = normalized_config | {"required_world_size": 1}
        peak_config = normalized_config | {"warmup_fraction": 0.0}
        _check_profile_runtime(args.directory / "profile-one/profile_receipt.json", one_config)
        _check_profile_runtime(
            args.directory / "profile-eight/profile_receipt.json", normalized_config
        )
        _check_profile_runtime(args.directory / "profile-peak/profile_receipt.json", peak_config)
        normalized_path = args.directory / "admission-normalized-config.json"
        binding_path = args.directory / "r4_admission_binding.json"
        if any(
            path.exists()
            for path in (normalized_path, args.directory / "approved_profile.json", binding_path)
        ):
            raise FileExistsError(
                "R4 admission evidence is immutable; begin a new attempt to retry"
            )
        with normalized_path.open("x", encoding="utf-8") as stream:
            json.dump(normalized_config, stream, indent=2, allow_nan=False)
            stream.write("\n")
        normalized_args = argparse.Namespace(**(vars(args) | {"config": normalized_path}))
        base.admit(normalized_args)
        binding = {
            "schema": "wnn.high-compute-r4-admission-binding.v1",
            "raw_config_path": str(args.config.resolve()),
            "raw_config_sha256": base.file_hash(args.config),
            "normalized_config_path": str(normalized_path.resolve()),
            "normalized_config_sha256": base.file_hash(normalized_path),
            "approved_profile_path": str((args.directory / "approved_profile.json").resolve()),
            "approved_profile_sha256": base.file_hash(args.directory / "approved_profile.json"),
        }
        base.write_new(binding_path, binding)


if __name__ == "__main__":
    main()
