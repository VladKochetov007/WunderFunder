"""Validate the exact H390 full-valid feature-export contract."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pins import (
    CANONICAL_SPLIT_SHA256,
    CHECKPOINT_SHA256,
    CONTROL_RUN,
    OLD_SPLIT_SHA256,
    OVERLAY_MANIFEST_SHA256,
    PREDICTOR_SHA256,
    ROW_GROUPS,
    SOURCE_MANIFEST_SHA256,
)
from source_alignment import read_pinned_json

EXPORT_PREP_REL = Path("research/campaign_2026-10-18/quality_gap_r1/export_preparation")
OUTPUT_NAME = "clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1_exportprep_r1"
CONTRACT_SCHEMA = "quality_gap.f0_strong_control_export_contract.r1"
EXPECTED_PROJECT_ROOT = Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/quality-gap-r1/export-preparation-r2/root"
)
EXPECTED_B_HOME = Path("/home/grinchenkoeg/WunderFunder")
EXPECTED_B_BASE = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
EXPECTED_RUNTIME_PYTHON = Path(
    "/home/grinchenkoeg/.local/share/uv/python/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13"
)
EXPECTED_RUNTIME_PYTHON_VERSION = "3.13.15 (main, Sep 24 2026, 17:55:50) [Clang 22.1.3 ]"
EXPECTED_RUNTIME_PYTHON_BYTES = 31_493_640
EXPECTED_RUNTIME_PYTHON_SHA256 = "20a5569a1bac8de02122347777b6cf2ae12ec1eafd42db990c6a0c5407f96b8e"
EXPECTED_RUNTIME_PACKAGES = {
    "numpy": "2.2.6",
    "polars": "1.44.2",
    "pyarrow": "19.0.1",
    "torch": "2.14.0+cu130",
}
EXPECTED_RESOURCE_ENVELOPE = {
    "guard_kind": "gpu",
    "cpus": 1,
    "reserve_bytes": 8_000_000_000,
    "allocator_bytes": 4_000_000_000,
    "gpu_free_memory_safety_reserve_bytes": 1_000_000_000,
    "runtime_limit_seconds": 7200,
    "admission_timeout_seconds": 300,
    "profile": "strong-control-export-unmeasured-r1",
    "profile_status": "UNMEASURED_FOR_THIS_CONTROL",
}
EXPECTED_UTILITY_NAMES = {
    "bindings.py",
    "capture.py",
    "launch.py",
    "prepare.py",
    "worker.py",
}
EXPECTED_UTILITY_IDENTITIES = {
    "bindings.py": (16528, "2cdc8456ea6c059b8db0c33169578250c5c1083060978fa61f941b92ab0b6cc4"),
    "prepare.py": (13166, "d2bafcca727bb3f72e444dfbf46bca68b733371030035d3286f68dd07c9ad623"),
    "launch.py": (9041, "18c96de0e8447db91648d3ce26390e6ce754c942f54b78c6dbddae61b13908f1"),
    "worker.py": (7914, "ee8029db0095e4b0f4c897c04a08ff4d555436eec91d3ac8b81709493355be9f"),
    "capture.py": (19140, "87b66ccd4be728bba72f8a9b789cceb0715263c93bfc0cc8a11cadbbf3a8ae93"),
}


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _expected_predictor_command(project_root: Path) -> list[str]:
    return [
        CONTROL_RUN,
        "valid",
        "--epoch",
        "5",
        "--output-name",
        OUTPUT_NAME,
        "--campaign-split-manifest",
        str(project_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"),
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
        "--campaign-full-valid-feature-only",
    ]


def validate_contract(path: Path, expected_sha256: str) -> tuple[dict[str, object], str]:
    """Check the exact full-valid feature-only H390 predictor contract."""
    if path.is_symlink():
        raise ValueError("export contract must be a regular file")
    contract = read_pinned_json(path, expected_sha256, "export contract")
    project = Path(str(contract.get("project_root", "")))
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != "READY_FOR_ROOT_GUARDED_GPU_EXPORT"
        or contract.get("owner") != "root"
        or not contract.get("attempt_id")
        or contract.get("contract_path") != str(path.resolve(strict=True))
        or project != EXPECTED_PROJECT_ROOT
    ):
        raise ValueError("export contract schema, status, owner, attempt, or path is invalid")
    pins = _mapping(contract.get("pins"), "contract pins")
    pinned = {
        "predictor_sha256": PREDICTOR_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "old_split_sha256": OLD_SPLIT_SHA256,
        "canonical_split_sha256": CANONICAL_SPLIT_SHA256,
        "overlay_manifest_sha256": OVERLAY_MANIFEST_SHA256,
    }
    for key, expected in pinned.items():
        if pins.get(key) != expected:
            raise ValueError(f"export contract source pin mismatch: {key}")
    source = _mapping(contract.get("source_snapshot"), "contract source snapshot")
    if (
        source.get("path") != str(project / "frozen_source_snapshot")
        or source.get("predictor_sha256") != PREDICTOR_SHA256
        or source.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("export contract snapshot does not bind frozen H390 predictor")
    if type(source.get("file_count")) is not int or source["file_count"] < 1:
        raise ValueError("contract lacks the frozen source inventory count")
    artifacts = _mapping(contract.get("artifacts"), "contract artifacts")
    checkpoint = _mapping(artifacts.get("control_checkpoint"), "contract control checkpoint")
    if checkpoint.get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("export contract checkpoint SHA-256 mismatch")
    if contract.get("resource_envelope") != EXPECTED_RESOURCE_ENVELOPE:
        raise ValueError("export contract resource envelope differs from the reviewed bound")
    execution = _mapping(contract.get("execution"), "contract execution")
    runtime = _mapping(contract.get("runtime"), "contract runtime")
    expected_execution_dir = project / EXPORT_PREP_REL / "export_execution_r1"
    expected_campaign_root = EXPECTED_B_BASE / "source-a266b036"
    expected_execution_paths = {
        "python": str(EXPECTED_B_HOME / ".venv/bin/python"),
        "limited": str(expected_campaign_root / "scripts/limited"),
        "guard": str(expected_campaign_root / "scripts/bandit_resources.py"),
        "guard_state": str(EXPECTED_B_BASE / "runtime"),
        "guard_unit_template": "wnn-bandit-{attempt}.service",
        "physical_tmp": str(EXPECTED_B_BASE / "research_tmp"),
    }
    if contract.get("execution_dir") != str(expected_execution_dir) or any(
        execution.get(key) != value for key, value in expected_execution_paths.items()
    ):
        raise ValueError("export contract execution utilities differ from the pinned B stage")
    expected_predictor = source.get("path")
    predictor = str(Path(str(expected_predictor)) / "scripts/predict_run.py")
    if (
        execution.get("predictor") != predictor
        or Path(str(execution.get("predictor"))).is_symlink()
        or execution.get("command") != _expected_predictor_command(project)
    ):
        raise ValueError(
            "export contract command or predictor path is not the exact feature-only run"
        )
    overlay_src = project / EXPORT_PREP_REL / "compatibility_overlay_r2/src"
    frozen_src = project / "frozen_source_snapshot/src"
    if execution.get("pythonpath") != f"{overlay_src}:{frozen_src}":
        raise ValueError(
            "export contract PYTHONPATH does not put reviewed overlay before frozen source"
        )
    if (
        runtime.get("python_executable") != str(EXPECTED_RUNTIME_PYTHON)
        or runtime.get("python_version") != EXPECTED_RUNTIME_PYTHON_VERSION
        or runtime.get("python_binary_bytes") != EXPECTED_RUNTIME_PYTHON_BYTES
        or runtime.get("python_binary_sha256") != EXPECTED_RUNTIME_PYTHON_SHA256
        or runtime.get("packages") != EXPECTED_RUNTIME_PACKAGES
        or runtime.get("bandit_resources_sha256")
        != "3732ffeef79e635c8867d83e618f8d51b2fa8beffd7badf11f60412a864210f7"
        or runtime.get("limited_sha256")
        != "d9bcfc3b671af3237b607e18087fde43fc7451b6482891e4abadd8580bfb8661"
    ):
        raise ValueError(
            "contract runtime interpreter or utilities differ from pinned B-stage evidence"
        )
    utility_code = _mapping(runtime.get("export_utility_code"), "runtime export utility hashes")
    if set(utility_code) != EXPECTED_UTILITY_NAMES:
        raise ValueError("contract runtime utility inventory differs from the reviewed exporter")
    for name, identity_value in utility_code.items():
        identity = _mapping(identity_value, f"runtime utility {name}")
        if (
            type(identity.get("bytes")) is not int
            or identity["bytes"] <= 0
            or not isinstance(identity.get("sha256"), str)
            or len(identity["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in identity["sha256"])
        ):
            raise ValueError(f"runtime utility identity is invalid: {name}")
        if (identity["bytes"], identity["sha256"]) != EXPECTED_UTILITY_IDENTITIES[name]:
            raise ValueError(f"runtime utility differs from the reviewed R3 source: {name}")
    expected_output = _mapping(contract.get("expected_output"), "contract expected output")
    if expected_output != {
        "split": "valid",
        "epoch": 5,
        "row_groups": ROW_GROUPS,
        "shape": [ROW_GROUPS, 20_000, 2],
        "dtype": "<f4",
        "bytes": 299_680_128,
        "grading_run": False,
        "training_run": False,
    }:
        raise ValueError("export contract output or no-training scope differs")
    output_paths = _mapping(contract.get("outputs_create_only"), "contract output paths")
    expected_prediction = project / "cache" / f"pred_{OUTPUT_NAME}_valid.npy"
    expected_sidecar = expected_prediction.with_suffix(".json")
    if (
        Path(str(output_paths.get("prediction"))) != expected_prediction
        or Path(str(output_paths.get("sidecar"))) != expected_sidecar
        or Path(str(output_paths.get("partial")))
        != expected_prediction.with_name(expected_prediction.name + ".partial")
        or Path(str(output_paths.get("family_prediction")))
        != project / "cache" / f"allout_{OUTPUT_NAME}_valid.npy"
        or Path(str(output_paths.get("stale_seqsum")))
        != project / "cache" / f"seqsums_{OUTPUT_NAME}.npz"
        or set(output_paths)
        != {"prediction", "sidecar", "partial", "family_prediction", "stale_seqsum"}
    ):
        raise ValueError("export contract create-only outputs differ from the pinned run")
    return contract, expected_sha256


def validate_export_output_paths(
    contract: dict[str, object], prediction_path: Path, sidecar_path: Path
) -> None:
    """Require supplied export files to be the exact canonical contract outputs."""
    outputs = _mapping(contract.get("outputs_create_only"), "contract output paths")
    for supplied, key in (
        (prediction_path, "prediction"),
        (sidecar_path, "sidecar"),
    ):
        expected = Path(str(outputs[key]))
        supplied_resolved = supplied.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
        if (
            not supplied.is_absolute()
            or not expected.is_absolute()
            or supplied.is_symlink()
            or expected.is_symlink()
            or supplied.absolute() != expected.absolute()
            or supplied.absolute() != supplied_resolved
            or expected.absolute() != expected_resolved
            or supplied_resolved != expected_resolved
        ):
            raise ValueError(f"supplied {key} path differs from the exact contract output")
