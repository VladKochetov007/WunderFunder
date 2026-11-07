"""Immutable receiving identities and fail-closed compact cohort evidence checks."""

from __future__ import annotations

import json
import math
from pathlib import Path

from wnn.multistage_search import canonical_json_sha256, file_sha256, read_json

SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
SOURCE_INVENTORY_SHA = "7d3324230effb050ff083b939e26ad638e5df5c5b72ea70b0f8aed169997db4a"
FAMILIES = ("gru", "gru_local", "mamba2", "ema_tabm_style")
ORACLE_TEST = "tests/test_compact_sequence_models.py::test_mamba2_real_backend_oracle_causality_chunk_reset_and_gradients"


def source_files(path: Path, expected_sha: str = SOURCE_INVENTORY_SHA) -> dict[str, str]:
    """Read the sender-hashed inventory without trusting its own claimed digest."""
    if file_sha256(path) != expected_sha:
        raise ValueError("trusted staged source inventory digest differs")
    value = read_json(path)
    entries = value.get("files", value)
    if isinstance(entries, list):
        entries = {entry["path"]: entry["sha256"] for entry in entries}
    if not isinstance(entries, dict) or not entries:
        raise ValueError("source inventory must contain a nonempty path/digest mapping")
    if any(
        not isinstance(name, str) or not isinstance(digest, str) for name, digest in entries.items()
    ):
        raise ValueError("source inventory entry is malformed")
    return entries


def verify_source(root: Path, files: dict[str, str]) -> str:
    """Verify staged bytes and reject additional importable Python under src/scripts."""
    for name, digest in files.items():
        path = root / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("source inventory path escapes its staged root")
        if path.is_symlink() or file_sha256(path) != digest:
            raise ValueError(f"staged source changed: {name}")
    expected = {
        name for name in files if name.endswith(".py") and name.startswith(("src/", "scripts/"))
    }
    actual = {
        path.relative_to(root).as_posix()
        for folder in (root / "src", root / "scripts")
        for path in folder.rglob("*.py")
    }
    if actual != expected:
        raise ValueError("staged source has additional or missing Python files")
    trainer = {
        name: digest
        for name, digest in files.items()
        if name.startswith("src/") and name.endswith(".py")
    }
    trainer["scripts/train_sequence.py"] = files["scripts/train_sequence.py"]
    return canonical_json_sha256(trainer)


def runtime_recipe(old: dict[str, object], new: dict[str, object]) -> None:
    """Allow only the declared runtime-r3 derivative, preserving the scientific recipe."""
    expected = dict(old)
    expected["batch_size"] = 16
    expected["monitoring"] = dict(old["monitoring"]) | {
        "diagnostics_every_steps": 20,
        "training_telemetry_every_steps": 10,
        "eval_batch_size": 32,
    }
    if new != expected or new["epochs"] != 12 or new["kd_weight"] != 0 or new["hint_weight"] != 0:
        raise ValueError("runtime-r3 recipe changed outside its declared amendment")


def verify_cell(panel: dict[str, object], cell: dict[str, object]) -> dict[str, object]:
    """Check immutable recipes and canonical normalized-config identity before launch."""
    from wnn.training.sequence_config import load_config

    for key in ("old_config", "config"):
        if file_sha256(Path(cell[key])) != cell[f"{key}_sha256"]:
            raise ValueError(f"cell {key} bytes changed")
    runtime_recipe(read_json(Path(cell["old_config"])), read_json(Path(cell["config"])))
    config = load_config(Path(cell["config"]))
    if canonical_json_sha256(config) != cell["normalized_config_sha256"]:
        raise ValueError("normalized runtime recipe changed")
    if file_sha256(Path(panel["split_manifest"])) != SPLIT_SHA:
        raise ValueError("official chronological split changed")
    return config


def check_native_profile(
    panel: dict[str, object], cell: dict[str, object], receipt: dict[str, object]
) -> dict[str, object]:
    """Require twenty finite real updates with exact source/config/member/device identity."""
    if receipt.get("status") != "profile_complete" or receipt.get("updates") != 20:
        raise ValueError("profile did not complete exactly twenty successful updates")
    if receipt.get("scientific_checkpoint_written") is not False or receipt.get("errors") != []:
        raise ValueError("profile contains errors or a scientific checkpoint")
    identity = receipt["identity"]
    data = identity["data"]
    if (
        identity["config_sha256"] != cell["normalized_config_sha256"]
        or identity["source_sha256"] != panel["trainer_source_sha256"]
    ):
        raise ValueError("profile source/config identity differs from the frozen panel")
    if identity["data_sha256"] != canonical_json_sha256(data) or identity["world_size"] != 1:
        raise ValueError("profile data digest or rank count differs")
    if (
        data["split_sha256"] != SPLIT_SHA
        or data["fit_members"] != panel["fit_members"]
        or data["evaluation_members"] != panel["evaluation_members"]
    ):
        raise ValueError("profile does not use the exact train-only/full-valid membership")
    if (
        data["normalization"]["scope"] != "all_fit"
        or data["normalization"]["sequence_count"] != 10325
        or data.get("teacher") is not None
    ):
        raise ValueError("profile preprocessing/teacher ancestry differs")
    devices = identity["runtime"]["devices"]
    if (
        len(devices) != 1
        or str(devices[0]["uuid"]).removeprefix("GPU-")
        != str(cell["gpu_uuid"]).removeprefix("GPU-")
        or "L4" not in devices[0]["name"]
    ):
        raise ValueError("profile used a different physical GPU")
    return identity


def approved_profile(panel: dict[str, object], cell: dict[str, object]) -> str:
    """Require an independently ROOT-approved copy; never approve or modify a profile."""
    from wnn.training.sequence_provenance import validate_profile_receipt

    native_path, approved_path = (
        Path(cell["profile_output"]) / "profile_receipt.json",
        Path(cell["approved_profile"]),
    )
    native, approved = read_json(native_path), read_json(approved_path)
    identity = check_native_profile(panel, cell, native)
    if approved.get("approved") is not True or approved.get("reviewer") != "ROOT":
        raise ValueError("separate ROOT profile approval is missing")
    if any(approved.get(key) != value for key, value in native.items() if key != "approved"):
        raise ValueError("approved profile changed native evidence")
    return validate_profile_receipt(approved_path, identity)


def oracle_ready(panel: dict[str, object], cell: dict[str, object]) -> bool:
    """Admit genuine Mamba only with a ROOT-bound passing actual-device oracle receipt."""
    if cell["family"] != "mamba2":
        return True
    path = Path(cell["oracle_receipt"])
    if not path.exists():
        return False
    value = read_json(path)
    if (
        value.get("status") != "passed"
        or value.get("exit_code") != 0
        or value.get("test") != ORACLE_TEST
        or value.get("gpu_uuid") != cell["gpu_uuid"]
        or value.get("source_inventory_sha256") != panel["source_inventory_sha256"]
    ):
        raise ValueError("Mamba oracle does not bind this passing genuine backend/device/source")
    evidence = value["evidence"]
    if file_sha256(Path(evidence["path"])) != evidence["sha256"]:
        raise ValueError("Mamba oracle raw evidence changed")
    return True


def evaluations(output: Path, *, complete: bool = False) -> list[dict[str, object]]:
    """Validate every complete full-valid journal event; retain raw all-step journals."""
    path = output / "monitor_evaluations.jsonl"
    if not path.exists():
        if complete:
            raise ValueError("completed fit has no full-valid journal")
        return []
    events = []
    with path.open("rb") as stream:
        for line in stream:
            if len(line) > 32768:
                raise ValueError("oversized full-valid journal event")
            if not line.endswith(b"\n"):
                if complete:
                    raise ValueError("completed full-valid journal has a partial record")
                break
            event = json.loads(line)
            metrics = event["metrics"]
            if (
                event["kind"] != "evaluation"
                or metrics["valid/sequence_count"] != 1873
                or metrics["valid/selected_rows"] != 3_528_273
            ):
                raise ValueError("full-valid journal coverage differs from1873/3528273")
            if event["epoch"] != len(events) + 1:
                raise ValueError("full-valid journal epoch is missing or repeated")
            if any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in metrics.values()
            ):
                raise ValueError("full-valid journal contains nonfinite metrics")
            events.append(event)
    if complete and len(events) != 12:
        raise ValueError("fixed-final fit requires twelve complete full-valid evaluations")
    return events
