"""Validate one terminal, production-geometry B16 student prediction profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

MAX_JSON_BYTES = 256_000
MAX_LOG_BYTES = 128_000
MAX_CHECKPOINT_BYTES = 64_000_000
PROFILE_BATCH_SIZE = 16
SOURCE_SEQUENCE_COUNT = 1873
SEQUENCE_ROWS = 20_000
OUTPUT_COLUMNS = 2
PROFILE_OUTPUT_BYTES = PROFILE_BATCH_SIZE * SEQUENCE_ROWS * OUTPUT_COLUMNS * 4 + 128
OBSERVED_VRAM_LIMIT_BYTES = 5_000_000_000
CUDA_ALLOCATOR_LIMIT_BYTES = 3_500_000_000
PROFILE_HOST_RESERVATION_BYTES = 5_000_000_000
REQUIRED_OBSERVERS = {
    "attempt_memory_events_final",
    "attempt_memory_peak_final",
    "attempt_systemd_runtime",
    "nvidia_smi_runtime",
    "nvidia_smi_final",
    "parent_hierarchical_memory_events_final",
}
OOM_FIELDS = ("oom", "oom_kill", "oom_group_kill")
ALLOWED_SHAPES = ((256, 2), (256, 1), (192, 2), (512, 1), (384, 2), (640, 1), (512, 2), (448, 3))
EVENT_DELTA_FIELDS = (
    "memory_event_deltas",
    "final_memory_event_deltas",
    "parent_memory_event_deltas",
    "final_parent_memory_event_deltas",
    "parent_hierarchical_memory_event_deltas",
    "final_parent_hierarchical_memory_event_deltas",
)


def sha256_file(path: Path, maximum_bytes: int | None = None) -> str:
    """Hash a file in bounded blocks, optionally enforcing a maximum size."""
    size = path.stat().st_size
    if size <= 0 or (maximum_bytes is not None and size > maximum_bytes):
        raise ValueError(f"{path} is empty or exceeds its {maximum_bytes}-byte hash bound")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    """Read a small JSON object only after checking its exact on-disk size."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{path} exceeds its {maximum_bytes}-byte JSON read bound")
    with path.open("rb") as source:
        raw = source.read(maximum_bytes + 1)
    if len(raw) != size:
        raise ValueError(f"{path} changed while being read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require_equal(actual: object, expected: object, label: str) -> None:
    """Fail closed when one value differs from its frozen identity."""
    if actual != expected:
        raise ValueError(f"{label} differs: expected {expected!r}, got {actual!r}")


def profile_record_from_log(path: Path) -> dict[str, Any]:
    """Find exactly one resource profile record in the bounded worker log."""
    size = path.stat().st_size
    if size <= 0 or size > MAX_LOG_BYTES:
        raise ValueError(f"worker log exceeds its {MAX_LOG_BYTES}-byte read bound")
    with path.open("rb") as source:
        raw = source.read(MAX_LOG_BYTES + 1)
    if len(raw) != size:
        raise ValueError("worker log changed while being read")
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if b"STUDENT_FRONTIER_PREDICTION_RESOURCE_PROFILE" not in line:
            continue
        candidate = json.loads(line)
        if isinstance(candidate, dict) and candidate.get("record") == (
            "STUDENT_FRONTIER_PREDICTION_RESOURCE_PROFILE"
        ):
            records.append(candidate)
    if len(records) != 1:
        raise ValueError(f"worker log has {len(records)} prediction-profile records; expected one")
    return records[0]


def validate_controller_attempt(
    attempt: dict[str, Any],
    attempt_id: str,
    contract_path: Path,
    contract_sha256: str,
    expected_command: list[str],
) -> dict[str, Any]:
    """Require terminal success, sound observers, no OOM and bounded resources."""
    for key, expected in {
        "id": attempt_id,
        "kind": "gpu",
        "status": "finished",
        "phase": "terminal",
        "exit_status": 0,
        "failure": None,
        "contract": str(contract_path.resolve()),
        "contract_sha256": contract_sha256,
        "command": expected_command,
    }.items():
        require_equal(attempt.get(key), expected, f"controller attempt {key}")
    reservation = attempt.get("reservation")
    if not isinstance(reservation, dict):
        raise ValueError("controller receipt lacks its resource reservation")
    require_equal(reservation.get("ram_bytes"), PROFILE_HOST_RESERVATION_BYTES, "profile host reserve")
    require_equal(reservation.get("cpus"), 1, "profile CPU reserve")
    peak_host = attempt.get("peak_ram_bytes")
    if type(peak_host) is not int or not 0 < peak_host <= PROFILE_HOST_RESERVATION_BYTES:
        raise ValueError("profile worker host peak is missing or exceeds its admission reserve")
    peak_vram = attempt.get("peak_observed_vram_bytes")
    if type(peak_vram) is not int or not 0 < peak_vram < OBSERVED_VRAM_LIMIT_BYTES:
        raise ValueError("profile observed VRAM is missing or reaches the frozen 5 GB ceiling")
    observer_status = attempt.get("observer_status")
    if not isinstance(observer_status, dict):
        raise ValueError("controller receipt lacks observer statuses")
    if any(status != "ok" for status in observer_status.values()):
        raise ValueError("controller receipt contains an unknown or failed observer status")
    for name in REQUIRED_OBSERVERS:
        require_equal(observer_status.get(name), "ok", f"observer {name}")
    observer_events = attempt.get("observer_events")
    if not isinstance(observer_events, list) or observer_events:
        raise ValueError("controller receipt has unknown or failed observer events")
    for field in EVENT_DELTA_FIELDS:
        deltas = attempt.get(field)
        if not isinstance(deltas, dict):
            raise ValueError(f"controller receipt lacks {field}")
        for name in OOM_FIELDS:
            require_equal(deltas.get(name), 0, f"{field}.{name}")
    log_path = Path(str(attempt.get("log", "")))
    if not log_path.is_file():
        raise ValueError("controller receipt points to a missing worker log")
    return {"peak_host_bytes": peak_host, "peak_observed_vram_bytes": peak_vram, "log": log_path}


def validate_profile_record(record: dict[str, Any], model: dict[str, Any], output_path: Path) -> None:
    """Check successful finite inference at the exact production batch geometry."""
    expected = {
        "run": model["run"],
        "split": "valid",
        "profile_only": True,
        "production_batch_size": PROFILE_BATCH_SIZE,
        "sequences_exported": PROFILE_BATCH_SIZE,
        "full_source_sequences": SOURCE_SEQUENCE_COUNT,
        "sequence_rows": SEQUENCE_ROWS,
        "finite_outputs": True,
        "prediction_path": str(output_path),
        "output_bytes": PROFILE_OUTPUT_BYTES,
    }
    for key, value in expected.items():
        require_equal(record.get(key), value, f"profile record {key}")
    allocated = record.get("peak_allocated_bytes")
    reserved = record.get("peak_reserved_bytes")
    elapsed = record.get("elapsed_seconds")
    if type(allocated) is not int or allocated <= 0:
        raise ValueError("profile record lacks a positive allocated CUDA peak")
    if type(reserved) is not int or not allocated <= reserved <= CUDA_ALLOCATOR_LIMIT_BYTES:
        raise ValueError("profile record reserved CUDA peak is invalid or exceeds its cap")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or elapsed <= 0:
        raise ValueError("profile record elapsed time is missing or nonfinite")


def validate_prediction_profile(
    root: Path,
    state: Path,
    profile_contract_path: Path,
    expected_contract_sha256: str,
    attempt_id: str,
    run: str,
    hidden: int,
    layers: int,
    output_name: str,
    fit_receipt_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Bind one profiled exporter execution to its fit, sources and first 16 rows."""
    contract = read_json(profile_contract_path)
    contract_sha256 = sha256_file(profile_contract_path, MAX_JSON_BYTES)
    require_equal(contract_sha256, expected_contract_sha256, "prediction-profile contract SHA256")
    require_equal(contract.get("id"), "student-frontier-f0-seed3-b16-prediction-profiles-r2", "profile contract id")
    require_equal(contract.get("status"), "prepared_for_root_admission", "profile contract status")
    require_equal(contract.get("project_root"), str(root), "profile project root")
    require_equal(contract.get("runtime"), str(state), "profile runtime path")

    models = contract.get("models")
    if not isinstance(models, list):
        raise ValueError("profile contract lacks its model allowlist")
    if [
        (item.get("hidden"), item.get("layers"))
        for item in models
        if isinstance(item, dict)
    ] != list(ALLOWED_SHAPES):
        raise ValueError("profile contract model list differs from the exact admitted shape order")
    matches = [
        model for model in models
        if isinstance(model, dict)
        and (model.get("run"), model.get("hidden"), model.get("layers")) == (run, hidden, layers)
    ]
    if len(matches) != 1:
        raise ValueError("run and shape are absent or duplicated in the frozen profile allowlist")
    model = matches[0]
    require_equal(model.get("profile_attempt_id"), attempt_id, "profile attempt identity")
    require_equal(model.get("output_name"), output_name, "profile output identity")
    require_equal(model.get("fit_receipt_path"), str(fit_receipt_path), "fit-validation receipt path")
    require_equal(
        model.get("output_sidecar_path"),
        str(Path(str(model["output_path"])).with_suffix(".json")),
        "profile sidecar path",
    )

    prediction = contract.get("prediction")
    if not isinstance(prediction, dict):
        raise ValueError("profile contract lacks prediction geometry")
    for key, value in {
        "batch_size": PROFILE_BATCH_SIZE,
        "sequence_rows": SEQUENCE_ROWS,
        "output_columns": OUTPUT_COLUMNS,
        "precision": "fp32",
        "chunk_rows": 4000,
        "feature_only": True,
        "state_reset": "new zero state per sequence; state carried through inference chunks",
    }.items():
        require_equal(prediction.get(key), value, f"prediction geometry {key}")

    source = contract.get("source")
    if not isinstance(source, dict):
        raise ValueError("profile contract lacks pinned inference sources")
    for field in ("upstream_predictor", "profiled_predictor_copy", "trainer_copy"):
        path = Path(str(source[field]))
        if sha256_file(path) != source[f"{field}_sha256"]:
            raise ValueError(f"pinned inference source changed: {path}")
    for relative_path, expected in source["model_source_sha256"].items():
        path = Path(str(source["source_worktree"])) / relative_path
        if sha256_file(path) != expected:
            raise ValueError(f"pinned model source changed: {path}")
    fit_contract_path = Path(str(source["fit_consumer_contract"]))
    require_equal(
        contract.get("consumer_contract"),
        str(fit_contract_path),
        "fit consumer contract path",
    )
    if sha256_file(fit_contract_path, MAX_JSON_BYTES) != source["fit_consumer_contract_sha256"]:
        raise ValueError("frozen fit consumer contract changed")
    require_equal(
        contract.get("consumer_contract_sha256"),
        source["fit_consumer_contract_sha256"],
        "fit consumer contract SHA256 pin",
    )
    fit_contract = read_json(fit_contract_path)
    fit_source_root = Path(str(fit_contract["source_root"]))
    require_equal(str(fit_source_root), source["fit_source_worktree"], "fit source worktree")
    if fit_contract.get("source_sha256") != source["fit_source_sha256"]:
        raise ValueError("profile contract fit-source list differs from the fit consumer contract")
    require_equal(
        source["trainer_copy_sha256"],
        source["fit_source_sha256"].get("scripts/train_rnn.py"),
        "copied trainer source SHA256",
    )
    for relative_path, expected in source["fit_source_sha256"].items():
        path = fit_source_root / relative_path
        if sha256_file(path) != expected:
            raise ValueError(f"pinned fit source changed: {path}")
        if relative_path != "scripts/train_rnn.py":
            require_equal(
                source["model_source_sha256"].get(relative_path),
                expected,
                f"predictor/fit source parity {relative_path}",
            )

    data = contract.get("data")
    if not isinstance(data, dict):
        raise ValueError("profile contract lacks data identities")
    split_path = Path(data["split_manifest_path"])
    if sha256_file(split_path, 2_000_000) != data["split_manifest_sha256"]:
        raise ValueError("fold-0 split manifest changed")
    targets_path = Path(data["valid_targets_path"])
    if targets_path.stat().st_size != data["valid_targets_bytes"]:
        raise ValueError("validation-target archive byte size changed")
    grader_manifest_path = Path(str(contract["grader_inputs_manifest"]))
    if sha256_file(grader_manifest_path, MAX_JSON_BYTES) != contract["grader_inputs_manifest_sha256"]:
        raise ValueError("frozen grader-input manifest changed")
    grader_manifest = read_json(grader_manifest_path)
    grader_target = grader_manifest["grader"]["targets"]
    require_equal(grader_target.get("path"), str(targets_path), "grader target path")
    require_equal(grader_target.get("sha256"), data["valid_targets_sha256"], "grader target SHA256 pin")
    require_equal(grader_target.get("bytes"), data["valid_targets_bytes"], "grader target byte pin")
    require_equal(
        grader_manifest["split"].get("sha256"),
        data["split_manifest_sha256"],
        "grader split SHA256 pin",
    )
    require_equal(
        grader_manifest["prediction"].get("upstream_path"),
        source["upstream_predictor"],
        "grader predictor source path",
    )
    require_equal(
        grader_manifest["prediction"].get("upstream_sha256"),
        source["upstream_predictor_sha256"],
        "grader predictor source SHA256",
    )
    with np.load(targets_path, allow_pickle=False) as target_archive:
        target_groups = np.asarray(target_archive["row_group"], dtype=np.int64)
    if target_groups.shape != (SOURCE_SEQUENCE_COUNT,):
        raise ValueError("validation target row-group order has changed shape")
    expected_groups = target_groups[:PROFILE_BATCH_SIZE].tolist()

    fit_receipt = read_json(fit_receipt_path)
    require_equal(fit_receipt.get("run"), run, "fit receipt run")
    require_equal(fit_receipt.get("shape"), [hidden, layers], "fit receipt shape")
    require_equal(fit_receipt.get("epoch"), 0, "fit receipt terminal epoch")
    checkpoint_path = Path(str(fit_receipt.get("checkpoint_path", "")))
    checkpoint_sha256 = fit_receipt.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str):
        raise ValueError("fit receipt lacks terminal checkpoint SHA256")
    require_equal(sha256_file(checkpoint_path, MAX_CHECKPOINT_BYTES), checkpoint_sha256, "terminal checkpoint SHA256")
    expected_checkpoint = root / "artifacts" / "runs" / run / "model_ep0.pt"
    require_equal(checkpoint_path.resolve(), expected_checkpoint.resolve(), "profile terminal checkpoint path")

    attempt_path = state / "attempts" / f"{attempt_id}.json"
    attempt = read_json(attempt_path)
    controller = validate_controller_attempt(
        attempt,
        attempt_id,
        profile_contract_path,
        contract_sha256,
        model["worker_command"],
    )
    output_path = Path(str(model["output_path"]))
    log_path = controller["log"]
    record = profile_record_from_log(log_path)
    validate_profile_record(record, model, output_path)

    sidecar_path = output_path.with_suffix(".json")
    output_sidecar = read_json(sidecar_path)
    expected_sidecar = {
        "run": run,
        "split": "valid",
        "epoch": 0,
        "checkpoint_sha256": checkpoint_sha256,
        "inference_precision": "fp32",
        "row_groups": expected_groups,
        "campaign_export_scope": "profile-first-production-batch-feature-only",
        "campaign_profile_only": True,
        "campaign_full_source_group_count": SOURCE_SEQUENCE_COUNT,
        "campaign_split_manifest_sha256": data["split_manifest_sha256"],
        "feature_only": True,
    }
    for key, value in expected_sidecar.items():
        require_equal(output_sidecar.get(key), value, f"prediction sidecar {key}")
    require_equal(output_sidecar.get("prediction_sha256"), sha256_file(output_path, PROFILE_OUTPUT_BYTES), "profile prediction SHA256")
    require_equal(output_path.stat().st_size, PROFILE_OUTPUT_BYTES, "profile prediction byte size")
    predictions = np.load(output_path, mmap_mode="r", allow_pickle=False)
    if predictions.shape != (PROFILE_BATCH_SIZE, SEQUENCE_ROWS, OUTPUT_COLUMNS):
        raise ValueError("profile prediction tensor has the wrong batch/sequence/output shape")
    if predictions.dtype != np.dtype(np.float32) or not np.isfinite(predictions).all():
        raise ValueError("profile predictions are not finite float32 values")
    del predictions

    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite profile-validation receipt: {receipt_path}")
    receipt = {
        "format": "wnn.student-frontier-prediction-profile-validation",
        "version": 1,
        "profile_contract_sha256": contract_sha256,
        "attempt_id": attempt_id,
        "run": run,
        "shape": [hidden, layers],
        "epoch": 0,
        "checkpoint_sha256": checkpoint_sha256,
        "batch_size": PROFILE_BATCH_SIZE,
        "sequence_rows": SEQUENCE_ROWS,
        "precision": "fp32",
        "chunk_rows": 4000,
        "row_groups": expected_groups,
        "row_groups_sha256": hashlib.sha256(np.asarray(expected_groups, dtype="<i8").tobytes()).hexdigest(),
        "prediction_path": str(output_path),
        "prediction_sha256": output_sidecar["prediction_sha256"],
        "controller_status": attempt["status"],
        "controller_exit_status": attempt["exit_status"],
        "peak_host_bytes": controller["peak_host_bytes"],
        "peak_observed_vram_bytes": controller["peak_observed_vram_bytes"],
        "peak_cuda_allocated_bytes": record["peak_allocated_bytes"],
        "peak_cuda_reserved_bytes": record["peak_reserved_bytes"],
        "oom_counters_zero": True,
        "all_outputs_finite": True,
        "validation": "passed; resource-only first B16 inference batch; no quality evaluation",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    """Validate a completed profile and emit the receipt required by export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--profile-contract", type=Path, required=True)
    parser.add_argument("--profile-contract-sha256", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate_prediction_profile(
        args.root.resolve(strict=True),
        args.state.resolve(strict=True),
        args.profile_contract.resolve(strict=True),
        args.profile_contract_sha256,
        args.attempt_id,
        args.run,
        args.hidden,
        args.layers,
        args.output_name,
        args.fit_receipt.resolve(strict=True),
        args.out.resolve(),
    )
    print(json.dumps({"run": receipt["run"], "validation": receipt["validation"]}))


if __name__ == "__main__":
    main()
