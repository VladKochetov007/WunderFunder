"""Seal a measured, user-authorized batch amendment for four PR32 fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/data/epanov_s/tabdata")
SOURCE = ROOT / "runtime/pr32-source-f1beeb4c"
COMMIT = "f1beeb4c210a9da01735e03b1ff6e27be878b873"
PACKET = (
    SOURCE
    / "research/experiment_drafts/b300-transformer-distillation-runtime/large_teacher_r3_muon30"
)
AUTHORITY_ID = "user-2026-10-24-max-throughput-batch-on-8-b300"
AMENDMENT_SCHEMA = "wnn.pr32.teacher-batch-amendment.v1"
SELECTION_SCHEMA = "wnn.pr32.batch-selection.v2"
CELLS = ("t1280_f0", "t1792_f0", "t1280_f2", "t1792_f2")
BASELINE_BATCHES = (8, 16, 32, 64)
SMALL_BATCHES = (*BASELINE_BATCHES, 96, 128)
LARGE_BATCHES = BASELINE_BATCHES
MIN_VRAM_HEADROOM_FRACTION = 0.10


def require(condition: object, message: str) -> None:
    """Enforce an admission condition even when Python runs with ``-O``."""
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash an immutable admission input."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject other top-level values."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: object) -> None:
    """Create a receipt without replacing a previous attempt."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def frozen_profile_recipe(cell: str, batch: int) -> dict[str, Any]:
    """Apply only the already-admitted runtime mode and chosen batch to R3."""
    expected = read_json(PACKET / "configs" / f"{cell}.json")
    require(
        expected["batch_size"] * expected["required_world_size"] == 16,
        f"{cell}: frozen source recipe no longer has global batch 16",
    )
    expected["batch_size"] = batch
    expected["required_world_size"] = 2
    expected["precision_mode"] = "bf16_bf16x9"
    expected["model"].update(
        checkpoint_blocks=False,
        attention_backend="efficient",
        compile_blocks=True,
    )
    return expected


def _finite_json(value: object, where: str) -> None:
    """Reject NaN and infinity anywhere in an update record."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        require(math.isfinite(value), f"{where}: nonfinite numeric value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, f"{where}.{key}")
        return
    raise TypeError(f"{where}: unsupported JSON value {type(value).__name__}")


def _warmed_batch_metrics(profile_dir: Path) -> dict[str, float]:
    """Recompute throughput and rank-max timing from all 12 update logs."""
    per_rank: list[dict[int, dict[str, Any]]] = []
    for rank in range(2):
        path = profile_dir / f"rank{rank}.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"{path}: row {index} is not a JSON object")
            _finite_json(row, f"{path}:{index + 1}")
        if any(type(row.get("step")) is not int for row in rows):
            raise ValueError(f"{path}: every profile step must be an integer")
        selected = {row["step"]: row for row in rows if 1 <= row["step"] <= 12}
        if set(selected) != set(range(1, 13)) or len(rows) != 12:
            raise ValueError(f"{path}: expected exactly the 12 profile updates")
        per_rank.append(selected)

    step_walls: list[float] = []
    rates: list[float] = []
    for step in range(6, 13):
        rows = [per_rank[rank][step] for rank in range(2)]
        forwarded = []
        for rank, row in enumerate(rows):
            if row.get("rank") != rank:
                raise ValueError(f"rank{rank}, step {step}: rank identity mismatch")
            for field in (
                "total_loss",
                "gradient_norm_preclip",
                "learning_rate",
                "optimizer_seconds",
            ):
                if field not in row:
                    raise ValueError(f"rank{rank}, step {step}: missing {field}")
            count = row.get("forwarded_rows")
            wall = row.get("window_seconds")
            if type(count) is not int or count <= 0:
                raise ValueError(f"rank{rank}, step {step}: invalid forwarded row count")
            if isinstance(wall, bool) or not isinstance(wall, (int, float)) or wall <= 0:
                raise ValueError(f"rank{rank}, step {step}: invalid update duration")
            forwarded.append(count)
        if forwarded[0] != forwarded[1]:
            raise ValueError(f"step {step}: DDP ranks disagree on global row count")
        wall = max(float(row["window_seconds"]) for row in rows)
        step_walls.append(wall)
        rates.append(forwarded[0] / wall)
    return {
        "median_rows_per_second": float(statistics.median(rates)),
        "median_rankmax_seconds": float(statistics.median(step_walls)),
    }


def _same_metric(actual: object, expected: float, label: str) -> None:
    """Require receipt metrics to match the raw profile logs within serialization noise."""
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or not math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-6)
    ):
        raise ValueError(f"selection {label} differs from recomputed profile evidence")


def _validate_profile_command(argv: list[str], *, expected_output: Path, label: str) -> None:
    """Bind the measured output to the pinned canonical trainer invocation."""
    entrypoint = (SOURCE / "scripts/train_sequence.py").resolve()
    require(
        str(entrypoint) in argv,
        f"{label}: profile command does not use the frozen trainer",
    )
    try:
        data_root = Path(argv[argv.index("--data-root") + 1]).resolve()
        output = Path(argv[argv.index("--output") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError(f"{label}: profile command lacks data/output binding") from error
    require(data_root == (ROOT / "data").resolve(), f"{label}: data root mismatch")
    require(output == expected_output.resolve(), f"{label}: output directory mismatch")


def _validate_profile_environment(operator: dict[str, Any], label: str) -> None:
    """Require the original precision and deterministic cuBLAS environment."""
    environment = operator.get("environment")
    require(isinstance(environment, dict), f"{label}: operator environment is missing")
    require(
        environment.get("CUBLAS_EMULATE_SINGLE_PRECISION") == "1"
        and environment.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        f"{label}: precision/determinism environment differs from the frozen profile",
    )


def validate_selection(
    selection: dict[str, Any],
    *,
    selected_small: int,
    selected_large: int,
    expected_source_identity: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Verify all sweep evidence and independently recompute its per-width choice."""
    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_provenance import canonical_sha256

    require(selection.get("schema") == SELECTION_SCHEMA, "selection schema mismatch")
    require(selection.get("source_commit") == COMMIT, "selection source commit mismatch")
    require(selection.get("engineering_only") is True, "selection must be engineering-only")
    require(
        selection.get("quality_equivalence_claim") is False,
        "selection must explicitly disclaim quality equivalence",
    )
    expected_selected = {"t1280": selected_small, "t1792": selected_large}
    require(
        selection.get("selected_per_rank_batch") == expected_selected,
        "selected batches differ from the reviewed selection receipt",
    )
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 8:
        raise ValueError("selection must bind the eight baseline width-by-batch candidates")

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("each selection candidate must be an object")
        model = candidate.get("model")
        batch = candidate.get("batch_size")
        if model not in {"t1280", "t1792"}:
            raise ValueError("candidate model must be t1280 or t1792")
        allowed = SMALL_BATCHES if model == "t1280" else LARGE_BATCHES
        if type(batch) is not int or batch not in allowed:
            raise ValueError(f"{model}: candidate batch size is not an approved profile size")
        key = (model, batch)
        if key in by_key:
            raise ValueError(f"duplicate batch selection candidate: {key}")
        by_key[key] = candidate

    required_keys = {
        (model, batch)
        for model, batches in (("t1280", BASELINE_BATCHES), ("t1792", BASELINE_BATCHES))
        for batch in batches
    }
    require(required_keys.issubset(by_key), "selection omits a baseline batch candidate")
    unexpected_keys = set(by_key) - required_keys
    require(
        unexpected_keys.issubset({("t1280", 96), ("t1280", 128)}),
        "selection contains an unsupported extension candidate",
    )
    measured: dict[str, list[dict[str, Any]]] = {"t1280": [], "t1792": []}
    for (model, batch), candidate in by_key.items():
        label = f"{model}/B{batch}"
        if "status" in candidate:
            require(
                candidate.get("status") in {"complete", "complete_low_headroom"},
                f"{label}: profile did not complete",
            )
        for field in ("profile_receipt", "operator_receipt"):
            path_value = candidate.get(field)
            if not isinstance(path_value, str):
                raise TypeError(f"{label}: {field} must be an absolute path")
            path = Path(path_value)
            if not path.is_absolute() or path.is_symlink() or not path.is_file():
                raise ValueError(f"{label}: {field} must name a regular absolute file")
            if ROOT / "runs" not in path.resolve().parents:
                raise ValueError(f"{label}: {field} must be under the owned runs root")
        profile_path = Path(candidate["profile_receipt"]).resolve(strict=True)
        operator_path = Path(candidate["operator_receipt"]).resolve(strict=True)
        profile_hash = sha(profile_path)
        operator_hash = sha(operator_path)
        require(
            candidate.get("profile_sha256") == profile_hash,
            f"{label}: profile receipt digest mismatch",
        )
        require(
            candidate.get("operator_sha256") == operator_hash,
            f"{label}: operator receipt digest mismatch",
        )
        if profile_path.name != "profile_receipt.json":
            raise ValueError(f"{label}: unexpected profile receipt filename")
        cell_dir = profile_path.parent.parent
        if operator_path.parent != cell_dir:
            raise ValueError(f"{label}: operator and profile are from different cells")

        operator = read_json(operator_path)
        base = f"{model}_f0"
        require(
            operator.get("cell") == cell_dir.name and isinstance(operator.get("cell"), str),
            f"{label}: initial-round operator cell mismatch",
        )
        require(
            operator.get("global_batch_sequences") == 2 * batch,
            f"{label}: global batch mismatch",
        )
        require(
            operator.get("engineering_only") is True,
            f"{label}: operator is not engineering-only",
        )
        require(operator.get("source_commit") == COMMIT, f"{label}: source commit mismatch")
        argv = operator.get("argv")
        if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
            raise TypeError(f"{label}: operator argv is missing or malformed")
        _validate_profile_command(argv, expected_output=profile_path.parent, label=label)
        _validate_profile_environment(operator, label)
        try:
            config_path = Path(argv[argv.index("--config") + 1]).resolve(strict=True)
            split_path = Path(argv[argv.index("--split-manifest") + 1]).resolve(strict=True)
        except (ValueError, IndexError) as error:
            raise ValueError(f"{label}: operator argv lacks config/split bindings") from error
        require(config_path.parent == cell_dir, f"{label}: config belongs to another cell")
        raw_config = read_json(config_path)
        require(
            raw_config == frozen_profile_recipe(base, batch),
            f"{label}: profile recipe changed",
        )
        config_file_hash = operator.get("config_file_sha256", operator.get("config_sha256"))
        require(
            sha(config_path) == config_file_hash,
            f"{label}: config SHA mismatch",
        )
        require(
            raw_config.get("required_world_size") == 2 and raw_config.get("batch_size") == batch,
            f"{label}: profile config batch/world size mismatch",
        )
        require(
            "--profile-updates" in argv and argv[argv.index("--profile-updates") + 1] == "12",
            f"{label}: initial-round command did not request 12 updates",
        )
        require(
            "--nproc-per-node=2" in argv,
            f"{label}: initial-round command was not W2",
        )
        require(
            split_path == (ROOT / "splits/fold0.json").resolve(),
            f"{label}: sweep must use F0",
        )

        profile = read_json(profile_path)
        _finite_json(profile, f"{label} profile receipt")
        identity = profile.get("identity")
        require(
            profile.get("receipt_schema") == "wnn.sequence-profile.v1",
            f"{label}: profile schema mismatch",
        )
        require(profile.get("status") == "profile_complete", f"{label}: profile incomplete")
        require(profile.get("approved") is False, f"{label}: profile approval was pre-set")
        require(
            profile.get("scientific_checkpoint_written") is False,
            f"{label}: profile wrote a checkpoint",
        )
        require(
            profile.get("updates") == 12
            and profile.get("errors") == []
            and profile.get("error_count") == 0,
            f"{label}: profile updates or finite-state check failed",
        )
        require(isinstance(identity, dict), f"{label}: profile identity is missing")
        require(
            identity.get("source_sha256") == expected_source_identity["sha256"]
            and identity.get("source_files") == expected_source_identity["files"],
            f"{label}: profile source identity differs from the frozen source",
        )
        canonical_config_hash = canonical_sha256(load_config(config_path))
        require(
            identity.get("config_sha256") == canonical_config_hash,
            f"{label}: profile normalized config mismatch",
        )
        require(identity.get("world_size") == 2, f"{label}: identity world size mismatch")
        data_identity = identity.get("data", {})
        require(
            data_identity.get("fold") == 0 and data_identity.get("split_sha256") == sha(split_path),
            f"{label}: profile must bind F0 split",
        )
        runtime = identity.get("runtime", {})
        require(runtime.get("world_size") == 2, f"{label}: runtime world size mismatch")
        devices = runtime.get("devices", [])
        gpu_rows = operator.get("gpus")
        op_gpus = (
            [row[1] for row in gpu_rows if isinstance(row, list) and len(row) > 1]
            if isinstance(gpu_rows, list)
            else []
        )
        require(
            isinstance(op_gpus, list) and len(op_gpus) == 2,
            f"{label}: operator GPU binding missing",
        )
        profile_gpus = [
            str(device.get("uuid", "")).removeprefix("GPU-").lower() for device in devices
        ]
        require(
            profile_gpus == [str(uuid).removeprefix("GPU-").lower() for uuid in op_gpus],
            f"{label}: profile GPU identity mismatch",
        )

        require(
            operator.get("exit_code") == 0 and operator.get("expired") is False,
            f"{label}: initial-round profile failed or timed out",
        )
        if "status" in candidate:
            require(
                candidate.get("status") in {"complete", "complete_low_headroom"},
                f"{label}: selection candidate status is not complete",
            )
        metrics = _warmed_batch_metrics(profile_path.parent)
        _same_metric(
            candidate.get("median_rows_per_second"),
            metrics["median_rows_per_second"],
            f"{label} median_rows_per_second",
        )
        _same_metric(
            candidate.get("median_rankmax_seconds"),
            metrics["median_rankmax_seconds"],
            f"{label} median_rankmax_seconds",
        )
        memory = profile.get("memory_by_rank")
        if not isinstance(memory, list) or len(memory) != 2:
            raise ValueError(f"{label}: per-rank memory evidence missing")
        peak_reserved = max(int(row["peak_reserved_bytes"]) for row in memory)
        total_values = {int(row["device_total_memory_bytes"]) for row in memory}
        require(len(total_values) == 1, f"{label}: GPU capacity differs across ranks")
        total_memory = total_values.pop()
        headroom_eligible = peak_reserved <= (1 - MIN_VRAM_HEADROOM_FRACTION) * total_memory
        require(
            type(candidate.get("eligible")) is bool and candidate["eligible"] is headroom_eligible,
            f"{label}: selection headroom eligibility differs from measured memory",
        )
        if (
            type(candidate.get("peak_reserved_bytes")) is not int
            or candidate["peak_reserved_bytes"] != peak_reserved
        ):
            raise ValueError(f"{label}: candidate peak VRAM differs from profile receipt")
        if (
            type(candidate.get("device_total_memory_bytes")) is not int
            or candidate["device_total_memory_bytes"] != total_memory
        ):
            raise ValueError(f"{label}: candidate device capacity differs from profile receipt")
        measured[model].append(
            {
                "batch_size": batch,
                **metrics,
                "peak_reserved_bytes": peak_reserved,
                "device_total_memory_bytes": total_memory,
                "headroom_eligible": headroom_eligible,
                "profile_receipt": str(profile_path),
                "profile_sha256": profile_hash,
                "operator_receipt": str(operator_path),
                "operator_sha256": operator_hash,
                "exit_code": operator["exit_code"],
                "expired": operator["expired"],
                "rank0_jsonl": str(profile_path.parent / "rank0.jsonl"),
                "rank0_jsonl_sha256": sha(profile_path.parent / "rank0.jsonl"),
                "rank1_jsonl": str(profile_path.parent / "rank1.jsonl"),
                "rank1_jsonl_sha256": sha(profile_path.parent / "rank1.jsonl"),
            }
        )

    failed_profiles = selection.get("failed_profiles")
    if not isinstance(failed_profiles, list):
        raise TypeError("selection failed_profiles must be a list, including when empty")
    failed_keys: set[tuple[str, int]] = set()
    failed_receipts: list[dict[str, Any]] = []
    for failed in failed_profiles:
        if not isinstance(failed, dict):
            raise TypeError("each failed profile record must be an object")
        model = failed.get("model")
        batch = failed.get("batch_size")
        status = failed.get("status")
        key = (model, batch)
        if model != "t1280" or type(batch) is not int or batch not in (96, 128):
            raise ValueError("only the bounded small-width B96/B128 attempts may fail")
        if key in failed_keys or key in by_key:
            raise ValueError(f"duplicate completed/failed profile attempt: {key}")
        if status not in {"oom", "failed"}:
            raise ValueError("failed profile status must be oom or failed")
        failed_keys.add(key)
        operator_value = failed.get("operator_receipt")
        log_value = failed.get("log")
        if not isinstance(operator_value, str) or not isinstance(log_value, str):
            raise TypeError("failed profiles must bind operator and log paths")
        operator_path = Path(operator_value)
        log_path = Path(log_value)
        for field, path_value in (("operator", operator_path), ("log", log_path)):
            if (
                not path_value.is_absolute()
                or path_value.is_symlink()
                or not path_value.is_file()
                or ROOT / "runs" not in path_value.resolve().parents
            ):
                raise ValueError(f"failed {field} must be a regular file under runs")
        operator_path = operator_path.resolve(strict=True)
        log_path = log_path.resolve(strict=True)
        require(
            failed.get("operator_sha256") == sha(operator_path),
            f"{key}: failed operator digest mismatch",
        )
        require(
            failed.get("log_sha256") == sha(log_path),
            f"{key}: failed log digest mismatch",
        )
        failed_operator = read_json(operator_path)
        base = "t1280_f0"
        require(
            failed_operator.get("cell") == operator_path.parent.name
            and isinstance(failed_operator.get("cell"), str),
            f"{key}: failed operator cell mismatch",
        )
        require(
            failed_operator.get("source_commit") == COMMIT
            and failed_operator.get("engineering_only") is True,
            f"{key}: failed operator source or purpose mismatch",
        )
        require(
            failed_operator.get("global_batch_sequences") == 2 * batch,
            f"{key}: failed operator global batch mismatch",
        )
        require(
            failed_operator.get("exit_code") != 0 or failed_operator.get("expired") is True,
            f"{key}: failed record actually has a successful process exit",
        )
        require(
            failed_operator.get("expired") is False or status == "failed",
            f"{key}: timed out profile cannot be labeled OOM",
        )
        argv = failed_operator.get("argv")
        if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
            raise TypeError(f"{key}: failed operator argv is missing or malformed")
        _validate_profile_command(
            argv,
            expected_output=operator_path.parent / "profile",
            label=str(key),
        )
        _validate_profile_environment(failed_operator, str(key))
        try:
            config_path = Path(argv[argv.index("--config") + 1]).resolve(strict=True)
            split_path = Path(argv[argv.index("--split-manifest") + 1]).resolve(strict=True)
        except (ValueError, IndexError) as error:
            raise ValueError(f"{key}: failed operator lacks config/split binding") from error
        require(
            config_path.parent == operator_path.parent
            and sha(config_path) == failed_operator.get("config_sha256"),
            f"{key}: failed profile config path or digest mismatch",
        )
        require(
            read_json(config_path) == frozen_profile_recipe(base, batch),
            f"{key}: failed profile recipe changed",
        )
        require(
            split_path == (ROOT / "splits/fold0.json").resolve()
            and "--nproc-per-node=2" in argv
            and "--profile-updates" in argv
            and argv[argv.index("--profile-updates") + 1] == "12",
            f"{key}: failed profile did not use F0/W2/12 updates",
        )
        require(
            failed_operator.get("log", str(operator_path.parent / "console.log")) == str(log_path),
            f"{key}: failed operator/log path mismatch",
        )
        log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
        if status == "oom" and not (
            "out of memory" in log_text or "cuda error: out of memory" in log_text
        ):
            raise ValueError(f"{key}: OOM label is not supported by the captured log")
        failed_receipts.append(
            {
                "model": model,
                "batch_size": batch,
                "status": status,
                "operator_receipt": str(operator_path),
                "operator_sha256": failed["operator_sha256"],
                "log": str(log_path),
                "log_sha256": failed["log_sha256"],
            }
        )

    required_extension_attempts = {("t1280", 96), ("t1280", 128)}
    observed_extension_attempts = {
        key for key in by_key if key in required_extension_attempts
    } | failed_keys
    require(
        observed_extension_attempts == required_extension_attempts,
        "selection must account for both B96 and B128 attempts as a success or hashed failure",
    )

    selected = {}
    for model, rows in measured.items():
        eligible = [row for row in rows if row["headroom_eligible"]]
        if not eligible:
            raise ValueError(f"{model}: no batch profile retains at least 10% VRAM headroom")
        best_rate = max(row["median_rows_per_second"] for row in eligible)
        near_best = [row for row in eligible if row["median_rows_per_second"] >= best_rate * 0.97]
        selected[model] = min(near_best, key=lambda row: row["batch_size"])["batch_size"]
        for row in rows:
            row["selected"] = row["batch_size"] == selected[model]
    require(
        expected_selected == selected,
        "selection does not choose the fastest eligible batch (3% tie favors smaller)",
    )
    return {
        "candidates": [row for model in ("t1280", "t1792") for row in measured[model]],
        "failed_profiles": failed_receipts,
    }


def operator_details(directory: Path) -> tuple[dict[str, Any], list[str], list[int]]:
    """Read either operator receipt casing and normalize its assigned devices."""
    path = next(
        (
            directory / name
            for name in ("operator.json", "OPERATOR.json")
            if (directory / name).is_file()
        ),
        None,
    )
    if path is None:
        raise FileNotFoundError(f"{directory}: missing operator receipt")
    operator = read_json(path)
    result = operator.get("result")
    result = result if isinstance(result, dict) else {}
    if not result and (directory / "RESULT.json").is_file():
        result = read_json(directory / "RESULT.json")
    exit_code = result.get("exit_code", operator.get("exit_code"))
    timed_out = result.get("timed_out", operator.get("timed_out", operator.get("expired", False)))
    status = result.get("status", operator.get("status"))
    require(exit_code == 0 and timed_out is False, f"{path}: operator did not exit cleanly")
    require(
        status in (None, "complete", "complete_low_headroom"),
        f"{path}: operator status is not complete",
    )
    uuids = operator.get("gpu_uuids")
    if not isinstance(uuids, list):
        rows = operator.get("gpus")
        if isinstance(rows, list):
            uuids = [row[1] for row in rows if isinstance(row, list) and len(row) > 1]
    cpus = operator.get("cpus")
    if (
        not isinstance(uuids, list)
        or len(uuids) != 2
        or any(not isinstance(uuid, str) or not uuid.startswith("GPU-") for uuid in uuids)
    ):
        raise ValueError(f"{path}: expected two explicit GPU UUIDs")
    if not isinstance(cpus, list) or len(cpus) != 8 or any(type(cpu) is not int for cpu in cpus):
        raise ValueError(f"{path}: expected eight explicit CPU IDs")
    return operator, uuids, cpus


def warmed_seconds(directory: Path) -> float:
    """Median slowest-rank update time for warmed updates 6 through 12."""
    per_rank: list[dict[int, dict[str, Any]]] = []
    for rank in range(2):
        rows = [
            json.loads(line)
            for line in (directory / "profile" / f"rank{rank}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        selected = {int(row["step"]): row for row in rows if 6 <= int(row["step"]) <= 12}
        if set(selected) != set(range(6, 13)):
            raise ValueError(f"{directory}: missing profile steps 6 through 12 on rank {rank}")
        per_rank.append(selected)
    walls = []
    for step in range(6, 13):
        rows = [per_rank[rank][step] for rank in range(2)]
        if len({int(row["forwarded_rows"]) for row in rows}) != 1:
            raise ValueError(f"{directory}: DDP ranks disagree on forwarded rows at step {step}")
        wall = max(float(row["window_seconds"]) for row in rows)
        if wall <= 0:
            raise ValueError(f"{directory}: nonpositive update duration at step {step}")
        walls.append(wall)
    return float(statistics.median(walls))


def main() -> None:
    """Approve only complete, source-bound, frozen-recipe profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-batch", type=int, choices=SMALL_BATCHES, required=True)
    parser.add_argument("--large-batch", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--profiles-root", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    operator_path = args.operator.resolve(strict=True)
    require(operator_path.is_file(), "receiving controller path must be a regular file")
    os.environ["CUBLAS_EMULATE_SINGLE_PRECISION"] = "1"
    sys.path.insert(0, str(SOURCE / "src"))
    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_provenance import (
        canonical_sha256,
        source_identity,
        validate_profile_receipt,
    )

    manifest = SOURCE / "TRANSPORT_MANIFEST.json"
    transport = read_json(manifest)
    require(transport.get("commit") == COMMIT, "transport manifest commit mismatch")
    members = transport.get("files")
    if not isinstance(members, dict) or not members:
        raise ValueError("transport manifest lacks frozen file digests")
    for name, expected in members.items():
        member = SOURCE / name
        if member.is_symlink() or not member.is_file() or sha(member) != expected:
            raise ValueError(f"frozen source member changed: {name}")
    identity = source_identity(SOURCE, SOURCE / "scripts/train_sequence.py")

    selection = read_json(args.selection_receipt.resolve(strict=True))
    selection_metrics = validate_selection(
        selection,
        selected_small=args.small_batch,
        selected_large=args.large_batch,
        expected_source_identity=identity,
    )

    previous_path = ROOT / "runs/pr32-production-admission-20261024-r2/plan.json"
    previous_plan = read_json(previous_path)
    require(
        previous_plan.get("source", {}).get("commit") == COMMIT,
        "prior plan source differs",
    )
    require(
        previous_plan.get("source", {}).get("root") == str(SOURCE),
        "prior plan source root differs",
    )
    previous_jobs = previous_plan.get("jobs")
    if not isinstance(previous_jobs, list) or {job.get("name") for job in previous_jobs} != set(
        CELLS
    ):
        raise ValueError("prior plan must contain the exact four frozen width/fold cells")

    destination = args.destination.resolve()
    require(
        ROOT / "runs" in destination.parents,
        "destination must be under the owned runs root",
    )
    destination.mkdir(parents=True, exist_ok=False)
    batch_map: dict[str, int] = {}
    jobs: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    host_memory_estimate = 0
    all_gpus: list[str] = []
    all_cpus: list[int] = []

    for cell in CELLS:
        directory = args.profiles_root.resolve(strict=True) / cell
        config_path = directory / "config.json"
        raw_config = read_json(config_path)
        batch = args.small_batch if cell.startswith("t1280_") else args.large_batch
        expected_config = frozen_profile_recipe(cell, batch)
        require(
            raw_config == expected_config,
            f"{cell}: config changed beyond approved batch/runtime fields",
        )
        config = load_config(config_path)
        require(
            config["batch_size"] == batch and config["required_world_size"] == 2,
            f"{cell}: batch/W2 mismatch",
        )
        require(config["epochs"] == 30, f"{cell}: run must remain 30 epochs")
        require(config["lr"] == 0.0003, f"{cell}: learning rate must remain 3e-4")
        require(config["warmup_fraction"] == 0.05, f"{cell}: warmup fraction must remain 5%")
        require(config["schedule"] == "warmup_cosine_v2", f"{cell}: cosine schedule changed")
        require(
            config["expected_fold"] == (0 if cell.endswith("f0") else 2),
            f"{cell}: fold mismatch",
        )

        profile_path = directory / "profile" / "profile_receipt.json"
        profile = read_json(profile_path)
        operator, gpu_uuids, cpus = operator_details(directory)
        require(operator.get("source_commit") == COMMIT, f"{cell}: operator source mismatch")
        require(
            operator.get("config_sha256") == sha(config_path),
            f"{cell}: operator config digest mismatch",
        )
        require(
            profile.get("receipt_schema") == "wnn.sequence-profile.v1",
            f"{cell}: profile schema mismatch",
        )
        require(profile.get("status") == "profile_complete", f"{cell}: profile incomplete")
        require(
            profile.get("approved") is False,
            f"{cell}: profile already has unexpected approval",
        )
        require(
            profile.get("updates") == 12 and profile.get("errors") == [],
            f"{cell}: expected 12 clean updates",
        )
        profile_identity = profile.get("identity")
        if not isinstance(profile_identity, dict):
            raise TypeError(f"{cell}: profile identity is missing or invalid")
        require(
            profile_identity.get("source_sha256") == identity["sha256"],
            f"{cell}: source identity mismatch",
        )
        require(
            profile_identity.get("source_files") == identity["files"],
            f"{cell}: frozen source-file map mismatch",
        )
        require(
            profile_identity.get("config_sha256") == canonical_sha256(config),
            f"{cell}: normalized config identity mismatch",
        )
        require(profile_identity.get("world_size") == 2, f"{cell}: profile was not W2")
        require(
            profile_identity.get("data", {}).get("fold") == config["expected_fold"],
            f"{cell}: profile fold mismatch",
        )
        require(
            profile_identity.get("data", {}).get("split_sha256")
            == sha(ROOT / "splits" / f"fold{config['expected_fold']}.json"),
            f"{cell}: profile split mismatch",
        )
        profile_devices = profile_identity.get("runtime", {}).get("devices", [])
        profile_uuids = [str(item.get("uuid", "")) for item in profile_devices]
        require(
            [uuid.removeprefix("GPU-").lower() for uuid in profile_uuids]
            == [uuid.removeprefix("GPU-").lower() for uuid in gpu_uuids],
            f"{cell}: profile device order differs from operator allocation",
        )
        log_text = (directory / "console.log").read_text(encoding="utf-8", errors="replace")
        require(
            "hit config.recompile_limit" not in log_text,
            f"{cell}: compiler recompile fallback logged",
        )
        require(
            "hit config.cache_size_limit" not in log_text,
            f"{cell}: compiler cache fallback logged",
        )
        memory = profile.get("memory_by_rank")
        if not isinstance(memory, list) or len(memory) != 2:
            raise ValueError(f"{cell}: per-rank memory receipt missing")
        require(
            all(
                row["peak_reserved_bytes"]
                <= (1 - MIN_VRAM_HEADROOM_FRACTION) * row["device_total_memory_bytes"]
                for row in memory
            ),
            f"{cell}: profile does not retain 10% VRAM headroom",
        )

        profile["approved"] = True
        profile["receiving_operator_approval"] = {
            "revision": "pr32-blackwell-batch-r4",
            "source_commit": COMMIT,
            "operator_sha256": sha(operator_path),
            "authority_id": AUTHORITY_ID,
            "scope": "Engineering admission for a user-authorized batch change; no quality equivalence.",
            "selection_receipt_sha256": sha(args.selection_receipt),
            "original_global_batch_sequences": 16,
            "global_batch_sequences": 2 * batch,
            "new_attempt_from_seed": True,
        }
        approved = destination / f"{cell}-approved.json"
        write(approved, profile)
        validate_profile_receipt(approved, profile_identity)

        split = ROOT / "splits" / f"fold{config['expected_fold']}.json"
        split_sha = sha(split)
        job = {
            "name": cell,
            "config": str(config_path),
            "config_file_sha256": sha(config_path),
            "config_sha256": canonical_sha256(config),
            "split_manifest": str(split),
            "split_sha256": split_sha,
            "profile_receipt": str(approved),
            "profile_sha256": sha(approved),
            "gpu_uuids": gpu_uuids,
            "cpus": cpus,
            "max_seconds": 7 * 86400,
        }
        jobs.append(job)
        all_gpus.extend(gpu_uuids)
        all_cpus.extend(cpus)

        seconds = warmed_seconds(directory)
        horizon = int(profile_identity["optimizer"]["total_updates"])
        require(horizon > 0, f"{cell}: optimizer horizon must be positive")
        measured.append(
            {
                "cell": cell,
                "world_size": 2,
                "per_rank_batch": batch,
                "global_batch_sequences": 2 * batch,
                "optimizer_updates": horizon,
                "steady_seconds_per_update": seconds,
                "fit_hours_from_steady_windows": seconds * horizon / 3600,
                "peak_reserved_gib_per_gpu": max(row["peak_reserved_bytes"] for row in memory)
                / 2**30,
                "profile_elapsed_seconds_including_compile": profile["elapsed_seconds"],
            }
        )
        host_memory_estimate += 24 * profile["model_parameters"]["total"] + sum(
            profile["host_peak_rss_bytes"]
        )
        batch_map[cell] = batch

    require(len(set(all_gpus)) == 8, "the four profiles must bind eight unique GPUs")
    require(len(set(all_cpus)) == 32, "the four profiles must bind 32 unique CPUs")

    data_root = previous_plan["data_root"]
    plan = {
        "schema": "wnn.pr32.teacher-cohort-batch.v1",
        "revision": "pr32-blackwell-batch-r4",
        "source": previous_plan["source"],
        "data_root": data_root,
        "data_files": previous_plan["data_files"],
        "run_root": str(ROOT / "runs/pr32-teachers-20261024-r4-batch"),
        "tmp_root": str(ROOT / "tmp/pr32-production-20261024-r4-batch"),
        "minimum_mem_available_bytes": 512 * 2**30,
        "minimum_disk_free_bytes": 200 * 2**30,
        "jobs": jobs,
        "batch_amendment": {
            "schema": AMENDMENT_SCHEMA,
            "authority_id": AUTHORITY_ID,
            "operator_sha256": sha(operator_path),
            "original_global_batch": 16,
            "per_job_batch": batch_map,
            "new_attempt_from_seed": True,
        },
        "batch_selection": {
            "schema": selection["schema"],
            "source_commit": selection["source_commit"],
            "selection_receipt": str(args.selection_receipt.resolve(strict=True)),
            "selection_receipt_sha256": sha(args.selection_receipt),
            "selected_per_rank_batch": selection["selected_per_rank_batch"],
            "verified_candidates": selection_metrics["candidates"],
            "failed_profiles": selection_metrics["failed_profiles"],
            "minimum_vram_headroom_fraction": MIN_VRAM_HEADROOM_FRACTION,
            "tie_rate_fraction": 0.03,
            "selection_rule": "For each width, choose the fastest profile with at least 10% VRAM headroom; within 3% of top throughput, choose the smaller batch.",
            "superseded_run": previous_plan["run_root"],
            "lr_policy": "Retain 3e-4 and 5% warmup; recalculate the cosine/update horizon from 30 epochs and the selected batch.",
            "quality_equivalence_claim": False,
        },
    }

    estimate = {
        "measured_cells": measured,
        "node_steady_fit_hours": max(row["fit_hours_from_steady_windows"] for row in measured),
        "planned_host_ram_gib": 512,
        "conservative_host_staging_plus_profile_rss_gib": host_memory_estimate / 2**30,
        "planned_disk_gib": 200,
        "logical_cpus": 32,
        "source_commit": COMMIT,
        "quality_equivalence_claim": False,
    }
    require(
        host_memory_estimate < 0.8 * 512 * 2**30,
        "projected staging exceeds host RAM headroom",
    )
    write(destination / "plan.json", plan)
    write(destination / "RESOURCE_ESTIMATE.json", estimate)
    print(
        json.dumps(
            {
                "plan": str(destination / "plan.json"),
                "plan_sha256": sha(destination / "plan.json"),
                "estimate": estimate,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
