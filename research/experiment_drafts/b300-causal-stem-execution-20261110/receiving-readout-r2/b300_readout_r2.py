"""Re-export and score six approved B300 endpoints, then run the canonical panel grader."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

READOUT_SCHEMA = "wnn.b300-pr74-fourpairs-readout-summary.r2"
CELL_SCHEMA = "wnn.b300-pr74-fourpairs-readout-cell.r2"
EXPORT_TIMEOUT_SECONDS = 2_400
SCORE_TIMEOUT_SECONDS = 1_200
PANEL_TIMEOUT_SECONDS = 600
PANEL_PACKET = Path("research/experiment_drafts/transformer-causal-input-stem-20261109")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--fit-admission-sha256", required=True)
    parser.add_argument("--r1-receiver", type=Path, required=True)
    parser.add_argument("--r2-receiver", type=Path, required=True)
    return parser


def _load_runtime(args: argparse.Namespace) -> tuple[Any, ...]:
    """Load R1 source gates and R2 descendant ownership from their frozen staged paths."""
    r2_path = str(args.r2_receiver.resolve(strict=True))
    if r2_path not in sys.path:
        sys.path.insert(0, r2_path)
    fit_driver = importlib.import_module("b300_fit_stream_r2")
    core, fits, driver, support, guard_type, runner, _ = fit_driver._import_r1(
        args.r1_receiver
    )
    return core, fits, driver, support, guard_type, runner, fit_driver


def _source_apis(source: Path) -> tuple[Any, Any, Any]:
    """Load only source-qualified split, export, and producer validators."""
    source_root = source.resolve(strict=True)
    source_path = str(source_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    splits = importlib.import_module("wnn.splits")
    cache = importlib.import_module("wnn.training.sequence_cache")
    lineage = importlib.import_module("wnn.training.sequence_cache_ensemble_lineage")
    expected = {
        splits: source_root / "src" / "wnn" / "splits.py",
        cache: source_root / "src" / "wnn" / "training" / "sequence_cache.py",
        lineage: source_root
        / "src"
        / "wnn"
        / "training"
        / "sequence_cache_ensemble_lineage.py",
    }
    if any(
        Path(module.__file__).resolve(strict=True) != path
        for module, path in expected.items()
    ):
        raise ValueError(
            "prediction validators were imported outside the frozen source capsule"
        )
    return (
        splits.load_campaign_split,
        cache.validate_export_schema,
        lineage.validate_producer,
    )


def _read_fit_events(path: Path) -> list[dict[str, Any]]:
    """Read a bounded complete monitor journal without accepting partial lines."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4_000_000:
        raise ValueError("fit monitor journal is missing, linked, or oversized")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("fit monitor journal has a partial trailing record")
    return [json.loads(line) for line in raw.splitlines()]


def _artifact_inventory(core: Any, fit_dir: Path) -> dict[str, tuple[int, str]]:
    """Recheck every hashed fit artifact before readout; reject links and extra files."""
    if (
        any(
            path.is_symlink()
            for path in (fit_dir.parent.parent, fit_dir.parent, fit_dir)
        )
        or not fit_dir.is_dir()
    ):
        raise ValueError("fit artifact directory is missing or linked")
    result: dict[str, tuple[int, str]] = {}
    for folder, directories, names in os.walk(fit_dir, followlinks=False):
        if any((Path(folder) / name).is_symlink() for name in directories):
            raise ValueError("fit artifact tree contains a linked directory")
        for name in names:
            path = Path(folder) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("fit artifact tree contains a link or non-file")
            result[path.relative_to(fit_dir).as_posix()] = (
                path.stat().st_size,
                core.sha256_file(path),
            )
    return result


def _verify_fit_bundle(
    *,
    core: Any,
    fits: Any,
    fit_driver: Any,
    authority: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    config_values: dict[str, dict[str, Any]],
    config_shas: dict[str, str],
    source_sha: str,
    root: Path,
    source: Path,
    control: Path,
    fit_admission_sha: str,
    first_profile: float,
    fit_deadline: float,
    wave_deadline: float,
) -> dict[str, dict[str, Any]]:
    """Require six on-time fixed endpoints, unchanged fit inventories and valid admission."""
    summary_path = control / "fit-r2-complete.json"
    summary = core.read_json(summary_path, 8_000_000)
    fit_owner = core.read_json(control / "fit-r2" / "owner.json", 64_000)
    fit_state = core.read_json(control / "fit-r2-status.json", 8_000_000)
    if (
        summary.get("status") != "complete"
        or fit_owner.get("status") != "complete"
        or fit_owner.get("fit_admission_sha256") != fit_admission_sha
        or fit_owner.get("source_sha256") != source_sha
        or fit_state.get("status") != "complete"
        or not isinstance(fit_state.get("summary"), dict)
        or core.canonical_sha256(fit_state["summary"]) != core.canonical_sha256(summary)
        or summary.get("source_sha256") != source_sha
        or summary.get("source_commit") != core.COMMIT
        or summary.get("authority_sha256")
        != core.sha256_file(control / "authority.json")
        or summary.get("fit_admission_sha256") != fit_admission_sha
        or summary.get("split_sha256") != core.SPLIT_SHA256
        or summary.get("first_profile_epoch") != first_profile
        or summary.get("fit_deadline_epoch") != fit_deadline
        or summary.get("wave_deadline_epoch") != wave_deadline
        or summary_path.stat().st_mtime > fit_deadline
    ):
        raise ValueError(
            "R2 fit summary is missing, late, or bound to another authority"
        )
    admission_path = Path(authority["fit_admission_path"])
    admitted = fit_driver._validate_admission(
        core=core,
        admission_path=admission_path,
        expected_sha=fit_admission_sha,
        authority_sha=core.sha256_file(control / "authority.json"),
        source_sha=source_sha,
        source=source,
        configs=configs,
        root=root,
    )
    outcomes = summary.get("fits")
    if not isinstance(outcomes, dict) or set(outcomes) != set(core.CELLS):
        raise ValueError("R2 fit summary does not contain all six cells")
    verified: dict[str, dict[str, Any]] = {}
    for cell in core.CELLS:
        fit_dir = root / "runs" / cell / "fit"
        result_path = control / "fits" / f"{cell}.json"
        result = core.read_json(result_path, 8_000_000)
        file_mtime = result_path.stat().st_mtime
        owned = result.get("owned_fit")
        if (
            result.get("schema") != "wnn.b300-pr74-fourpairs-fit-result.r1"
            or result.get("cell") != cell
            or result.get("status") != "fit_complete"
            or outcomes[cell].get("status") != "fit_complete"
            or result.get("source_sha256") != source_sha
            or result.get("source_commit") != core.COMMIT
            or result.get("authority_sha256")
            != core.sha256_file(control / "authority.json")
            or result.get("config_sha256") != config_shas[cell]
            or result.get("raw_config_sha256") != configs[cell]["raw_sha256"]
            or result.get("fit_admission_sha256") != fit_admission_sha
            or result.get("pair") != configs[cell]["pair"]
            or result.get("gpu_uuids")
            != [str(uuid).removeprefix("GPU-") for uuid in configs[cell]["gpu_uuids"]]
            or result.get("fixed_epochs") != 4
            or result.get("global_updates") != 3_280
            or result.get("fit_members") != core.TRAIN_MEMBERS
            or result.get("evaluation_members") != core.VALID_MEMBERS
            or result.get("scored_rows") != core.SCORED_ROWS
            or result.get("completed_at_epoch", fit_deadline + 1) > fit_deadline
            or file_mtime > fit_deadline
            or not isinstance(owned, dict)
            or owned.get("status") != "completed"
            or owned.get("exit_code") != 0
            or owned.get("timed_out") is not False
            or core.canonical_sha256(outcomes[cell]) != core.canonical_sha256(result)
        ):
            raise ValueError(f"fit result is not an on-time zero-exit endpoint: {cell}")
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"fit receipt has no artifact inventory: {cell}")
        expected_files: dict[str, tuple[int, str]] = {}
        for item in artifacts:
            if not isinstance(item, dict):
                raise TypeError(f"malformed artifact inventory entry: {cell}")
            relative = item.get("path")
            size = item.get("bytes")
            digest = item.get("sha256")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or type(size) is not int
                or size < 0
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or relative in expected_files
            ):
                raise ValueError(
                    f"invalid or duplicate artifact inventory entry: {cell}"
                )
            expected_files[relative] = (size, digest)
        if _artifact_inventory(core, fit_dir) != expected_files:
            raise ValueError(f"fit output changed after its result receipt: {cell}")
        uuids = [str(uuid).removeprefix("GPU-") for uuid in configs[cell]["gpu_uuids"]]
        evidence = fits._fit_metrics(
            root,
            cell,
            config_values[cell],
            config_shas[cell],
            source_sha,
            uuids,
        )
        approved_profile = root / "runs" / cell / "approved_profile.json"
        if approved_profile.is_symlink() or not approved_profile.is_file():
            raise ValueError(f"approved profile receipt is missing or linked: {cell}")
        approved_profile_sha = core.sha256_file(approved_profile)
        if (
            evidence["model_sha256"] != result.get("model_sha256")
            or evidence["terminal_sha256"] != result.get("terminal_sha256")
            or evidence["metadata_sha256"] != result.get("metadata_sha256")
            or evidence["monitor_sha256"] != result.get("monitor_sha256")
            or approved_profile_sha != admitted[cell]["approved_receipt_sha256"]
        ):
            raise ValueError(f"fit evidence differs from its receipt for {cell}")
        verified[cell] = result
    return verified


def _score_agrees_with_monitor(score: dict[str, Any], monitor: dict[str, Any]) -> bool:
    """Compare all pooled endpoint channels at the pinned FP32 tolerance."""
    values = score.get("score")
    if not isinstance(values, dict):
        return False
    for scorer_key, monitor_key in (
        ("wp", "valid/wp"),
        ("t0", "valid/wp/t0"),
        ("t1", "valid/wp/t1"),
    ):
        actual = values.get(scorer_key)
        expected = monitor.get(monitor_key)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or isinstance(expected, bool)
            or not isinstance(expected, (int, float))
            or not math.isfinite(float(expected))
            or abs(float(actual) - float(expected)) > 1e-5
        ):
            return False
    return True


def _readout_cell(
    *,
    core: Any,
    cell: str,
    authority_config: dict[str, Any],
    root: Path,
    source: Path,
    python: Path,
    data_root: Path,
    split_path: Path,
    control: Path,
    readout_control: Path,
    support: Any,
    guard: Any,
    owner: dict[str, int | str],
    state_path: Path,
    state_lock: threading.Lock,
    deadline: float,
    split: Any,
    validate_export_schema: Any,
    validate_producer: Any,
    source_sha: str,
    fit_admission_sha: str,
    config_sha: str,
    normalized_config: dict[str, Any],
    fit_result: dict[str, Any],
) -> dict[str, Any]:
    """Run exact world2 prediction export and canonical independent scoring for one cell."""
    output = root / "runs" / cell / "final-readout-r2"
    predictions = output / "predictions"
    score_path = output / "score.json"
    result_path = readout_control / f"{cell}.json"
    if any(path.exists() or path.is_symlink() for path in (output, result_path)):
        raise FileExistsError(
            f"R2 readout already exists for {cell}; retries are disabled"
        )
    checkpoint = root / "runs" / cell / "fit" / "model.pt"
    if (
        checkpoint.is_symlink()
        or core.sha256_file(checkpoint) != fit_result["model_sha256"]
    ):
        raise ValueError(f"fixed endpoint changed before readout: {cell}")
    output.mkdir(parents=True)
    uuids = [str(uuid).removeprefix("GPU-") for uuid in authority_config["gpu_uuids"]]
    export_command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        str(source / "scripts" / "predict_sequence.py"),
        "--checkpoint",
        str(checkpoint),
        "--data-root",
        str(data_root),
        "--split-manifest",
        str(split_path),
        "--output",
        str(predictions),
        "--membership",
        "evaluation",
    ]
    export = support._run_stage(
        python=python,
        source=source,
        root=root,
        control=readout_control,
        guard=guard,
        owner=owner,
        cell=cell,
        stage="predict-world2",
        output=predictions,
        command=export_command,
        gpu_uuids=uuids,
        deadline=deadline,
        timeout=EXPORT_TIMEOUT_SECONDS,
        state_path=state_path,
        state_lock=state_lock,
    )
    manifest_path = predictions / "manifest.json"
    manifest = core.read_json(manifest_path, 32_000_000)
    validate_export_schema(manifest, "evaluation")
    producer = manifest.get("producer")
    fit_metadata = core.read_json(
        root / "runs" / cell / "fit" / "run_metadata.json", 1_000_000
    )
    fit_provenance = fit_metadata.get("provenance")
    fitting = [("train", group) for group in split.train_groups]
    expected_fit_members = [list(member) for member in fitting]
    expected_evaluation_members = [
        ["valid", group] for group in split.evaluation_groups
    ]
    fit_data = fit_provenance.get("data") if isinstance(fit_provenance, dict) else None
    validate_producer(producer, split.sha256, fitting)
    members = [
        (item.get("split"), item.get("row_group")) for item in manifest["sequences"]
    ]
    expected = [("valid", group) for group in split.evaluation_groups]
    if (
        export.get("status") != "completed"
        or core.sha256_file(checkpoint) != fit_result["model_sha256"]
        or manifest.get("schema") != "sequence-predictions-v1"
        or manifest.get("complete") is not True
        or manifest.get("world_size") != 2
        or manifest.get("split_sha256") != core.SPLIT_SHA256
        or len(members) != len(expected)
        or set(members) != set(expected)
        or len(set(members)) != len(members)
        or producer.get("checkpoint_sha256") != fit_result["model_sha256"]
        or producer.get("config") != normalized_config
        or producer.get("split_sha256") != core.SPLIT_SHA256
        or not isinstance(fit_provenance, dict)
        or not isinstance(fit_data, dict)
        or fit_data.get("fit_members") != expected_fit_members
        or fit_data.get("evaluation_members") != expected_evaluation_members
        or fit_data.get("teacher") is not None
        or fit_data.get("reference") is not None
        or producer.get("provenance") != fit_provenance
        or producer.get("provenance", {}).get("world_size") != 2
        or producer.get("provenance", {}).get("data_sha256")
        != fit_result["data_sha256"]
        or producer.get("provenance", {}).get("identity", {}).get("config_sha256")
        != config_sha
        or producer.get("teacher") is not None
        or producer.get("reference") is not None
        or producer.get("provenance", {}).get("identity", {}).get("source_sha256")
        != source_sha
    ):
        raise ValueError(
            f"prediction export is incomplete or bound to another fit: {cell}"
        )
    if any(
        not isinstance(item.get("file"), str) or Path(item["file"]).name != item["file"]
        for item in manifest["sequences"]
    ):
        raise ValueError(f"prediction manifest contains a non-flat filename: {cell}")
    expected_names = {
        "manifest.json",
        "rank-0.json",
        "rank-1.json",
        *(Path(item["file"]).name for item in manifest["sequences"]),
    }
    observed_paths = list(predictions.iterdir())
    if (
        any(path.is_symlink() or not path.is_file() for path in observed_paths)
        or {path.name for path in observed_paths} != expected_names
    ):
        raise ValueError(
            f"prediction export contains missing or unexpected files: {cell}"
        )
    score_command = [
        str(python),
        str(source / "scripts" / "score_sequence.py"),
        "--data-root",
        str(data_root),
        "--split-manifest",
        str(split_path),
        "--predictions",
        str(predictions),
        "--output",
        str(score_path),
    ]
    scored = support._run_stage(
        python=python,
        source=source,
        root=root,
        control=readout_control,
        guard=guard,
        owner=owner,
        cell=cell,
        stage="score-canonical",
        output=output,
        command=score_command,
        gpu_uuids=[],
        deadline=deadline,
        timeout=SCORE_TIMEOUT_SECONDS,
        state_path=state_path,
        state_lock=state_lock,
    )
    score = core.read_json(score_path, 8_000_000)
    validate_producer(score.get("producer"), split.sha256, fitting)
    event_path = root / "runs" / cell / "fit" / "monitor_evaluations.jsonl"
    events = _read_fit_events(event_path)
    if (
        scored.get("status") != "completed"
        or core.sha256_file(checkpoint) != fit_result["model_sha256"]
        or score.get("interpretation")
        != "Exposed fold; fixed endpoint, not checkpoint selection or public score."
        or score.get("split_sha256") != split.sha256
        or score.get("sequences") != 1_873
        or score.get("score", {}).get("selected_rows") != 3_528_273
        or score.get("score", {}).get("metric_id") != "global_wp_clip2_v1"
        or score.get("producer") != producer
        or not isinstance(score.get("component_centered_moments"), dict)
        or len(events) != 4
        or not _score_agrees_with_monitor(score, events[-1].get("metrics", {}))
    ):
        raise ValueError(f"canonical score differs from full-valid monitor for {cell}")
    result = {
        "schema": CELL_SCHEMA,
        "cell": cell,
        "status": "readout_complete",
        "source_sha256": source_sha,
        "source_commit": core.COMMIT,
        "source_manifest_sha256": core.MANIFEST_SHA256,
        "fit_admission_sha256": fit_admission_sha,
        "split_sha256": core.SPLIT_SHA256,
        "fit_model_sha256": fit_result["model_sha256"],
        "config_sha256": config_sha,
        "fit_result_sha256": core.sha256_file(control / "fits" / f"{cell}.json"),
        "prediction_manifest_sha256": core.sha256_file(manifest_path),
        "score_sha256": core.sha256_file(score_path),
        "gpu_uuids": uuids,
        "prediction_world_size": 2,
        "sequences": 1_873,
        "scored_rows": 3_528_273,
        "score": score["score"],
        "monitor_match_abs_tolerance": 1e-5,
        "automatic_retry": False,
    }
    core.atomic_json(control / f"{cell}.json", result)
    return result


def _panel_view(
    *,
    core: Any,
    root: Path,
    source: Path,
    python: Path,
    split_path: Path,
    data_root: Path,
    control: Path,
    readout_control: Path,
    support: Any,
    guard: Any,
    owner: dict[str, int | str],
    state_path: Path,
    state_lock: threading.Lock,
    deadline: float,
) -> dict[str, Any]:
    """Run canonical compare_panel against a separate read-only view of fit endpoints."""
    panel_root = readout_control / "panel-input"
    if panel_root.exists() or panel_root.is_symlink():
        raise FileExistsError("panel input already exists; no readout retry is allowed")
    for cell in core.CELLS:
        shadow = panel_root / "runs" / cell
        shadow_fit = shadow / "fit"
        shadow_fit.mkdir(parents=True, exist_ok=False)
        original_fit = root / "runs" / cell / "fit"
        model = original_fit / "model.pt"
        target = shadow_fit / "model.pt"
        try:
            os.link(model, target)
        except OSError:
            shutil.copyfile(model, target)
        shutil.copyfile(
            original_fit / "terminal_status.json", shadow_fit / "terminal_status.json"
        )
        shutil.copyfile(
            root / "runs" / cell / "final-readout-r2" / "score.json",
            shadow_fit / "score.json",
        )
        shutil.copyfile(
            root / "runs" / cell / "approved_profile.json",
            shadow / "approved_profile.json",
        )
    comparison_script = source / PANEL_PACKET / "compare_panel.py"
    if comparison_script.is_symlink() or not comparison_script.is_file():
        raise FileNotFoundError(
            "frozen canonical compare_panel.py is missing or linked"
        )
    result = support._run_stage(
        python=python,
        source=source,
        root=root,
        control=readout_control,
        guard=guard,
        owner=owner,
        cell="panel",
        stage="compare-panel-canonical",
        output=panel_root,
        command=[
            str(python),
            str(comparison_script),
            "--run-root",
            str(panel_root),
            "--data-root",
            str(data_root),
            "--split-manifest",
            str(split_path),
        ],
        gpu_uuids=[],
        deadline=deadline,
        timeout=PANEL_TIMEOUT_SECONDS,
        state_path=state_path,
        state_lock=state_lock,
    )
    comparison_path = panel_root / "comparison.json"
    comparison = core.read_json(comparison_path, 8_000_000)
    expected_hashes = {
        cell: {
            "checkpoint_sha256": result["fit_model_sha256"],
            "score_sha256": core.sha256_file(
                root / "runs" / cell / "final-readout-r2" / "score.json"
            ),
            "approved_profile_sha256": core.sha256_file(
                root / "runs" / cell / "approved_profile.json"
            ),
        }
        for cell, result in (
            (name, core.read_json(readout_control / f"{name}.json", 128_000))
            for name in core.CELLS
        )
    }
    expected_scores = {
        cell: {
            "primary": core.read_json(
                root / "runs" / cell / "final-readout-r2" / "score.json", 8_000_000
            )["score"]
        }
        for cell in core.CELLS
    }
    if (
        result.get("status") != "completed"
        or comparison.get("schema") != "wnn.input-stem-comparison.v1"
        or comparison.get("status") != "complete_exposed_fixed4E_screen"
        or set(comparison.get("scores", {})) != set(core.CELLS)
        or comparison.get("receipt_sha256") != expected_hashes
        or comparison.get("scores") != expected_scores
    ):
        raise ValueError("canonical compare_panel did not complete for all six cells")
    return {
        "comparison_sha256": core.sha256_file(comparison_path),
        "comparison_path": str(comparison_path),
        "status": comparison["status"],
        "exposed_validation": True,
        "world_size_prediction": 2,
    }


def main(argv: list[str] | None = None) -> int:
    """Re-export only after all fits have completed; never train or modify fit artifacts."""
    args = _parser().parse_args(argv)
    core, fits, driver, support, Guard, _runner, fit_driver = _load_runtime(args)
    (
        authority,
        configs,
        config_values,
        config_shas,
        _source_files,
        root,
        source,
        data_root,
        split_path,
        python,
    ) = driver._prepare(args)
    control = root / "control" / "b300-driver-r1"
    authority_sha = core.sha256_file(control / "authority.json")
    if authority_sha != args.authority_sha256:
        raise ValueError("R1 authority receipt differs from the readout command")
    readout_control = control / "readout-r2"
    if readout_control.is_symlink():
        raise ValueError("readout control directory may not be a symlink")
    readout_control.mkdir(parents=True, exist_ok=True)
    if any((readout_control / name).exists() for name in ("owner.json", "status.json")):
        raise FileExistsError(
            "R2 readout already has an owner/status; retries are disabled"
        )
    lock_stream = (control / "phase.lock").open("a+b")
    import fcntl

    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_stream.close()
        raise RuntimeError(
            "fit/readout controller still owns the shared run-root lock"
        ) from error
    try:
        owner = support._identity(root)
        state_path = readout_control / "status.json"
        core.atomic_json(
            readout_control / "owner.json",
            {
                "schema": "wnn.b300-pr74-readout-r2-owner.v1",
                "owner": owner,
                "source_sha256": core.SOURCE_SHA256,
                "source_manifest_sha256": core.MANIFEST_SHA256,
                "authority_sha256": authority_sha,
                "fit_admission_sha256": args.fit_admission_sha256,
                "started_epoch": time.time(),
            },
        )
        core.atomic_json(
            state_path,
            {
                "schema": "wnn.b300-pr74-readout-r2-status.v1",
                "status": "running",
                "owner": owner,
                "cells": {},
                "started_epoch": time.time(),
            },
        )
        guard = Guard(root, readout_control, support._caps(authority))
        first, fit_deadline, wave_deadline = support._read_clock(
            support._clock_path(root, None)
        )
        if time.time() >= wave_deadline:
            raise TimeoutError("original six-hour collection deadline has passed")
        split_loader, export_validator, producer_validator = _source_apis(source)
        split = split_loader(split_path, data_root)
        if (
            split.sha256 != core.SPLIT_SHA256
            or split.valid_groups
            or len(split.train_groups) != core.TRAIN_MEMBERS
            or len(split.evaluation_groups) != core.VALID_MEMBERS
        ):
            raise ValueError("readout data membership differs from the official split")
        fit_results = _verify_fit_bundle(
            core=core,
            fits=fits,
            fit_driver=fit_driver,
            authority=authority,
            configs=configs,
            config_values=config_values,
            config_shas=config_shas,
            source_sha=core.SOURCE_SHA256,
            root=root,
            source=source,
            control=control,
            fit_admission_sha=args.fit_admission_sha256,
            first_profile=first,
            fit_deadline=fit_deadline,
            wave_deadline=wave_deadline,
        )
        state_lock = threading.Lock()
        outcomes: dict[str, dict[str, Any]] = {}

        def run_cell(cell: str) -> dict[str, Any]:
            current_admission = fit_driver._validate_admission(
                core=core,
                admission_path=Path(authority["fit_admission_path"]),
                expected_sha=args.fit_admission_sha256,
                authority_sha=authority_sha,
                source_sha=core.SOURCE_SHA256,
                source=source,
                configs=configs,
                root=root,
            )
            if cell not in current_admission:
                raise ValueError("fit admission no longer includes this readout cell")
            return _readout_cell(
                core=core,
                cell=cell,
                authority_config=configs[cell],
                root=root,
                source=source,
                python=python,
                data_root=data_root,
                split_path=split_path,
                control=control,
                readout_control=readout_control,
                support=support,
                guard=guard,
                owner=owner,
                state_path=state_path,
                state_lock=state_lock,
                deadline=wave_deadline,
                split=split,
                validate_export_schema=export_validator,
                validate_producer=producer_validator,
                source_sha=core.SOURCE_SHA256,
                fit_admission_sha=args.fit_admission_sha256,
                config_sha=config_shas[cell],
                normalized_config=config_values[cell],
                fit_result=fit_results[cell],
            )

        stream_cells = [cell for cells in core.FIT_STREAMS.values() for cell in cells]
        if (
            len(core.FIT_STREAMS) != 4
            or len(stream_cells) != len(core.CELLS)
            or set(stream_cells) != set(core.CELLS)
            or any(len(cells) not in (1, 2) for cells in core.FIT_STREAMS.values())
        ):
            raise ValueError("frozen four-pair stream map is incomplete or duplicated")
        if time.time() >= wave_deadline:
            raise TimeoutError(
                "original six-hour collection deadline passed during fit audit"
            )
        with ThreadPoolExecutor(
            max_workers=len(core.FIT_STREAMS), thread_name_prefix="readout-pair-r2"
        ) as pool:
            futures = {
                pool.submit(fit_driver._pair_stream, pair, cells, run_cell): (
                    pair,
                    cells,
                )
                for pair, cells in core.FIT_STREAMS.items()
            }
            for future in as_completed(futures):
                pair, cells = futures[future]
                try:
                    outcomes.update(future.result())
                except Exception as error:  # noqa: BLE001 - record all cell failures.
                    for cell in cells:
                        outcomes.setdefault(
                            cell,
                            {
                                "schema": CELL_SCHEMA,
                                "cell": cell,
                                "pair_stream": pair,
                                "status": "readout_failed",
                                "error": f"{type(error).__name__}: {error}",
                                "automatic_retry": False,
                            },
                        )
                core.atomic_json(
                    readout_control / "progress.json",
                    {
                        "completed_cells": sorted(outcomes),
                        "cells": outcomes,
                        "updated_epoch": time.time(),
                    },
                    replace=True,
                )
        if set(outcomes) == set(core.CELLS) and all(
            item.get("status") == "readout_complete" for item in outcomes.values()
        ):
            panel = _panel_view(
                core=core,
                root=root,
                source=source,
                python=python,
                split_path=split_path,
                data_root=data_root,
                control=control,
                readout_control=readout_control,
                support=support,
                guard=guard,
                owner=owner,
                state_path=state_path,
                state_lock=state_lock,
                deadline=wave_deadline,
            )
            status = "complete_exposed_fixed4E_screen"
        else:
            panel = None
            status = "partial_or_failed"
        summary = {
            "schema": READOUT_SCHEMA,
            "status": status,
            "source_sha256": core.SOURCE_SHA256,
            "source_commit": core.COMMIT,
            "source_manifest_sha256": core.MANIFEST_SHA256,
            "authority_sha256": authority_sha,
            "fit_admission_sha256": args.fit_admission_sha256,
            "split_sha256": core.SPLIT_SHA256,
            "first_profile_epoch": first,
            "fit_deadline_epoch": fit_deadline,
            "wave_deadline_epoch": wave_deadline,
            "prediction_world_size": 2,
            "pair_streams": {
                pair: list(cells) for pair, cells in core.FIT_STREAMS.items()
            },
            "cells": outcomes,
            "panel": panel,
            "automatic_retry": False,
            "interpretation": "Official validation was exposed in prior screens; readout is not fresh confirmation or a public score.",
        }
        core.atomic_json(readout_control / "complete.json", summary)
        state = core.read_json(state_path, 4_000_000)
        state.update(status=status, completed_epoch=time.time(), summary=summary)
        core.atomic_json(state_path, state, replace=True)
        owner_record = core.read_json(readout_control / "owner.json", 64_000)
        owner_record.update(status=status, completed_epoch=state["completed_epoch"])
        core.atomic_json(readout_control / "owner.json", owner_record, replace=True)
        return 0 if status == "complete_exposed_fixed4E_screen" else 2
    except Exception as error:
        state_path = readout_control / "status.json"
        if state_path.is_file() and not state_path.is_symlink():
            state = core.read_json(state_path, 4_000_000)
            state.update(
                status="failed",
                completed_epoch=time.time(),
                error=f"{type(error).__name__}: {error}",
            )
            core.atomic_json(state_path, state, replace=True)
        owner_path = readout_control / "owner.json"
        if owner_path.is_file() and not owner_path.is_symlink():
            owner_record = core.read_json(owner_path, 64_000)
            owner_record.update(
                status="failed",
                completed_epoch=time.time(),
                error=f"{type(error).__name__}: {error}",
            )
            core.atomic_json(owner_path, owner_record, replace=True)
        raise
    finally:
        import fcntl

        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
