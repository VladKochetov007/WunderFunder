"""Shared identity and input validation for the bounded primary-refit actor."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from primary_refit_actor_support_r4 import (
    AUXILIARY_SHA256,
    DEADLINE_SHA256,
    MAX_FIT_ATTEMPTS,
    MAX_PROFILE_ATTEMPTS,
    POLICY_SHA256,
    PRIMARY_SHA256,
    REVIEW_PRECISION_SHA256,
    REVIEW_SUPPORT_SHA256,
    REVIEWER_SHA256,
    SOURCE4_SHA256,
    read_json,
    sha256,
    verify_source4,
)
from primary_refit_profile_review_support import uuid as normalize_uuid

POLICY_NAME = "PRIMARY_REFIT_PROFILE_POLICY_R5.json"
REVIEWER_NAME = "review_primary_refit_profile_r5.py"
REVIEW_SUPPORT_NAME = "primary_refit_profile_review_support.py"
REVIEW_PRECISION_NAME = "primary_refit_precision_contract_r5.py"


def _packet(operator_root: Path) -> Path:
    packet = operator_root / "research/experiment_drafts/multistage-optuna-b300-20261031"
    if not packet.is_dir() or packet.is_symlink():
        raise ValueError("operator root does not contain the frozen multistage packet")
    for name in (POLICY_NAME, REVIEWER_NAME, REVIEW_SUPPORT_NAME, REVIEW_PRECISION_NAME):
        path = packet / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"frozen primary-refit profile input is missing or unsafe: {name}")
    if sha256(packet / POLICY_NAME) != POLICY_SHA256:
        raise ValueError("Stage 4 profile policy bytes differ from the frozen pin")
    if (
        sha256(packet / REVIEWER_NAME) != REVIEWER_SHA256
        or sha256(packet / REVIEW_SUPPORT_NAME) != REVIEW_SUPPORT_SHA256
        or sha256(packet / REVIEW_PRECISION_NAME) != REVIEW_PRECISION_SHA256
    ):
        raise ValueError("profile reviewer/support differs from its frozen policy pins")
    return packet


def _admission(runtime_root: Path, run_dir: Path) -> dict[str, Any]:
    candidates = (
        runtime_root / "receipts/ADMISSION_R2.json",
        run_dir / "receipts/ADMISSION_R2.json",
    )
    path = next((item for item in candidates if item.is_file() and not item.is_symlink()), None)
    if path is None:
        raise FileNotFoundError(
            "ADMISSION_R2.json was not found under runtime or campaign receipts"
        )
    admission = read_json(path.resolve(strict=True), "ADMISSION_R2")
    if (
        admission.get("schema") != "wnn.multistage-search-admission.v1"
        or admission.get("status") != "approved"
    ):
        raise ValueError("campaign admission must be an approved root-issued record")
    return admission


def _request(prepared_root: Path, fit_id: str) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    if re.fullmatch(r"primary-refit-\d{2}-s1t\d{6}-[a-z0-9_-]+", fit_id) is None:
        raise ValueError("fit ID is not a prepared Stage 4 refit identifier")
    selection_path = prepared_root / "selection_receipt.json"
    request_path = prepared_root / "requests" / f"{fit_id}.json"
    selection = read_json(selection_path, "primary-refit selection receipt")
    request = read_json(request_path, "prepared primary-refit request")
    request_sha = sha256(request_path)
    primary = request.get("primary_split")
    if (
        not isinstance(primary, Mapping)
        or primary.get("sha256") != PRIMARY_SHA256
        or primary.get("fit_train_sequences") != 10325
        or primary.get("fit_valid_sequences") != 0
        or primary.get("evaluation_sequences") != 1873
    ):
        raise ValueError(
            "refit requires 10325 official-train fitting and 1873 valid evaluation sequences"
        )
    selected = selection.get("selected")
    if (
        selection.get("schema") != "wnn.multistage-primary-refit-selection.v1"
        or selection.get("status") != "prepared_pending_review_and_profiles"
        or selection.get("selection_metric") != "fixed-final inner_train_proxy pooled WP"
        or selection.get("selection_used_primary_valid") is not False
        or selection.get("primary_split_sha256") != PRIMARY_SHA256
        or selection.get("auxiliary_split_sha256") != AUXILIARY_SHA256
        or not isinstance(selected, list)
        or not 2 <= len(selected) <= 4
        or request.get("schema") != "wnn.multistage-primary-refit-request.v1"
        or request.get("fit_id") != fit_id
        or not any(
            isinstance(row, dict)
            and row.get("fit_id") == fit_id
            and row.get("request_path") == str(request_path.resolve())
            and row.get("request_sha256") == request_sha
            for row in selected
        )
    ):
        raise ValueError("request is not one of the frozen <=4 Stage 4 selections")
    return selection_path, selection, request, request_sha


def _campaign_inputs(run_dir: Path, runtime_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    budget = read_json(run_dir / "budget.json", "campaign budget")
    deadline = read_json(run_dir / "campaign_deadline.json", "campaign deadline")
    if (
        budget.get("schema") != "wnn.multistage-budget.v1"
        or budget.get("max_profile_attempts") != MAX_PROFILE_ATTEMPTS
        or budget.get("max_fit_attempts") != MAX_FIT_ATTEMPTS
        or deadline.get("schema") != "wnn.multistage-deadline.v1"
        or deadline.get("campaign_wall_seconds") != 21600
        or deadline.get("collection_reserve_seconds") != 1200
        or deadline.get("reserve_start_epoch") != 1793424069.6589406
        or deadline.get("deadline_epoch") != 1793425269.6589406
        or sha256(run_dir / "campaign_deadline.json") != DEADLINE_SHA256
    ):
        raise ValueError("shared campaign budget/deadline differs from the frozen six-hour policy")
    if not (runtime_root / "receipts").is_dir():
        raise FileNotFoundError("runtime receipt directory is missing")
    return budget, deadline


def _source_and_config(request: Mapping[str, object]) -> tuple[Path, Path, list[str]]:
    training_source, config = request.get("training_source"), request.get("config")
    profile = request.get("profile")
    command = profile.get("argv") if isinstance(profile, Mapping) else None
    if not isinstance(training_source, Mapping) or not isinstance(config, Mapping):
        raise ValueError("prepared request lacks config/source identities")
    if (
        not isinstance(command, list)
        or len(command) < 2
        or not all(isinstance(x, str) for x in command)
    ):
        raise ValueError("prepared profile command is malformed")
    trainer = Path(command[1]).resolve(strict=True)
    source_root = trainer.parent.parent.resolve(strict=True)
    config_path = Path(str(config.get("path"))).resolve(strict=True)
    if (
        trainer.name != "train_sequence.py"
        or trainer.parent.name != "scripts"
        or sha256(config_path) != config.get("sha256")
    ):
        raise ValueError("prepared Source4 trainer/config path or exact config SHA changed")
    verify_source4(source_root, training_source)
    if training_source.get("sha256") != SOURCE4_SHA256:
        raise ValueError("request is not pinned to the fresh Source4 trainer")
    primary = request.get("primary_split")
    if not isinstance(primary, Mapping) or not isinstance(profile, Mapping):
        raise ValueError("prepared profile lacks primary split identities")
    split_path = Path(str(primary.get("path"))).resolve(strict=True)
    if sha256(split_path) != PRIMARY_SHA256:
        raise ValueError("prepared profile is not using the exact official primary split")
    if command.count("--data-root") != 1:
        raise ValueError("prepared profile must have exactly one data root")
    data_root = Path(command[command.index("--data-root") + 1]).resolve(strict=True)
    expected_command = [
        command[0],
        str(trainer),
        "--config",
        str(config_path),
        "--data-root",
        str(data_root),
        "--split-manifest",
        str(split_path),
        "--output",
        str(Path(str(profile.get("output"))).resolve()),
        "--profile-updates",
        "50",
    ]
    fit = request.get("fit")
    if (
        command != expected_command
        or profile.get("updates") != 50
        or profile.get("max_wall_seconds") != 180
        or profile.get("primary_valid_score_used_for_admission") is not False
        or not isinstance(fit, Mapping)
        or fit.get("max_wall_seconds") != 7200
        or fit.get("resume") is not False
        or fit.get("initializer") is not None
        or fit.get("teacher_or_cache") is not None
        or fit.get("transferred_auxiliary_weights") is not False
    ):
        raise ValueError("prepared profile/refit differs from the bounded fresh Source4 recipe")
    return source_root, config_path, command


def _runtime_paths(runtime_root: Path, fit_id: str) -> dict[str, Path]:
    paths = {
        "logs": runtime_root / "logs",
        "receipts": runtime_root / "receipts",
        "requests": runtime_root / "requests",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _nvml(device: Mapping[str, object]) -> dict[str, object]:
    from wnn.multistage_search_worker_support import _nvml_memory_snapshot

    record = _nvml_memory_snapshot(str(device["cuda_visible_devices"]))
    if record.get("status") != "observed":
        raise RuntimeError("NVML pre-profile observation was unavailable")
    record["uuid"] = record.get("device_uuid")
    if (
        normalize_uuid(record.get("uuid")) != normalize_uuid(device.get("uuid"))
        or record.get("total_memory_bytes") != device.get("total_memory_bytes")
        or int(record.get("free_memory_bytes", 0)) < 32 * (1 << 30)
    ):
        raise RuntimeError("profile device lacks its exact UUID or 32 GiB pre-profile headroom")
    return record
