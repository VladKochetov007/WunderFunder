"""Qualify the four already-launched full-primary refits for report-only scoring."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from report_multistage_primary_ensemble_worker import (
    PRIMARY_SHA256,
    _canonical_sha256,
    _read_json,
    _sha256,
)

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/experiment_drafts/multistage-optuna-b300-20261031"
HARD_DEADLINE = 1793425269.6589406
SOURCE4 = "c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81"


def artifact(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"report evidence must be a regular file: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def configured_final(
    child: Mapping[str, object], terminal: Mapping[str, object], epochs: int
) -> bool:
    """A censored or early-stopped endpoint cannot become a configured-final model."""
    return (
        child.get("status") == "completed"
        and type(child.get("exit_code")) is int
        and child.get("exit_code") == 0
        and child.get("timed_out") is False
        and terminal.get("status") == "completed"
        and terminal.get("completion_kind") == "max_epochs"
        and terminal.get("completed_epochs") == terminal.get("requested_epochs") == epochs
    )


def validate_weights(
    frozen: Mapping[str, object],
    selection: Mapping[str, object],
    selection_sha: str,
    stage4: Mapping[str, object],
    stage4_sha: str,
    first_fit_epoch: float,
) -> dict[str, list[float]]:
    """Bind both fixed vectors to all four selected IDs before the first new fit."""
    selected = selection.get("selected")
    if not isinstance(selected, list) or len(selected) != 4:
        raise ValueError("full-primary report requires the four frozen selections")
    ids = [row["fit_id"] for row in selected]
    if (
        frozen.get("schema") != "wnn.primary-refit-frozen-ensembles.v1"
        or frozen.get("selection_receipt_sha256") != selection_sha
        or frozen.get("stage4_result_sha256") != stage4_sha
        or frozen.get("frozen_before_new_refit_fitting") is not True
        or frozen.get("primary_valid_used_for_weights") is not False
        or frozen.get("fit_ids") != ids
        or len(set(ids)) != 4
        or type(frozen.get("frozen_epoch")) not in (int, float)
        or not math.isfinite(frozen["frozen_epoch"])
        or not frozen["frozen_epoch"] < first_fit_epoch
    ):
        raise ValueError("ensemble weights were not frozen for these four pre-fit selections")
    members = stage4.get("ensemble", {}).get("members", [])
    original = {row["trial_result_path"]: row["weight"] for row in members}
    direct = [original[row["source_result_path"]] for row in selected]
    if (
        any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in direct
        )
        or sum(direct) <= 0
    ):
        raise ValueError("original Stage4 direct weights are invalid")
    expected = [float(value / sum(direct)) for value in direct]
    result = {}
    for name, target in (("equal_weight", [0.25] * 4), ("stage4_direct_renormalized", expected)):
        vector = frozen.get(name)
        if (
            not isinstance(vector, list)
            or len(vector) != 4
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in vector
            )
            or abs(sum(vector) - 1) > 1e-10
            or any(
                abs(value - wanted) > 1e-12 for value, wanted in zip(vector, target, strict=True)
            )
        ):
            raise ValueError(f"frozen {name} differs from its predeclared four-member vector")
        result[name] = [float(value) for value in vector]
    return result


def require_processes_gone(owner: Mapping[str, object], child: Mapping[str, object]) -> None:
    if os.name != "posix":
        raise RuntimeError("actual fit termination must be checked on the POSIX receiving host")
    for identity in (
        {key: owner.get(key) for key in ("pid", "pgid", "birth")},
        child.get("child", {}),
    ):
        if (
            not isinstance(identity, Mapping)
            or type(identity.get("pid")) is not int
            or type(identity.get("pgid")) is not int
            or identity["pid"] <= 0
            or identity["pgid"] != identity["pid"]
            or not isinstance(identity.get("birth"), str)
            or not identity["birth"].isdecimal()
        ):
            raise ValueError("owned fit process identity is malformed")
        if owned_group_active(identity):
            raise RuntimeError(
                "a full-primary fit guardian or child is still active; scoring refused"
            )


def owned_group_active(identity: Mapping[str, object], proc_root: Path = Path("/proc")) -> bool:
    """Ignore reaped/zombie tuples while rejecting live members of their owned group."""
    for path in proc_root.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            raw = (path / "stat").read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        close = raw.rfind(")")
        fields = raw[close + 2 :].split() if close >= 0 else []
        if len(fields) > 19 and fields[0] not in {"Z", "X"} and int(fields[2]) == identity["pgid"]:
            return True
    return False


def bind_final_model(terminal: Mapping[str, object], checkpoint: Mapping[str, object]) -> None:
    if terminal.get("model_sha256") != checkpoint.get("sha256"):
        raise ValueError(
            "configured-final model SHA differs from native terminal_status.model_sha256"
        )


def _member(runtime: Path, prepared: Path, fit_id: str, index: int, gpu: str) -> dict[str, object]:
    from primary_refit_actor_common_r4 import _request, _source_and_config
    from primary_refit_actor_fit_r4 import _primary_profile_data, _validate_root_approval
    from primary_refit_actor_support_r4 import POLICY_SHA256, REVIEWER_SHA256

    selection_path, _, request, request_sha = _request(prepared, fit_id)
    source_root, config_path, _ = _source_and_config(request)
    fit_plan, profile_plan = request["fit"], request["profile"]
    output = runtime / "runs" / fit_id
    profile_dir = runtime / "profiles" / f"{fit_id}-profile"
    if (
        Path(fit_plan["output"]).resolve() != output.resolve()
        or Path(profile_plan["output"]).resolve() != profile_dir.resolve()
    ):
        raise ValueError("prepared full-primary paths escaped the declared runtime")
    paths = {
        "native_profile": profile_dir / "profile_receipt.json",
        "engineering_review": profile_dir / "engineering_review.json",
        "profile_attempt": profile_dir / "profile_attempt_receipt.json",
        "approved_profile": profile_dir / "approved_profile_receipt.json",
        "fit_request": runtime / "requests" / f"stage4-primary-refit-{fit_id}.json",
        "owner": runtime / "receipts" / f"stage4-primary-refit-{fit_id}.owner.json",
        "child": runtime / "receipts" / f"stage4-primary-refit-{fit_id}.child.json",
        "metadata": output / "run_metadata.json",
    }
    if not paths["metadata"].exists():
        del paths["metadata"]
    values = {name: _read_json(path, name, 32_000_000) for name, path in paths.items()}
    native, review, attempt, approved = (
        values[name]
        for name in ("native_profile", "engineering_review", "profile_attempt", "approved_profile")
    )
    fit, owner, child = (values[name] for name in ("fit_request", "owner", "child"))
    owner_command = owner.get("command")
    actor_manifest = _read_json(
        PACKET / "PRIMARY_REFIT_ACTOR_R4_MANIFEST.json", "frozen R4 actor manifest"
    )
    if (
        not isinstance(owner_command, list)
        or len(owner_command) != 4
        or owner_command[0] != profile_plan["argv"][0]
        or Path(owner_command[1]).name != "primary_refit_fit_guardian_r4.py"
        or owner_command[2:] != ["--request", str(paths["fit_request"].resolve())]
        or _sha256(Path(owner_command[1]))
        != actor_manifest["files"]["primary_refit_fit_guardian_r4.py"]
        or owner.get("command_sha256")
        != hashlib.sha256(json.dumps(owner_command, separators=(",", ":")).encode()).hexdigest()
    ):
        raise ValueError("owned fit guardian command or frozen R4 source identity differs")
    approval_path = Path(approved["root_native_approval"]["path"])
    approval = _read_json(approval_path, "root approval")
    _validate_root_approval(
        approval, request, request_sha, paths["native_profile"], paths["engineering_review"]
    )
    paths["root_approval"] = approval_path
    identity = native.get("identity", {})
    metadata = values.get("metadata")
    if metadata is None:
        from wnn.training.sequence_config import load_config

        metadata = {
            "config": load_config(config_path),
            "source": {
                "sha256": identity.get("source_sha256"),
                "files": identity.get("source_files"),
            },
            "provenance": {
                key: identity.get(key)
                for key in ("data", "data_sha256", "runtime", "world_size", "model_parameters")
            },
        }
    provenance = metadata.get("provenance", {})
    attempt_id = f"stage4-primary-refit-{fit_id}"
    if (
        native.get("status") != "profile_complete"
        or native.get("approved") is not False
        or identity.get("source_sha256") != SOURCE4
        or not _primary_profile_data(identity.get("data", {}))
        or review.get("status") != "engineering_checks_passed"
        or review.get("approved") is not False
        or review.get("policy_sha256") != POLICY_SHA256
        or review.get("reviewer_script_sha256") != REVIEWER_SHA256
        or review.get("native_profile_sha256") != _sha256(paths["native_profile"])
        or review.get("request_sha256") != request_sha
        or review.get("selection_receipt_sha256") != _sha256(selection_path)
        or review.get("attempt_receipt_sha256") != _sha256(paths["profile_attempt"])
        or not review.get("checks")
        or not all(value is True for value in review["checks"].values())
        or attempt.get("status") != "completed"
        or attempt.get("profile_sha256") != _sha256(paths["native_profile"])
        or approved.get("approved") is not True
        or approved.get("identity") != identity
        or approved["root_native_approval"].get("sha256") != _sha256(approval_path)
        or approved.get("external_review", {}).get("sha256") != _sha256(paths["engineering_review"])
        or metadata.get("source", {}).get("sha256") != SOURCE4
        or metadata.get("source", {}).get("files") != identity.get("source_files")
        or _canonical_sha256(metadata.get("config")) != identity.get("config_sha256")
        or any(
            provenance.get(key) != identity.get(key)
            for key in ("data", "data_sha256", "runtime", "world_size", "model_parameters")
        )
        or fit.get("command") != fit_plan.get("argv_after_approval_and_reservation")
        or fit.get("source_root") != str(source_root)
        or fit.get("source_sha256") != SOURCE4
        or fit.get("primary_split_sha256") != PRIMARY_SHA256
        or fit.get("config_sha256") != _sha256(config_path)
        or fit.get("selection_receipt_sha256") != _sha256(selection_path)
        or fit.get("profile_sha256") != _sha256(paths["native_profile"])
        or fit.get("engineering_review_sha256") != _sha256(paths["engineering_review"])
        or fit.get("root_native_approval_sha256") != _sha256(approval_path)
        or fit.get("fit_attempt_id") != attempt_id
        or fit.get("stage") != 4
        or owner.get("budget_attempt_id") != attempt_id
        or owner.get("fit_id") != fit_id
        or owner.get("approval_sha256") != _sha256(approval_path)
        or owner.get("profile_sha256") != fit.get("profile_sha256")
        or owner.get("stage") != 4
        or child.get("schema") != "wnn.multistage-owned-process.v1"
        or child.get("status")
        not in {"completed", "failed", "timed_out", "progress_callback_failed"}
        or child.get("command") != fit.get("command")
        or child.get("stage") != 4
        or child.get("mode") != "fit"
        or child.get("gpu_key") != fit.get("gpu_key")
        or child.get("cwd") != str(source_root)
        or child.get("output_dir") != str(output.resolve())
        or child.get("owner") != {key: owner.get(key) for key in ("pid", "pgid", "birth")}
        or child.get("command_sha256")
        != hashlib.sha256(json.dumps(fit["command"], separators=(",", ":")).encode()).hexdigest()
    ):
        raise ValueError(
            f"refit {fit_id} is not bound to its approved Source4 inputs and exact owned terminal"
        )
    require_processes_gone(owner, child)
    terminal_path = output / "terminal_status.json"
    terminal = _read_json(terminal_path, "native terminal") if terminal_path.is_file() else {}
    config = metadata["config"]
    final = "metadata" in values and configured_final(child, terminal, int(config["epochs"]))
    model = output / "model.pt"
    if final and (not model.is_file() or model.is_symlink()):
        final = False
    checkpoint = artifact(model) if final else None
    if checkpoint is not None:
        bind_final_model(terminal, checkpoint)
    artifacts = {name: artifact(path) for name, path in paths.items()}
    native_epoch_index = None
    for name in ("terminal_status.json", "epoch_index.json"):
        path = output / name
        if path.is_file():
            artifacts[name] = artifact(path)
            if name == "epoch_index.json":
                native_epoch_index = _read_json(path, "native epoch index").get("completed_epochs")
    return {
        "index": index,
        "fit_id": fit_id,
        "family": request["selected_by"]["family"],
        "gpu_uuid": gpu,
        "configured_final_complete": final,
        "native_terminal": terminal,
        "child_status": child["status"],
        "child_exit_code": child.get("exit_code"),
        "native_metadata_available": "metadata" in values,
        "native_epoch_index_completed_epochs": native_epoch_index,
        "partial_model_artifact": artifact(model) if not final and model.is_file() else None,
        "first_fit_epoch": child["started_at_epoch"],
        "config": config,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "identity": identity,
        "artifacts": artifacts,
        "fit_output": str(output.resolve()),
        "checkpoint": checkpoint,
        "partial_policy": "retained epochs are reported as partial; never scored as configured final or resumed",
    }


def collect_inputs(
    runtime: Path, prepared: Path, gpus: Sequence[str]
) -> tuple[list[dict[str, object]], dict[str, list[float]], dict[str, object]]:
    sys.path[:0] = [str(PACKET), str(ROOT / "src")]
    from primary_refit_actor_common_r4 import _admission, _campaign_inputs, _packet
    from primary_refit_actor_support_r4 import load_device_inventory, resolve_device
    from primary_refit_profile_review_support import uuid as normalize_uuid

    _packet(ROOT)
    budget, _ = _campaign_inputs(runtime / "campaign-r2", runtime)
    admission = _admission(runtime, runtime / "campaign-r2")
    _, inventory = load_device_inventory(
        runtime, admission, PACKET / "receiving/B300_DEVICE_INVENTORY_ACTUAL_R2.json"
    )
    allowed = {
        normalize_uuid(resolve_device(admission, inventory, str(i))["uuid"]) for i in range(4)
    }
    if {normalize_uuid(gpu) for gpu in gpus} != allowed or None in allowed:
        raise ValueError("only the exact admitted inventoried GPUs0..3 can score these four refits")
    selection_path = prepared / "selection_receipt.json"
    selection = _read_json(selection_path, "selection", 32_000_000)
    selected = selection.get("selected")
    if (
        not isinstance(selected, list)
        or len(selected) != 4
        or len(gpus) != 4
        or len(set(gpus)) != 4
    ):
        raise ValueError("four frozen refits and four distinct assigned GPU UUIDs are required")
    members = [
        _member(runtime, prepared, row["fit_id"], index, gpus[index])
        for index, row in enumerate(selected)
    ]
    for member in members:
        attempts = [
            row
            for row in budget["fit_attempts"]
            if row.get("attempt_id") == f"stage4-primary-refit-{member['fit_id']}"
        ]
        expected = {
            "completed": "completed",
            "failed": "failed",
            "timed_out": "censored_timeout",
            "progress_callback_failed": "guardian_failed",
        }[member["child_status"]]
        if (
            len(attempts) != 1
            or attempts[0].get("stage") != 4
            or attempts[0].get("status") != expected
            or attempts[0].get("exit_code") != member["child_exit_code"]
        ):
            raise ValueError(
                "owned child terminal outcome differs from the finalized scientific-fit ledger"
            )
        member["budget_attempt"] = attempts[0]
    stage4_path = runtime / "campaign-r2/stage4/ensemble_result.json"
    weights_path = prepared / "frozen_primary_refit_ensembles.json"
    weights = validate_weights(
        _read_json(weights_path, "frozen ensembles"),
        selection,
        _sha256(selection_path),
        _read_json(stage4_path, "Stage4 result", 64_000_000),
        _sha256(stage4_path),
        min(member["first_fit_epoch"] for member in members),
    )
    request = _read_json(
        prepared / "requests" / f"{members[0]['fit_id']}.json", "prepared request", 32_000_000
    )
    argv = request["profile"]["argv"]
    registry = Path(request["profile"]["environment_template"]["WNN_PORTABLE_DATA_MANIFEST"])
    policy = _read_json(PACKET / "PRIMARY_REFIT_PROFILE_POLICY_R5.json", "R5 policy")
    if _sha256(registry) != policy["portable_data_registry"]["sha256"]:
        raise ValueError(
            "primary evaluation portable registry differs from the pinned training/profile registry"
        )
    return (
        members,
        weights,
        {
            "selection": artifact(selection_path),
            "frozen_weights": artifact(weights_path),
            "stage4": artifact(stage4_path),
            "data_root": argv[argv.index("--data-root") + 1],
            "primary_split": request["primary_split"],
            "portable_data_registry": artifact(registry),
            "fit_budget": artifact(runtime / "campaign-r2/budget.json"),
        },
    )
