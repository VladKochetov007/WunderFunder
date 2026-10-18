from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

REMOTE_ROOT = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
F3_PLAN_REMOTE_PATH = REMOTE_ROOT / (
    "research/campaign_2026-10-18/priority25_execution_r4/"
    "f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json"
)
FOLD3_SHA256 = "21c7fb52e8e69f38126362cbc5dc60dbc46539041d649b9206ec35e0f21ada1f"
PLAN_SHA256 = "b303146cf9db05e0a3bddf8f245d3c9b880d055ce7e97ebe5dd13510b5f4a2d8"
REGISTRY_SHA256 = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
ARCHIVE_MANIFEST_SHA256 = "9a90e46958eaed2784f1f12e6a501292e4c865b1bc6d07e3f2bd233271da7341"
SOURCE_REVISION = "a266b036b02e8e012c6eb31072291dad618a169d"
SOURCE_ARCHIVE_SHA256 = "288b4c9dd67b48114240bbf593901fc539de15aa709a1aa63388dccf64a4478a"
SOURCE_FILE_COUNT = 138
STEPS_PER_EPOCH = 14_880
BATCHES_PER_EPOCH = 1_488
SCHEDULE_UPDATES = 89_280
SCHEDULE_BATCHES = 8_928
TRAIN_GROUPS = 10_539
FIT_VALID_GROUPS = 1_368
DATASET_LENGTH = 11_907

ARTIFACTS = {
    "stage1": (
        "model_ep0.pt",
        "model_ep1.pt",
        "provenance.json",
        "training_state.pt",
        "history.json",
    ),
    "terminal": (
        "model_ep5.pt",
        "model.pt",
        "provenance.json",
        "training_state.pt",
        "history.json",
    ),
}


class CheckpointMismatch(ValueError):
    """A checkpoint or collection receipt violates the frozen F3 contract."""


def require(condition: bool, label: str) -> None:
    """Fail closed with the name of the violated invariant."""
    if not condition:
        raise CheckpointMismatch(f"{label} differs")


def mapping(value: object, label: str) -> Mapping[str, object]:
    """Require a JSON/state mapping without normalizing its producer schema."""
    require(isinstance(value, Mapping), f"{label} is not a mapping")
    return cast(Mapping[str, object], value)


def read_json(path: Path) -> dict[str, object]:
    """Read one JSON object from a local evidence file."""
    value = json.loads(path.read_text())
    return cast(dict[str, object], mapping(value, f"{path.name} JSON"))


def sha256_file(path: Path) -> str:
    """Hash one file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arm_spec(plan: Mapping[str, object], arm: str, phase: str) -> dict[str, object]:
    """Resolve exact run, attempt, command and output names from the F3 plan."""
    require(arm in ("wide", "deep"), "arm")
    require(phase in ARTIFACTS, "phase")
    pair = mapping(plan["pair_recipe"], "pair recipe")
    arms = cast(list[dict[str, object]], pair["arms"])
    matches = [row for row in arms if row["name"] == arm]
    require(len(matches) == 1, "plan arm identity")
    step_id = f"stage1_{arm}" if phase == "stage1" else f"continuation_{arm}"
    steps = cast(list[dict[str, object]], plan["steps"])
    step_matches = [row for row in steps if row["step_id"] == step_id]
    require(len(step_matches) == 1, f"plan step {step_id}")
    step = step_matches[0]
    run = str(step["run_name"])
    if phase == "terminal":
        require(run.endswith("_e6"), "terminal run suffix")
    return {
        "arm": arm,
        "phase": phase,
        "run": run,
        "attempt": step["attempt"],
        "command": step["expected_command"],
        "hidden": matches[0]["hidden"],
        "layers": matches[0]["layers"],
        "parameters": matches[0]["trainable_parameters"],
        "stage1_run": matches[0]["run_base"],
        "artifacts": ARTIFACTS[phase],
        "epochs": 6,
        "next_epoch": {"stage1": 2, "terminal": 6}[phase],
        "completed_epochs": {"stage1": 2, "terminal": 6}[phase],
    }


def validate_plan(path: Path) -> dict[str, object]:
    """Bind support tools to the immutable, unadmitted F3 plan bytes."""
    require(sha256_file(path) == PLAN_SHA256, "F3 plan SHA256")
    plan = read_json(path)
    require(plan["state"] == "UNADMITTED", "plan admission state")
    require(plan["launch_command_executed"] is False, "plan execution metadata")
    return plan


def _flag(command: list[str], name: str) -> str:
    """Read a required value from one frozen command vector."""
    require(name in command and command.index(name) + 1 < len(command), f"command {name}")
    return command[command.index(name) + 1]


def validate_attempt(
    record_path: Path, plan: Mapping[str, object], spec: Mapping[str, object]
) -> dict[str, object]:
    """Require exact plan-command identity and phase-appropriate terminal status."""
    record = read_json(record_path)
    command = cast(list[str], spec["command"])
    require(record.get("id") == spec["attempt"], "attempt id")
    require(record.get("command") == command, "attempt command")
    require(record.get("unit") == f"wnn-bandit-{spec['attempt']}.service", "attempt unit")
    require(record.get("contract") == str(F3_PLAN_REMOTE_PATH), "attempt contract path")
    require(record.get("contract_sha256") == PLAN_SHA256, "attempt contract SHA256")
    source_map = mapping(record.get("source_sha256"), "attempt source map")
    require(bool(source_map), "attempt source map empty")
    require(
        all(
            isinstance(key, str)
            and key
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for key, digest in source_map.items()
        ),
        "attempt source map shape",
    )
    require(
        source_map.get(str(F3_PLAN_REMOTE_PATH)) == PLAN_SHA256,
        "attempt source map F3 plan SHA256",
    )
    source_manifest_sha = record.get("source_manifest_sha256")
    require(
        isinstance(source_manifest_sha, str)
        and len(source_manifest_sha) == 64
        and all(character in "0123456789abcdef" for character in source_manifest_sha),
        "attempt source manifest SHA256",
    )
    expected_source_manifest_sha = hashlib.sha256(
        json.dumps(dict(source_map), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        source_manifest_sha == expected_source_manifest_sha,
        "attempt source manifest/map consistency",
    )
    require(
        str(record.get("working_directory", "")).endswith(f"/source-{SOURCE_REVISION[:8]}"),
        "attempt source root",
    )
    require(record.get("status") == "finished", "attempt status")
    require(
        type(record.get("exit_status")) is int and record.get("exit_status") == 0,
        "attempt exit status",
    )
    for flag, value in {
        "--name": spec["run"],
        "--seed": "1001",
        "--valid-fold": "3",
        "--split-manifest": str(REMOTE_ROOT / "splits/fold3.json"),
        "--precision": "fp32",
        "--hidden": str(spec["hidden"]),
        "--layers": str(spec["layers"]),
        "--epochs": "6",
        "--checkpoint-policy": "fixed",
        "--checkpoint-epoch": "5",
    }.items():
        require(_flag(command, flag) == value, f"command {flag}")
    require("--no-eval-during-fit" in command, "fit-only validation mode")
    require(
        "--stop-after-epoch" in command
        if spec["phase"] == "stage1"
        else "--resume-state" in command,
        "phase command mode",
    )
    if spec["phase"] == "stage1":
        require(_flag(command, "--stop-after-epoch") == "1", "stage1 epoch cap")
    if spec["phase"] == "terminal":
        expected_parent = (
            REMOTE_ROOT / "data/artifacts/runs" / str(spec["stage1_run"]) / "training_state.pt"
        )
        require(
            _flag(command, "--resume-state") == str(expected_parent), "continuation parent path"
        )
    return {
        "id": spec["attempt"],
        "status": record["status"],
        "exit_status": record.get("exit_status"),
        "command": command,
        "contract": record["contract"],
        "contract_sha256": record["contract_sha256"],
        "source_manifest_sha256": source_manifest_sha,
        "source_sha256": dict(source_map),
        "record_sha256": sha256_file(record_path),
    }


def expected_counts(epochs: int) -> dict[str, int]:
    """Return exact F3 optimizer counters for the given completed epochs."""
    updates = epochs * STEPS_PER_EPOCH
    batches = epochs * BATCHES_PER_EPOCH
    return {
        "optimizer_attempts": updates,
        "amp_skipped_updates": 0,
        "optimizer_completed_updates": updates,
        "ema_updates": 0,
        "completed_batches": batches,
    }


def validate_history(history: object, completed_epochs: int) -> list[dict[str, object]]:
    """Validate complete fit-only epoch summaries and corrected counter schemas."""
    require(isinstance(history, list) and len(history) == completed_epochs, "history length")
    rows = cast(list[dict[str, object]], history)
    for epoch, row in enumerate(rows):
        require(
            row.get("epoch") == epoch and row.get("evaluation") == "withheld",
            f"history epoch {epoch}",
        )
        require(
            math.isfinite(float(row["train_seconds"])) and float(row["train_seconds"]) > 0,
            f"epoch {epoch} duration",
        )
        diagnostics = mapping(row["update_diagnostics"], f"epoch {epoch} diagnostics")
        one = expected_counts(1)
        cumulative = expected_counts(epoch + 1)
        require(diagnostics.get("epoch") == one, f"epoch {epoch} counters")
        require(diagnostics.get("cumulative") == cumulative, f"epoch {epoch} cumulative counters")
        grad = mapping(diagnostics["grad_stats"], f"epoch {epoch} gradient stats")
        require(grad.get("epoch") == epoch, f"epoch {epoch} gradient epoch")
        require(grad.get("nonfinite_steps") == 0, f"epoch {epoch} nonfinite gradients")
        fit = mapping(row["fit_summary"], f"epoch {epoch} fit summary")
        require(math.isfinite(float(fit["objective_mean"])), f"epoch {epoch} objective")
        require(fit.get("optimizer_windows") == STEPS_PER_EPOCH, f"epoch {epoch} optimizer windows")
        require(fit.get("optimizer_updates") == one, f"epoch {epoch} fit update dictionary")
        rates = cast(list[dict[str, object]], fit["learning_rate_by_group"])
        require(
            len(rates) == 1
            and all(math.isfinite(float(rates[0][key])) for key in ("first", "last", "mean")),
            f"epoch {epoch} learning rates",
        )
    return rows


def validate_parent_headroom(
    current_bytes: int,
    maximum_bytes: int,
    *,
    worker_reserve_bytes: int = 8_000_000_000,
    support_reserve_bytes: int = 512_000_000,
) -> None:
    """Require live slice headroom for one worker plus bounded support process."""
    require(current_bytes >= 0 and maximum_bytes > 0, "parent cgroup memory values")
    require(
        current_bytes + worker_reserve_bytes + support_reserve_bytes <= maximum_bytes,
        "parent cgroup memory headroom",
    )


def validate_systemd_runtime_limit(value: str) -> None:
    """Require the provider's observed systemd two-minute duration encoding."""
    require(value == "2min", "collector systemd runtime limit")


def is_resume_model_checkpoint(name: str, phase: str) -> bool:
    """Identify only the endpoint snapshot that must equal the live resume model."""
    require(phase in ARTIFACTS, "checkpoint phase")
    endpoint_epoch = {"stage1": 1, "terminal": 5}[phase]
    return name == f"model_ep{endpoint_epoch}.pt" or (phase == "terminal" and name == "model.pt")
