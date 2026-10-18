from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import cast

REMOTE_ROOT = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
FOLD1_SHA256 = "d8e9748d770bb3af525943b36b1801e6fd72d78259cc6b823d4661ffd78367d9"
PLAN_SHA256 = "96cfb67b3447866651ae04638166f17dccb9729b3d4a9340f1b72b8d7b089f96"
REGISTRY_SHA256 = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
ARCHIVE_MANIFEST_SHA256 = "9a90e46958eaed2784f1f12e6a501292e4c865b1bc6d07e3f2bd233271da7341"
SOURCE_REVISION = "a266b036b02e8e012c6eb31072291dad618a169d"
SOURCE_ARCHIVE_SHA256 = "288b4c9dd67b48114240bbf593901fc539de15aa709a1aa63388dccf64a4478a"
SOURCE_FILE_COUNT = 138
STEPS_PER_EPOCH = 14_930
BATCHES_PER_EPOCH = 1_493
SCHEDULE_UPDATES = 89_580
SCHEDULE_BATCHES = 8_958
TRAIN_GROUPS = 10_533
FIT_VALID_GROUPS = 1_416
DATASET_LENGTH = 11_949

ARTIFACTS = {
    "epoch0": ("model_ep0.pt", "provenance.json", "training_state.pt"),
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
    """A checkpoint or collection receipt violates the frozen F1 contract."""


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
    """Resolve exact run, attempt, command and output names from the F1 plan."""
    require(arm in ("wide", "deep"), "arm")
    require(phase in ARTIFACTS, "phase")
    pair = mapping(plan["pair_recipe"], "pair recipe")
    arms = cast(list[dict[str, object]], pair["arms"])
    matches = [row for row in arms if row["name"] == arm]
    require(len(matches) == 1, "plan arm identity")
    step_id = f"stage1_{arm}" if phase in ("epoch0", "stage1") else f"continuation_{arm}"
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
        "next_epoch": {"epoch0": 1, "stage1": 2, "terminal": 6}[phase],
        "completed_epochs": {"epoch0": 1, "stage1": 2, "terminal": 6}[phase],
    }


def validate_plan(path: Path) -> dict[str, object]:
    """Bind support tools to the immutable, unadmitted F1 plan bytes."""
    require(sha256_file(path) == PLAN_SHA256, "F1 plan SHA256")
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
    require(
        str(record.get("working_directory", "")).endswith(f"/source-{SOURCE_REVISION[:8]}"),
        "attempt source root",
    )
    if spec["phase"] == "epoch0":
        require(record.get("status") in ("running", "finished"), "epoch0 attempt status")
        if record.get("status") == "running":
            require(record.get("exit_status") is None, "running attempt exit status")
        else:
            require(
                type(record.get("exit_status")) is int and record["exit_status"] == 0,
                "epoch0 terminal exit status",
            )
    else:
        require(record.get("status") == "finished", "attempt status")
        require(
            type(record.get("exit_status")) is int and record["exit_status"] == 0,
            "attempt exit status",
        )
    for flag, value in {
        "--name": spec["run"],
        "--seed": "1001",
        "--valid-fold": "1",
        "--split-manifest": str(REMOTE_ROOT / "splits/fold1.json"),
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
        if spec["phase"] in ("epoch0", "stage1")
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
        "record_sha256": sha256_file(record_path),
    }


def expected_counts(epochs: int) -> dict[str, int]:
    """Return exact F1 optimizer counters for the given completed epochs."""
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


def validate_optional_epoch0_binding(
    receipt: Mapping[str, object] | None,
    arm: str,
    run: str,
    model_sha256: str,
) -> bool:
    """Bind an earlier epoch0 receipt when present without requiring it."""
    if receipt is None:
        return False
    require(
        (receipt.get("schema"), receipt.get("arm"), receipt.get("phase"), receipt.get("run"))
        == ("priority25.f1_checkpoint_collection.v1", arm, "epoch0", run),
        "prior epoch0 collection identity",
    )
    artifacts = mapping(receipt["artifacts"], "prior epoch0 artifacts")
    model = mapping(artifacts["model_ep0.pt"], "prior epoch0 model")
    require(model.get("sha256") == model_sha256, "stage1 epoch0 model binding")
    return True


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
    endpoint_epoch = {"epoch0": 0, "stage1": 1, "terminal": 5}[phase]
    return name == f"model_ep{endpoint_epoch}.pt" or (phase == "terminal" and name == "model.pt")


def validate_adam_state_descriptors(
    parameter_shapes: list[tuple[int, ...]],
    states: list[Mapping[str, object]],
    expected_step_value: int,
) -> None:
    """Reject malformed Adam moments or step counters using CPU descriptors."""
    expected_shapes = Counter(parameter_shapes)
    observed_shapes: Counter[tuple[int, ...]] = Counter()
    require(len(states) == len(parameter_shapes), "Adam state parameter count")
    for index, state in enumerate(states):
        require(
            state["state_fields"] == ("exp_avg", "exp_avg_sq", "step"),
            f"Adam state {index} fields",
        )
        require(
            state["step_numel"] == 1 and state["step_finite"] is True, f"Adam state {index} step"
        )
        require(
            float(state["step_value"]) == expected_step_value,
            f"Adam state {index} step value",
        )
        require(state["exp_avg_dtype"] == "torch.float32", f"Adam state {index} exp_avg dtype")
        require(
            state["exp_avg_sq_dtype"] == "torch.float32", f"Adam state {index} exp_avg_sq dtype"
        )
        require(state["moments_finite"] is True, f"Adam state {index} moments finite")
        first = cast(tuple[int, ...], state["exp_avg_shape"])
        second = cast(tuple[int, ...], state["exp_avg_sq_shape"])
        require(first == second, f"Adam state {index} moment shapes")
        observed_shapes[first] += 1
    require(observed_shapes == expected_shapes, "Adam moment/model parameter shapes")


def validate_pair_contracts(left: Mapping[str, object], right: Mapping[str, object]) -> None:
    """Require identical F1 data/runtime recipe except declared architecture fields."""
    left_options = mapping(left["options"], "left options")
    right_options = mapping(right["options"], "right options")
    differences = {
        key
        for key in set(left_options) | set(right_options)
        if left_options.get(key) != right_options.get(key)
    }
    require(differences == {"hidden", "layers"}, "paired option differences")
    require(
        {"hidden": left_options["hidden"], "layers": left_options["layers"]}
        == {"hidden": 512, "layers": 1},
        "wide architecture",
    )
    require(
        {"hidden": right_options["hidden"], "layers": right_options["layers"]}
        == {"hidden": 384, "layers": 2},
        "deep architecture",
    )
    left_rest, right_rest = dict(left), dict(right)
    left_rest.pop("options")
    right_rest.pop("options")
    require(left_rest == right_rest, "paired non-option contract")
