from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

JSONDict = dict[str, Any]  # Controller and producer JSON are dynamic but schema-checked.


def sha256_file(path: Path) -> str:
    """Hash a frozen source or output without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash JSON-compatible values using the controller's canonical encoding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> JSONDict:
    """Read a required JSON object and reject other top-level shapes."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    """Write a gate receipt atomically so launch intent survives interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as output:
        temporary = Path(output.name)
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def finite_tree(value: object, path: str = "root") -> None:
    """Reject non-finite numeric diagnostics, including nested lists/maps."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value in {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, f"{path}[{index}]")


def verify_policy(root: Path, packet: JSONDict) -> tuple[JSONDict, JSONDict, JSONDict]:
    """Bind this gate to the exact admitted queue, root allocation and sources."""
    queue_path = root / packet["queue_path"]
    authority_path = root / packet["authority_path"]
    queue_bytes = queue_path.read_bytes()
    authority_bytes = authority_path.read_bytes()
    if hashlib.sha256(queue_bytes).hexdigest() != packet["queue_sha256"]:
        raise ValueError("frozen two-fit queue hash changed")
    if hashlib.sha256(authority_bytes).hexdigest() != packet["authority_sha256"]:
        raise ValueError("root allocation hash changed")
    queue = json.loads(queue_bytes)
    authority = json.loads(authority_bytes)
    profile_plan_path = root / packet["profile_plan_path"]
    if sha256_file(profile_plan_path) != packet["base_recipe_plan_sha256"]:
        raise ValueError("frozen profile/recipe plan hash changed")
    profile_plan = read_json(profile_plan_path)
    if queue.get("status") != "finite_two_fit_queue_prepared":
        raise ValueError("queue is not the admitted finite two-fit queue")
    if queue.get("root_authority_sha256") != packet["authority_sha256"]:
        raise ValueError("queue does not bind the root allocation")
    if Path(queue.get("root_authority", "")).resolve() != authority_path.resolve():
        raise ValueError("queue authority path differs from the frozen root allocation")
    if queue.get("profile_authority_sha256") != packet["profile_authority_sha256"]:
        raise ValueError("queue profile authority differs from the gate packet")
    profile_authority_path = Path(queue["profile_authority"]).resolve(strict=True)
    if sha256_file(profile_authority_path) != packet["profile_authority_sha256"]:
        raise ValueError("profile admission hash changed")
    profile_authority = read_json(profile_authority_path)
    if (
        profile_authority.get("status") != "admitted_profile_only"
        or profile_authority.get("plan_sha256") != packet["base_recipe_plan_sha256"]
    ):
        raise ValueError("profile authority does not admit the frozen profile plan")
    expected_profile_attempt = (
        root / packet["runtime_relative"] / "attempts" / f"{packet['profile_attempt']}.json"
    ).resolve()
    if Path(queue.get("profile_attempt", "")).resolve() != expected_profile_attempt:
        raise ValueError("queue profile attempt path differs from the gate packet")
    if queue.get("base_recipe_plan_sha256") != packet["base_recipe_plan_sha256"]:
        raise ValueError("queue recipe-plan identity changed")
    if profile_plan.get("source_pins_sha256") != queue.get("source_pins_sha256"):
        raise ValueError("profile plan and fit queue source pins differ")
    profile_steps = profile_plan.get("sequence", [])
    if not profile_steps or profile_steps[0].get("attempt_id") != packet["profile_attempt"]:
        raise ValueError("profile plan does not name the admitted real-device profile")
    if queue.get("fit_count_maximum") != 2 or queue.get("scientific_recipe_unchanged") is not True:
        raise ValueError("queue no longer represents exactly two unchanged-recipe fits")
    if authority.get("status") != "conditional_exact_successor_authorized":
        raise ValueError("root allocation does not authorize this successor")
    if authority.get("basis_plan_sha256") != packet["base_recipe_plan_sha256"]:
        raise ValueError("root allocation basis differs from the frozen fit plan")
    expected_authority = {
        "maximum_fits": 2,
        "cpu_cores": 2,
        "memory_max_bytes": 9_000_000_000,
        "observed_total_vram_max_bytes": 5_000_000_000,
        "swap_bytes": 0,
        "runtime_seconds_per_fit": 7200,
        "aggregate_fit_runtime_seconds": 14400,
    }
    if any(authority.get(key) != value for key, value in expected_authority.items()):
        raise ValueError("root fit limits differ from the frozen two-fit contract")
    disk = authority.get("disk_allocation", {})
    expected_disk = {
        "maximum_new_bytes_per_fit": 500_000_000,
        "minimum_free_bytes_before_each_fit": 2_000_000_000,
        "large_predictions_or_cache_exports": False,
    }
    if any(disk.get(key) != value for key, value in expected_disk.items()):
        raise ValueError("root disk allocation differs from the checkpoint-only fit contract")
    queue_disk = queue.get("disk_gate", {})
    if any(queue_disk.get(key) != value for key, value in expected_disk.items()):
        raise ValueError("queue disk gate differs from the root allocation")
    steps = queue.get("steps")
    if not isinstance(steps, list) or len(steps) != 2:
        raise ValueError("queue must contain exactly seed3 then seed4")
    if [step.get("attempt_id") for step in steps] != packet["attempt_ids"]:
        raise ValueError("queue attempt order differs from the gate packet")
    if any(step.get("maximum_new_bytes") != 500_000_000 for step in steps):
        raise ValueError("fit output cap differs from the root allocation")
    if [step.get("queue_index") for step in steps] != [1, 2]:
        raise ValueError("queue fit order is not the exact planned index-1/index-2 pair")
    if [step.get("run_name") for step in steps] != ["bd26_w384raw_s3_f1", "bd26_w384raw_s4_f1"]:
        raise ValueError("queue run names differ from the admitted seed3/seed4 pair")
    expected_resources = {
        "cpu_quota_percent": 200,
        "memory_max_bytes": 9_000_000_000,
        "swap_max_bytes": 0,
        "observed_total_vram_max_bytes": 5_000_000_000,
        "timeout_seconds": 7200,
        "exclusive_local_gpu": True,
    }
    if any(step.get("resource_request") != expected_resources for step in steps):
        raise ValueError("fit resource request differs from the admitted envelope")
    if queue.get("source_pins_sha256") != packet.get("source_pins_sha256"):
        raise ValueError("gate packet and queue source pins differ")
    if any(step.get("source_pins_sha256") != queue["source_pins_sha256"] for step in steps):
        raise ValueError("seed source pins differ within the finite queue")
    for relative, expected in queue["source_pins_sha256"].items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"source pin changed: {relative}")
    if sha256_file(root / packet["gate_source_relative"]) != packet["gate_source_sha256"]:
        raise ValueError("postfit gate source hash differs from the frozen packet")
    if sha256_file(root / packet["test_source_relative"]) != packet["test_source_sha256"]:
        raise ValueError("postfit gate test source hash differs from the frozen packet")
    for relative, expected in packet["control_source_pins_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"durable controller source pin changed: {relative}")
    expected_commands = packet["command_sha256_canonical_json"]
    for step, expected_hash in zip(steps, expected_commands, strict=True):
        if canonical_sha256(step.get("command")) != expected_hash:
            raise ValueError(f"frozen command hash changed for {step['attempt_id']}")
        if step.get("command_sha256_canonical_json") != expected_hash:
            raise ValueError(f"queue command receipt differs for {step['attempt_id']}")
    return queue, authority, profile_plan


def verify_attempt_record(record: JSONDict, step: JSONDict, authority_sha256: str) -> None:
    """Require exact command, completed unit and bounded non-OOM resources."""
    verify_attempt_binding(record, step, authority_sha256)
    if record.get("status") != "finished" or record.get("exit_status") != 0:
        raise ValueError("seed3 attempt did not finish successfully")
    events = record.get("memory_event_deltas", {})
    if any(int(events.get(key, -1)) != 0 for key in ("oom", "oom_kill", "oom_group_kill")):
        raise ValueError("seed3 attempt recorded an OOM event")
    if int(record.get("peak_ram_bytes", 0)) > 9_000_000_000:
        raise ValueError("seed3 exceeded its cgroup memory limit")
    if int(record.get("peak_observed_vram_bytes", 0)) > 5_000_000_000:
        raise ValueError("seed3 exceeded its observed VRAM limit")
    if not record.get("unit") or not record.get("finished_utc"):
        raise ValueError("seed3 terminal unit receipt is incomplete")


def verify_attempt_binding(record: JSONDict, step: JSONDict, authority_sha256: str) -> None:
    """Check identity and exact command for either an active or terminal attempt."""
    expected_hash = step["command_sha256_canonical_json"]
    if record.get("id") != step["attempt_id"]:
        raise ValueError("attempt record ID differs from the frozen queue")
    if record.get("command") != step["command"]:
        raise ValueError("seed3 executed command differs from the frozen queue")
    if canonical_sha256(record["command"]) != expected_hash:
        raise ValueError("seed3 attempt command hash differs from the frozen queue")
    if record.get("contract_sha256") != authority_sha256:
        raise ValueError("seed3 attempt is not bound to the admitted root allocation")


def verify_unit(unit: str) -> dict[str, str]:
    """Confirm systemd saw a successful terminal result for the fit unit."""
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=ActiveState,Result,ExecMainStatus,SubState",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if values.get("ActiveState") != "inactive" or values.get("Result") != "success":
        raise ValueError(f"seed3 systemd unit is not a successful terminal unit: {values}")
    if values.get("ExecMainStatus") != "0":
        raise ValueError(f"seed3 systemd main process did not exit zero: {values}")
    return values


def verify_profile_log(path: Path, authority: JSONDict) -> JSONDict:
    """Recheck the real-device finite-update receipt bound by the profile."""
    records = [
        line.removeprefix("PROFILE_UPDATE_EVIDENCE ")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("PROFILE_UPDATE_EVIDENCE ")
    ]
    if len(records) != 1:
        raise ValueError("profile log must contain exactly one update-evidence record")
    evidence = json.loads(records[0])
    expected = authority.get("profile_result", {})
    if (
        evidence.get("optimizer_attempts") != expected.get("completed_updates")
        or evidence.get("optimizer_completed_updates") != expected.get("completed_updates")
        or evidence.get("amp_skipped_updates") != expected.get("skipped_updates")
        or evidence.get("valid_for_throughput") is not True
        or evidence.get("loss_finite") is not True
        or evidence.get("gradient_finite") is not True
        or evidence.get("counts_consistent") is not True
    ):
        raise ValueError("actual profile update evidence differs from root admission")
    return evidence


def verify_provenance_membership(
    step: dict[str, Any],
    provenance: JSONDict,
    split: Any,
    source_pins: dict[str, str],
    require_selected_epoch: bool,
) -> dict[str, int]:
    """Verify exact fold1 membership and that no evaluation component was fitted."""
    args = provenance.get("args", {})
    expected = {
        "name": step["run_name"],
        "device": "cuda",
        "arch": "gru",
        "cell": "gru",
        "hidden": 384,
        "layers": 2,
        "dropout": 0.0,
        "seed": 3,
        "raw_only": True,
        "loss": "wmse",
        "loss_norm": "chunk",
        "train_source": "train+valid",
        "train_seqs": 10607,
        "valid_fold": 1,
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
        "batch": 16,
        "chunk": 2000,
        "lr": 0.001,
        "wd": 0.01,
        "optim": "adamw",
        "sched": "onecycle",
        "precision": "fp16",
        "save_training_state": True,
    }
    mismatches = {
        key: (args.get(key), value) for key, value in expected.items() if args.get(key) != value
    }
    if mismatches:
        raise ValueError(f"provenance recipe differs from the frozen command: {mismatches}")
    manifest_relative = "research/campaign_2026-09-29/artifacts/splits/fold1.json"
    manifest_hash = source_pins[manifest_relative]
    if provenance.get("split_manifest_sha256") != manifest_hash:
        raise ValueError("provenance split-manifest hash differs from the frozen fold1 source")
    if args.get("split_manifest_sha256") != manifest_hash:
        raise ValueError("parsed trainer arguments do not bind the frozen fold1 manifest")
    if Path(str(args.get("split_manifest", ""))).name != "fold1.json":
        raise ValueError("parsed trainer arguments do not name the fold1 split")
    if require_selected_epoch and provenance.get("selected_epoch") != 3:
        raise ValueError("fixed epoch-3 checkpoint was not selected")
    if provenance.get("run_identity_source_sha256") != source_pins["src/wnn/run_identity.py"]:
        raise ValueError("provenance run-identity implementation differs from the frozen source")
    fit = provenance.get("fit_row_groups", {})
    train = list(map(int, fit.get("train", [])))
    valid = list(map(int, fit.get("valid", [])))
    evaluation = list(map(int, provenance.get("evaluation_row_groups", [])))
    if set(train) != set(split.train_groups) or set(valid) != set(split.valid_groups):
        raise ValueError("producer fit memberships do not equal the frozen fold1 fit sets")
    if len(set(train)) != len(train) or len(set(valid)) != len(valid):
        raise ValueError("producer fit memberships contain duplicate row groups")
    if len(set(evaluation)) != len(evaluation):
        raise ValueError("producer evaluation membership contains duplicate row groups")
    if len(set(train)) != len(train) or len(set(valid)) != len(valid):
        raise ValueError("producer fit memberships contain duplicate row groups")
    if len(set(evaluation)) != len(evaluation):
        raise ValueError("producer evaluation membership contains duplicate row groups")
    if set(evaluation) != set(split.evaluation_groups):
        raise ValueError("recorded evaluation membership differs from the full fold1 holdout")
    split.assert_fit_groups(train, valid)
    from wnn.run_identity import group_identity

    identity = provenance.get("fit_membership_identity", {})
    if identity.get("train") != group_identity(train) or identity.get("valid") != group_identity(
        valid
    ):
        raise ValueError("ordered fit membership identities do not match the recorded groups")
    evaluated_components = {
        component
        for source, group, component in split.components
        if source == "valid" and group in set(split.evaluation_groups)
    }
    component_of = {(source, group): component for source, group, component in split.components}
    fitted_components = {component_of[("train", group)] for group in train} | {
        component_of[("valid", group)] for group in valid
    }
    if fitted_components & evaluated_components:
        raise ValueError("fit/evaluation connected components overlap")
    return {"train": len(train), "valid": len(valid), "evaluation": len(evaluation)}


def directory_bytes(path: Path, maximum: int) -> int:
    """Reject symlinks and account every produced byte under the admitted cap."""
    total = 0
    for current, directories, filenames in os.walk(path, followlinks=False):
        for name in directories:
            child = Path(current) / name
            if child.is_symlink():
                raise ValueError(f"unexpected symlink in fit output: {child}")
        for name in filenames:
            child = Path(current) / name
            if child.is_symlink() or not child.is_file():
                raise ValueError(f"unexpected non-regular fit output: {child}")
            total += child.stat().st_size
            if total > maximum:
                raise ValueError(f"fit output exceeds the {maximum}-byte cap")
    return total


def finite_checkpoint(path: Path) -> JSONDict:
    """Load the producer's inference checkpoint on CPU and check every tensor."""
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("epoch") != 3:
        raise ValueError("model_ep3 is not a fixed epoch-3 inference checkpoint")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("epoch-3 checkpoint has no model state")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint state {name} is not a tensor")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"checkpoint state {name} contains non-finite weights")
    return checkpoint


def verify_output(root: Path, step: JSONDict, split: Any, source_pins: dict[str, str]) -> JSONDict:
    """Audit checkpoint, training state, provenance, diagnostics and disk use."""
    output = root / step["expected_output_path"]
    if not output.is_dir():
        raise FileNotFoundError(f"fit output directory is missing: {output}")
    total_bytes = directory_bytes(output, int(step["maximum_new_bytes"]))
    required = ("model_ep3.pt", "model.pt", "provenance.json", "history.json", "training_state.pt")
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required final fit artifacts are missing: {missing}")
    provenance = read_json(output / "provenance.json")
    checkpoint_map = provenance.get("checkpoint_sha256", {})
    for name in ("model_ep3.pt", "model.pt"):
        if checkpoint_map.get(name) != sha256_file(output / name):
            raise ValueError(f"checkpoint SHA differs from producer provenance: {name}")
    checkpoint = finite_checkpoint(output / "model_ep3.pt")
    if checkpoint.get("args", {}).get("name") != step["run_name"]:
        raise ValueError("epoch-3 checkpoint run name differs")
    if checkpoint.get("args", {}).get("seed") != 3:
        raise ValueError("epoch-3 checkpoint seed differs")
    if checkpoint.get("args") != provenance.get("args"):
        raise ValueError("checkpoint arguments differ from the producer sidecar")
    embedded = checkpoint.get("provenance", {})
    verify_provenance_membership(step, provenance, split, source_pins, True)
    # The producer snapshots embedded provenance before assigning selected_epoch
    # to its sidecar; checkpoint.epoch above is the authoritative embedded epoch.
    verify_provenance_membership(step, embedded, split, source_pins, False)
    if embedded.get("fit_membership_identity") != provenance.get("fit_membership_identity"):
        raise ValueError("checkpoint ancestry membership differs from the sidecar")
    selected = finite_checkpoint(output / "model.pt")
    selected_state = selected.get("model", {})
    if selected.get("epoch") != 3 or selected_state.keys() != checkpoint["model"].keys():
        raise ValueError("selected model.pt is not the matching epoch-3 checkpoint")
    if selected.get("args") != provenance.get("args"):
        raise ValueError("selected model.pt arguments differ from the producer sidecar")
    if any(
        not torch.equal(selected_state[name], checkpoint["model"][name])
        for name in checkpoint["model"]
    ):
        raise ValueError("selected model.pt tensors differ from model_ep3.pt")
    training = torch.load(output / "training_state.pt", map_location="cpu", weights_only=False)
    if (
        not isinstance(training, dict)
        or training.get("format") != "wnn.epoch-training-state"
        or training.get("version") != 1
        or training.get("boundary") != "epoch"
        or training.get("next_epoch") != 4
        or training.get("contract", {}).get("epochs") != 4
    ):
        raise ValueError("saved training state is not the final epoch-3 continuation boundary")
    contract = training["contract"]
    if contract.get("epochs") != 4 or contract.get("fit_membership") != provenance.get(
        "fit_membership_identity"
    ):
        raise ValueError("training-state contract differs from the fixed fold1 fit membership")
    expected_contract_sources = {
        "trainer": source_pins["scripts/train_rnn.py"],
        "nn.py": source_pins["src/wnn/nn.py"],
        "splits.py": source_pins["src/wnn/splits.py"],
        "optim.py": source_pins["src/wnn/optim.py"],
        "run_identity.py": source_pins["src/wnn/run_identity.py"],
    }
    contract_sources = contract.get("sources", {})
    if any(contract_sources.get(key) != value for key, value in expected_contract_sources.items()):
        raise ValueError("training-state source contract differs from the frozen fit sources")
    contract_options = contract.get("options", {})
    if any(
        contract_options.get(key) != value
        for key, value in {
            "seed": 3,
            "valid_fold": 1,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 3,
            "no_eval_during_fit": True,
        }.items()
    ):
        raise ValueError("training-state recipe contract differs from the fixed fold1 fit")
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != 4:
        raise ValueError("history does not contain exactly four fixed-epoch records")
    if [entry.get("epoch") for entry in history] != list(range(4)):
        raise ValueError("history epoch indices differ from the fixed four-epoch recipe")
    if any(entry.get("evaluation") != "withheld" for entry in history):
        raise ValueError("fit history contains an unexpected evaluation score")
    finite_tree(history)
    diagnostics = provenance.get("update_diagnostics", [])
    if len(diagnostics) != 4:
        raise ValueError("update diagnostics do not contain four completed epochs")
    totals = {"optimizer_attempts": 0, "optimizer_completed_updates": 0, "amp_skipped_updates": 0}
    for index, entry in enumerate(diagnostics):
        if entry.get("epoch_index") != index:
            raise ValueError("update diagnostics epoch order is invalid")
        epoch = entry.get("epoch", {})
        attempts = int(epoch.get("optimizer_attempts", -1))
        completed = int(epoch.get("optimizer_completed_updates", -1))
        skipped = int(epoch.get("amp_skipped_updates", -1))
        if attempts <= 0 or completed <= 0 or skipped < 0 or attempts != completed + skipped:
            raise ValueError(f"invalid optimizer update counters in epoch {index}")
        if int(epoch.get("completed_batches", 0)) <= 0:
            raise ValueError(f"no completed batches recorded in epoch {index}")
        for key in totals:
            totals[key] += int(epoch[key])
    if (
        diagnostics[-1].get("cumulative", {}).get("optimizer_attempts")
        != totals["optimizer_attempts"]
    ):
        raise ValueError("cumulative optimizer-attempt count does not reconcile")
    finite_tree(diagnostics)
    return {
        "output_bytes": total_bytes,
        "checkpoint_sha256": checkpoint_map["model_ep3.pt"],
        "model_sha256": checkpoint_map["model.pt"],
        "training_state_sha256": sha256_file(output / "training_state.pt"),
        "training_state_bytes": (output / "training_state.pt").stat().st_size,
        "fit_membership_counts": {
            "train": len(provenance["fit_row_groups"]["train"]),
            "valid": len(provenance["fit_row_groups"]["valid"]),
            "evaluation": len(provenance["evaluation_row_groups"]),
        },
        "optimizer_totals": totals,
    }


def launch_seed4(
    root: Path,
    authority_path: Path,
    runtime: Path,
    step: JSONDict,
) -> JSONDict:
    """Use the existing guarded resource controller for the one successor fit."""
    attempt_path = runtime / "attempts" / f"{step['attempt_id']}.json"
    if attempt_path.exists():
        raise FileExistsError(f"seed4 attempt already exists; refusing a relaunch: {attempt_path}")
    output = root / step["expected_output_path"]
    if output.exists():
        raise FileExistsError(f"seed4 output already exists; refusing overwrite: {output}")
    stop_marker = runtime / "STOP_ADMISSION.json"
    if stop_marker.exists():
        raise RuntimeError("new admission is closed; seed4 remains unlaunched")
    if shutil.disk_usage(root).free < 2_000_000_000:
        raise RuntimeError("free disk fell below the admitted 2 GB before seed4")
    command = [
        sys.executable,
        str(root / "scripts/bandit_resources.py"),
        "--root",
        str(root),
        "--state",
        str(runtime),
        "launch",
        "--attempt",
        step["attempt_id"],
        "--kind",
        "gpu",
        "--cpus",
        "2",
        "--reserve-bytes",
        "9000000000",
        "--profile",
        "FOLD1-GRU384-B16-RAW-SEED4-FIXED-EP3-CHECKPOINT",
        "--contract",
        str(authority_path),
        "--admission-timeout",
        "45",
        "--",
        *step["command"],
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=50)
    if result.returncode != 0:
        raise RuntimeError(f"guarded seed4 launch failed: {result.stderr[-4000:]}")
    launched = read_json(attempt_path)
    if launched.get("command") != step["command"]:
        raise ValueError("guarded seed4 record does not match its frozen command")
    if launched.get("status") not in {"reserved", "waiting", "running"}:
        raise RuntimeError(f"seed4 guardian did not reserve a live fit: {launched.get('status')}")
    if launched.get("contract_sha256") != hashlib.sha256(authority_path.read_bytes()).hexdigest():
        raise ValueError("seed4 guardian record is not bound to the root allocation")
    return {
        "status": launched["status"],
        "unit": launched["unit"],
        "heartbeat_utc": launched["heartbeat_utc"],
    }


def run(packet_path: Path, expected_packet_sha256: str, preflight_only: bool) -> None:
    """Validate exact seed3 outputs and, only on a complete pass, admit seed4."""
    if sha256_file(packet_path) != expected_packet_sha256:
        raise ValueError("postfit gate packet hash differs from the frozen sequence command")
    packet = read_json(packet_path)
    root = Path(packet["root"]).resolve(strict=True)
    sys.path.insert(0, str(root / "src"))
    from wnn.splits import load_campaign_split

    queue, authority, profile_plan = verify_policy(root, packet)
    runtime = (root / packet["runtime_relative"]).resolve(strict=True)
    expected_runtime = root / "research/campaign_2026-10-06/bandit_continuation/runtime"
    if runtime != expected_runtime.resolve(strict=True):
        raise ValueError("runtime path differs from the admitted campaign controller")
    profile = read_json(runtime / "attempts" / f"{packet['profile_attempt']}.json")
    profile_step = profile_plan["sequence"][0]
    profile_step = {
        "attempt_id": profile_step["attempt_id"],
        "command": profile_step["command"],
        "command_sha256_canonical_json": canonical_sha256(profile_step["command"]),
    }
    verify_attempt_record(profile, profile_step, packet["profile_authority_sha256"])
    if Path(str(profile.get("log", ""))).resolve() != Path(queue["profile_log"]).resolve():
        raise ValueError("profile attempt log path differs from the frozen queue")
    profile_sources = profile.get("source_sha256", {})
    required_profile_sources = {
        **{
            relative: queue["source_pins_sha256"][relative]
            for relative in (
                "scripts/train_rnn.py",
                "src/wnn/nn.py",
                "src/wnn/splits.py",
                "src/wnn/optim.py",
                "src/wnn/run_identity.py",
                "research/campaign_2026-09-29/artifacts/splits/fold1.json",
            )
        },
        **packet["control_source_pins_sha256"],
    }
    for relative, expected in required_profile_sources.items():
        if profile_sources.get(relative) != expected:
            raise ValueError(f"profile attempt source manifest differs for {relative}")
    if authority.get("profile_result", {}).get("status") != "passed":
        raise ValueError("root allocation no longer records the hardware profile as passed")
    profile_unit = verify_unit(str(profile["unit"]))
    if (
        profile.get("peak_ram_bytes", 0) > 9_000_000_000
        or profile.get("peak_observed_vram_bytes", 0) > 5_000_000_000
    ):
        raise ValueError("profile resource usage exceeds the admitted envelope")
    profile_log = Path(queue["profile_log"])
    profile_evidence = verify_profile_log(profile_log, authority)
    profile_result = authority["profile_result"]
    if (
        profile.get("peak_ram_bytes") != profile_result.get("ram_peak_bytes")
        or profile.get("peak_observed_vram_bytes") != profile_result.get("observed_vram_peak_bytes")
        or profile_result.get("oom_events") != 0
    ):
        raise ValueError("actual profile resource receipt differs from the root's measured profile")
    profile_output = profile_plan["sequence"][0].get("output_path")
    if profile_output:
        profile_output_path = root / profile_output
        if profile_output_path.exists() and any(profile_output_path.glob("model*.pt")):
            raise ValueError("profile unexpectedly produced a model checkpoint")
    step3, step4 = queue["steps"]
    attempt3_path = runtime / "attempts" / f"{step3['attempt_id']}.json"
    attempt3 = read_json(attempt3_path)
    verify_attempt_binding(attempt3, step3, packet["authority_sha256"])
    recorded_sources = attempt3.get("source_sha256", {})
    required_recorded_sources = {
        "scripts/train_rnn.py",
        "src/wnn/nn.py",
        "src/wnn/splits.py",
        "src/wnn/optim.py",
        "src/wnn/run_identity.py",
        "research/campaign_2026-09-29/artifacts/splits/fold1.json",
    }
    for relative in required_recorded_sources:
        if recorded_sources.get(relative) != queue["source_pins_sha256"].get(relative):
            raise ValueError(f"seed3 attempt source manifest differs for {relative}")
    for relative, expected in packet["control_source_pins_sha256"].items():
        if recorded_sources.get(relative) != expected:
            raise ValueError(f"seed3 attempt controller source differs for {relative}")
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < 2_000_000_000:
        raise RuntimeError("free disk is below the admitted 2 GB before the successor gate")
    seed4_attempt_path = runtime / "attempts" / f"{step4['attempt_id']}.json"
    seed4_output_path = root / step4["expected_output_path"]
    if seed4_attempt_path.exists() or seed4_output_path.exists():
        raise FileExistsError("seed4 attempt/output exists; refusing to overwrite or duplicate")
    if (runtime / "STOP_ADMISSION.json").exists():
        raise RuntimeError("new admission is closed; sequence must drain without seed4")
    if preflight_only:
        print(
            json.dumps(
                {
                    "status": "static_preflight_passed_no_launch",
                    "seed3_status": attempt3.get("status"),
                    "seed3_command_sha256": step3["command_sha256_canonical_json"],
                    "profile_update_evidence": profile_evidence,
                    "source_pins_sha256": queue["source_pins_sha256"],
                    "control_source_pins_sha256": packet["control_source_pins_sha256"],
                    "disk_free_bytes": free_bytes,
                    "seed4_attempt_absent": True,
                    "seed4_output_absent": True,
                    "disk_pressure_context": packet["disk_pressure_context"],
                },
                indent=2,
            )
        )
        return
    verify_attempt_record(attempt3, step3, packet["authority_sha256"])
    fit_unit = verify_unit(str(attempt3["unit"]))
    split_path = root / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
    split = load_campaign_split(split_path, root)
    if (
        split.sha256
        != packet["source_pins_sha256"]["research/campaign_2026-09-29/artifacts/splits/fold1.json"]
    ):
        raise ValueError("loaded fold1 membership manifest differs from its frozen source pin")
    output_result = verify_output(root, step3, split, queue["source_pins_sha256"])
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < 2_000_000_000:
        raise RuntimeError("free disk is below the admitted 2 GB before seed4")
    checks = {
        "seed3_attempt": step3["attempt_id"],
        "seed3_command_sha256": step3["command_sha256_canonical_json"],
        "seed3_systemd": fit_unit,
        "profile_systemd": profile_unit,
        "profile_update_evidence": profile_evidence,
        "disk_pressure_context": packet["disk_pressure_context"],
        "fold1_fit_membership": output_result["fit_membership_counts"],
        "seed3_outputs": output_result,
        "disk_free_bytes_before_seed4": free_bytes,
        "seed4_attempt_absent": not seed4_attempt_path.exists(),
        "seed4_output_absent": not seed4_output_path.exists(),
        "source_pins_sha256": queue["source_pins_sha256"],
        "control_source_pins_sha256": packet["control_source_pins_sha256"],
        "authority_sha256": packet["authority_sha256"],
        "queue_sha256": packet["queue_sha256"],
    }
    receipt_path = root / packet["receipt_relative"]
    if receipt_path.exists():
        raise FileExistsError(
            f"postfit receipt already exists; refusing a second gate run: {receipt_path}"
        )
    receipt = {
        "schema": "wnn.fold1_gru384_seed3_postfit_gate.r1",
        "status": "seed4_launch_intent_durable",
        "created_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "seed4_attempt": step4["attempt_id"],
        "seed4_command_sha256": step4["command_sha256_canonical_json"],
    }
    atomic_json(receipt_path, receipt)
    try:
        launch = launch_seed4(root, root / packet["authority_path"], runtime, step4)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        receipt.update(
            status="seed4_launch_needs_review",
            launch_error=f"{type(error).__name__}: {error}",
            finished_utc=datetime.now(UTC).isoformat(),
        )
        atomic_json(receipt_path, receipt)
        raise
    receipt.update(
        status="seed4_guarded_attempt_reserved",
        seed4_guardian=launch,
        finished_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


def main() -> int:
    """Run static preflight or the exact terminal seed3 gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.packet.resolve(strict=True), args.packet_sha256, args.preflight_only)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print(f"SEED3_POSTFIT_GATE_FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
