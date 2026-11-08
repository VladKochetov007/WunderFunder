"""Read actual owned-command and legacy run evidence for the native dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.tl3_dashboard_bridge import (
    BridgeError,
    _atomic_json,
    _canonical_sha256,
    _read_json,
    _sha256_file,
)
from scripts.tl3_finite17_dashboard_core import RUN_ID, SHA256, _cli_args
from scripts.tl3_finite17_dashboard_evidence import _verify_provenance
from scripts.tl3_finite17_dashboard_process import _proc

# Legacy/request JSON varies by producer and is checked at this read-only boundary.
JsonObject = dict[str, Any]
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
VIEW_SCHEMA = "wnn.heterogeneous-native-dashboard.v1"
W18 = "tl3c_w18final_distilled_s5"


def read_nodes(root: Path) -> list[JsonObject]:
    """Read six actual frozen nodes and the separate original W18 recovery recipe."""
    path = root / "nodes-r1.json"
    if path.is_symlink() or path.stat().st_size > 1 << 20:
        raise BridgeError("unsafe heterogeneous node recipe")
    value = json.loads(path.read_text(encoding="utf-8"))
    nodes = value.get("nodes") if isinstance(value, dict) else value
    if not isinstance(nodes, list) or len(nodes) != 6:
        raise BridgeError("heterogeneous observer requires exactly six scientific nodes")
    w18 = _read_json(root / "w18-node-r1.json", 1 << 20)
    w18 = w18.get("node", w18)
    result = nodes + [w18]
    runs = [node.get("run") for node in result]
    if len(set(runs)) != 7 or W18 not in runs:
        raise BridgeError("observer run identities are incomplete or duplicated")
    for node in result:
        run, args = node.get("run"), node.get("args")
        if not isinstance(run, str) or not RUN_ID.fullmatch(run) or not isinstance(args, list):
            raise BridgeError("malformed run name or scientific arguments")
        if run != W18 and not run.startswith("het6_"):
            raise BridgeError("observer cannot adopt an unrelated scientific run")
        cli = _cli_args(args)
        epochs = 2 if run == W18 else 4
        if cli.get("name") != run or int(cli.get("epochs", 0)) != epochs:
            raise BridgeError("node name or configured horizon differs")
        node.update(epochs=epochs, endpoint=epochs - 1, args_sha256=_canonical_sha256(args))
    return result


def split_members(nodes: list[JsonObject]) -> tuple[list[int], list[int]]:
    """Bind every view to the fixed full official chronological split."""
    paths = {str(_cli_args(node["args"])["split_manifest"]) for node in nodes}
    if len(paths) != 1:
        raise BridgeError("observer recipes do not share one primary split")
    path = Path(paths.pop())
    if _sha256_file(path) != SPLIT_SHA256:
        raise BridgeError("observer split differs from the admitted primary boundary")
    split = _read_json(path)
    train, valid = split["fit"]["train"], split["evaluation"]["valid"]
    if len(train) != 10325 or split["fit"]["valid"] or len(valid) != 1873:
        raise BridgeError("observer requires full10325train/1873valid membership")
    return train, valid


def requests(root: Path, nodes: list[JsonObject]) -> dict[str, JsonObject]:
    """Join reviewed fit requests to their actual matching owned status records."""
    paths = sorted((root / "requests").glob("*.json"))
    if len(paths) > 256:
        raise BridgeError("request inventory exceeds its bounded observer scope")
    by_run, result = {node["run"]: node for node in nodes}, {}
    for path in paths:
        request = _read_json(path, 1 << 20)
        if request.get("phase") not in {"fit", "scientific_fit"}:
            continue
        command = request.get("command")
        if not isinstance(command, list):
            raise BridgeError("fit request omits command arguments")
        index = next(
            (
                i
                for i, value in enumerate(command)
                if str(value).replace("\\", "/").endswith("/scripts/train_rnn.py")
            ),
            None,
        )
        if index is None:
            continue
        cli = _cli_args(command[index + 1 :])
        run = cli.get("name")
        if run not in by_run:
            continue
        if command[index + 1 :] != by_run[run]["args"]:
            raise BridgeError(f"fit request changes the frozen arguments: {run}")
        status_path = root / "status" / path.name
        status = _read_json(status_path, 1 << 20) if status_path.exists() else {}
        request_sha = _sha256_file(path)
        command_sha = _canonical_sha256(command)
        if status and (
            status.get("schema") != "wnn.owned-command.v1"
            or status.get("request_sha256") != request_sha
            or status.get("command_sha256") != command_sha
            or status.get("log") != request.get("log")
            or status.get("source_identity") != request.get("source_identity")
        ):
            raise BridgeError(f"owned fit status differs from its request: {run}")
        item = {
            "request": request,
            "status": status,
            "request_sha256": request_sha,
            "request_path": str(path),
            "source_root": str(Path(command[index]).parent.parent),
            "mtime_ns": path.stat().st_mtime_ns,
        }
        if (
            run in result
            and result[run]["status"].get("status") == "running"
            and status.get("status") == "running"
        ):
            raise BridgeError(f"multiple owned fits claim the same scientific run: {run}")
        if run not in result or item["mtime_ns"] > result[run]["mtime_ns"]:
            result[run] = item
    return result


def owned_process(binding: JsonObject) -> tuple[int, str] | None:
    """Verify birth ticks, session and CUDA identity without signaling or exposing env."""
    status, request = binding["status"], binding["request"]
    pid, uuid = status.get("pid"), status.get("gpu_uuid")
    if status.get("status") != "running" or type(pid) is not int or not isinstance(uuid, str):
        return None
    process = _proc(pid, Path("/proc"))
    if (
        process is None
        or process["state"] in {"Z", "X"}
        or process["start_ticks"] != str(status.get("start_ticks"))
    ):
        return None
    if process["pgrp"] != status.get("process_group") or process["session"] != status.get(
        "session"
    ):
        return None
    expected_environment = request.get("env", {})
    if process["env"].get("CUDA_VISIBLE_DEVICES") != uuid or process["env"].get(
        "WNN_PROJECT_ROOT"
    ) != expected_environment.get("WNN_PROJECT_ROOT"):
        return None
    command = process["cmdline"]
    script = str(Path(binding["source_root"]) / "scripts/train_rnn.py")
    index = next((i for i, value in enumerate(command) if value == script), None)
    requested_index = request["command"].index(script)
    if index is None or command[index + 1 :] != request["command"][requested_index + 1 :]:
        return None
    return pid, uuid


def view_metadata(
    runs: Path, root: Path, node: JsonObject, binding: JsonObject | None, state: str
) -> Path:
    """Create an owned native view; preserve the original W18 observer's view untouched."""
    run = node["run"]
    name = run if run != W18 else f"{run}_recovery_20261030"
    view = runs / name
    if view.is_symlink() or (view.exists() and not view.is_dir()):
        raise BridgeError("dashboard view must be an owned direct-child directory")
    marker = {
        "schema": VIEW_SCHEMA,
        "root": str(root),
        "run": run,
        "args_sha256": node["args_sha256"],
        "split_sha256": SPLIT_SHA256,
    }
    if not view.exists():
        view.mkdir(mode=0o700)
    metadata_path = view / "run_metadata.json"
    if (
        metadata_path.exists()
        and _read_json(metadata_path).get("provenance", {}).get("heterogeneous_observer") != marker
    ):
        raise BridgeError(f"dashboard view has another owner: {name}")
    cli = _cli_args(node["args"])
    architecture = "tcn" if cli.get("arch") == "tcn" else cli.get("cell", "gru")
    _atomic_json(
        metadata_path,
        {
            "source": {
                "campaign": "Heterogeneous teachers vs fresh TL3",
                "role": node.get("role"),
                "run": run,
            },
            "config": {
                "run": run,
                "cell_state": state,
                "phase": "fit",
                "epochs": node["epochs"],
                "endpoint": node["endpoint"],
                "expected_updates": node["expected_updates"],
                "architecture": architecture,
                "hidden": int(cli.get("hidden", 256)),
                "layers": int(cli.get("layers", 2)),
                "effective_args": cli,
                "parents": node.get("parents", []),
            },
            "provenance": {
                "heterogeneous_observer": marker,
                "request_sha256": None if binding is None else binding["request_sha256"],
                "source_identity": None
                if binding is None
                else binding["request"].get("source_identity"),
                "fitting_train_sequences": 10325,
                "fitting_valid_sequences": 0,
                "full_valid_sequences": 1873,
                "full_valid_scored_rows": 3528273,
                "train_eval": {"sequences": 32, "mask": "all_after_warmup"},
            },
            "synthetic": False,
        },
    )
    for filename in ("monitor_events.jsonl", "monitor_evaluations.jsonl"):
        target = view / filename
        if target.is_symlink():
            raise BridgeError("dashboard journal cannot be a symlink")
        if not target.exists():
            target.touch(mode=0o600, exist_ok=False)
    return view


def evaluation_plan(
    binding: JsonObject,
    node: JsonObject,
    run_dir: Path,
    train: list[int],
    valid: list[int],
    verified: set[tuple[str, str]],
) -> tuple[JsonObject, JsonObject] | None:
    """Bind completed native epoch metrics to actual source/provenance without model loads."""
    path = run_dir / "provenance.json"
    if not path.exists():
        return None
    provenance = _read_json(path)
    implementation = provenance.get("trainer_implementation", {})
    source_sha = implementation.get("sha256")
    files = implementation.get("files")
    if (
        not isinstance(source_sha, str)
        or not SHA256.fullmatch(source_sha)
        or not isinstance(files, dict)
        or _canonical_sha256(files) != source_sha
    ):
        raise BridgeError("actual run trainer source inventory is malformed")
    source_root = Path(binding["source_root"]).resolve(strict=True)
    key = str(source_root), source_sha
    if key not in verified:
        for relative, expected in files.items():
            source = (source_root / relative).resolve(strict=True)
            if not source.is_relative_to(source_root) or _sha256_file(source) != expected:
                raise BridgeError("actual trainer source files differ from saved provenance")
        verified.add(key)
    plan = {
        "sha256": binding["request_sha256"],
        "source_identity": {"training_source_sha256": source_sha},
        "split": {"split_manifest_sha256": SPLIT_SHA256},
    }
    _verify_provenance(plan, node, provenance, train, valid)
    return plan, provenance
