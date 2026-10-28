from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from scripts.tl3_finite17_dashboard_core import (
        MAX_LOG_BYTES,
        PROGRESS,
        TRAIN_GROUPS,
        BridgeError,
        _canonical_sha256,
        _cli_args,
        _same_arg,
    )
except ModuleNotFoundError:
    from tl3_finite17_dashboard_core import (
        MAX_LOG_BYTES,
        PROGRESS,
        TRAIN_GROUPS,
        BridgeError,
        _canonical_sha256,
        _cli_args,
        _same_arg,
    )


def _parse_log(path: Path) -> dict[str, Any]:
    """Parse complete online, gradient and VALID records from one bounded fit log."""
    records: dict[str, Any] = {"progress": [], "gradients": {}, "valid": []}
    if path.is_symlink() or not path.is_file():
        return records
    size = path.stat().st_size
    if size > MAX_LOG_BYTES:
        raise BridgeError("finite fit log exceeded its size bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise BridgeError("finite fit log changed while being read")
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith((b"\n", b"\r")):
        lines.pop()
    for raw_line in lines:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        match = PROGRESS.fullmatch(line)
        if match:
            point = {
                "epoch": int(match["epoch"]),
                "batch": int(match["batch"]),
                "total": int(match["total"]),
                "loss": float(match["loss"]),
                "lr": float(match["lr"]),
                "rate": float(match["rate"]),
            }
            if all(math.isfinite(point[key]) for key in ("loss", "lr", "rate")):
                records["progress"].append(point)
        elif line.startswith(("GRAD_STATS ", "VALID ")):
            prefix = "GRAD_STATS " if line.startswith("GRAD_STATS ") else "VALID "
            value = json.loads(line[len(prefix) :])
            if not isinstance(value, dict) or type(value.get("epoch")) is not int:
                raise BridgeError(f"malformed {prefix.strip()} fit record")
            if prefix.startswith("GRAD"):
                records["gradients"][value["epoch"]] = value
            else:
                records["valid"].append(value)
    return records


def _verify_provenance(
    plan: dict[str, Any],
    node: dict[str, Any],
    provenance: dict[str, Any],
    train_groups: list[int],
    valid_groups: list[int],
) -> dict[str, Any]:
    """Join actual run source, fit/eval membership and explicit trainer args to plan."""
    if provenance.get("split_manifest_sha256") != plan["split"]["split_manifest_sha256"]:
        raise BridgeError(f"{node['id']}: run split hash differs from plan")
    identity = provenance.get("trainer_implementation")
    expected_source = plan["source_identity"]["training_source_sha256"]
    if (
        not isinstance(identity, dict)
        or identity.get("combined_sha256", identity.get("sha256")) != expected_source
    ):
        raise BridgeError(f"{node['id']}: run source hash differs from plan")
    fit = provenance.get("fit_row_groups")
    if not isinstance(fit, dict):
        raise BridgeError(f"{node['id']}: run omits fit row groups")
    fitted, fit_valid, evaluated = (
        fit.get("train"),
        fit.get("valid"),
        provenance.get("evaluation_row_groups"),
    )
    if (
        not isinstance(fitted, list)
        or len(fitted) != len(train_groups)
        or len(set(fitted)) != len(fitted)
        or set(fitted) != set(train_groups)
        or fit_valid != []
        or not isinstance(evaluated, list)
        or len(evaluated) != len(valid_groups)
        or len(set(evaluated)) != len(evaluated)
        or set(evaluated) != set(valid_groups)
    ):
        raise BridgeError(f"{node['id']}: run membership is not full train-only/full-valid")
    args = provenance.get("args")
    if not isinstance(args, dict):
        raise BridgeError(f"{node['id']}: run provenance omits trainer arguments")
    expected = _cli_args(node["args"])
    expected.update({"eval_source": "vfold", "valid_fold": 0, "n_valid_folds": 1})
    expected["checkpoint_epoch"] = int(node["endpoint"])
    for key, value in expected.items():
        if key not in args or not _same_arg(value, args[key]):
            raise BridgeError(f"{node['id']}: saved trainer argument differs: {key}")
    if (
        args.get("name") != node["run"]
        or args.get("train_source") != "train"
        or args.get("train_seqs") != TRAIN_GROUPS
        or args.get("full_valid") is not True
        or args.get("eval_source") != "vfold"
        or args.get("valid_fold") != 0
        or args.get("n_valid_folds") != 1
        or args.get("no_eval_during_fit") is not False
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != node["endpoint"]
        or args.get("epochs") != node["epochs"]
        or args.get("train_eval_seqs") != 32
        or args.get("precision") != "fp32"
    ):
        raise BridgeError(f"{node['id']}: effective recipe differs from frozen plan")
    return args


def _fit_log_path(
    task: dict[str, Any], node: dict[str, Any], state: Path, plan: dict[str, Any]
) -> Path:
    """Check the driver's immutable fit task and derive its exact log path."""
    if (
        task.get("kind") != "fit"
        or task.get("node_id") != node["id"]
        or task.get("source_sha256") != plan["source_identity"]["training_source_sha256"]
        or task.get("args_sha256") != node["args_sha256"]
    ):
        raise BridgeError(f"{node['id']}: fit task identity differs from plan")
    command = task.get("command")
    if not isinstance(command, list) or task.get("command_sha256") != _canonical_sha256(command):
        raise BridgeError(f"{node['id']}: fit command hash is invalid")
    index = next(
        (
            i
            for i, value in enumerate(command)
            if value.replace("\\", "/").endswith("/scripts/train_rnn.py")
        ),
        None,
    )
    if index is None or command[index + 1 :] != node["args"]:
        raise BridgeError(f"{node['id']}: task arguments differ from plan")
    path = Path(str(task.get("log", "")))
    expected = state / "logs" / f"fit__{node['id']}.log"
    if path != expected or path.is_symlink():
        raise BridgeError(f"{node['id']}: task log path differs from receipt")
    return path
