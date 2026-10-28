from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

try:
    from scripts import tl3_dashboard_bridge as _bridge
except ModuleNotFoundError:
    import tl3_dashboard_bridge as _bridge

BridgeError = _bridge.BridgeError
_append_event = _bridge._append_event
_append_evaluation = _bridge._append_evaluation
_atomic_json = _bridge._atomic_json
_canonical_sha256 = _bridge._canonical_sha256
_event = _bridge._event
_flatten_gradients = _bridge._flatten_gradients
_read_events = _bridge._read_events
_read_json = _bridge._read_json
_sha256_file = _bridge._sha256_file

PLAN_SCHEMA = "wnn.tl3-finite17-plan.v1"
STATE_SCHEMA = "wnn.tl3-finite17-state.v1"
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PROGRESS = re.compile(
    r"^ep (?P<epoch>\d+) batch (?P<batch>\d+)/(?P<total>\d+) "
    r"loss (?P<loss>[-+0-9.eE]+) lr (?P<lr>[-+0-9.eE]+) seq/s (?P<rate>[-+0-9.eE]+)$"
)
TRAIN_GROUPS, VALID_GROUPS, VALID_ROWS = 10_325, 1_873, 3_528_273
ADOPTED_NODE = "fin256_s3"
MAX_LOG_BYTES = 64 << 20


def _cli_args(tokens: list[str]) -> dict[str, object]:
    """Decode the plan's explicit argparse options for provenance comparison."""
    result: dict[str, object] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise BridgeError("finite node arguments contain a positional value")
        key = token[2:].replace("-", "_")
        if key in result:
            raise BridgeError(f"finite node repeats {token}")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            result[key] = tokens[index + 1]
            index += 2
        else:
            result[key] = True
            index += 1
    return result


def _same_arg(expected: object, actual: object) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, bool) or actual is None:
        return False
    if str(expected) == str(actual):
        return True
    try:
        left, right = float(expected), float(actual)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    )


def _plan_and_split(
    root: Path, state: Path
) -> tuple[dict[str, Any], dict[str, Any], list[int], list[int]] | None:
    """Read finite plan/status and verify the official chronological memberships."""
    plan_path, status_path = state / "plan.json", state / "status.json"
    if not plan_path.exists() or not status_path.exists():
        return None
    plan, status = _read_json(plan_path), _read_json(status_path)
    plan_hash = plan.get("sha256")
    unhashed = dict(plan)
    unhashed.pop("sha256", None)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan_hash != _canonical_sha256(unhashed)
        or plan.get("root") != str(root)
        or plan.get("state_dir") != str(state)
        or status.get("schema") != STATE_SCHEMA
        or status.get("plan_sha256") != plan_hash
    ):
        raise BridgeError("finite plan/status identity differs from the supplied paths")
    split, source, nodes = plan.get("split"), plan.get("source_identity"), plan.get("nodes")
    if not isinstance(split, dict) or not isinstance(source, dict) or not isinstance(nodes, list):
        raise BridgeError("finite plan omits split, source or node records")
    source_sha, split_sha = source.get("training_source_sha256"), split.get("split_manifest_sha256")
    if not isinstance(source_sha, str) or not SHA256.fullmatch(source_sha):
        raise BridgeError("finite plan trainer source hash is malformed")
    if not isinstance(split_sha, str) or not SHA256.fullmatch(split_sha):
        raise BridgeError("finite plan split hash is malformed")
    ids = [node.get("id") for node in nodes if isinstance(node, dict)]
    if len(ids) != 17 or len(set(ids)) != 17 or ADOPTED_NODE not in ids:
        raise BridgeError("finite plan is not the expected 17-node graph")
    manifest_path = Path(str(split.get("split_manifest"))).resolve(strict=True)
    if _sha256_file(manifest_path) != split_sha:
        raise BridgeError("official split manifest differs from the finite plan")
    manifest = _read_json(manifest_path)
    fit, evaluation = manifest.get("fit"), manifest.get("evaluation")
    if (
        manifest.get("schema") != "wnn.chronological-primary.v1"
        or not isinstance(fit, dict)
        or not isinstance(evaluation, dict)
    ):
        raise BridgeError("finite plan does not use the chronological primary manifest")
    train, fit_valid, valid = fit.get("train"), fit.get("valid"), evaluation.get("valid")
    if (
        not isinstance(train, list)
        or not isinstance(fit_valid, list)
        or not isinstance(valid, list)
        or len(train) != TRAIN_GROUPS
        or fit_valid
        or len(valid) != VALID_GROUPS
        or len(set(train)) != TRAIN_GROUPS
        or len(set(valid)) != VALID_GROUPS
        or set(train).intersection(valid)
        or split.get("fit_train_count") != TRAIN_GROUPS
        or split.get("fit_valid_count") != 0
        or split.get("evaluation_valid_count") != VALID_GROUPS
    ):
        raise BridgeError("finite plan does not contain exact train-only/full-valid membership")
    return plan, status, list(map(int, train)), list(map(int, valid))


def _run_identity(plan: dict[str, Any], node: dict[str, Any]) -> str:
    return _canonical_sha256(
        {
            "plan_sha256": plan["sha256"],
            "node_id": node["id"],
            "run": node["run"],
            "args_sha256": node["args_sha256"],
            "source_sha256": plan["source_identity"]["training_source_sha256"],
            "split_sha256": plan["split"]["split_manifest_sha256"],
        }
    )


def _context(
    plan: dict[str, Any], node: dict[str, Any], checkpoint: str | None = None
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "run": node["run"],
        "node_id": node["id"],
        "run_identity_sha256": _run_identity(plan, node),
        "plan_sha256": plan["sha256"],
        "source_sha256": plan["source_identity"]["training_source_sha256"],
        "split_manifest_sha256": plan["split"]["split_manifest_sha256"],
        "source_timestamp_available": False,
    }
    if checkpoint:
        context["checkpoint_sha256"] = checkpoint
    return context


def _view(
    runs: Path, plan: dict[str, Any], node: dict[str, Any], node_state: dict[str, Any]
) -> Path:
    """Create or verify one owned direct-child view and refresh its config metadata."""
    run = str(node["run"])
    if not RUN_ID.fullmatch(run) or ".." in run:
        raise BridgeError("finite plan contains an unsafe run name")
    view = runs / run
    if view.is_symlink():
        raise BridgeError(f"dashboard view cannot be a symlink: {run}")
    marker = {
        "schema": "wnn.tl3-finite17-native-dashboard.v1",
        "plan_sha256": plan["sha256"],
        "source_sha256": plan["source_identity"]["training_source_sha256"],
        "split_sha256": plan["split"]["split_manifest_sha256"],
        "args_sha256": node["args_sha256"],
        "node_id": node["id"],
        "run": run,
    }
    if not view.exists():
        view.mkdir(mode=0o700)
    elif not view.is_dir() or view.resolve(strict=True).parent != runs:
        raise BridgeError(f"dashboard view is not a direct child: {run}")
    metadata_path = view / "run_metadata.json"
    if metadata_path.exists():
        old = _read_json(metadata_path, 1 << 20)
        provenance = old.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("native_bridge") != marker:
            raise BridgeError(f"dashboard view belongs to another run: {run}")
    config = {
        "node_id": node["id"],
        "run": run,
        "cell_state": {"not_started": "queued", "complete": "completed"}.get(
            str(node_state.get("state")), node_state.get("state")
        ),
        "phase": node_state.get("phase"),
        "epochs": node["epochs"],
        "endpoint": node["endpoint"],
        "expected_updates": node["expected_updates"],
        "parents": node.get("parents", []),
        "args": node["args"],
        "args_sha256": node["args_sha256"],
    }
    _atomic_json(
        metadata_path,
        {
            "source": {"campaign": "TL3 chronological finite17", "node_id": node["id"]},
            "config": config,
            "provenance": {
                "native_bridge": marker,
                "run_identity_sha256": _run_identity(plan, node),
                "trainer_source_sha256": marker["source_sha256"],
                "split_manifest_sha256": marker["split_sha256"],
                "fit_membership": {"source": "train", "groups": TRAIN_GROUPS},
                "official_valid": {"groups": VALID_GROUPS, "selected_rows": VALID_ROWS},
                "train_eval": {
                    "sequences": 32,
                    "row_mask": "all_after_warmup",
                    "source_order": "sorted fit train prefix",
                },
            },
            "synthetic": False,
        },
    )
    for filename in ("monitor_events.jsonl", "monitor_evaluations.jsonl"):
        path = view / filename
        if path.is_symlink():
            raise BridgeError(f"dashboard event file cannot be a symlink: {run}/{filename}")
        if not path.exists():
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
    return view
