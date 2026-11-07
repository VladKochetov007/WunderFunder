from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path

from scripts.sequence_monitor_mirror_io import apply_snapshot, atomic_json
from scripts.sequence_rank0_view import update_rank0_view

LOGGER = logging.getLogger(__name__)
STATE_SCHEMA = "wnn.sequence-monitor-mirror-state.v1"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("mirror configuration/state must be a JSON object")
    return value


def _rpc_payload(result: dict[str, object], request_id: str) -> dict[str, object]:
    if result.get("id") != request_id:
        raise ValueError("RPC result identifier differs from its owned request")
    output = str(result["output"])
    begin, end = f"WNN_RPC_BEGIN {request_id}", f"WNN_RPC_END {request_id} ok"
    start, stop = output.find(begin), output.rfind(end)
    if start < 0 or stop < start:
        raise ValueError("RPC did not return a successful complete collector frame")
    value = json.loads(output[start + len(begin) : stop].strip())
    if not isinstance(value, dict):
        raise ValueError("RPC collector payload must be an object")
    return value


def _request(node: dict[str, object], state: dict[str, object], now: float) -> None:
    rpc = Path(node["rpc_root"]).resolve(strict=True)
    requests, results = rpc / "requests", rpc / "results"
    if (
        requests.is_symlink()
        or results.is_symlink()
        or not requests.is_dir()
        or not results.is_dir()
    ):
        raise ValueError("RPC directories must already exist as direct local directories")
    request_id = f"sequence-mirror-{int(now * 1000):013d}-{uuid.uuid4().hex[:12]}"
    host = str(node["host"])
    code = (
        "import json,runpy\n"
        f"collector=runpy.run_path({str(node['collector_path'])!r})['collect_runs']\n"
        f"state=json.loads({json.dumps(state.get('cursors', {}))!r})\n"
        f"print(json.dumps(collector({str(node['run_root'])!r},{host!r},state),sort_keys=True))\n"
    )
    atomic_json(requests / f"{request_id}.json", {"code": code})
    state.update(pending_id=request_id, queued_at=now)


def _acknowledge_owned_request(
    node: dict[str, object], request_id: str, response: dict[str, object]
) -> None:
    """Compact this bridge's reply while keeping the daemon's result marker present.

    Removing a completed request/result can race the daemon's directory snapshot
    and replay a request or terminate its stream. Both filenames remain present.
    Native bytes and cursor receipts were durably published before this acknowledgement.
    """
    if not request_id.startswith("sequence-mirror-") or "/" in request_id or "\\" in request_id:
        raise ValueError("refusing to remove an unowned RPC request")
    rpc = Path(node["rpc_root"]).resolve(strict=True)
    parent = (rpc / "results").resolve(strict=True)
    path = parent / f"{request_id}.json"
    if path.is_symlink() or path.resolve().parent != parent:
        raise ValueError("owned RPC transport path escaped its direct directory")
    output = str(response.get("output", "")).encode()
    atomic_json(
        path,
        {
            "id": request_id,
            "time": response.get("time"),
            "bridge_consumed": True,
            "output_bytes": len(output),
            "output_sha256": hashlib.sha256(output).hexdigest(),
        },
    )


def tick(config: dict[str, object], state: dict[str, object], now: float) -> bool:
    """Queue one read-only delta per host and apply complete replies without SSH."""
    dashboard = Path(config["dashboard_root"])
    native = Path(config["native_root"])
    hosts = state.setdefault("hosts", {})
    completed = False
    for node in config["hosts"]:
        host = str(node["host"])
        current = hosts.setdefault(host, {"cursors": {}, "rank_views": {}})
        pending = current.get("pending_id")
        if pending:
            result_path = Path(node["rpc_root"]) / "results" / f"{pending}.json"
            try:
                response = _read_json(result_path)
            except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
                continue
            try:
                payload = _rpc_payload(response, pending)
                cursors = apply_snapshot(payload, dashboard, native, current["cursors"], host)
            except (OSError, ValueError, KeyError) as exc:
                current.update(
                    last_rpc_error={"id": pending, "at": now, "error": str(exc)[:300]},
                    cursors={},
                    next_poll_unix=now + config.get("interval", 15),
                )
                current.pop("pending_id", None)
                atomic_json(Path(config["state_path"]), state)
                _acknowledge_owned_request(node, pending, response)
                LOGGER.exception("source mirror reply rejected for host %s", host)
                continue
            current["cursors"] = cursors
            if host == "b300":
                for run in payload["runs"]:
                    run_id = run["run_id"]
                    current["rank_views"][run_id] = update_rank0_view(
                        native / run_id, dashboard / run_id, current["rank_views"].get(run_id, {})
                    )
            current.update(
                last_applied_unix=now,
                last_collected_unix=payload["collected_at_unix"],
                last_errors=payload["errors"],
                next_poll_unix=now + config.get("interval", 15),
            )
            current.pop("pending_id", None)
            if host not in config.setdefault("_done", []):
                config["_done"].append(host)
            completed = True
            # Save cursor ownership before compacting this owned transport reply.
            atomic_json(Path(config["state_path"]), state)
            _acknowledge_owned_request(node, pending, response)
        once_done = config.get("_once") and host in config.get("_done", [])
        if (
            not once_done
            and not current.get("pending_id")
            and now >= current.get("next_poll_unix", 0)
        ):
            _request(node, current, now)
    return completed


def main() -> None:
    """Run the bounded local two-host file-queue bridge; open no SSH connection."""
    parser = argparse.ArgumentParser(
        description="Mirror source-native monitoring via existing file RPC"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=86_400)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 777_600:
        parser.error("max-seconds must be between 1 and 777600")
    config = _read_json(args.config)
    config["_once"] = args.once
    config["_done"] = []
    if not 5 <= config.get("interval", 15) <= 60 or not 1 <= len(config["hosts"]) <= 2:
        parser.error("configuration needs one/two hosts and interval between 5 and 60 seconds")
    for name in ("dashboard_root", "native_root"):
        root = Path(config[name])
        if root.is_symlink():
            raise ValueError("local mirror roots cannot be symlinks")
        root.mkdir(parents=True, exist_ok=True)
    state_path = Path(config["state_path"])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = _read_json(state_path) if state_path.exists() else {"schema": STATE_SCHEMA, "hosts": {}}
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("unsupported bridge state schema")
    deadline = time.monotonic() + args.max_seconds
    while time.monotonic() < deadline:
        try:
            completed = tick(config, state, time.time())
            state.pop("last_error", None)
            if args.once and completed and len(set(config["_done"])) == len(config["hosts"]):
                atomic_json(state_path, state)
                return
        except (OSError, ValueError, KeyError) as exc:
            state["last_error"] = {"at": time.time(), "error": str(exc)[:300]}
            LOGGER.exception(
                "monitor mirror cycle failed; source liveness timestamps are unchanged"
            )
        atomic_json(state_path, state)
        time.sleep(min(1, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
