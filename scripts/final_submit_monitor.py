"""Publish bounded train-only final-fit status and read-only dashboard views."""

from __future__ import annotations

import argparse
import atexit
import gzip
import hashlib
import json
import math
import os
import re
import signal
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROGRESS = re.compile(
    r"^ep (\d+) batch (\d+)/(\d+) loss ([-+0-9.eE]+) lr ([-+0-9.eE]+) seq/s ([-+0-9.eE]+)$"
)
BRANCH = "results/tabdata-final-submit-20261031"
PREFIX = "research/experiment_drafts/final-submit-all-labeled-20261031"
REPO = "/repos/VladKochetov007/WunderFunder"


def canonical(value: object) -> bytes:
    """Serialize finite public evidence without credentials or raw input rows."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path: Path) -> dict[str, Any]:
    """Read a regular bounded JSON record owned by this task."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_000_000:
        raise ValueError(f"unsafe or missing report record: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("expected a report object")
    return value


def verify_import_source(source: Path) -> None:
    """Check archived Python bytes before importing code that will read credentials."""
    capsule = source.parent / "inputs/final-submit-source-39d83f739.tar.gz"
    if (
        hashlib.sha256(capsule.read_bytes()).hexdigest()
        != "8381c0d6a6333cbc32dca2187516994c5a80832193bcf2531bd606511bc7ccb4"
    ):
        raise ValueError("credential reader source capsule differs")
    expected: set[str] = set()
    with tarfile.open(capsule) as archive:
        for member in archive.getmembers():
            if (
                not member.isfile()
                or not member.name.endswith(".py")
                or not member.name.startswith(("scripts/", "src/"))
            ):
                continue
            expected.add(member.name)
            file = source / member.name
            if file.is_symlink() or not file.is_file():
                raise ValueError("unsafe imported Python source")
            stream = archive.extractfile(member)
            if (
                stream is None
                or hashlib.sha256(stream.read()).digest()
                != hashlib.sha256(file.read_bytes()).digest()
            ):
                raise ValueError("imported Python source differs from the fixed capsule")
    actual = {
        p.relative_to(source).as_posix()
        for folder in ("scripts", "src")
        for p in (source / folder).rglob("*.py")
    }
    if actual != expected:
        raise ValueError("importable Python source inventory differs from the capsule")


def view(root: Path, node: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Expose actual native events or explicitly converted legacy progress lines."""
    run = node["run"]
    output = root / "data/artifacts/runs" / run
    journal = output / "monitor_events.jsonl"
    metadata_path = output / "run_metadata.json"
    metadata: dict[str, Any] = {
        "source": "final-fit-all-labels",
        "config": {
            "node_id": node["id"],
            "epochs": node["epochs"],
            "validation": "none; all labels fit",
        },
        "provenance": {
            "independent_score_claim": "forbidden",
            "fit_sequences": 12480,
            "independent_evaluation_sequences": 0,
        },
    }
    events: list[dict[str, Any]] = []
    if metadata_path.is_file():
        raw = read_json(metadata_path)
        metadata["config"] = raw.get("config", metadata["config"])
        metadata["provenance"]["source_sha256"] = raw.get("source", {}).get("sha256")
    if journal.is_file() and not journal.is_symlink():
        if journal.stat().st_size > 16_000_000:
            raise ValueError("native journal exceeds declared report bound")
        for line in journal.read_bytes().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue  # A writer may have an unfinished final line.
            if isinstance(record, dict) and record.get("schema") == "wnn.monitor.v1":
                if record.get("kind") == "evaluation":
                    raise ValueError("unexpected evaluation journal in all-labels final fit")
                events.append(record)
    else:
        log = root / "control/final-submit-chain/logs/fits" / f"{node['id']}.log"
        if log.is_file() and not log.is_symlink():
            if log.stat().st_size > 16_000_000:
                raise ValueError("legacy journal exceeds declared report bound")
            wall = datetime.fromtimestamp(log.stat().st_mtime, UTC).isoformat()
            for line in log.read_text(errors="replace").splitlines():
                match = PROGRESS.match(line)
                if match is None:
                    continue
                epoch, batch, total = map(int, match.group(1, 2, 3))
                loss, lr, rate = map(float, match.group(4, 5, 6))
                if total <= 0 or not all(math.isfinite(v) for v in (loss, lr, rate)):
                    continue
                events.append(
                    {
                        "schema": "wnn.monitor.v1",
                        "kind": "train_step",
                        "step": epoch * total + batch,
                        "epoch": epoch + batch / total,
                        "wall_time": wall,
                        "metrics": {
                            "train/loss": loss,
                            "train/lr": lr,
                            "throughput/sequences_per_second": rate,
                        },
                        "context": {
                            "step_unit": "whole-sequence batches",
                            "time_basis": "log mtime; per-line timestamp unavailable",
                            "objective": "frozen legacy training objective",
                            "independent_score_claim": "forbidden",
                        },
                    }
                )
    status = state.get("nodes", {}).get(node["id"], {}).get("status", "pending")
    return {
        "run_id": "final-" + node["id"],
        "metadata": metadata,
        "events": events,
        "status": status,
    }


def snapshot(root: Path) -> dict[str, Any]:
    """Read the exact fixed plan and durable state; do not launch any computation."""
    plan_path = root / "inputs/final-plan-r2.json"
    plan = read_json(plan_path)
    if (
        hashlib.sha256(plan_path.read_bytes()).hexdigest()
        != "72aab31bbc4e40de8712bc599b7d6ad922b8b4a5de44d16225d355e8d1e17e2f"
    ):
        raise ValueError("unexpected final plan")
    state_path = root / "control/final-submit-chain/controller-state.json"
    state = read_json(state_path) if state_path.exists() else {"phase": "waiting_for_director"}
    if state_path.exists():
        expected = {
            "plan_sha256": plan["plan_sha256"],
            "source_capsule_sha256": "8381c0d6a6333cbc32dca2187516994c5a80832193bcf2531bd606511bc7ccb4",
            "source_commit": "39d83f73926a459269bcd8511eec08ce9a6e02fd",
            "source_validation_sha256": "1e8a8a794a0b8d9677a76030e570681074503d38aa2d54b80e36067566b94ce0",
            "membership_sha256": "3f766945e00a2fa5edc6230d9c0bed125da112584f11e3d82458167a011bf543",
            "fit_members_sha256": "7a67382ec10984283b50dd4f90bda235933d24a614146c3c00808b0372608300",
        }
        identity = state.get("identity", {})
        if state.get("schema") != "wnn.final-submit-chain-state.v1" or any(
            identity.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("controller state does not bind the exact final plan and membership")
    nodes = list(plan["nodes"]) + [
        {"id": "source4_lstm192x1", "run": "final-lstm192x1", "epochs": 8}
    ]
    runs = [view(root, node, state) for node in nodes]
    return {
        "schema": "wnn.final-submit-live-snapshot.v1",
        "plan_sha256": plan["plan_sha256"],
        "state": state,
        "runs": runs,
        "independent_score_claim": "forbidden",
    }


def main() -> None:
    """Mirror changed task evidence and comment hourly or upon terminal fit changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    verify_import_source(args.source_root)
    sys.path[:0] = [str(args.source_root / "scripts"), str(args.source_root / "src")]
    import quality_horizon_publish as publisher
    from galam_l4_quality8.publisher import read_token, remove_owned_token
    from multistage_github_node_api import GitHubAPI

    token, token_identity = read_token(args.token_file)
    atexit.register(remove_owned_token, args.token_file, token_identity)

    def stop_reporter(signum: int, frame: object) -> None:
        """Exit only this reporter; its registered cleanup removes its own credential."""
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_reporter)
    api = GitHubAPI(token)
    if api.request("GET", "/user")["login"] != "SemyonEpanov":
        raise ValueError("unexpected GitHub publication identity")
    publisher.BRANCH, publisher.PREFIX = BRANCH, PREFIX
    deadline = read_json(args.runtime_root / "CONTRACT_R1.json")["hard_deadline_epoch"]
    state_path = args.runtime_root / "receipts/final-monitor-publish-state.json"
    sent = read_json(state_path) if state_path.exists() else {"terminal": [], "hour": -1}
    while time.time() < deadline:
        body = snapshot(args.runtime_root)
        raw = canonical(body)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != sent.get("snapshot_sha256"):
            compressed = gzip.compress(raw, mtime=0)
            manifest = {
                "schema": "wnn.final-submit-live-transfer.v1",
                "payload_sha256": digest,
                "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                "bytes": len(raw),
            }
            commit = publisher.commit_files(
                api,
                {
                    "live/snapshot.json.gz": compressed,
                    "live/snapshot.manifest.json": canonical(manifest),
                },
            )
            sent.update(snapshot_sha256=digest, commit=commit)
        terminal = sorted(
            (run["run_id"], run["status"])
            for run in body["runs"]
            if run["status"] in {"complete", "failed", "reconciliation_required"}
        )
        hour = int(time.time() // 3600)
        if [list(v) for v in terminal] != sent["terminal"] or hour != sent["hour"]:
            completed = sum(status == "complete" for _, status in terminal)
            text = f"Final-only all-labels refit: {completed}/18 graph nodes complete. Phase `{body['state'].get('phase', 'unknown')}`. Fixed TL3 + LSTM recipes, 12,480 fitting sequences, no independent validation score. CPU serving eligibility is still under engineering checks. [Live evidence](https://github.com/VladKochetov007/WunderFunder/tree/{BRANCH}/{PREFIX}/live)."
            api.request("POST", REPO + "/issues/54/comments", {"body": text})
            sent.update(terminal=[list(v) for v in terminal], hour=hour)
        if state_path.is_symlink():
            raise ValueError("publisher state must not be a symlink")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".final-monitor-", dir=state_path.parent
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(sent))
        os.replace(temporary_name, state_path)
        if args.once or len(terminal) == 18:
            return
        time.sleep(min(120, max(0, deadline - time.time())))


if __name__ == "__main__":
    main()
