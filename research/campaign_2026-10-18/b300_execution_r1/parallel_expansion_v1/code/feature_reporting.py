"""Publish only independently verified native-feature pair grades to new PRs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from feature_report_core import canonical_sha, collect, report_body

ROOT = Path("/data/epanov_s/tabdata")
REPOSITORY = "VladKochetov007/WunderFunder"
EXPERIMENT_IDS = {
    "age": "E-B300-PR4-AGE-SUCCESSOR-V1",
    "innovation": "E-B300-PR14-INNOVATION-SUCCESSOR-V1",
}


def _sha(path: Path) -> str:
    """Hash one file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _helpers(config: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Load the frozen GitHub and archive helpers from their pinned package."""
    directory = Path(config["github_code_dir"]).resolve(strict=True)
    manifest_path = directory / "CODE_MANIFEST.json"
    if _sha(manifest_path) != config["github_code_manifest_sha256"]:
        raise ValueError("GitHub helper manifest differs from the authorized pin")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("GitHub helper manifest contains a non-flat source member")
        path = directory / name
        if path.is_symlink() or _sha(path) != digest:
            raise ValueError("GitHub helper changed after acceptance: " + name)
    sys.path.insert(0, str(directory))
    from github_api import REPOSITORY as API_REPOSITORY
    from github_api import GitHub, GitHubError, read_token
    from package_evidence import ensure_archives
    from render_report import marker
    from report_state import append_event, read_json, write_json

    if API_REPOSITORY != REPOSITORY:
        raise ValueError("Pinned GitHub helper repository differs")
    return (GitHub, GitHubError, read_token, ensure_archives), (marker, append_event, read_json, write_json), directory, sys.modules["render_report"]


def verify_configuration(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Bind the two new PRs, feature runtime and reporter implementation before auth."""
    if path.is_symlink() or _sha(path) != expected_sha256:
        raise ValueError("Feature reporter configuration differs from its unit pin")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("root") != str(ROOT) or config.get("repository") != REPOSITORY:
        raise ValueError("Feature reporter root or repository differs from its fixed scope")
    validate_targets(config.get("targets", {}))
    code_files = config.get("reporter_files", {})
    if set(code_files) != {"feature_reporting.py", "feature_report_core.py"}:
        raise ValueError("Feature reporter source manifest is incomplete")
    for name, digest in code_files.items():
        source = Path(__file__).resolve().parent / name
        if source.is_symlink() or _sha(source) != digest:
            raise ValueError("Feature reporter source changed after acceptance: " + name)
    if not config.get("execute_enabled"):
        raise ValueError("Automatic feature reporting is not authorized in this configuration")
    return config


def validate_targets(targets: dict[str, Any]) -> None:
    """Reject source PRs, duplicate targets or incomplete successor mappings."""
    if set(targets) != set(EXPERIMENT_IDS):
        raise ValueError("Feature reporter must target only age and innovation successor PRs")
    numbers = []
    for family, target in targets.items():
        if (
            target.get("experiment_id") != EXPERIMENT_IDS[family]
            or type(target.get("number")) is not int
            or target["number"] <= 0
            or target["number"] in {4, 14}
            or not target.get("head_ref")
            or not target.get("evidence_url", "").startswith("https://")
        ):
            raise ValueError("Feature successor PR mapping is incomplete or drifted")
        numbers.append(target["number"])
    if len(set(numbers)) != 2:
        raise ValueError("Age and innovation successors must use distinct pull requests")


def _own_comments(comments: list[dict[str, Any]], login: str) -> list[dict[str, Any]]:
    """Reconcile only comments authored by the configured publisher."""
    return [comment for comment in comments if comment.get("user", {}).get("login") == login]


def _target_metadata(client: Any, target: dict[str, Any]) -> dict[str, Any]:
    """Require the exact open successor PR before any comment write."""
    metadata = client.request("GET", f"/repos/{REPOSITORY}/pulls/{target['number']}")
    if (
        metadata["state"] != "open"
        or metadata["base"]["repo"]["full_name"] != REPOSITORY
        or metadata["base"]["ref"] != "research-team"
        or metadata["head"]["repo"]["full_name"] != REPOSITORY
        or metadata["head"]["ref"] != target["head_ref"]
    ):
        raise ValueError("Feature PR identity, branch or open state differs from its pin")
    return metadata


def _publish_snapshot(
    root: Path,
    client: Any,
    helpers: tuple[Any, Any, Any, Any],
    utility: tuple[Any, Any, Any, Any],
    snapshot: dict[str, Any],
    target: dict[str, Any],
    state: dict[str, Any],
    login: str,
    reason: str,
) -> dict[str, Any]:
    """Reconcile markers, archive endpoints and publish bounded pair batches."""
    _, _, _, ensure_archives = helpers
    marker, _, _, _ = utility
    experiment, number = snapshot["experiment_id"], target["number"]
    _target_metadata(client, target)
    comments = _own_comments(client.comments(number), login)
    acknowledged = state.setdefault("pairs", {})
    for row in snapshot["completed"]:
        pair_marker = marker(
            experiment,
            "pair",
            canonical_sha({"pair_id": row["pair_id"], "receipt_sha256": row["receipt_sha256"]}),
        )
        if any(pair_marker in comment.get("body", "") for comment in comments):
            acknowledged[row["pair_id"]] = row["receipt_sha256"]
    pending = [
        row
        for row in snapshot["completed"]
        if acknowledged.get(row["pair_id"]) != row["receipt_sha256"]
    ]
    result_urls = state.setdefault("result_urls", [])
    for start in range(0, len(pending), 8):
        rows = copy.deepcopy(pending[start : start + 8])
        for row in rows:
            for endpoint in (row["control"], row["feature"]):
                endpoint.update(ensure_archives(root, endpoint))
        batch_marker = marker(experiment, "results", canonical_sha(rows))
        found = next((comment for comment in comments if batch_marker in comment.get("body", "")), None)
        if found is None:
            body = report_body(snapshot, target["evidence_url"], rows=rows)
            found = client.post(number, body)
            comments.append(dict(found, user={"login": login}))
        for row in rows:
            acknowledged[row["pair_id"]] = row["receipt_sha256"]
        if found["html_url"] not in result_urls:
            result_urls.append(found["html_url"])
    from render_report import canonical_sha as helper_canonical_sha

    snapshot_sha = helper_canonical_sha(snapshot)
    status_marker = marker(experiment, "status")
    status = next((comment for comment in comments if status_marker in comment.get("body", "")), None)
    if reason == "hourly" or pending or status is None or state.get("snapshot_sha256") != snapshot_sha:
        body = report_body(snapshot, target["evidence_url"], rows=None)
        status = client.post(number, body) if status is None else client.patch(status["id"], body)
        state["status_url"] = status["html_url"]
    state["snapshot_sha256"] = snapshot_sha
    state["last_success_unix"] = time.time()
    return state


def _load_snapshots(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect each successor independently so one invalid family cannot hide the other."""
    try:
        candidates = collect(root, config)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        candidates = []
        collection_error = f"{type(error).__name__}: {error}"
    else:
        collection_error = ""
    snapshots = []
    for family, target in config["targets"].items():
        snapshot = next(
            (row for row in candidates if row["experiment_id"] == target["experiment_id"]), None
        )
        if snapshot is None:
            snapshot = {
                "experiment_id": target["experiment_id"],
                "expected_cells": 30,
                "expected_per_arm": 30,
                "completed": [],
                "active": 0,
                "pending": 30,
                "terminal": False,
                "source": {},
                "contract_note": "Collection unavailable; no result was inferred from missing evidence.",
                "errors": [collection_error or "Expected experiment snapshot is missing."],
            }
        if snapshot["experiment_id"] != EXPERIMENT_IDS[family]:
            raise ValueError("Feature collector returned the wrong experiment identity")
        snapshots.append(snapshot)
    return snapshots


def run(config: dict[str, Any], reason: str, *, dry_run: bool = False) -> int:
    """Run one bounded hourly/completion reporting pass; never launch science work."""
    helpers, utility, _, _ = _helpers(config)
    GitHub, GitHubError, read_token, ensure_archives = helpers
    _, append_event, read_json, write_json = utility
    root = Path(config["root"]).resolve(strict=True)
    if dry_run:
        snapshots = _load_snapshots(root, config)
        print(json.dumps(snapshots, sort_keys=True, indent=2, allow_nan=False))
        return 0
    state_root = root / "state/feature-reporting-v1"
    if state_root.is_symlink():
        raise ValueError("Feature reporter state root cannot be a symlink")
    state_root.mkdir(parents=True, exist_ok=True)
    lock_path = state_root / "publisher.lock"
    if lock_path.is_symlink():
        raise ValueError("Feature reporter lock cannot be a symlink")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return 0
    cursor_path = state_root / "cursor.json"
    events_path = state_root / "publications.jsonl"
    try:
        snapshots = _load_snapshots(root, config)
        cursor = read_json(cursor_path, {"schema": "tabdata-feature-reporting-cursor.v1", "experiments": {}})
        if time.time() < cursor.get("cooldown_until_unix", 0):
            return 0
        outbox_dir = state_root / "outbox"
        outbox_dir.mkdir(exist_ok=True)
        for snapshot in snapshots:
            outbox_sha = canonical_sha(snapshot)
            outbox = outbox_dir / f"{outbox_sha}.json"
            if not outbox.exists():
                write_json(outbox, snapshot)
            elif read_json(outbox) != snapshot:
                raise ValueError("Existing durable outbox identity differs")
        client = GitHub(read_token(Path(config["token_file"])), request_limit=32)
        identity = client.request("GET", "/user")
        login = config["publisher_login"]
        if identity.get("login") != login:
            raise ValueError("Authenticated GitHub identity differs from the authorized publisher")
        failures = []
        by_experiment = {snapshot["experiment_id"]: snapshot for snapshot in snapshots}
        for _family, target in config["targets"].items():
            snapshot = by_experiment[target["experiment_id"]]
            name = snapshot["experiment_id"]
            current = cursor.setdefault("experiments", {}).setdefault(name, {})
            try:
                _publish_snapshot(
                    root,
                    client,
                    helpers,
                    utility,
                    snapshot,
                    target,
                    current,
                    login,
                    reason,
                )
                write_json(cursor_path, cursor)
                append_event(
                    events_path,
                    "published",
                    experiment=name,
                    status_url=current.get("status_url"),
                    result_urls=current.get("result_urls", []),
                    outbox_sha256=canonical_sha(snapshot),
                )
            except (GitHubError, RuntimeError, ValueError, OSError, KeyError) as error:
                failures.append(name)
                delay = max(
                    getattr(error, "retry_after", 0),
                    3600 if getattr(error, "status", 0) in {403, 429} else 60,
                )
                cursor["cooldown_until_unix"] = time.time() + delay
                write_json(cursor_path, cursor)
                append_event(
                    events_path,
                    "deferred",
                    experiment=name,
                    error_type=type(error).__name__,
                    http_status=getattr(error, "status", None),
                )
        terminal = all(snapshot.get("terminal") is True for snapshot in snapshots)
        expired = time.time() > config["expires_unix"]
        if not failures and (terminal or expired):
            units = config["stop_units"]
            if len(units) != 2 or any(not unit.startswith("tabdata-feature-reporting-") for unit in units):
                raise ValueError("Completion may stop only the two named feature reporting units")
            subprocess.run(["systemctl", "--user", "stop", *units], timeout=15, check=True)
            append_event(events_path, "triggers_stopped", terminal=terminal, expired=expired)
        write_json(
            state_root / "LAST_RUN.json",
            {
                "time_unix": time.time(),
                "terminal": terminal,
                "failures": failures,
                "scored_pairs": {s["experiment_id"]: len(s["completed"]) for s in snapshots},
            },
        )
        return 1 if failures else 0
    finally:
        os.close(descriptor)


def main() -> int:
    """Validate the service pin and publish one explicit reasoned update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--reason", choices=("hourly", "completion"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = verify_configuration(args.config, args.config_sha256)
    return run(config, args.reason, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
