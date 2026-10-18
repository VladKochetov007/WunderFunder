"""Reconcile experiment comments against verified, fully evaluated run receipts."""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import time
from pathlib import Path

from github_api import REPOSITORY, GitHub, GitHubError, read_token
from package_evidence import ensure_archives
from render_report import canonical_sha, marker, report, validate_snapshot
from report_state import append_event, read_json, write_json

LOGGER = logging.getLogger(__name__)
ROOT = Path("/data/epanov_s/tabdata")
STOP_UNITS = ("tabdata-github-updates.timer", "tabdata-github-completions.path")


def verify_configuration(path: Path, expected: str) -> dict:
    """Bind the exact authorized targets and source before loading credentials."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Reporter configuration differs from the service identity")
    value = read_json(path)
    if value.get("root") != str(ROOT) or value.get("repository") != REPOSITORY:
        raise ValueError("Reporter root or repository differs from its fixed scope")
    if set(value["targets"]) != {"E-B300-R01", "PR15"}:
        raise ValueError("Reporter target experiment set differs")
    if value["targets"]["PR15"]["number"] != 15:
        raise ValueError("Geometry results must remain bound to the original PR15")
    if value["targets"]["E-B300-R01"]["number"] in {2, 15}:
        raise ValueError("R01 requires its own fixed-budget contract, not PR2 matched compute")
    packet = Path(__file__).resolve().parent
    if (
        hashlib.sha256((packet / "CODE_MANIFEST.json").read_bytes()).hexdigest()
        != value["code_manifest_sha256"]
    ):
        raise ValueError("Reporter code manifest differs from the fixed configuration")
    manifest = read_json(packet / "CODE_MANIFEST.json")
    for relative, digest in manifest["files"].items():
        if Path(relative).name != relative or not relative.endswith(".py"):
            raise ValueError("Reporter manifest member is not a flat Python source file")
        if hashlib.sha256((packet / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Reporter source changed after acceptance: " + relative)
    return value


def own_comments(comments: list[dict], login: str) -> list[dict]:
    """Never edit a teammate's comment, even if it contains a copied marker."""
    return [row for row in comments if row.get("user", {}).get("login") == login]


def publish_snapshot(
    client: GitHub, snapshot: dict, target: dict, state: dict, login: str, reason: str
) -> dict:
    """Reconcile before POST so an uncertain previous acknowledgment cannot duplicate a report."""
    number = target["number"]
    metadata = client.request("GET", f"/repos/{REPOSITORY}/pulls/{number}")
    if (
        metadata["base"]["ref"] != "research-team"
        or metadata["base"]["repo"]["full_name"] != REPOSITORY
        or metadata["head"]["repo"]["full_name"] != REPOSITORY
        or metadata["head"]["ref"] != target["head_ref"]
        or metadata["state"] != "open"
    ):
        raise ValueError("Experiment PR identity or open state differs from the accepted mapping")
    comments = own_comments(client.comments(number), login)
    acknowledged = state.setdefault("endpoints", {})
    for row in snapshot["completed"]:
        endpoint_marker = marker(
            snapshot["experiment_id"],
            "endpoint",
            canonical_sha({"id": row["id"], "receipt_sha256": row["receipt_sha256"]}),
        )
        if any(endpoint_marker in comment.get("body", "") for comment in comments):
            acknowledged[row["id"]] = row["receipt_sha256"]
    pending = sorted(
        [
            row
            for row in snapshot["completed"]
            if acknowledged.get(row["id"]) != row["receipt_sha256"]
        ],
        key=lambda row: row["id"],
    )
    if target.get("archive_results"):
        pending = [dict(row, **ensure_archives(ROOT, row)) for row in pending]
    result_urls = state.setdefault("result_urls", [])
    for start in range(0, len(pending), 8):
        rows = pending[start : start + 8]
        identifier = marker(snapshot["experiment_id"], "results", canonical_sha(rows))
        existing = next((row for row in comments if identifier in row.get("body", "")), None)
        if existing is None:
            existing = client.post(
                number, report(snapshot, new_rows=rows, evidence_url=target["evidence_url"])
            )
            comments.append(dict(existing, user={"login": login}))
        for row in rows:
            acknowledged[row["id"]] = row["receipt_sha256"]
        if existing["html_url"] not in result_urls:
            result_urls.append(existing["html_url"])
    signature = canonical_sha(snapshot)
    status_marker = marker(snapshot["experiment_id"], "status")
    status = next((row for row in comments if status_marker in row.get("body", "")), None)
    if reason == "hourly" or pending or status is None or state.get("snapshot_sha256") != signature:
        body = report(snapshot, new_rows=None, evidence_url=target["evidence_url"])
        status = client.post(number, body) if status is None else client.patch(status["id"], body)
        state["status_url"] = status["html_url"]
        state["snapshot_sha256"] = signature
    state["last_success_unix"] = time.time()
    return state


def collect_snapshots(root: Path) -> list[dict]:
    """Use receipt-only collectors; never invoke a trainer, exporter or grader."""
    from collect_pr15 import collect as collect_pr15
    from collect_r01 import collect as collect_r01

    snapshots = []
    for collector, name, cells in ((collect_r01, "E-B300-R01", 60), (collect_pr15, "PR15", 120)):
        try:
            snapshot = collector(root)
            validate_snapshot(snapshot)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
            snapshot = dict(
                experiment_id=name,
                expected_cells=cells,
                expected_per_arm=30,
                completed=[],
                active=0,
                pending=0,
                terminal=False,
                source={},
                contract_note="Collection unavailable; counts are unknown, not evidence of stopped fits.",
                errors=["Collector unavailable: " + type(error).__name__],
            )
        snapshots.append(snapshot)
    return snapshots


def run(config: dict, reason: str, *, dry_run: bool) -> int:
    """Persist reports before any write, then publish only to the two accepted PRs."""
    import fcntl

    state_root = ROOT / "state/github-reporting-v1"
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / "publisher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOGGER.info("Another reporter owns publication; the hourly reconciler remains active")
            return 0
        cursor_path = state_root / "CURSOR.json"
        cursor = read_json(cursor_path, {"experiments": {}, "cooldown_until_unix": 0})
        if cursor.get("cooldown_until_unix", 0) > time.time():
            return 0
        snapshots = collect_snapshots(ROOT)
        outbox = dict(observed_unix=time.time(), reason=reason, snapshots=snapshots)
        outbox_path = state_root / "outbox" / (canonical_sha(snapshots) + ".json")
        write_json(outbox_path, outbox)
        if dry_run:
            write_json(state_root / "DRY_RUN.json", outbox)
            LOGGER.info(
                "Verified reporting dry run: %s",
                [(s["experiment_id"], len(s["completed"])) for s in snapshots],
            )
            return 0
        if not config.get("execute_enabled"):
            raise ValueError("Automatic publication is not enabled in the accepted configuration")
        client = GitHub(read_token(Path(config["token_file"])))
        user = client.request("GET", "/user")
        if user["login"] != config["publisher_login"]:
            raise ValueError("GitHub publisher identity differs from the accepted account")
        failures = []
        for snapshot in snapshots:
            name = snapshot["experiment_id"]
            try:
                existing = cursor["experiments"].setdefault(name, {})
                publish_snapshot(
                    client, snapshot, config["targets"][name], existing, user["login"], reason
                )
                write_json(cursor_path, cursor)
                append_event(
                    state_root / "publications.jsonl",
                    "published",
                    experiment=name,
                    status_url=existing["status_url"],
                    result_urls=existing["result_urls"],
                    outbox_sha256=canonical_sha(snapshots),
                )
            except (GitHubError, RuntimeError, ValueError, OSError) as error:
                failures.append(name)
                delay = max(
                    getattr(error, "retry_after", 0),
                    3600 if getattr(error, "status", 0) in {403, 429} else 60,
                )
                cursor["cooldown_until_unix"] = time.time() + delay
                write_json(cursor_path, cursor)
                append_event(
                    state_root / "publications.jsonl",
                    "deferred",
                    experiment=name,
                    error_type=type(error).__name__,
                    http_status=getattr(error, "status", None),
                    outbox=str(outbox_path),
                )
                LOGGER.error(
                    "Publication deferred for %s (%s); evidence retained",
                    name,
                    type(error).__name__,
                )
        terminal = all(snapshot.get("terminal") is True for snapshot in snapshots)
        expired = time.time() > config["expires_unix"]
        if not failures and (terminal or expired):
            subprocess.run(["systemctl", "--user", "stop", *STOP_UNITS], timeout=15, check=True)
            append_event(
                state_root / "publications.jsonl",
                "triggers_stopped",
                terminal=terminal,
                expired=expired,
            )
        write_json(
            state_root / "LAST_RUN.json",
            dict(
                time_unix=time.time(),
                terminal=terminal,
                failures=failures,
                scored={s["experiment_id"]: len(s["completed"]) for s in snapshots},
            ),
        )
        return 1 if failures else 0


def main() -> int:
    """Run one bounded hourly or completion-driven reporting pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--reason", choices=("hourly", "completion"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(
        verify_configuration(args.config, args.config_sha256), args.reason, dry_run=args.dry_run
    )


if __name__ == "__main__":
    raise SystemExit(main())
