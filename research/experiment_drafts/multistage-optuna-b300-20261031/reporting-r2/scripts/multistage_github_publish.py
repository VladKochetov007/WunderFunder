"""Publish sanitized native dashboard snapshots to the dedicated results branch."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from galam_l4_quality8.publisher import read_token, remove_owned_token  # noqa: E402
from multistage_github_common import (  # noqa: E402
    canonical_json,
    make_manifest,
    make_snapshot,
    sha256_bytes,
)
from multistage_github_node_api import (  # noqa: E402
    APIError,
    GitHubAPI,
    comment_text,
    commit_snapshot,
    post_comment,
    remote_snapshot_sha,
    results_branch_head,
    verify_publication_context,
)
from multistage_github_source import collect_snapshot  # noqa: E402
from multistage_github_state import load_state, save_state  # noqa: E402

POLL_SECONDS = 120
TERMINAL_STATES = {"completed", "early_stopped", "failed", "error", "cancelled"}
LOG = logging.getLogger("multistage_github_publish")


def _terminal_markers(runs: list[dict[str, object]]) -> list[tuple[str, str, str, str]]:
    """Find one last terminal status per exact native metadata+journal identity."""
    found: dict[str, tuple[str, str, str, str]] = {}
    for run in runs:
        journal = run["event_journal"]
        events = journal["events"]
        if not isinstance(events, list):
            continue
        canonical_metadata_sha = run.get("canonical_metadata_sha256")
        journal_sha = journal.get("source_sha256")
        if not isinstance(canonical_metadata_sha, str) or not isinstance(journal_sha, str):
            raise ValueError("terminal run lacks exact metadata or event-journal identity")
        fit_key = sha256_bytes(
            canonical_json(
                {
                    "canonical_metadata_sha256": canonical_metadata_sha,
                    "event_journal_sha256": journal_sha,
                }
            )
        )
        final_terminal: tuple[str, str] | None = None
        for event in events:
            if event.get("kind") != "status":
                continue
            context = event.get("context")
            status = context.get("status") if isinstance(context, dict) else None
            if status in TERMINAL_STATES:
                marker = hashlib.sha256(canonical_json(event)).hexdigest()
                final_terminal = str(status), marker
        if final_terminal is None:
            continue
        record = (fit_key, str(run["run_id"]), final_terminal[0], final_terminal[1])
        previous = found.get(fit_key)
        if previous is not None and previous[2:] != record[2:]:
            raise ValueError("duplicate endpoint views disagree on their terminal fit status")
        if previous is None or record[1] < previous[1]:
            found[fit_key] = record
    return [found[key] for key in sorted(found)]


def _completed_count(runs: list[dict[str, object]]) -> int:
    """Count distinct exact fits with completed or declared early-stop status."""
    return sum(
        1 for _, _, status, _ in _terminal_markers(runs) if status in {"completed", "early_stopped"}
    )


def _publish_cycle(
    args: argparse.Namespace,
    api: GitHubAPI,
    state: dict[str, object],
    code_sha: str,
) -> tuple[list[dict[str, object]], str]:
    """Publish one changed snapshot and any due PR comments, preserving restart state."""
    runs = collect_snapshot(args.runs_root)
    payload, compressed, body = make_snapshot(runs)
    digest = sha256_bytes(canonical_json(body))
    if remote_snapshot_sha(api) != digest:
        manifest = make_manifest(payload, compressed, datetime.now(UTC).isoformat())
        commit_sha = commit_snapshot(api, code_sha, compressed, manifest)
        state["commit_sha"] = commit_sha
        state["snapshot_sha256"] = digest
        save_state(args.state_file, state)
        LOG.info("published changed snapshot runs=%d commit=%s", len(runs), commit_sha[:12])
    else:
        commit_sha = str(state.get("commit_sha", "")) or str(results_branch_head(api) or "")
        if commit_sha:
            state["commit_sha"] = commit_sha
    if commit_sha:
        seen = {str(item) for item in state.get("terminal_seen", [])}
        for fit_key, run_id, status, event_sha in _terminal_markers(runs):
            marker = f"multistage-b300-terminal-fit-{fit_key[:20]}-{event_sha[:16]}"
            if marker not in seen:
                post_comment(
                    api,
                    comment_text(
                        commit_sha,
                        len(runs),
                        _completed_count(runs),
                        f"terminal fit `{run_id}`: `{status}`",
                    ),
                    marker,
                )
                seen.add(marker)
        state["terminal_seen"] = sorted(seen)
        hour = int(time.time() // 3600)
        if state.get("last_hour_comment") != hour:
            post_comment(
                api,
                comment_text(
                    commit_sha, len(runs), _completed_count(runs), "hourly progress update"
                ),
                f"multistage-b300-hourly-{hour}",
            )
            state["last_hour_comment"] = hour
        save_state(args.state_file, state)
    return runs, commit_sha


def run(args: argparse.Namespace) -> None:
    """Poll changed native dashboard journals and publish until the fixed deadline."""
    token, identity = read_token(args.token_file)
    api = GitHubAPI(token)
    state = load_state(args.state_file)
    code_sha = verify_publication_context(api)
    while True:
        try:
            runs, commit_sha = _publish_cycle(args, api, state, code_sha)
        except APIError as error:
            if error.status != 429 and error.status < 500:
                raise
            LOG.warning("temporary GitHub API failure (HTTP %d); retrying", error.status)
        except OSError as error:
            LOG.warning("native dashboard input is changing or unavailable; retrying: %s", error)
        else:
            if time.time() >= args.deadline_epoch:
                if commit_sha:
                    marker = f"multistage-b300-final-{int(args.deadline_epoch)}"
                    post_comment(
                        api,
                        comment_text(
                            commit_sha,
                            len(runs),
                            _completed_count(runs),
                            "final deadline snapshot",
                        ),
                        marker,
                    )
                if not remove_owned_token(args.token_file, identity):
                    raise RuntimeError(
                        "staged publisher credential changed before exact-inode cleanup"
                    )
                LOG.info("deadline reached; final comment posted and staged credential removed")
                return
        time.sleep(min(args.poll_seconds, max(0.0, args.deadline_epoch - time.time())))


def main() -> None:
    """Parse bounded node-side inputs and run the fixed-repository publisher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--deadline-epoch", type=float, default=1793425269.65894)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    args = parser.parse_args()
    if not 30 <= args.poll_seconds <= 600:
        parser.error("--poll-seconds must be between 30 and 600")
    if args.deadline_epoch <= 0:
        parser.error("--deadline-epoch must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
