"""Verify GitHub snapshots and mirror native run files for the local dashboard."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import time
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

from multistage_github_common import (
    MANIFEST_PATH,
    MAX_COMPRESSED_BYTES,
    PARENT_CODE_BRANCH,
    REPOSITORY,
    RESULTS_BRANCH,
    SNAPSHOT_PATH,
    canonical_json,
    is_sha256,
    sha256_bytes,
    verify_snapshot,
)
from multistage_github_local_files import (  # noqa: E402
    atomic_write,
    ensure_local_mirror_root,
    safe_run_directory,
)
from publish_checkpoint_releases import GitHub

LOG = logging.getLogger("multistage_github_mirror")
DEFAULT_DEADLINE = 1793425269.65894
POLL_SECONDS = 120
MAX_MANIFEST_BYTES = 1_048_576


def _request_file(github: GitHub, path: str) -> bytes | None:
    """Fetch one exact allowlisted Contents path or its Git blob, verifying Git identity."""
    if path not in {SNAPSHOT_PATH, MANIFEST_PATH}:
        raise ValueError("GitHub mirror requested a path outside the snapshot allowlist")
    query = urllib.parse.urlencode({"ref": RESULTS_BRANCH})
    endpoint_path = urllib.parse.quote(path, safe="/.-")
    endpoint = f"/repos/{REPOSITORY}/contents/{endpoint_path}?{query}"
    try:
        record = github.request("GET", endpoint)
    except RuntimeError as error:
        if "HTTP404" in str(error):
            return None
        raise
    if not isinstance(record, dict) or record.get("type") != "file" or record.get("path") != path:
        raise ValueError("GitHub returned a file outside the approved snapshot path")
    blob_sha = record.get("sha")
    if not isinstance(blob_sha, str) or len(blob_sha) != 40:
        raise ValueError("GitHub Contents response lacks a Git blob SHA")
    content = record.get("content")
    encoding = record.get("encoding")
    if encoding == "base64" and isinstance(content, str):
        raw = base64.b64decode(content, validate=False)
    else:
        blob = github.request("GET", f"/repos/{REPOSITORY}/git/blobs/{blob_sha}")
        if (
            not isinstance(blob, dict)
            or blob.get("sha") != blob_sha
            or blob.get("encoding") != "base64"
        ):
            raise ValueError("GitHub returned an unexpected Git blob")
        blob_content = blob.get("content")
        if not isinstance(blob_content, str):
            raise ValueError("GitHub blob response has no encoded content")
        raw = base64.b64decode(blob_content, validate=False)
    limit = MAX_COMPRESSED_BYTES if path == SNAPSHOT_PATH else MAX_MANIFEST_BYTES
    if len(raw) > limit:
        raise ValueError("GitHub snapshot file exceeds its mirror byte bound")
    git_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if git_sha != blob_sha or record.get("size") != len(raw):
        raise ValueError("GitHub Contents bytes differ from their Git blob identity")
    return raw


def fetch_snapshot(github: GitHub) -> tuple[dict[str, object], dict[str, object], bytes] | None:
    """Fetch both fixed snapshot files and verify their content and manifest bindings."""
    manifest_raw = _request_file(github, MANIFEST_PATH)
    if manifest_raw is None:
        return None
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict):
        raise ValueError("GitHub snapshot manifest is not an object")
    compressed = _request_file(github, SNAPSHOT_PATH)
    if compressed is None:
        raise ValueError("manifest exists but the bound snapshot blob is missing")
    body, payload = verify_snapshot(compressed, manifest)
    return body, manifest, payload


def _branch_head_sha(github: GitHub) -> str | None:
    """Read the dedicated branch ref so unchanged snapshots need no blob download."""
    branch = urllib.parse.quote(RESULTS_BRANCH, safe="/")
    try:
        value = github.request("GET", f"/repos/{REPOSITORY}/git/ref/heads/{branch}")
    except RuntimeError as error:
        if "HTTP404" in str(error):
            return None
        raise
    branch_object = value.get("object", {}) if isinstance(value, dict) else {}
    sha = branch_object.get("sha") if isinstance(branch_object, dict) else None
    if not isinstance(sha, str) or len(sha) != 40:
        raise ValueError("GitHub returned a malformed results branch ref")
    return sha


def _cached_receipt_sha(runs_root: Path, branch_head: str | None) -> str | None:
    """Reuse a previously verified local snapshot only while its branch commit is unchanged."""
    if branch_head is None:
        return None
    path = runs_root / ".multistage_github_snapshot.json"
    if not path.exists():
        return None
    info = path.lstat()
    if path.is_symlink() or info.st_size > 1_048_576:
        raise ValueError("local snapshot receipt is unsafe or oversized")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        isinstance(value, dict)
        and value.get("branch_head_sha") == branch_head
        and is_sha256(value.get("snapshot_sha256"))
    ):
        return str(value["snapshot_sha256"])
    return None


def _write_run(root: Path, run: dict[str, object]) -> None:
    """Atomically write the three native dashboard files under a validated run ID."""
    run_dir = safe_run_directory(root, run.get("run_id"))
    metadata = run.get("metadata")
    event_journal = run.get("event_journal")
    evaluation_journal = run.get("evaluation_journal")
    if (
        not isinstance(metadata, dict)
        or not isinstance(event_journal, dict)
        or not isinstance(evaluation_journal, dict)
    ):
        raise ValueError("verified snapshot has an incomplete run record")
    metadata_bytes = (
        json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    atomic_write(run_dir / "run_metadata.json", metadata_bytes)
    for filename, journal in (
        ("monitor_events.jsonl", event_journal),
        ("monitor_evaluations.jsonl", evaluation_journal),
    ):
        events = journal.get("events")
        if not isinstance(events, list):
            raise ValueError("verified snapshot journal has no events")
        lines = b"".join(canonical_json(event) + b"\n" for event in events)
        atomic_write(run_dir / filename, lines)


def mirror_once(github: GitHub, runs_root: Path) -> str | None:
    """Download, verify, and apply one immutable snapshot without touching source runs."""
    branch_head = _branch_head_sha(github)
    cached_sha = _cached_receipt_sha(runs_root, branch_head)
    if cached_sha is not None:
        return cached_sha
    result = fetch_snapshot(github)
    if result is None:
        return None
    body, manifest, payload = result
    runs = body.get("runs")
    if not isinstance(runs, list):
        raise ValueError("verified snapshot has no run list")
    ensure_local_mirror_root(runs_root)
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("verified snapshot contains a malformed run")
        _write_run(runs_root, run)
    receipt = {
        "schema": "wnn.multistage-github-local-mirror.v1",
        "repository": REPOSITORY,
        "branch": RESULTS_BRANCH,
        "parent_code_branch": PARENT_CODE_BRANCH,
        "branch_head_sha": branch_head,
        "snapshot_sha256": manifest["snapshot_sha256"],
        "manifest_sha256": sha256_bytes(canonical_json(manifest)),
        "json_sha256": sha256_bytes(payload),
        "run_count": len(runs),
        "mirrored_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write(runs_root / ".multistage_github_snapshot.json", canonical_json(receipt) + b"\n")
    return str(manifest["snapshot_sha256"])


def run(args: argparse.Namespace) -> None:
    """Poll the fixed results branch until the campaign's shared deadline."""
    github = GitHub(args.git_root.resolve(strict=True))
    user = github.request("GET", "/user")
    if not isinstance(user, dict) or user.get("login") != "SemyonEpanov":
        raise ValueError("local Git credential does not identify SemyonEpanov")
    ensure_local_mirror_root(args.runs_root)
    last_sha = ""
    while True:
        try:
            digest = mirror_once(github, args.runs_root)
        except (OSError, RuntimeError) as error:
            LOG.warning("local mirror fetch will retry: %s", error)
            digest = None
        if digest is not None and digest != last_sha:
            LOG.info("verified local dashboard mirror snapshot=%s", digest[:12])
            last_sha = digest
        now = time.time()
        if now >= args.deadline_epoch:
            return
        time.sleep(min(args.poll_seconds, max(0.0, args.deadline_epoch - now)))


def main() -> None:
    """Run the local mirror using the existing Git Credential Manager session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-root", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--deadline-epoch", type=float, default=DEFAULT_DEADLINE)
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
