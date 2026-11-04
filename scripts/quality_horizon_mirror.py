"""Mirror verified quality-horizon native journals into the existing local dashboard."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import logging
import time
import urllib.parse
from pathlib import Path

from multistage_github_local_files import atomic_write
from publish_checkpoint_releases import GitHub

REPO = "/repos/VladKochetov007/WunderFunder"
BRANCH = "results/tabdata-multistage-b300-20261031"
PREFIX = "research/experiment_drafts/multistage-optuna-b300-20261031/quality-execution-r1/live/"
ALLOWED = {"qh-a1302", "qh-b1301", "qh-b1302", "qh-c1301", "qh-c1302"}
LOG = logging.getLogger(__name__)


def digest(raw: bytes) -> str:
    """Verify the scientific transport hash."""
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    """Serialize native events without changing numerical values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fetch_file(api: GitHub, name: str, commit: str) -> bytes:
    """Fetch a bounded exact-commit blob and verify Git's byte identity."""
    query = urllib.parse.urlencode({"ref": commit})
    record = api.request("GET", REPO + "/contents/" + PREFIX + name + "?" + query)
    blob = record
    if blob.get("encoding") != "base64":
        blob = api.request("GET", REPO + "/git/blobs/" + record["sha"])
    raw = base64.b64decode(blob["content"])
    assert len(raw) < 3_000_000 and record["size"] == len(raw)
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == record["sha"]
    )
    return raw


def mirror_once(api: GitHub, root: Path, old_commit: str | None) -> str:
    """Keep the primary-valid scope and actual journal coverage visible."""
    commit = api.request("GET", REPO + "/git/ref/heads/" + BRANCH)["object"]["sha"]
    if commit == old_commit:
        return commit
    manifest = json.loads(fetch_file(api, "native_snapshot.manifest.json", commit))
    packed = fetch_file(api, "native_snapshot.json.gz", commit)
    assert digest(packed) == manifest["compressed_sha256"]
    with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
        raw = stream.read(32_000_001)
    assert len(raw) <= 32_000_000 and len(raw) == manifest["payload_bytes"]
    assert digest(raw) == manifest["payload_sha256"]
    body = json.loads(raw)
    assert body["schema"] == "wnn.quality-horizon-native-snapshot.v1"
    assert len(body["runs"]) <= 5
    for run in body["runs"]:
        name = run["run_id"]
        assert name in ALLOWED and run["phase"] == "primary_batch_tbptt_ablation"
        metadata = run["metadata"]
        assert digest(canonical(metadata)) == run["metadata_canonical_sha256"]
        data = metadata["provenance"]["data"]
        assert data["split_sha256"] == body["contract"]["primary_split_sha256"]
        assert len(data["fit_members"]) == 10325 and all(
            x[0] == "train" for x in data["fit_members"]
        )
        assert len(data["evaluation_members"]) == 1873 and all(
            x[0] == "valid" for x in data["evaluation_members"]
        )
        assert data["evaluation_source"] == data["evaluation_role"] == "valid"
        target = root / name
        target.mkdir(exist_ok=True)
        assert not target.is_symlink()
        atomic_write(target / "run_metadata.json", canonical(metadata) + b"\n")
        for key, filename in [
            ("event_journal", "monitor_events.jsonl"),
            ("evaluation_journal", "monitor_evaluations.jsonl"),
        ]:
            journal = run[key]
            assert journal["retained_event_count"] == len(journal["events"])
            atomic_write(
                target / filename, b"".join(canonical(x) + b"\n" for x in journal["events"])
            )
        atomic_write(
            target / "mirror_provenance.json",
            canonical(
                {
                    "commit": commit,
                    "metadata_source_sha256": run["metadata_source_sha256"],
                    "event_journal": {
                        k: v for k, v in run["event_journal"].items() if k != "events"
                    },
                    "evaluation_journal": {
                        k: v for k, v in run["evaluation_journal"].items() if k != "events"
                    },
                }
            ),
        )
    LOG.info("mirrored %s quality views at %s", len(body["runs"]), commit[:12])
    return commit


def main() -> None:
    """Update the already running dashboard through authenticated GitHub snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    args.mirror_root.mkdir(parents=True, exist_ok=True)
    api = GitHub(Path(__file__).resolve().parents[1])
    last = None
    while time.time() < args.deadline_epoch:
        try:
            last = mirror_once(api, args.mirror_root, last)
        except (OSError, ValueError, RuntimeError) as error:
            LOG.warning("mirror awaiting a valid snapshot: %s", error)
        time.sleep(min(60, max(0, args.deadline_epoch - time.time())))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
