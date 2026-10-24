"""Publish small allowlisted validation receipts, excluding weights and private arrays."""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path
from typing import Any

REPO = "/repos/VladKochetov007/WunderFunder"
BRANCH = "experiment/pr32-b300-evidence-20261024"
PREFIX = "research/experiment_drafts/b300-transformer-distillation-runtime/execution_20261024"


def publish_evidence(api: Any, root: Path, digest: str) -> dict:
    """Append immutable metric/code receipts to the existing evidence branch."""
    stamp = root / "EVIDENCE_PUBLISHED.json"
    if stamp.exists():
        previous = json.loads(stamp.read_text())
        if previous.get("digest") == digest:
            return previous
    files = {f"operator/{p.name}": p for p in Path(__file__).parent.glob("*.py")}
    for path in [root / "INITIAL_CAPTURE.json", *root.glob("*-job.json")]:
        files["validation/" + path.name] = path
    for result in (root / "evaluations").glob("*/inode-*/RESULT.json"):
        if json.loads(result.read_text()).get("status") != "complete":
            continue
        for name in (
            "RESULT.json",
            "PROGRESS.jsonl",
            "WORKER_EXIT.json",
            "INDEPENDENT_METRIC_CHECK.json",
        ):
            path = result.parent / name
            if path.is_file():
                files["validation/" + path.relative_to(root).as_posix()] = path
    manifest = {
        "schema": "wnn.pr32.validation-evidence.v1",
        "created_unix": time.time(),
        "digest": digest,
        "files": {},
        "excludes": ["weights", "targets", "predictions", "credentials"],
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for relative, path in sorted(files.items()):
            content = path.read_bytes()
            manifest["files"][relative] = {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            item = tarfile.TarInfo(relative)
            item.size, item.mode = len(content), 0o644
            archive.addfile(item, io.BytesIO(content))
    compressed = gzip.compress(buffer.getvalue(), mtime=0)
    manifest.update(
        archive_sha256=hashlib.sha256(compressed).hexdigest(), archive_bytes=len(compressed)
    )
    name = f"validation-{int(manifest['created_unix'])}"
    head = api.request("GET", REPO + "/git/ref/heads/" + BRANCH)["object"]["sha"]
    entries = []
    for extension, content in [
        (".tar.gz", compressed),
        (".json", (json.dumps(manifest, indent=2) + "\n").encode()),
    ]:
        blob = api.request(
            "POST",
            REPO + "/git/blobs",
            {"content": base64.b64encode(content).decode(), "encoding": "base64"},
        )
        entries.append(
            {
                "path": f"{PREFIX}/{name}{extension}",
                "mode": "100644",
                "type": "blob",
                "sha": blob["sha"],
            }
        )
    tree = api.request("GET", REPO + "/git/commits/" + head)["tree"]["sha"]
    created_tree = api.request("POST", REPO + "/git/trees", {"base_tree": tree, "tree": entries})
    commit = api.request(
        "POST",
        REPO + "/git/commits",
        {
            "message": "Record intermediate PR32 validation metrics and provenance",
            "tree": created_tree["sha"],
            "parents": [head],
        },
    )
    api.request("PATCH", REPO + "/git/refs/heads/" + BRANCH, {"sha": commit["sha"], "force": False})
    receipt = {
        "digest": digest,
        "commit": commit["sha"],
        "branch": BRANCH,
        "manifest_path": f"{PREFIX}/{name}.json",
        "archive_sha256": manifest["archive_sha256"],
        "archive_bytes": len(compressed),
    }
    temporary = stamp.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(stamp)
    return receipt
