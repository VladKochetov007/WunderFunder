"""Restore only hash-pinned own archive files without loading model tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from http.client import HTTPMessage
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

Json = dict[str, Any]  # Archive manifests contain heterogeneous file descriptors.
REPO = "hft-kagglers/wnn-challenge2-model-archive"
REVISION = "167452c72a8e994b280446d2fca93f7c08e16f66"
MAX_TOTAL = 40_000_000


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Never carry a Hub credential to a redirected host."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Strip authorization after every redirect and require HTTPS."""
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError("unsafe redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def sha(path: Path) -> str:
    """Hash a bounded archive file without deserializing it."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(value: str) -> PurePosixPath:
    """Reject absolute paths and traversal before constructing archive paths."""
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in {".", ".."} for p in path.parts):
        raise ValueError("unsafe relative path")
    return path


def copy_verified(stream: Any, target: Path, size: int, digest: str) -> None:
    """Publish a streamed file only after exact size/hash validation.

    Args:
        stream: Binary response or test stream implementing read.
        target: New archive destination; existing outputs are never overwritten.
        size: Expected exact byte length.
        digest: Expected SHA256 from the frozen archive inventory.
    """
    part = target.with_name(target.name + ".partial")
    hasher = hashlib.sha256()
    count = 0
    with part.open("xb") as output:
        while block := stream.read(min(262144, size - count + 1)):
            count += len(block)
            if count > size:
                raise ValueError("download exceeds frozen size")
            hasher.update(block)
            output.write(block)
        output.flush()
        os.fsync(output.fileno())
    if count != size or hasher.hexdigest() != digest:
        raise ValueError("download size or hash mismatch; partial retained")
    os.link(part, target)
    part.unlink()


def restore(manifest_path: Path, manifest_sha: str, destination: Path) -> Json:
    """Restore a finite list using existing Hub authentication in memory only."""
    if manifest_path.stat().st_size > 128000 or sha(manifest_path) != manifest_sha:
        raise ValueError("manifest identity mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["repo_id"] != REPO or manifest["revision"] != REVISION:
        raise ValueError("unexpected archive identity")
    files = manifest["files"]
    if len(files) != 10 or sum(f["bytes"] for f in files) != manifest["total_bytes"]:
        raise ValueError("unexpected file list")
    if not 0 < manifest["total_bytes"] <= MAX_TOTAL:
        raise ValueError("archive exceeds bounded allocation")
    for item in files:
        relative(item["local_path"])
        relative(item["hub_path"])
        if item["bytes"] <= 0 or len(item["sha256"]) != 64:
            raise ValueError("invalid file descriptor")
    destination.mkdir(parents=True, exist_ok=False)
    token = (Path.home() / ".cache/huggingface/token").read_text().strip()
    opener = urllib.request.build_opener(SafeRedirect())
    restored = []
    for item in files:
        target = destination / item["local_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{item['hub_path']}"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with opener.open(request, timeout=25) as response:
            copy_verified(response, target, item["bytes"], item["sha256"])
        restored.append({**item, "destination": str(target), "verified": True})
    receipt = {
        "schema": "quality_gap.control_archive_restore.v1",
        "status": "restored_hash_verified_no_tensor_load",
        "observed_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": manifest_sha,
        "script_sha256": sha(Path(__file__)),
        "repo_id": REPO,
        "revision": REVISION,
        "total_bytes": sum(item["bytes"] for item in files),
        "files": restored,
        "scope": "Own existing archive only; no model load, prediction, fitting or credential copy.",
    }
    with (destination / "RESTORE_RECEIPT.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main() -> None:
    """Run bounded restoration; redact URLs and credentials from failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = restore(args.manifest, args.manifest_sha256, args.destination)
    except Exception as error:
        status = {"status": "failed", "exception_type": type(error).__name__}
        if isinstance(error, urllib.error.HTTPError):
            status["http_status"] = str(error.code)
        print(json.dumps(status), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({"status": receipt["status"], "total_bytes": receipt["total_bytes"]}))


if __name__ == "__main__":
    main()
