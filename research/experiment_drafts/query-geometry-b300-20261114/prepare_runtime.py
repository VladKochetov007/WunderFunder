"""Materialize a pinned native runtime with this packet's isolated experiment patch."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
PIN = "8626000a4b965763a70011a0291a891aa6535d7e"
BASE = "research/experiment_drafts/transformer-qk-normalization-20261109"
# Receipts contain heterogeneous, nested native identity records.


def read(path: Path) -> dict[str, Any]:
    """Read bounded regular JSON metadata, rejecting symlinks and non-object roots."""
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size < 64 << 20:
        raise ValueError(f"expected bounded regular JSON metadata: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    """Hash exact source bytes without importing or executing model code."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    """Use the native provenance canonical JSON hash convention."""
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(data.encode()).hexdigest()


def git(directory: Path, *args: str) -> str:
    """Read a Git object or packet identity with a bounded subprocess."""
    return subprocess.check_output(["git", "-C", directory, *args], text=True, timeout=30).strip()


def file_map(root: Path) -> dict[str, str]:
    """Inventory the complete isolated runtime, excluding only Git administration."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("runtime must be a regular directory")
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] == ".git":
            continue
        if path.is_symlink():
            raise ValueError(f"symlink in isolated runtime: {relative}")
        if path.is_file():
            files[relative.as_posix()] = sha256(path)
    return files


def packet_identity() -> str:
    """Require a clean published packet before accepting its source manifest."""
    if git(PACKET, "status", "--porcelain", "--untracked-files=all", "--", "."):
        raise ValueError("receiver packet must be a clean published Git checkout")
    return git(PACKET, "rev-parse", "HEAD")


def manifest() -> dict[str, Any]:
    """Bind the patch and added modules to the committed complete file inventory."""
    value = read(PACKET / "runtime_manifest.json")
    payload = file_map(PACKET / "payload")
    if (
        value.get("schema") != "wnn.query-geometry.runtime.v1"
        or value.get("base_commit") != PIN
        or value.get("patch_sha256") != sha256(PACKET / "runtime.patch")
        or value.get("payload_sha256") != payload
        or value.get("files_sha256") != canonical(value["files"])
    ):
        raise ValueError("runtime manifest does not bind this patch and payload")
    source = {
        name: digest
        for name, digest in value["files"].items()
        if (name.startswith("src/") and name.endswith(".py")) or name == "scripts/train_sequence.py"
    }
    if value.get("source_sha256") != canonical(source):
        raise ValueError("manifest source identity differs from the native hash convention")
    return value


def verify_runtime(runtime: Path) -> dict[str, Any]:
    """Verify every effective source, config and helper before native imports."""
    value = manifest()
    if file_map(runtime) != value["files"]:
        raise ValueError("isolated runtime has changed, added or missing files")
    return value


def publish(path: Path, value: dict[str, Any]) -> None:
    """Create an immutable receipt without replacing any earlier evidence."""
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, path)


def materialize(base: Path, runs: Path) -> None:
    """Extract immutable Git objects into a fresh private tree; preserve source checkouts."""
    packet_commit = packet_identity()
    contract = manifest()
    if git(base, "rev-parse", f"{PIN}^{{commit}}") != PIN:
        raise ValueError("WF_BASE must contain the pinned PR75 Git object")
    runtime = runs / "runtime"
    receipt = runs / "runtime_preparation.json"
    if runtime.exists() or receipt.exists():
        raise FileExistsError("runtime preparation already exists; use the verify stage")
    with tempfile.TemporaryDirectory(dir=runs) as temporary:
        work = Path(temporary)
        destination = work / "runtime"
        destination.mkdir()
        archive = work / "native.tar"
        subprocess.run(
            [
                "git",
                "-C",
                base,
                "archive",
                "--format=tar",
                "-o",
                archive,
                PIN,
                "src",
                "scripts",
                "utils.py",
                BASE,
            ],
            check=True,
            timeout=120,
        )
        subprocess.run(["tar", "-xf", archive, "-C", destination], check=True, timeout=120)
        subprocess.run(["git", "-C", destination, "init", "--quiet"], check=True, timeout=30)
        subprocess.run(
            ["git", "-C", destination, "apply", "--whitespace=error", PACKET / "runtime.patch"],
            check=True,
            timeout=30,
        )
        for relative in contract["payload_sha256"]:
            target = destination / "src" / relative
            if target.exists():
                raise FileExistsError(f"payload unexpectedly replaces native file: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PACKET / "payload" / relative, target)
        verify_runtime(destination)
        destination.rename(runtime)
    publish(
        receipt,
        {
            "schema": "wnn.query-geometry.prepared.v1",
            "packet_commit": packet_commit,
            "base_commit": PIN,
            "runtime_manifest_sha256": sha256(PACKET / "runtime_manifest.json"),
            "source_sha256": contract["source_sha256"],
            "files_sha256": contract["files_sha256"],
        },
    )


def verify_preparation(runtime: Path) -> dict[str, Any]:
    """Reject artifacts from another packet revision or altered prepared sources."""
    contract = verify_runtime(runtime)
    value = read(runtime.parent / "runtime_preparation.json")
    expected = {
        "schema": "wnn.query-geometry.prepared.v1",
        "packet_commit": packet_identity(),
        "base_commit": PIN,
        "runtime_manifest_sha256": sha256(PACKET / "runtime_manifest.json"),
        "source_sha256": contract["source_sha256"],
        "files_sha256": contract["files_sha256"],
    }
    if value != expected:
        raise ValueError("preparation receipt differs from the source-qualified packet")
    return value


if __name__ == "__main__":
    if sys.platform != "linux":
        raise RuntimeError("runtime preparation is a Linux receiver operation")
    run_root = Path(os.environ["WF_RUNS"]).resolve(strict=True)
    if sys.argv[1:] == ["prepare"]:
        materialize(Path(os.environ["WF_BASE"]).resolve(strict=True), run_root)
    elif sys.argv[1:] == ["verify"]:
        verify_preparation(run_root / "runtime")
    else:
        raise ValueError("expected prepare or verify")
