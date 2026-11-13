"""Materialize the immutable native base and this packet's patch in the run store."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

BASE_COMMIT = "33542502a1404a2843359b3999fa34f3a157bf3f"
PACKET = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    """Hash a producer file without loading its payload into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    """Parse heterogeneous canonical run metadata from a bounded regular JSON file."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 << 20:
        raise ValueError(f"Expected a regular metadata file: {path}")
    obj = json.loads(path.read_bytes())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return obj


def git(directory: Path, *args: str) -> str:
    """Read the checked-out identity through Git."""
    return subprocess.check_output(["git", "-C", directory, *args], text=True, timeout=30).strip()


def source_contract(runtime: Path, manifest: dict[str, Any]) -> None:
    """Verify the effective fit, prediction and scoring files before native imports."""
    sources = sorted((runtime / "src").rglob("*.py")) + [runtime / "scripts/train_sequence.py"]
    if any(path.is_symlink() for path in sources) or runtime.is_symlink():
        raise ValueError("Expected a regular, private runtime tree")
    files = {path.relative_to(runtime).as_posix(): sha256(path) for path in sources}
    canonical = json.dumps(
        files, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Training source differs from the frozen patched native runtime")
    for rel, expected in manifest["entrypoints"].items():
        path = runtime / rel
        if path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"Native entrypoint differs: {rel}")
    base = Path(os.environ["WF_BASE"])
    blob = git(base, "rev-parse", f"{BASE_COMMIT}:utils.py")
    if git(base, "hash-object", str(runtime / "utils.py")) != blob:
        raise ValueError("Shared native utils.py differs from the pinned source")


def materialize(base: Path, runs: Path, manifest: dict[str, Any]) -> None:
    """Export only native project sources to a new directory; keep existing checkouts intact."""
    if git(base, "rev-parse", "HEAD") != BASE_COMMIT:
        raise ValueError("WF_BASE must be the pinned standalone KD source")
    runtime = runs / "runtime"
    if runtime.exists():
        source_contract(runtime, manifest)
        return
    patch = PACKET / "runtime.patch"
    if sha256(patch) != manifest["patch_sha256"] or manifest["base_commit"] != BASE_COMMIT:
        raise ValueError("The B03 patch and native source do not match the manifest")
    with tempfile.TemporaryDirectory(dir=runs) as work:
        tmp = Path(work)
        dest = tmp / "runtime"
        dest.mkdir()
        archive = tmp / "native.tar"
        subprocess.run(
            [
                "git",
                "-C",
                base,
                "-c",
                "core.autocrlf=false",
                "archive",
                "--format=tar",
                "-o",
                archive,
                BASE_COMMIT,
                "src",
                "scripts",
                "utils.py",
            ],
            check=True,
        )
        subprocess.run(["tar", "-xf", archive, "-C", dest], check=True)
        # An isolated temporary repository makes git apply operate at this exact root.
        subprocess.run(["git", "-C", dest, "init", "--quiet"], check=True)
        subprocess.run(["git", "-C", dest, "apply", "--whitespace=nowarn", patch], check=True)
        source_contract(dest, manifest)
        dest.rename(runtime)


def main() -> None:
    """Check source bytes before every stage, materializing them only on first check."""
    runs = Path(os.environ["WF_RUNS"]).resolve()
    if git(PACKET, "status", "--porcelain", "--untracked-files=all", "--", "."):
        raise ValueError("Receiver packet must be a clean Git checkout")
    manifest = read(PACKET / "runtime_manifest.json")
    if sys.argv[1] == "check":
        materialize(Path(os.environ["WF_BASE"]), runs, manifest)
    else:
        source_contract(runs / "runtime", manifest)


if __name__ == "__main__":
    main()
