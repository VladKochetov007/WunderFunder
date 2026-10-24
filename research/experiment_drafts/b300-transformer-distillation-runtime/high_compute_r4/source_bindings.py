"""Source inventories bound into an R4 campaign."""

from __future__ import annotations

from typing import Any

from campaign_io import (
    PACKET_ROOT,
    R3_INPUTS,
    R3_ROOT,
    REPO_ROOT,
    canonical_sha256,
    require,
    sha256_file,
)


def runtime_source_files() -> dict[str, str]:
    """Hash every Python source file plus the sequence training entrypoint."""
    paths = sorted((REPO_ROOT / "src").rglob("*.py"))
    paths.append(REPO_ROOT / "scripts/train_sequence.py")
    return {path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in paths}


def packet_source_files() -> dict[str, str]:
    """Hash static packet inputs while omitting bytecode and runtime output."""
    return {
        path.relative_to(PACKET_ROOT).as_posix(): sha256_file(path)
        for path in sorted(PACKET_ROOT.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def r3_source_files() -> dict[str, str]:
    """Hash the frozen R3 recipe and configs that define A and G's reference."""
    return {relative: sha256_file(R3_ROOT / relative) for relative in R3_INPUTS}


def source_identity(files: dict[str, str]) -> dict[str, object]:
    """Bind an explicit source map to its canonical digest."""
    return {"sha256": canonical_sha256(files), "source_files": files}


def _validate_source_identity(name: str, stored: dict[str, Any], current: dict[str, str]) -> None:
    """Require exact per-file source hashes and the canonical aggregate digest."""
    require(stored.get("source_files") == current, f"{name} files changed since init")
    require(stored.get("sha256") == canonical_sha256(current), f"{name} hash changed since init")
