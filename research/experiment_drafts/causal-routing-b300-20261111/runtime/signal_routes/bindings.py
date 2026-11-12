"""Immutable experiment, data, and source identities for Signal Routes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

RUNTIME_COMMIT = "8626000a4b965763a70011a0291a891aa6535d7e"
RUNTIME_MANIFEST_SHA256 = "b6c2004cd5b916a8976973217b944d44f79464a557309b31c42f8912ef7711bf"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
SEEDS = (205201, 205202)
VARIANTS = (
    "plain",
    "attention_additive",
    "attention_differential",
    "ffn_static",
    "ffn_conditional",
)
PACKET = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    """Hash a regular evidence file in bounded chunks."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value: object) -> str:
    """Hash canonical JSON data using the runtime's stable encoding."""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Read bounded JSON metadata and reject non-object or symlink inputs."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32_000_000:
        raise ValueError(f"missing, unsafe, or oversized metadata: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_new(path: Path, value: object) -> None:
    """Create immutable JSON evidence without replacing an earlier attempt."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def admit_runtime(root: Path) -> Path:
    """Admit only the clean, pinned PR75 runtime before importing canonical modules."""
    root = root.resolve(strict=True)
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if commit != RUNTIME_COMMIT:
        raise ValueError(f"runtime must be pinned to {RUNTIME_COMMIT}, got {commit}")
    subprocess.run(
        ["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", "src", "scripts", "utils.py"],
        check=True,
        capture_output=True,
        timeout=10,
    )
    untracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "src", "scripts"],
        text=True,
        timeout=10,
    )
    if any(line.endswith(".py") for line in untracked.splitlines()):
        raise ValueError("pinned runtime contains untracked Python modules")
    if any(name == "wnn" or name.startswith("wnn.") for name in sys.modules):
        raise RuntimeError("admit the canonical runtime before importing wnn")
    sys.path[:0] = [str(root / "src"), str(root)]
    return root


def source_identity(root: Path) -> dict[str, object]:
    """Bind all pinned canonical Python sources and every packet runtime module."""
    runtime_files: dict[str, str] = {}
    canonical_paths = sorted((root / "src").rglob("*.py"))
    canonical_paths.extend((root / "scripts/train_sequence.py", root / "utils.py"))
    for path in canonical_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"canonical runtime source is not a regular file: {path}")
        runtime_files[f"runtime/{path.relative_to(root).as_posix()}"] = digest(path)
    manifest_path = PACKET / "sources/runtime_source_files.json"
    if digest(manifest_path) != RUNTIME_MANIFEST_SHA256:
        raise ValueError("pinned canonical runtime file manifest has changed")
    manifest = read_object(manifest_path)
    expected_runtime = manifest.get("files")
    if (
        manifest.get("runtime_commit") != RUNTIME_COMMIT
        or not isinstance(expected_runtime, dict)
        or manifest.get("source_sha256") != canonical(expected_runtime)
        or runtime_files != expected_runtime
    ):
        raise ValueError(
            "canonical runtime worktree is incomplete or differs from the pinned manifest"
        )
    files = dict(runtime_files)
    for path in sorted((PACKET / "runtime").rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"packet runtime source is not a regular file: {path}")
        files[f"signal_routes/{path.relative_to(PACKET).as_posix()}"] = digest(path)
    return {"sha256": canonical(files), "files": files}


def load_cell(path: Path) -> tuple[dict[str, Any], Path]:
    """Validate a frozen cell and resolve its seed-specific native PR75 recipe."""
    cell = read_object(path)
    required = {"schema", "id", "variant", "seed", "native_config", "native_sha256"}
    variant, seed = cell.get("variant"), cell.get("seed")
    if (
        set(cell) != required
        or cell.get("schema") != "signal-routes.cell.v1"
        or variant not in VARIANTS
        or type(seed) is not int
        or seed not in SEEDS
        or cell.get("id") != f"sr_{variant}_s{seed}"
    ):
        raise ValueError("cell differs from the frozen Signal Routes panel")
    relative = cell.get("native_config")
    if (
        not isinstance(relative, str)
        or ".." in Path(relative).parts
        or Path(relative).is_absolute()
    ):
        raise ValueError("cell native_config must be a safe packet-relative path")
    candidate = PACKET / relative
    if candidate.is_symlink():
        raise ValueError("native config cannot be a symlink")
    native_path = candidate.resolve(strict=True)
    if not native_path.is_relative_to(PACKET / "configs"):
        raise ValueError("native config must resolve inside the packet configs directory")
    if digest(native_path) != cell.get("native_sha256"):
        raise ValueError("cell native config digest mismatch")
    native = read_object(native_path)
    expected = read_object(PACKET / "sources/native_control.json")
    expected["seed"] = seed
    if native != expected:
        raise ValueError("native config differs from the PR75 control except for its seed")
    return cell, native_path
