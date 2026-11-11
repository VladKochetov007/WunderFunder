from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# JSON recipes are heterogeneous, with strict outer fields validated below.
RUNTIME_COMMIT = "8626000a4b965763a70011a0291a891aa6535d7e"
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
PACKET = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    """Hash a source or evidence file without loading large payloads at once."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value: object) -> str:
    """Hash a JSON value with the canonical runtime encoding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Read bounded regular JSON metadata, rejecting symlink inputs."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32_000_000:
        raise ValueError(f"missing, unsafe or oversized metadata: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_new(path: Path, value: object) -> None:
    """Write new evidence once, preserving an existing attempt."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def admit_runtime(root: Path) -> Path:
    """Import only the exact clean external PR75 runtime, without changing that checkout."""
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
    extra = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "src", "scripts"],
        text=True,
        timeout=10,
    )
    if any(line.endswith(".py") for line in extra.splitlines()):
        raise ValueError("runtime contains untracked Python modules")
    if any(name == "wnn" or name.startswith("wnn.") for name in sys.modules):
        raise RuntimeError("runtime pin must be admitted before importing wnn")
    sys.path[:0] = [str(root / "src"), str(root)]
    return root


def source_identity(root: Path) -> dict[str, object]:
    """Bind both immutable trainer sources and this experiment's adapters/entrypoints."""
    paths = sorted((root / "src").rglob("*.py"))
    paths += [root / "scripts/train_sequence.py", root / "utils.py"]
    files = {f"runtime/{path.relative_to(root)}": digest(path) for path in paths}
    for path in sorted((PACKET / "runtime").rglob("*.py")):
        files[f"impact81/{path.relative_to(PACKET)}"] = digest(path)
    return {"sha256": canonical(files), "files": files}


def load_cell(path: Path) -> tuple[dict[str, Any], Path]:
    """Resolve a named paired cell and its unmodified canonical training recipe."""
    cell = read_object(path)
    fields = {"schema", "id", "card", "arm", "variant", "seed", "native_config", "native_sha256"}
    if set(cell) != fields or cell["schema"] != "impact81.cell.v1":
        raise ValueError("unsupported experiment cell descriptor")
    variants = {"01": "ple", "02": "groups", "03": "target_adapters"}
    if cell["card"] not in variants or cell["arm"] not in ("control", "treatment"):
        raise ValueError("only the first three paired screens are implemented")
    expected = "plain" if cell["arm"] == "control" else variants[cell["card"]]
    if cell["variant"] != expected or cell["seed"] not in (204101, 204102):
        raise ValueError("cell variant or paired seed differs from the declared panel")
    if cell["id"] != f"ip{cell['card']}_{cell['arm']}_s{cell['seed']}":
        raise ValueError("cell ID must encode its card, arm and seed")
    native = (PACKET / cell["native_config"]).resolve(strict=True)
    if not native.is_relative_to(PACKET / "configs") or digest(native) != cell["native_sha256"]:
        raise ValueError("native config path or hash mismatch")
    raw = read_object(native)
    if raw["seed"] != cell["seed"]:
        raise ValueError("native config seed differs from the cell")
    expected_raw = read_object(PACKET / "sources/transformer_plain_control.json")
    expected_raw["seed"] = cell["seed"]
    if raw != expected_raw:
        raise ValueError("PR81 freezes every native training setting except its paired seed")
    return cell, native
