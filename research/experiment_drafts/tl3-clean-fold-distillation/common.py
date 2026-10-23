"""Source and supplied-data bindings for the bounded TL3 student comparison."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Receiver JSON retains canonical heterogeneous provenance records.
PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[2]
DATA_FILES = (
    "datasets/train.parquet",
    "datasets/valid.parquet",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
    "cache/chains.npz",
    "cache/chain_folds.npz",
)


def digest(path: Path) -> str:
    """Hash artifact bytes without loading a large payload into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read bounded JSON metadata, never a model or supplied dataset."""
    if not 0 < path.stat().st_size <= 32 * (1 << 20):
        raise ValueError(f"metadata size outside the 32 MiB bound: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_new(path: Path, value: dict[str, Any]) -> None:
    """Publish a create-only receipt; partial or existing attempts are preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_assets(path: Path) -> dict[str, Any]:
    """Load receiver paths and retain their exact manifest identity."""
    assets = read_json(path)
    assets["_path"] = str(path.resolve())
    assets["_sha256"] = digest(path)
    if not Path(assets["project_root"]).is_absolute():
        raise ValueError("project_root must be an absolute mounted supplied-data workspace")
    if set(assets["splits"]) != {"F0", "F2"}:
        raise ValueError("exactly F0 and F2 must be bound")
    return assets


def output_root(assets: dict[str, Any]) -> Path:
    """Return the experiment's separate physical-disk evidence directory."""
    return Path(assets["project_root"]) / "research_tmp/tl3-clean-kd-r1"


def source_identity() -> dict[str, str]:
    """Bind all canonical Python, adapters and dependency metadata used by this packet."""
    paths = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/**/*.py"), *PACKET.glob("*.py")]
    paths += [PACKET / "recipe.json", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    files = {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(paths))}
    return {
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }


def command_environment(assets: dict[str, Any]) -> dict[str, str]:
    """Isolate imports, supplied data and one-worker resource controls."""
    env = os.environ.copy()
    env.update(
        WNN_PROJECT_ROOT=assets["project_root"],
        PYTHONPATH=f"{ROOT / 'src'}:{ROOT}",
        WNN_THREADS="2",
        WNN_LOADER_WORKERS="0",
        WNN_LOADER_RAM_GB="6",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="2",
        NVIDIA_TF32_OVERRIDE="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    portable = assets.get("portable_registry")
    if portable:
        env["WNN_PORTABLE_DATA_MANIFEST"] = portable["path"]
    else:
        env.pop("WNN_PORTABLE_DATA_MANIFEST", None)
    return env


def preflight(assets: dict[str, Any]) -> dict[str, Any]:
    """Reject source/data/split drift before every receiver phase, without fitting."""
    from wnn.splits import load_campaign_split

    recipe = read_json(PACKET / "recipe.json")
    bound = assets.get("binding")
    if not bound:
        raise ValueError("bind the receiver assets first; the example is not launchable")
    identity = source_identity()
    if identity != bound["source"] or digest(PACKET / "recipe.json") != bound["recipe_sha256"]:
        raise ValueError("source commit, implementation or frozen recipe changed")
    root = Path(assets["project_root"]).resolve(strict=True)
    for name in DATA_FILES:
        entry = bound["files"][name]
        path = root / name
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
            entry["bytes"],
            entry["mtime_ns"],
            entry["ctime_ns"],
        ):
            raise ValueError(f"immutable receiver data changed after content binding: {name}")
    portable = assets.get("portable_registry")
    if portable and digest(Path(portable["path"])) != portable["sha256"]:
        raise ValueError("portable source registry changed")
    if not portable:
        os.environ.pop("WNN_PORTABLE_DATA_MANIFEST", None)
    os.environ.update(command_environment(assets))
    for cell in recipe["cells"]:
        entry = assets["splits"][cell["id"]]
        if (
            digest(Path(entry["path"])) != cell["split_sha256"]
            or entry["sha256"] != cell["split_sha256"]
        ):
            raise ValueError("split bytes differ from the frozen component fold")
        split = load_campaign_split(Path(entry["path"]), root)
        if split.fold != cell["fold"]:
            raise ValueError("fold index does not match the frozen cell")
        split.assert_fit_groups(split.train_groups, split.valid_groups)
    return {
        "source_sha256": identity["sha256"],
        "recipe_sha256": bound["recipe_sha256"],
        "assets_sha256": assets["_sha256"],
    }


def bind_assets(source: Path, destination: Path) -> None:
    """Seal supplied bytes once on the receiver before profiles or scientific fitting."""
    assets = load_assets(source)
    if assets.get("binding"):
        raise ValueError("assets already bound; do not silently replace their identity")
    files = {}
    root = Path(assets["project_root"]).resolve(strict=True)
    for name in DATA_FILES:
        path = root / name
        stat = path.stat()
        files[name] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "sha256": digest(path),
        }
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino) != (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_ino,
        ):
            raise ValueError(f"source changed during content hashing: {name}")
    assets["binding"] = {
        "source": source_identity(),
        "recipe_sha256": digest(PACKET / "recipe.json"),
        "files": files,
        "python": sys.version,
    }
    preflight(assets)
    for key in ("_path", "_sha256"):
        assets.pop(key)
    write_new(destination, assets)
