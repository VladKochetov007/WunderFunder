"""Frozen native-feature recipes and durable receipts for the B300 expansion."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
SOURCE_REVISION = "a266b036b02e8e012c6eb31072291dad618a169d"
R01_MANIFEST = "95f82d442eefe425dde6574222b037ba1deba2d3ae36534aeec41e31e450874f"
PR15_MANIFEST = "8b3bc9b636ce192c6da9ababace1c5a1a59e58ebcbbe3394fe7f2ae940704eb9"
GRADER_SHA256 = "56ffc693da52668726bab1f7008e31753a76aed3ff5d79089167785079d9eb31"
ARMS = {
    "group": ("age", "group_age", "none", 323),
    "field": ("age", "field_age", "none", 323),
    "moments": ("innovation", "causal_innovations", "newest", 291),
    "innovations": ("innovation", "causal_innovations", "none", 291),
}


def sha(path: Path) -> str:
    """Hash one immutable artifact with bounded resident memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    """Read an object-valued JSON receipt without following a receipt symlink."""
    if path.is_symlink():
        raise ValueError("Receipt file cannot be a symlink: " + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object: " + str(path))
    return value


def write(path: Path, value: dict, *, fresh: bool = False) -> None:
    """Persist a complete fsynced receipt and preserve existing scientific outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (fresh and path.exists()):
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=".feature-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if fresh:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def event(state: Path, kind: str, **fields: object) -> None:
    """Append one durable controller event without reconstructing prior attempts."""
    state.mkdir(parents=True, exist_ok=True)
    with (state / "events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(dict(event=kind, time_unix=time.time(), **fields), sort_keys=True) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def legacy(root: Path) -> dict:
    """Import the unchanged canonical command/workspace and process-envelope helpers."""
    r01 = root / "runtime/b300-r01-v1/code"
    pr15 = root / "runtime/pr15-host-v2"
    sys.path[:0] = [str(r01), str(pr15)]
    return {
        name: importlib.import_module(name)
        for name in (
            "adapter_common",
            "fit_recipe",
            "fit_workspace",
            "worker",
            "checkpoint_identity",
        )
    }


def verify_source(root: Path) -> dict:
    """Verify both inherited actor packets and every pinned canonical source member."""
    for folder, name, digest in [
        ("b300-r01-v1", "EXECUTION_MANIFEST.json", R01_MANIFEST),
        ("pr15-host-v2", "QUEUE_EXECUTION_MANIFEST.json", PR15_MANIFEST),
    ]:
        packet = root / "runtime" / folder
        manifest = packet / name
        if sha(manifest) != digest:
            raise ValueError("Inherited execution manifest changed: " + folder)
        for relative, expected in read(manifest)["files"].items():
            path = (packet / relative).resolve(strict=True)
            if not path.is_relative_to((root / "runtime").resolve()) or sha(path) != expected:
                raise ValueError("Inherited runtime member changed: " + relative)
    modules = legacy(root)
    a = modules["adapter_common"]
    source = root / "runtime/source-a266-lf-subset-v1"
    identity = a.verify_source_manifest(source, a._json(a._SOURCE_MANIFEST))
    tree = a.verify_deployed_tree_manifest(
        source, a._json(a._TREE_MANIFEST), manifest_sha256=sha(a._TREE_MANIFEST)
    )
    grader = root / "runtime/b300-r01-v1/grader/grade_endpoint.py"
    if sha(grader) != GRADER_SHA256:
        raise ValueError("Official grader source changed")
    return dict(commit=SOURCE_REVISION, trainer=identity, tree=tree, grader_sha256=GRADER_SHA256)


def verify_config(path: Path, expected: str) -> dict:
    """Bind a service to its exact authority, recipes, targets and implementation."""
    if sha(path) != expected:
        raise ValueError("Expansion configuration differs from service authority")
    config = read(path)
    if config["root"] != str(ROOT) or config["scientific_precision"] != "strict_fp32":
        raise ValueError("Expansion root or precision differs")
    packet = path.parent
    for name, digest in config["code_files"].items():
        if Path(name).name != name or sha(packet / "code" / name) != digest:
            raise ValueError("Expansion code member changed: " + name)
    return config


def training_arguments(root: Path, job: dict, mode: str = "fit") -> list[str]:
    """Append only the declared feature treatment to the original six-epoch R01 recipe."""
    family, feature, control, _ = ARMS[job["arm"]]
    if job["family"] != family:
        raise ValueError("Arm belongs to another experiment")
    modules = legacy(root)
    a = modules["adapter_common"]
    argv = modules["fit_recipe"].build_training_argv(
        a._json(a._RECIPE),
        arm="wide",
        fold=job["fold"],
        seed=job["seed"],
        run_name=job["id"],
        split_manifest=root / "splits" / f"fold{job['fold']}.json",
        mode=mode,
    )
    return argv + ["--research-features", feature, "--research-feature-control", control]
