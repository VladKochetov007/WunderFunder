from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from wnn.training.sequence_provenance import canonical_sha256, file_sha256

SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
OLD_MANIFEST_SHA = "7d3324230effb050ff083b939e26ad638e5df5c5b72ea70b0f8aed169997db4a"
DIAGNOSTIC_FILES = {
    "src/wnn/training/sequence_monitor_gates.py",
    "src/wnn/training/sequence_monitor_diagnostics.py",
    "src/wnn/training/sequence_config_validation.py",
}


def pinned_json(item: dict[str, str]) -> dict[str, object]:
    """Read one explicitly hashed regular JSON artifact, never its self-claimed hash."""
    path = Path(item["path"])
    if path.is_symlink() or not path.is_file() or file_sha256(path) != item["sha256"]:
        raise ValueError(f"pinned artifact changed: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("pinned JSON artifact must contain an object")
    return value


def write_json_new(path: Path, value: object) -> str:
    """Create and flush one receipt without overwriting any existing evidence."""
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    return file_sha256(path)


def same_payload(left: object, right: object) -> bool:
    """Compare all tensor bytes, RNG arrays and heterogeneous checkpoint state."""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.stride() == right.stride()
            and torch.equal(
                left.contiguous().reshape(-1).view(torch.uint8),
                right.contiguous().reshape(-1).view(torch.uint8),
            )
        )
    if isinstance(left, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.strides == right.strides
            and left.tobytes() == right.tobytes()
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_payload(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            same_payload(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def verify_capsules(request: dict[str, object]) -> tuple[dict[str, str], dict[str, str]]:
    """Verify native bytes and the exact three-file diagnostic/admission transition."""
    inventories = []
    for label in ("old", "new"):
        spec = request[f"{label}_source_manifest"]
        if label == "old" and spec["sha256"] != OLD_MANIFEST_SHA:
            raise ValueError("parent inventory is not the frozen620-file source-r1")
        files = pinned_json(spec)["files"]
        root = Path(request[f"{label}_source_root"]).resolve(strict=True)
        if not isinstance(files, dict) or len(files) != 620:
            raise ValueError("source manifest must contain exactly620 files")
        for name, sha in files.items():
            relative = PurePosixPath(name)
            candidate = root.joinpath(*relative.parts)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or candidate.is_symlink()
                or not candidate.resolve().is_relative_to(root)
                or file_sha256(candidate) != sha
            ):
                raise ValueError(f"source capsule file changed: {name}")
        expected = {n for n in files if n.endswith(".py") and n.startswith(("src/", "scripts/"))}
        actual = {
            p.relative_to(root).as_posix()
            for base in (root / "src", root / "scripts")
            for p in base.rglob("*.py")
        }
        if expected != actual:
            raise ValueError("source capsule contains extra or missing importable files")
        inventories.append(files)
    old, new = inventories
    if (
        old.keys() != new.keys()
        or {name for name in old if old[name] != new[name]} != DIAGNOSTIC_FILES
    ):
        raise ValueError("source diff is not exactly the three approved files")
    patch = pinned_json(request["source_patch_receipt"])
    if (
        patch.get("parent_inventory_sha256") != OLD_MANIFEST_SHA
        or patch.get("new_inventory_sha256") != request["new_source_manifest"]["sha256"]
        or patch.get("changed_files")
        != {
            name: {"parent_sha256": old[name], "new_sha256": new[name]} for name in DIAGNOSTIC_FILES
        }
    ):
        raise ValueError("source patch receipt differs from verified native bytes")
    return old, new


def trainer_files(files: dict[str, str]) -> dict[str, str]:
    """Use exactly the canonical trainer's source-identity membership."""
    return {
        name: sha
        for name, sha in files.items()
        if name.startswith("src/") and name.endswith(".py") or name == "scripts/train_sequence.py"
    }


def validate_transition(
    old_config: dict[str, object],
    new_config: dict[str, object],
    old: dict[str, object],
    new: dict[str, object],
) -> None:
    """Permit only the explicit source diagnostics and retention-envelope amendment."""
    if old_config.get("checkpoint_retention") != "all_epochs" or new_config != (
        old_config | {"checkpoint_retention": "final_only"}
    ):
        raise ValueError("configuration changed beyond all_epochs→final_only")
    if (
        old_config.get("seed") not in (101001, 101002)
        or old_config.get("epochs") != 12
        or old_config.get("batch_size") != 16
        or old_config.get("model", {}).get("architecture") != "gru"
        or "early_stopping" in old_config.get("monitoring", {})
    ):
        raise ValueError("transition applies only to the two fixed12E B16 GRU cells")
    allowed = {"source_files", "source_sha256", "config_sha256"}
    if {k: v for k, v in old.items() if k not in allowed} != {
        k: v for k, v in new.items() if k not in allowed
    }:
        raise ValueError("data/runtime/training identity changed outside whitelist")
    for config, identity in ((old_config, old), (new_config, new)):
        if (
            identity["config_sha256"] != canonical_sha256(config)
            or identity["source_sha256"] != canonical_sha256(identity["source_files"])
            or identity["data_sha256"] != canonical_sha256(identity["data"])
            or identity.get("world_size") != 1
        ):
            raise ValueError("transition identity digest or world size differs")
    before, after = old["source_files"], new["source_files"]
    if (
        before.keys() != after.keys()
        or {n for n in before if before[n] != after[n]} != DIAGNOSTIC_FILES
    ):
        raise ValueError("profile source diff is not the exact diagnostic whitelist")


def copy_prefix(source: Path, target: Path, offset: int) -> dict[str, object]:
    """Copy only an exact checkpointed native journal prefix into a fresh directory."""
    if type(offset) is not int or not 0 <= offset <= 1 << 30:
        raise ValueError("checkpointed journal offset is outside its bound")
    if source.is_symlink() or not source.is_file() or source.stat().st_size < offset:
        raise ValueError("source journal is missing its checkpointed byte prefix")
    digest = hashlib.sha256()
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        remaining = offset
        last = b""
        while remaining:
            block = incoming.read(min(remaining, 1 << 20))
            if not block:
                raise ValueError("journal was truncated while copying its prefix")
            outgoing.write(block)
            digest.update(block)
            remaining -= len(block)
            last = block[-1:]
        if offset and last != b"\n":
            raise ValueError("safe journal offset is not a complete native JSONL boundary")
        outgoing.flush()
        os.fsync(outgoing.fileno())
    return {"path": str(source), "bytes": offset, "sha256": digest.hexdigest()}


def validate_evaluations(path: Path, epochs: int) -> None:
    """Require the preserved full-valid prefix to cover each safe epoch exactly."""
    events = []
    with path.open("rb") as stream:
        for line in stream:
            if len(line) > 32768 or not line.endswith(b"\n"):
                raise ValueError("full-valid prefix contains an oversized or partial event")
            event = json.loads(line)
            metrics = event["metrics"]
            if (
                event.get("kind") != "evaluation"
                or event.get("epoch") != len(events) + 1
                or metrics.get("valid/sequence_count") != 1873
                or metrics.get("valid/selected_rows") != 3528273
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values()
                )
            ):
                raise ValueError("full-valid prefix differs from exact1873/3528273 coverage")
            events.append(event)
    if len(events) != epochs:
        raise ValueError("full-valid prefix does not reach the checkpoint's safe whole epoch")
