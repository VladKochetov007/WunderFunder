from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Heterogeneous JSON receipts are checked explicitly at their trust boundaries.
PACKET = Path(__file__).resolve().parent
REPO = PACKET.parents[2]
CELLS = tuple(f"{role}_s{seed}" for seed in (1001, 1002) for role in ("teacher", "labels", "kd"))


def require(condition: bool, reason: str) -> None:
    """Reject invalid operator input without assertions that Python can disable."""
    if not condition:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    """Read bounded JSON metadata, never model weights or prediction arrays."""
    require(path.stat().st_size <= 32_000_000, f"Oversized JSON: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def digest(value: object) -> str:
    """Match the trainer's stable canonical JSON digest."""
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def file_hash(path: Path) -> str:
    """Hash large artifacts in bounded blocks without loading tensors."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_new(path: Path, value: dict[str, Any]) -> None:
    """Create immutable evidence and refuse to replace any earlier result."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def normalized_config(cell: str) -> dict[str, Any]:
    """Read literal canonical defaults without importing Torch or model modules."""
    require(cell in CELLS, "Unknown experimental cell")
    tree = ast.parse((REPO / "src/wnn/training/sequence_config.py").read_text())
    defaults = {
        node.target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in ("DEFAULTS", "MODEL_DEFAULTS")
    }
    raw = read(PACKET / "configs" / f"{cell}.json")
    config = defaults["DEFAULTS"] | raw
    if config["optimizer"] == "muon" and "adamw_beta2" not in raw:
        config["adamw_beta2"] = 0.95
    config["model"] = defaults["MODEL_DEFAULTS"] | raw["model"]
    config["model"]["precision_mode"] = config["precision_mode"]
    return config


def source_files() -> dict[str, str]:
    """Match the canonical trainer's whole-source and entrypoint binding."""
    paths = sorted((REPO / "src").rglob("*.py")) + [REPO / "scripts/train_sequence.py"]
    return {path.relative_to(REPO).as_posix(): file_hash(path) for path in paths}


def packet_files() -> dict[str, str]:
    """Bind executable packet configuration and dispatch code to preparation."""
    paths = sorted(path for path in PACKET.rglob("*") if path.suffix in (".py", ".json", ".sh"))
    return {path.relative_to(REPO).as_posix(): file_hash(path) for path in paths}


def guard(root: Path, expected_commit: str) -> None:
    """Require a clean pinned Linux receiver and an external output directory."""
    require(sys.platform == "linux", "Receiver commands require Linux; do not run on the Mac")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", expected_commit)), "Pin a full 40-character commit")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    require(head == expected_commit, "Receiver HEAD differs from --expected-commit")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    require(not status.strip(), "Receiver checkout must be clean, including untracked files")
    require(not root.is_relative_to(REPO), "Run root must be outside the checkout")
    require(root != Path("/"), "Use a dedicated fresh run root")
    require(os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") == "1", "Set cuBLAS emulation=1")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Set deterministic workspace")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    require(
        len(visible) == 2 and all(visible) and len(set(visible)) == 2, "Select exactly two GPUs"
    )


def binding(root: Path, data_root: Path, expected_commit: str) -> dict[str, Any]:
    """Recheck preparation, packet and split identity before every receiver action."""
    guard(root, expected_commit)
    value = read(root / "packet.json")
    require(value.get("schema") == "wnn.source-kd-packet.v1", "Wrong packet schema")
    require(value.get("git_commit") == expected_commit, "Packet commit changed")
    require(value.get("data_root") == str(data_root), "Packet data root changed")
    require(value.get("packet_files") == packet_files(), "Packet code/config changed")
    require(value.get("source_files") == source_files(), "Trainer source changed")
    require(value.get("split_sha256") == file_hash(root / "split.json"), "Split changed")
    return value


def members(root: Path) -> tuple[list[list[object]], list[list[object]]]:
    """Return the source-separated manifest's exact fitting and evaluation members."""
    split = read(root / "split.json")
    require(split.get("schema") == "wnn.sequence-source-holdout.v1", "Source holdout required")
    fit = split["fit"]
    evaluation = split["evaluation"]
    require(fit["valid"] == [] and fit["train"], "Fit must contain train only")
    require(evaluation.get("train", []) == [] and evaluation["valid"], "Evaluate valid only")
    return [["train", group] for group in fit["train"]], [
        ["valid", group] for group in evaluation["valid"]
    ]
