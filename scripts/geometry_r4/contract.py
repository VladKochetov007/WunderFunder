"""Frozen R4 recipe, input identity and child-command construction."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# JSON receipts intentionally carry schema-versioned, heterogeneous payloads.
ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "research/campaign_2026-10-17/geometry_from_scratch_r4"
PLAN = CAMPAIGN / "plan.json"
ADMISSION = CAMPAIGN / "admission_r3.json"
PROGRESS = CAMPAIGN / "progress.jsonl"
RESOURCES = CAMPAIGN / "resources.jsonl"
LOCK = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime/gpu.lock"
PREDICTIONS = CAMPAIGN / "predictions"
SCORES = CAMPAIGN / "scores"
LOGS = CAMPAIGN / "logs"
FIT_TIMEOUT_SECONDS = 6 * 60 * 60
PREDICTION_TIMEOUT_SECONDS = 60 * 60
RESOURCE_SAMPLE_SECONDS = 60
MIN_GPU_FREE_MIB = 4096
SEQUENCE_LENGTH = 20_000
TARGET_COUNT = 2
PREDICTION_CHUNK_GROUPS = 8


def now_utc() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Durably append one JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_manifest(admission: dict[str, Any]) -> dict[str, str]:
    """Return the exact code file hashes frozen by the admission receipt."""
    sources = admission.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("admission receipt is missing source hashes")
    return {str(path): str(digest) for path, digest in sources.items()}


def verify_frozen_inputs(*, verify_data: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject any plan, split or implementation drift after admission."""
    admission = read_json(ADMISSION)
    plan = read_json(PLAN)
    if admission.get("experiment_id") != "E-GEOMETRY-FROM-SCRATCH-20261017-R4":
        raise ValueError("R4 admission receipt names a different experiment")
    if admission.get("status") != "ready_for_launch":
        raise ValueError("R4 admission receipt is not ready for launch")
    if sha256_file(PLAN) != admission.get("plan_sha256"):
        raise ValueError("R4 plan hash differs from the admitted recipe")
    for relative, expected in source_manifest(admission).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen training source changed: {relative}")
    for relative, expected in admission.get("split_manifest_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen split changed: {relative}")
    if verify_data:
        for relative, expected in admission.get("input_sha256", {}).items():
            path = ROOT / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"frozen experiment input changed: {relative}")
    return plan, admission


def evaluation_groups(fold: int) -> tuple[list[int], dict[int, int]]:
    """Read the exact evaluation membership and component map for one fold."""
    split_path = ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
    manifest = read_json(split_path)
    groups = [int(value) for value in manifest["evaluation"]["valid"]]
    component_of = {
        int(row["row_group"]): int(row["component"]) for row in manifest["sequences"]["valid"]
    }
    if not groups or len(groups) != len(set(groups)):
        raise ValueError(f"fold {fold} evaluation groups must be nonempty and unique")
    if any(group not in component_of for group in groups):
        raise ValueError(f"fold {fold} component map is incomplete")
    return groups, component_of


def run_identity(pair_index: int, plan: dict[str, Any]) -> tuple[int, int, list[str]]:
    """Resolve one preregistered seed, fold and paired execution order."""
    repetition = plan["paired_repetitions"]
    seed = 42000 + pair_index
    fold = pair_index % 4
    arms = ["control", "features"]
    if pair_index % 2:
        arms.reverse()
    if repetition["pairs"] != 30 or repetition["fold_for_pair_index"] != (
        "pair_index modulo 4; fold counts are 8, 8, 7, 7"
    ):
        raise ValueError("R4 allocation differs from the registered 30-pair design")
    return fold, seed, arms


def run_name(fold: int, seed: int, arm: str) -> str:
    """Return a stable artifact name for one paired endpoint."""
    return f"geometry_r4_f{fold}_s{seed}_{arm}"


def fit_command(name: str, fold: int, seed: int, arm: str, resume: Path | None) -> list[str]:
    """Build the frozen full-data training command for one endpoint."""
    mode = "all" if arm == "control" else "none"
    command = [
        sys.executable,
        "-u",
        "scripts/train_rnn.py",
        "--name",
        name,
        "--device",
        "cuda",
        "--hidden",
        "512",
        "--layers",
        "1",
        "--batch",
        "8",
        "--chunk",
        "2000",
        "--epochs",
        "3",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "2",
        "--no-eval-during-fit",
        "--save-training-state",
        "--loss",
        "wmse",
        "--optim",
        "adamw",
        "--sched",
        "onecycle",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--clip",
        "1.0",
        "--precision",
        "fp32",
        "--grad-stats",
        "--deterministic",
        "--seed",
        str(seed),
        "--split-manifest",
        f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json",
        "--train-source",
        "train+valid",
        "--train-seqs",
        "10607",
        "--research-features",
        "causal_geometry_discrete",
        "--research-feature-control",
        mode,
    ]
    if resume is not None:
        command.extend(("--resume-state", str(resume)))
    return command


def prediction_command(name: str, fold: int) -> list[str]:
    """Build the post-fit exact evaluation-group prediction command."""
    return [
        sys.executable,
        "-u",
        "scripts/predict_run.py",
        name,
        "valid",
        "--campaign-split-manifest",
        f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json",
        "--campaign-evaluation-only",
        "--device",
        "cuda",
        "--output-name",
        name,
        "--epoch",
        "2",
    ]


def score_command(name: str, fold: int, prediction: Path) -> list[str]:
    """Build the frozen grader command for an archived prediction array."""
    return [
        sys.executable,
        "-u",
        "scripts/score_run.py",
        "--run",
        name,
        "--split",
        f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json",
        "--epoch",
        "2",
        "--prediction",
        str(prediction.resolve()),
    ]


def command_environment() -> dict[str, str]:
    """Pin CPU thread counts and physical-disk temporary paths for children."""
    environment = os.environ.copy()
    scratch = "/home/vlad/development/wunder/research_tmp"
    environment.update(
        {
            "TMPDIR": scratch,
            "TMP": scratch,
            "TEMP": scratch,
            "WNN_TMP_ROOT": scratch,
            "TORCHINDUCTOR_CACHE_DIR": f"{scratch}/torchinductor",
            "WNN_LOADER_WORKERS": "0",
            "WNN_LOADER_RAM_GB": "1.5",
            "WNN_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "1",
            "ARROW_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment
