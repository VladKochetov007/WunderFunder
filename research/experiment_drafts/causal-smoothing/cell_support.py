from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from cell_config import ROOT, read_json, sha256, training_dir, write_json


def process_environment(root: Path, require_gpu: bool) -> tuple[dict[str, str], dict[str, Any]]:
    """Capture runtime dependencies and refuse a missing CUDA device for model work."""
    env = os.environ.copy()
    env["WNN_PROJECT_ROOT"] = str(root)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT / "src"), env.get("PYTHONPATH", "")))
    )
    packages = sorted(
        f"{name}=={distribution.version}"
        for distribution in distributions()
        if (name := distribution.metadata.get("Name"))
    )
    runtime: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "installed_distributions": packages,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "WNN_THREADS",
                "WNN_LOADER_WORKERS",
                "WNN_LOADER_RAM_GB",
                "WNN_GPU_MEMORY_BYTES",
                "WNN_GPU_MEMORY_RESERVE_BYTES",
            )
        },
    }
    if require_gpu:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode or not result.stdout.strip():
            raise RuntimeError("CUDA GPU is required; nvidia-smi found no visible device")
        runtime["nvidia_smi"] = result.stdout.strip().splitlines()
    return env, runtime


def begin_stage(path: Path, receipt: dict[str, Any]) -> None:
    """Create a new stage receipt before execution; prior attempts are never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt.update(status="running", started_at=time.time())
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finish_stage(path: Path, receipt: dict[str, Any], error: str | None = None) -> None:
    """Persist a terminal complete/error stage state."""
    receipt.update(status="error" if error else "complete", finished_at=time.time())
    if error:
        receipt["error"] = error
    write_json(path, receipt)


def execute_logged(command: list[str], env: dict[str, str], log_path: Path) -> tuple[int, str]:
    """Capture full subprocess output in a stage-specific log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return result.returncode, str(log_path)


def verify_fit(
    root: Path,
    name: str,
    training: dict[str, Any],
    recipe_sha: str,
    arm: str,
    fold: int,
    seed: int,
    split: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Require complete observer events, canonical history, and the fixed checkpoint."""
    out = training_dir(root, name)
    metadata = read_json(out / "run_metadata.json")
    if metadata.get("source", {}).get("packet", {}).get("recipe.json") != recipe_sha:
        raise ValueError("trainer metadata is not bound to this recipe revision")
    for key in (
        "epochs",
        "batch",
        "chunk",
        "update_rows",
        "detach_rows",
        "hidden",
        "layers",
        "dropout",
        "precision",
        "optim",
        "lr",
        "wd",
        "beta2",
        "clip",
        "sched",
        "warmup_frac",
        "wsd_decay",
        "wsd_final",
        "wsd_shape",
        "loss",
        "checkpoint_policy",
        "checkpoint_epoch",
        "no_eval_during_fit",
        "save_training_state",
    ):
        if metadata.get("config", {}).get(key) != training[key]:
            raise ValueError(f"trainer metadata differs from recipe: {key}")
    config = metadata.get("config", {})
    control = "raw_zero" if arm == "raw-plus-control" else "none"
    fixed_config = {
        "name": name,
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "seed": seed,
        "valid_fold": fold,
        "train_source": "train+valid",
        "train_seqs": split["train_groups"],
        "full_valid": True,
        "no_eval_during_fit": True,
        "device": "cuda",
        "research_features": "causal_ewma",
        "research_feature_control": control,
    }
    if any(config.get(key) != value for key, value in fixed_config.items()):
        raise ValueError(
            "trainer architecture, arm, or fitting membership differs from the cell contract"
        )
    provenance = metadata.get("provenance", {}).get("trainer", {})
    runtime = metadata.get("provenance", {})
    if (
        runtime.get("precision") != "fp32"
        or runtime.get("tf32") is not False
        or not runtime.get("device")
    ):
        raise ValueError("trainer metadata does not confirm strict FP32 on a named CUDA device")
    trainer_source = metadata.get("source", {}).get("training", {})
    packet_source = metadata.get("source", {}).get("packet", {})
    for relative, digest in trainer_source.get("files", {}).items():
        if source["files"].get(relative) != digest:
            raise ValueError(f"trainer source hash differs from the cell binding: {relative}")
    for filename in ("train.py", "training_report.py", "recipe.json"):
        relative = f"research/experiment_drafts/causal-smoothing/{filename}"
        if packet_source.get(filename) != source["files"].get(relative):
            raise ValueError(f"packet source hash differs from the cell binding: {filename}")
    if provenance.get("split_manifest_sha256") != split["sha256"]:
        raise ValueError("trainer provenance is not bound to the selected split manifest")
    manifest = read_json(Path(split["path"]))
    fitted_groups = provenance.get("fit_row_groups", {})
    if set(fitted_groups.get("train", [])) != set(manifest["fit"]["train"]) or set(
        fitted_groups.get("valid", [])
    ) != set(manifest["fit"]["valid"]):
        raise ValueError("trainer provenance does not contain the full frozen fitting membership")
    history_path = out / "history.json"
    if history_path.stat().st_size > 5_000_000:
        raise ValueError("fit history exceeds 5 MB")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (out / "update_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    metrics = [
        json.loads(line)
        for line in (out / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    diagnostics = (
        [row.get("update_diagnostics", {}) for row in history] if isinstance(history, list) else []
    )
    total = sum(row.get("epoch", {}).get("optimizer_attempts", 0) for row in diagnostics)
    checkpoint = out / f"model_ep{training['checkpoint_epoch']}.pt"
    if (
        not isinstance(history, list)
        or len(history) != training["epochs"]
        or not checkpoint.is_file()
        or not (out / "model.pt").is_file()
        or not events
        or len(events) != total
        or len(metrics) != total
    ):
        raise ValueError("fit lacks its fixed checkpoint or complete update history")
    for step, event in enumerate(events, 1):
        if (
            event.get("status") != "optimizer_completed"
            or event.get("step") != step
            or event.get("optimizer_attempts") != step
            or event.get("optimizer_completed") != step
            or event.get("nonfinite_updates") != 0
        ):
            raise ValueError(f"update event {step} is incomplete or nonfinite")
    for step, metric in enumerate(metrics, 1):
        if metric.get("step") != step:
            raise ValueError(f"metric event {step} is missing or out of order")
    cumulative = diagnostics[-1].get("cumulative", {})
    if (
        cumulative.get("optimizer_attempts") != total
        or cumulative.get("optimizer_completed_updates") != total
        or cumulative.get("amp_skipped_updates") != 0
    ):
        raise ValueError("observer count differs from canonical epoch history")
    checkpoint_provenance = read_json(out / "provenance.json")
    expected_checkpoint_sha = checkpoint_provenance.get("checkpoint_sha256", {}).get(
        checkpoint.name
    )
    if checkpoint_provenance.get("selected_epoch") != training[
        "checkpoint_epoch"
    ] or expected_checkpoint_sha != sha256(checkpoint):
        raise ValueError("saved checkpoint is not the predeclared fixed epoch")
    return {
        "run_name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "updates": total,
        "epochs": len(history),
        "events": len(events),
        "metrics": len(metrics),
    }


def stage_digest(receipt: dict[str, Any]) -> str:
    """Hash stable source, recipe, data, split, cell, and run bindings."""
    keys = (
        "arm",
        "fold",
        "seed",
        "attempt",
        "recipe_sha256",
        "source_identity",
        "dataset_identity_sha256",
        "split",
        "run_name",
    )
    outputs = {
        "checkpoint_sha256": receipt.get("fit_evidence", {}).get("checkpoint_sha256"),
        "prediction_sha256": receipt.get("prediction", {}).get("sha256"),
        "score_prediction_sha256": receipt.get("score", {}).get("prediction_sha256"),
    }
    raw = json.dumps(
        {"identity": {key: receipt.get(key) for key in keys}, "outputs": outputs},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def shell(command: list[str]) -> str:
    """Render an exact CLI for planning receipts."""
    return shlex.join(command)
