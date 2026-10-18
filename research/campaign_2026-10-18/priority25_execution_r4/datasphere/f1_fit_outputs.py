"""Artifact collection for complete and interrupted F1/seed-1001 fits."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from f1_fit_recipe import ARMS, FOLD1_SHA256, prediction_args
from f1_fit_staging import read_object
from profile_staging import sha256_file


# Canonical run and prediction manifests carry heterogeneous JSON fields.
def validate_completed_fit(run_dir: Path, arm: dict[str, object]) -> dict[str, Any]:
    """Require the fixed endpoint and ordered fit-only epoch summaries before export."""
    names = ("model_ep5.pt", "model.pt", "training_state.pt", "history.json", "provenance.json")
    for name in names:
        if not (run_dir / name).is_file():
            raise ValueError(f"completed fit is missing {name}")
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != 6:
        raise ValueError("completed fit must retain six fit-only epoch summaries")
    for epoch, record in enumerate(history):
        if not isinstance(record, dict):
            raise ValueError("epoch history entries must be objects")
        summary = record.get("fit_summary")
        objective = summary.get("objective_mean") if isinstance(summary, dict) else None
        if record.get("epoch") != epoch or type(objective) not in (int, float):
            raise ValueError("epoch history lacks ordered fit-only mean objective telemetry")
        if not math.isfinite(float(objective)) or record.get("evaluation") != "withheld":
            raise ValueError("fit history objective must be finite and evaluation withheld")
        updates = summary.get("optimizer_updates", {})
        if not isinstance(updates, dict) or not updates:
            raise ValueError("epoch fit summary lacks optimizer update counts")
        rates = summary.get("learning_rate_by_group", [])
        if not isinstance(rates, list) or not rates:
            raise ValueError("epoch fit summary lacks scheduler learning-rate observations")
    provenance = read_object(run_dir / "provenance.json", "run provenance")
    if provenance.get("split_manifest_sha256") != FOLD1_SHA256:
        raise ValueError("completed fit provenance does not bind to fold 1")
    args = provenance.get("args")
    if not isinstance(args, dict):
        raise ValueError("completed fit provenance lacks canonical training arguments")
    if (
        args.get("name") != arm["name"]
        or args.get("arch") != "gru"
        or args.get("cell") != "gru"
        or args.get("enc") != "linear"
        or args.get("hidden") != arm["hidden"]
        or args.get("layers") != arm["layers"]
        or args.get("precision") != "fp32"
        or args.get("split_manifest_sha256") != FOLD1_SHA256
        or args.get("init")
    ):
        raise ValueError("completed fit provenance is not a cold strict-FP32 run")
    expected_args = {
        "valid_fold": 1,
        "n_valid_folds": 4,
        "full_valid": True,
        "valid_repeat": 1,
        "valid_train_fraction": 1.0,
        "device": "cuda",
        "epochs": 6,
        "checkpoint_epoch": 5,
        "train_source": "train+valid",
        "train_seqs": 10607,
        "eval_source": "vfold",
        "batch": 8,
        "chunk": 2000,
        "seed": 1001,
        "batch_order": "seeded",
        "loss": "wmse",
        "loss_norm": "chunk",
        "dropout": 0.0,
        "rnn_init": "default",
        "optim": "adamw",
        "lr": 0.0001,
        "beta2": 0.999,
        "wd": 0.01,
        "sched": "onecycle",
        "warmup_frac": 0.05,
        "head_lr_mult": 1.0,
        "clip": 1.0,
        "grad_stats": True,
        "pin_memory": False,
        "checkpoint_policy": "fixed",
        "no_eval_during_fit": True,
        "save_training_state": True,
        "resume_state": None,
        "final_model_adaptation": False,
        "final_distill_initializer": False,
        "inherit_init_preprocessing": False,
        "class_balance": False,
        "prop_cache": False,
        "distill_family_weight": 0.0,
        "distill_boot_h": 0,
        "fused_adam": True,
        "sync_free": True,
        "deterministic": True,
    }
    if any(args.get(key) != value for key, value in expected_args.items()):
        raise ValueError("completed fit provenance differs from the fixed F1 endpoint")
    if (
        args.get("distill")
        or args.get("distill_cache_root")
        or args.get("propensity")
        or args.get("prop_input")
    ):
        raise ValueError("completed fit unexpectedly uses teacher or propensity ancestry")
    return {
        name: {"bytes": (run_dir / name).stat().st_size, "sha256": sha256_file(run_dir / name)}
        for name in names
    }


def run_prediction(
    arm: dict[str, object],
    split_path: Path,
    source_root: Path,
    data_root: Path,
    output: Path,
    deadline: float,
    *,
    expected_evaluation_groups: Sequence[int],
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Run the canonical feature-only exporter in a terminable child process."""
    env = os.environ.copy()
    env.update(
        {
            "WNN_PROJECT_ROOT": str(data_root),
            "WNN_PORTABLE_DATA_MANIFEST": str(data_root / "portable-sources.json"),
            "NVIDIA_TF32_OVERRIDE": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "WNN_THREADS": "1",
        }
    )
    source_path = str(source_root / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source_path, env.get("PYTHONPATH", ""))))
    log_path = output / f"{arm['name']}.predict.log"
    command = [
        sys.executable,
        str(source_root / "scripts/predict_run.py"),
        *prediction_args(arm, split_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=source_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("fit pair reached its internal deadline before prediction")
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            _terminate_prediction(process)
            raise TimeoutError("canonical prediction exporter reached the fit deadline") from error
        except BaseException:
            _terminate_prediction(process)
            raise
    if code != 0:
        raise RuntimeError(f"canonical prediction exporter exited with status {code}")
    prediction = data_root / "cache" / f"pred_{arm['name']}_valid.npy"
    sidecar = prediction.with_suffix(".json")
    metadata = read_object(sidecar, "prediction sidecar")
    validate_prediction_sidecar(
        metadata,
        arm,
        expected_evaluation_groups,
        expected_checkpoint_sha256,
        prediction,
    )
    return {
        "prediction": {"bytes": prediction.stat().st_size, "sha256": sha256_file(prediction)},
        "sidecar": {"bytes": sidecar.stat().st_size, "sha256": sha256_file(sidecar)},
        "row_groups": metadata["row_groups"],
        "evaluation_membership_sha256": metadata["evaluation_membership_sha256"],
        "log": log_path.name,
    }


def validate_prediction_sidecar(
    metadata: dict[str, Any],
    arm: dict[str, object],
    expected_evaluation_groups: Sequence[int],
    expected_checkpoint_sha256: str,
    prediction: Path,
) -> None:
    """Bind exporter output to canonical epoch, checkpoint and ordered fold membership."""
    expected_groups = [int(group) for group in expected_evaluation_groups]
    expected_membership_sha256 = hashlib.sha256(
        b"".join(group.to_bytes(8, byteorder="little", signed=True) for group in expected_groups)
    ).hexdigest()
    groups = metadata.get("row_groups")
    if (
        metadata.get("run") != arm["name"]
        or metadata.get("split") != "valid"
        or metadata.get("epoch") != 5
        or metadata.get("checkpoint_sha256") != expected_checkpoint_sha256
        or metadata.get("split_manifest_sha256") != FOLD1_SHA256
        or metadata.get("campaign_split_manifest_sha256") != FOLD1_SHA256
        or metadata.get("campaign_export_scope") != "evaluation"
        or metadata.get("feature_only") is not True
        or metadata.get("inference_precision") != "fp32"
        or not isinstance(groups, list)
        or any(type(group) is not int for group in groups)
        or groups != expected_groups
        or metadata.get("evaluation_membership_sha256") != expected_membership_sha256
        or metadata.get("prediction_sha256") != sha256_file(prediction)
    ):
        raise ValueError("prediction sidecar does not bind the required fold-1 FP32 endpoint")


def _terminate_prediction(process: subprocess.Popen[bytes]) -> None:
    """Terminate only the exporter process group before artifact packaging."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def collect_artifacts(
    data_root: Path | None, output: Path, profile_spec: Path, fit_spec: Path
) -> list[str]:
    """Copy available checkpoints, resume state, history, predictions and identities."""
    errors: list[str] = []
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    for path in (profile_spec, fit_spec):
        try:
            if path.is_file():
                shutil.copy2(path, inputs / path.name)
        except (InterruptedError, TimeoutError):
            raise
        except OSError as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
    if data_root is None or not data_root.is_dir():
        return errors
    for path in (data_root / "splits/fold1.json", data_root / "portable-sources.json"):
        try:
            if path.is_file():
                shutil.copy2(path, inputs / path.name)
        except (InterruptedError, TimeoutError):
            raise
        except OSError as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
    for arm in ARMS:
        name = str(arm["name"])
        source = data_root / "artifacts/runs" / name
        if source.is_dir():
            destination = output / "artifacts/runs" / name
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination, dirs_exist_ok=True)
            except (InterruptedError, TimeoutError):
                raise
            except OSError as error:
                errors.append(f"{name} run directory: {type(error).__name__}: {error}")
        for suffix in (".npy", ".npy.partial", ".json"):
            source = data_root / "cache" / f"pred_{name}_valid{suffix}"
            if source.is_file():
                try:
                    destination = output / "cache" / source.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                except (InterruptedError, TimeoutError):
                    raise
                except OSError as error:
                    errors.append(f"{source.name}: {type(error).__name__}: {error}")
    return errors
