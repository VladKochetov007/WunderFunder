"""Sequential, resumable train/predict/grade orchestration for R4."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .artifacts import (
    checkpoint_complete,
    prediction_complete,
    prediction_paths,
    score_complete,
    score_path,
    verify_pair,
)
from .contract import (
    FIT_TIMEOUT_SECONDS,
    LOCK,
    LOGS,
    PLAN,
    PREDICTION_TIMEOUT_SECONDS,
    PROGRESS,
    ROOT,
    append_jsonl,
    evaluation_groups,
    fit_command,
    now_utc,
    prediction_command,
    read_json,
    run_identity,
    run_name,
    score_command,
    sha256_file,
    verify_frozen_inputs,
)
from .runtime import (
    cgroup_directory,
    cgroup_values,
    execute_phase,
    gpu_values,
    validate_service_budget,
)


# Experiment plans and provenance are intentionally schema-versioned JSON.
def train_endpoint(
    fold: int,
    seed: int,
    arm: str,
    directory: Path,
    admission: dict[str, Any],
) -> None:
    """Train or epoch-resume one endpoint and verify its fixed checkpoint."""
    name = run_name(fold, seed, arm)
    output = ROOT / "artifacts/runs" / name
    if checkpoint_complete(name):
        return
    resume = output / "training_state.pt"
    command = fit_command(name, fold, seed, arm, resume if resume.is_file() else None)
    execute_phase(
        name,
        "train",
        command,
        LOGS / f"{name}.train.log",
        FIT_TIMEOUT_SECONDS,
        directory,
    )
    if not checkpoint_complete(name):
        raise RuntimeError(f"fixed epoch-2 checkpoint is incomplete for {name}")
    provenance = read_json(output / "provenance.json")
    if provenance.get("seed") != seed or provenance.get("selected_epoch") != 2:
        raise RuntimeError(f"checkpoint seed/epoch identity mismatch for {name}")
    if (
        provenance.get("split_manifest_sha256")
        != admission["split_manifest_sha256"][
            f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
        ]
    ):
        raise RuntimeError(f"checkpoint split identity mismatch for {name}")


def predict_endpoint(
    fold: int,
    seed: int,
    arm: str,
    groups: Sequence[int],
    split_sha256: str,
    directory: Path,
) -> None:
    """Export exact held-out predictions and move them into the indexed campaign."""
    name = run_name(fold, seed, arm)
    target, target_sidecar = prediction_paths(fold, seed, arm)
    checkpoint = ROOT / "artifacts/runs" / name / "model_ep2.pt"
    checkpoint_sha = sha256_file(checkpoint)
    if prediction_complete(target, target_sidecar, groups, split_sha256, name, 2, checkpoint_sha):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    run_name_value = name
    command = prediction_command(name, fold)
    execute_phase(
        run_name_value,
        "predict",
        command,
        LOGS / f"{name}.predict.log",
        PREDICTION_TIMEOUT_SECONDS,
        directory,
    )
    source = ROOT / "cache" / f"pred_{name}_valid.npy"
    source_sidecar = source.with_suffix(".json")
    if not prediction_complete(
        source, source_sidecar, groups, split_sha256, name, 2, checkpoint_sha
    ):
        raise RuntimeError(f"evaluation prediction is incomplete for {name}")
    source.replace(target)
    source_sidecar.replace(target_sidecar)
    if not prediction_complete(
        target, target_sidecar, groups, split_sha256, name, 2, checkpoint_sha
    ):
        raise RuntimeError(f"archived evaluation prediction failed verification for {name}")


def score_endpoint(
    fold: int,
    seed: int,
    arm: str,
    split_sha256: str,
    directory: Path,
) -> dict[str, Any]:
    """Grade one fixed prediction, archive its official score and verify identity."""
    name = run_name(fold, seed, arm)
    prediction, sidecar = prediction_paths(fold, seed, arm)
    checkpoint_sha = sha256_file(ROOT / "artifacts/runs" / name / "model_ep2.pt")
    groups, _ = evaluation_groups(fold)
    if not prediction_complete(prediction, sidecar, groups, split_sha256, name, 2, checkpoint_sha):
        raise ValueError(f"cannot score an unverified prediction for {name}")
    prediction_sha = sha256_file(prediction)
    destination = score_path(fold, seed, arm)
    if score_complete(destination, name, prediction_sha, split_sha256):
        return read_json(destination)
    execute_phase(
        name,
        "score",
        score_command(name, fold, prediction),
        LOGS / f"{name}.score.log",
        PREDICTION_TIMEOUT_SECONDS,
        directory,
    )
    source = ROOT / "artifacts/runs" / name / "score.json"
    payload = read_json(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    if not score_complete(destination, name, prediction_sha, split_sha256):
        raise RuntimeError(f"official score receipt failed verification for {name}")
    append_jsonl(
        PROGRESS,
        {
            "timestamp_utc": now_utc(),
            "event": "endpoint_scored",
            "run": name,
            "score": payload,
            "score_archive": str(destination.relative_to(ROOT)),
        },
    )
    return payload


def lock_gpu() -> Any:
    """Acquire the shared GPU lock for the entire sequence; never queue silently."""
    handle = LOCK.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("shared GPU lock is held by another worker") from error
    return handle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan, admission = verify_frozen_inputs()
    repetitions = int(plan["paired_repetitions"]["pairs"])
    if repetitions != 30:
        raise ValueError("the registered R4 matrix must contain 30 paired repetitions")
    directory = cgroup_directory()
    validate_service_budget(directory)
    lock = lock_gpu()
    try:
        if args.dry_run:
            for pair_index in range(repetitions):
                fold, seed, arms = run_identity(pair_index, plan)
                for arm in arms:
                    name = run_name(fold, seed, arm)
                    print(
                        json.dumps(
                            {
                                "pair_index": pair_index,
                                "fold": fold,
                                "seed": seed,
                                "arm": arm,
                                "train_command": fit_command(name, fold, seed, arm, None),
                                "prediction_command": prediction_command(name, fold),
                            }
                        )
                    )
            return 0
        plan, admission = verify_frozen_inputs(verify_data=True)
        append_jsonl(
            PROGRESS,
            {
                "timestamp_utc": now_utc(),
                "event": "matrix_start_or_resume",
                "pairs": repetitions,
                "plan_sha256": sha256_file(PLAN),
                "cgroup": cgroup_values(directory),
                "gpu": gpu_values(),
            },
        )
        for pair_index in range(repetitions):
            verify_frozen_inputs()
            fold, seed, arms = run_identity(pair_index, plan)
            groups, _ = evaluation_groups(fold)
            split_relative = f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
            split_sha = admission["split_manifest_sha256"][split_relative]
            print(
                f"pair {pair_index + 1}/{repetitions}: fold={fold}, seed={seed}, order={arms}",
                flush=True,
            )
            for arm in arms:
                train_endpoint(fold, seed, arm, directory, admission)
                predict_endpoint(fold, seed, arm, groups, split_sha, directory)
                score_endpoint(fold, seed, arm, split_sha, directory)
            expected_updates = int(
                plan["training"]["expected_optimizer_updates_per_fit_by_fold"][str(fold)]
            )
            match = verify_pair(fold, seed, expected_updates)
            scores = {
                arm: read_json(score_path(fold, seed, arm)) for arm in ("control", "features")
            }
            append_jsonl(
                PROGRESS,
                {
                    "timestamp_utc": now_utc(),
                    "event": "pair_complete",
                    "pair_index": pair_index,
                    "fold": fold,
                    "seed": seed,
                    "arm_order": arms,
                    "control_wp": scores["control"]["wp"],
                    "features_wp": scores["features"]["wp"],
                    "paired_wp_delta": scores["features"]["wp"] - scores["control"]["wp"],
                    **match,
                },
            )
            print(f"pair {pair_index + 1}/{repetitions} complete", flush=True)
        final_resources = {**cgroup_values(directory), "gpu": gpu_values()}
        append_jsonl(
            PROGRESS,
            {
                "timestamp_utc": now_utc(),
                "event": "matrix_complete",
                "pairs": repetitions,
                "resources": final_resources,
            },
        )
        return 0
    except Exception as error:
        append_jsonl(
            PROGRESS,
            {
                "timestamp_utc": now_utc(),
                "event": "matrix_stopped_for_diagnosis",
                "error_type": type(error).__name__,
                "error": str(error),
                "resources": cgroup_values(directory),
            },
        )
        raise
    finally:
        lock.close()
