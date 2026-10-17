"""Run one frozen geometry/discrete B300 fit or measured profile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from common import (
    CONTROL_BY_ARM,
    PROFILE_CONFIG,
    ROOT,
    checkout_commit,
    execution_environment,
    gpu_identity,
    pair_identity,
    prediction_command,
    score_command,
    sha256_file,
    train_command,
    utc_now,
    validate_mounts,
    write_json,
)
from execution import clean_profile_id, parse_profile_events, run_phase, validate_profile_receipt
from matrix import validate_frozen_matrix


def parse_args() -> argparse.Namespace:
    """Parse one fit or one measured profile invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-index", type=int, default=0, help="paired cell index, 0..29")
    parser.add_argument("--arm", choices=tuple(CONTROL_BY_ARM), default="combined")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="return an existing terminal cell receipt without refitting",
    )
    parser.add_argument("--profile-steps", type=int, default=8)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-receipt", type=Path)
    parser.add_argument("--portable-manifest", type=Path)
    return parser.parse_args()


def run_profile(
    args: argparse.Namespace,
    root: Path,
    registry: Path,
    commit: str,
    device: dict[str, Any],
) -> Path:
    """Run one bounded profile and save a machine-readable per-card receipt."""
    if (
        args.profile_steps != PROFILE_CONFIG["profile_measured_updates"]
        or args.profile_warmup != PROFILE_CONFIG["profile_warmup_updates"]
    ):
        raise ValueError("use the frozen two-warmup/eight-measured-update profile")
    fold, seed = pair_identity(args.pair_index)
    identity = f"b300_profile_{clean_profile_id(device['gpu_id'])}"
    log_path = root / "artifacts/geometry_b300/profiles" / f"{identity}.log"
    if log_path.exists():
        raise FileExistsError(
            f"preserve the existing profile log and choose another GPU label: {log_path}"
        )
    command = train_command(identity, fold, seed, "none")
    command.extend(
        ("--profile-steps", str(args.profile_steps), "--profile-warmup", str(args.profile_warmup))
    )
    environment = execution_environment(root, registry)
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        print("B300_PROFILE_DEVICE " + json.dumps(device, sort_keys=True), file=log, flush=True)
        result = subprocess.run(
            command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    ended = time.time()
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    events = parse_profile_events(log_path)
    resources = events["resources"]
    updates = events["updates"]
    strict_fp32 = (
        resources.get("precision") == "fp32"
        and resources.get("matmul_tf32") is False
        and resources.get("cudnn_tf32") is False
        and resources.get("float32_matmul_precision") == "highest"
    )
    profile = {
        "schema": "causal_geometry_b300_card_profile_v1",
        "source_commit": commit,
        "profile_config": PROFILE_CONFIG,
        "device": device,
        "strict_fp32": strict_fp32,
        "finite_profile": bool(updates.get("valid_for_throughput")),
        "started_epoch": started,
        "finished_epoch": ended,
        "log": str(log_path.relative_to(root)),
        "log_sha256": sha256_file(log_path),
        **events,
    }
    if not strict_fp32 or not profile["finite_profile"]:
        raise RuntimeError("profile did not prove strict FP32 finite optimizer updates")
    output = log_path.with_suffix(".json")
    write_json(output, profile)
    return output


def run_cell(args: argparse.Namespace, root: Path, registry: Path, commit: str) -> Path:
    """Fit, predict and score one preregistered arm in one source-qualified fold."""
    if not 0 <= args.pair_index < 30:
        raise ValueError("pair index must be between 0 and 29")
    if args.profile_receipt is None:
        raise ValueError("full matrix cells require --profile-receipt from the eight-GPU profile")
    validate_profile_receipt(args.profile_receipt, commit)
    fold, seed = pair_identity(args.pair_index)
    arm = args.arm
    name = f"geometry_b300_f{fold}_s{seed}_{arm}"
    run_dir = root / "artifacts/runs" / name
    record_path = root / "artifacts/geometry_b300/cells" / f"{name}.json"
    profile_path = args.profile_receipt.resolve()
    expected_identity = (commit, args.pair_index, fold, seed, arm)
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        observed_identity = (
            record.get("source_commit"),
            record.get("pair_index"),
            record.get("fold"),
            record.get("seed"),
            record.get("arm"),
        )
        if (
            record.get("schema") != "causal_geometry_b300_cell_v1"
            or observed_identity != expected_identity
        ):
            raise ValueError(
                f"existing cell receipt belongs to a different source or cell: {record_path}"
            )
        if record.get("finished_utc"):
            if args.skip_completed:
                return record_path
            raise FileExistsError(f"cell is already complete; preserve its receipt: {record_path}")
    else:
        # Receipt fields mirror heterogeneous phase logs and trainer JSON records.
        record: dict[str, Any] = {
            "schema": "causal_geometry_b300_cell_v1",
            "source_commit": commit,
            "profile_receipt": str(profile_path),
            "profile_receipt_sha256": sha256_file(profile_path),
            "pair_index": args.pair_index,
            "fold": fold,
            "seed": seed,
            "arm": arm,
            "control": CONTROL_BY_ARM[arm],
            "split_manifest": f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json",
            "device": gpu_identity(),
            "started_utc": utc_now(),
            "phases": {},
        }
    environment = execution_environment(root, registry)
    if (run_dir / "model_ep2.pt").is_file():
        record["phases"]["train"] = {
            "attempts": record["phases"].get("train", {}).get("attempts", []),
            "status": "existing_fixed_checkpoint",
        }
    else:
        command = train_command(name, fold, seed, CONTROL_BY_ARM[arm])
        state_path = run_dir / "training_state.pt"
        if state_path.is_file():
            command.extend(("--resume-state", str(state_path)))
        elif run_dir.exists():
            raise RuntimeError(f"partial run has no resumable training state: {run_dir}")
        run_phase(
            "train",
            command,
            root / "artifacts/geometry_b300/logs" / f"{name}.train.log",
            environment,
            record,
            record_path,
        )
    run_phase(
        "predict",
        prediction_command(name, fold),
        root / "artifacts/geometry_b300/logs" / f"{name}.predict.log",
        environment,
        record,
        record_path,
    )
    run_phase(
        "score",
        score_command(name, fold),
        root / "artifacts/geometry_b300/logs" / f"{name}.score.log",
        environment,
        record,
        record_path,
    )
    artifacts = {}
    artifact_paths = {
        "fixed_checkpoint": run_dir / "model_ep2.pt",
        "training_state": run_dir / "training_state.pt",
        "provenance": run_dir / "provenance.json",
        "history": run_dir / "history.json",
        "score": run_dir / "score.json",
        "prediction": root / "cache" / f"pred_{name}_valid.npy",
        "prediction_metadata": root / "cache" / f"pred_{name}_valid.json",
    }
    for label, path in artifact_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required run artifact is missing: {path}")
        artifacts[label] = {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path) if label != "prediction" else None,
        }
    prediction_metadata = json.loads(artifact_paths["prediction_metadata"].read_text())
    artifacts["prediction"]["sha256"] = prediction_metadata.get("prediction_sha256")
    if artifacts["prediction"]["sha256"] is None:
        raise ValueError("prediction sidecar does not record the prediction array hash")
    record["artifacts"] = artifacts
    record["finished_utc"] = utc_now()
    record["score_receipt"] = str((run_dir / "score.json").relative_to(root))
    write_json(record_path, record)
    return record_path


def main() -> int:
    """Validate mounts and run one profile or one frozen matrix cell."""
    args = parse_args()
    root = Path(os.environ.get("WNN_PROJECT_ROOT", ROOT)).resolve()
    if root != ROOT:
        raise ValueError("run this adapter from the mounted project checkout")
    validate_frozen_matrix(root)
    commit = checkout_commit(root)
    fold, _ = pair_identity(args.pair_index)
    registry = validate_mounts(root, fold, args.portable_manifest)
    device = gpu_identity()
    output = (
        run_profile(args, root, registry, commit, device)
        if args.profile_only
        else run_cell(args, root, registry, commit)
    )
    print(json.dumps({"receipt": str(output), "source_commit": commit}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
