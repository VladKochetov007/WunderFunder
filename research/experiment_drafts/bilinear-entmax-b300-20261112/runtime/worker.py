"""Execute one Bilinear Entmax stage after admitting the exact PR75 runtime."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from bilinear_entmax.bindings import (
    PENDING_FULL_SHAPE_ACCEPTANCE,
    SPLIT_SHA256,
    VARIANTS,
    admit_runtime,
    digest,
    load_cell,
    source_identity,
    write_new,
)

TIMEOUT_LIMITS = {
    "prepare": 14_400,
    "check": 3_600,
    "profile": 3_600,
    "fit": 28_800,
    "export": 7_200,
    "score": 7_200,
}


def _parser() -> argparse.ArgumentParser:
    """Parse the bounded stage contract shared with ``launch.py``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check", "profile", "fit", "export", "score"))
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--preprocessing", type=Path)
    parser.add_argument("--checks-receipt", type=Path)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--profile-receipt", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    return parser


def _config_and_data(args: argparse.Namespace, native_path: Path) -> tuple[dict[str, Any], Any]:
    """Construct canonical PR75 data and reject any split outside the pinned roster."""
    from wnn.training.sequence_fit_admission import validate_fit_reference_inputs

    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_data import build_data

    config = load_config(native_path)
    validate_fit_reference_inputs(config, None, None, None)
    data = build_data(config, args.data_root, args.split_manifest, None, None)
    if data.split.sha256 != SPLIT_SHA256:
        raise ValueError("loaded sequence split differs from the frozen official split")
    return config, data


def _check(args: argparse.Namespace, runtime_root: Path) -> None:
    """Run the shared all-variant receiver probes on one visible CUDA device."""
    import torch

    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise ValueError("receiver checks require exactly one visible, available CUDA device")
    if torch.distributed.is_initialized():
        raise ValueError("receiver checks run in one ordinary process, not torchrun")
    if (
        os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") != "1"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("receiver checks require the pinned BF16x9 CUDA environment")

    cell, native_path = load_cell(args.cell)
    config, data = _config_and_data(args, native_path)
    from bilinear_entmax.preprocessing import load_preprocessing

    preprocessing_path = args.preprocessing.resolve(strict=True)
    artifact = load_preprocessing(preprocessing_path, data)
    artifact_sha256 = digest(preprocessing_path)
    config["bilinear_entmax"] = {
        "schema": "bilinear-entmax.model.v1",
        "cell": cell,
        "preprocessing": artifact,
        "preprocessing_sha256": artifact_sha256,
    }
    from bilinear_entmax.receiver_checks import run_receiver_checks

    result = run_receiver_checks(config, artifact, artifact_sha256, torch.device("cuda:0"))
    if not isinstance(result, dict):
        raise TypeError("receiver check runner must return a JSON object")
    required = {
        "schema": "bilinear-entmax.receiver-checks.v1",
        "status": "passed",
        "mechanism_contract": "bilinear-entmax.mechanisms.v1",
        "checked_variants": sorted(VARIANTS),
        "preprocessing_sha256": artifact_sha256,
        "precision_mode": config["precision_mode"],
        "full_shape_acceptance": PENDING_FULL_SHAPE_ACCEPTANCE,
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError("receiver check runner did not pass the shared mechanism contract")
    receipt = dict(result)
    receipt["initiating_cell"] = cell
    receipt["source_identity"] = source_identity(runtime_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output, receipt)


def main() -> None:
    """Dispatch one prepared stage without changing canonical fit or scoring code."""
    args = _parser().parse_args()
    if not args.owner.strip() or not 1 <= args.timeout_seconds <= TIMEOUT_LIMITS[args.mode]:
        raise ValueError("stage requires a named owner and timeout within its frozen bound")
    cell, native_path = load_cell(args.cell)
    if cell["variant"] not in VARIANTS:
        raise ValueError("cell variant is outside the frozen Bilinear Entmax panel")
    runtime_root = admit_runtime(args.runtime_root)

    if args.mode == "prepare":
        _, data = _config_and_data(args, native_path)
        from bilinear_entmax.preprocessing import prepare

        prepare(data, args.output)
    elif args.mode == "check":
        if args.preprocessing is None:
            raise ValueError("check requires a train-only preprocessing artifact")
        _check(args, runtime_root)
    elif args.mode in ("profile", "fit"):
        if args.preprocessing is None or args.checks_receipt is None:
            raise ValueError("profile and fit require preprocessing and all-variant checks")
        if args.mode == "profile" and args.profile_updates != 50:
            raise ValueError("engineering profile must use exactly 50 updates")
        if args.mode == "fit" and args.profile_receipt is None:
            raise ValueError("full fit requires an approved profile receipt")
        from bilinear_entmax.fitting import run

        run(args, runtime_root)
    elif args.mode == "export":
        if args.checkpoint is None:
            raise ValueError("export requires a fixed endpoint checkpoint")
        if os.environ.get("WORLD_SIZE") != "2":
            raise ValueError("endpoint export requires the two-rank torchrun launcher")
        from bilinear_entmax.exporting import export_endpoint

        export_endpoint(
            args.checkpoint,
            args.data_root,
            args.split_manifest,
            args.output,
            runtime_root,
        )
    elif args.mode == "score":
        if args.predictions is None:
            raise ValueError("score requires a complete endpoint prediction export")
        from wnn.training.sequence_score import score_export

        args.output.parent.mkdir(parents=True, exist_ok=True)
        score_export(args.data_root, args.split_manifest, args.predictions, args.output)
    else:
        raise AssertionError(f"unhandled stage: {args.mode}")


if __name__ == "__main__":
    main()
