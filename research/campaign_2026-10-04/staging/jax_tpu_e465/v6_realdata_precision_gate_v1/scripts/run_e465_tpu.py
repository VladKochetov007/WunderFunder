"""Run the staged full-shape E465 TPU profile or fixed one-epoch fit."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

STAGE_ROOT = Path(__file__).resolve().parents[1]


def _configure_environment() -> None:
    """Set backend and host thread limits before importing JAX or NumPy."""
    os.environ["JAX_PLATFORMS"] = "tpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def main() -> None:
    """Parse the immutable worker inputs, then import the JAX worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("profile", "fit"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--v6-bundle", type=Path, required=True)
    parser.add_argument("--fit-authorization", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-report", type=Path)
    parser.add_argument("--profile-batches", choices=(2,), type=int, default=2)
    args = parser.parse_args()
    if args.phase == "fit" and (args.profile_report is None or args.fit_authorization is None):
        parser.error("fit requires a measured profile and its root fit authorization")
    if args.phase == "profile" and (args.profile_report is not None or args.fit_authorization is not None):
        parser.error("profile mode cannot consume a profile report or fit authorization")

    worker_started = time.monotonic()
    _configure_environment()
    sys.path.insert(0, str(STAGE_ROOT / "src"))
    from wnn_jax.e465_gate import (
        E465_MAX_WORKER_SECONDS,
        source_hashes,
        validate_e463_v6_bundle,
        validate_fit_authorization,
        validate_profile_report,
    )

    preflight_identity = validate_e463_v6_bundle(
        args.v6_bundle,
        args.input_manifest,
        args.project_root,
    )
    source_identity = source_hashes(STAGE_ROOT)
    if args.phase == "fit":
        profile_report = json.loads(args.profile_report.read_text(encoding="utf-8"))
        validate_profile_report(
            profile_report,
            input_manifest_sha256=preflight_identity["v6_input_identity"][
                "input_manifest_sha256"
            ],
            source_identity=source_identity,
            preflight_identity=preflight_identity,
        )
        validate_fit_authorization(
            args.fit_authorization,
            profile_path=args.profile_report,
            profile_report=profile_report,
            input_manifest_sha256=preflight_identity["v6_input_identity"]["input_manifest_sha256"],
            source_identity=source_identity,
            preflight_identity=preflight_identity,
            max_worker_seconds=E465_MAX_WORKER_SECONDS,
        )

    from wnn_jax.e465_tpu_worker import MAX_WALL_SECONDS, run_worker

    if MAX_WALL_SECONDS != E465_MAX_WORKER_SECONDS:
        parser.error("runner and worker whole-job limits differ")
    report = run_worker(
        phase=args.phase,
        project_root=args.project_root,
        input_manifest_path=args.input_manifest,
        preflight_bundle_path=args.v6_bundle,
        output_dir=args.output_dir,
        stage_root=STAGE_ROOT,
        worker_started_monotonic=worker_started,
        profile_report_path=args.profile_report,
        fit_authorization_path=args.fit_authorization,
        profile_batches=args.profile_batches,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(args.output_dir),
                "elapsed_seconds": report.get("elapsed_seconds"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
