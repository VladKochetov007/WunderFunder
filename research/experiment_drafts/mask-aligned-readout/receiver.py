"""Fail-closed receiver commands for the four-fit mask-aligned readout study."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from receiver_assets import (
    PACKET,
    REPO_ROOT,
    Cell,
    ReceiverAssets,
    load_json,
    output_root,
    prediction_name,
    recipe_cells,
    run_preflight,
)

AUDIT_PATH = PACKET / "audit.py"
RECIPE_PATH = PACKET / "recipe.json"


def fit_command(cell: Cell, arm: str, split_path: Path) -> list[str]:
    """Build the exact canonical training invocation for one paired head fit."""
    mask_flag = ["--mask-weight"] if arm == "is_scored_weighted" else []
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_rnn.py"),
        "--name",
        cell.run_name(arm),
        "--device",
        "cuda",
        "--init",
        f"{cell.parent_run}@ep3",
        "--inherit-init-preprocessing",
        "--split-manifest",
        str(split_path),
        "--train-source",
        "valid",
        "--valid-train-fraction",
        "1.0",
        "--freeze",
        "inp,gru",
        "--loss",
        "wmse",
        "--aux-next",
        "0.3",
        "--aux-struct",
        "0.3",
        "--since-weight",
        "32",
        "--since-floor",
        "0.25",
        "--episode-weight",
        "0",
        "--class-weight=",
        "--propensity=",
        "--valid-true-mask=-1",
        "--lr",
        "0.0005",
        "--wd",
        "0.01",
        "--optim",
        "adamw",
        "--sched",
        "onecycle",
        "--epochs",
        "1",
        "--batch",
        "32",
        "--chunk",
        "2000",
        "--precision",
        "fp32",
        "--capture-execution-runtime",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "0",
        "--no-eval-during-fit",
        "--seed",
        str(cell.seed),
        *mask_flag,
    ]


def audit_command(
    assets: Path,
    stage: str,
    cell: Cell,
    arm: str = "",
    receipt: Path | None = None,
    profile_receipt: Path | None = None,
) -> list[str]:
    """Build an audit invocation with all receiver paths explicit."""
    command = [sys.executable, str(AUDIT_PATH), stage, "--assets", str(assets), "--fold", cell.fold]
    if arm:
        command += ["--arm", arm]
    if receipt:
        command += ["--receipt", str(receipt)]
    if profile_receipt:
        command += ["--profile-receipt", str(profile_receipt)]
    return command


def ensure_unwritten(paths: list[Path]) -> None:
    """Reject existing outputs to prevent accidental overwrite or implicit resume."""
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "outputs already exist; inspect before continuing: " + ", ".join(existing)
        )


def command_environment(root: Path) -> dict[str, str]:
    """Set the data root and make this checkout's canonical Python source importable."""
    environment = os.environ.copy()
    environment["WNN_PROJECT_ROOT"] = str(root)
    source = str(REPO_ROOT / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(value for value in (source, existing) if value)
    return environment


def _fit_outputs(assets: ReceiverAssets, cells: tuple[Cell, ...]) -> tuple[list[Path], list[Path]]:
    """Enumerate every child run directory and prefit receipt before any work starts."""
    child_dirs = [
        assets.root / "artifacts" / "runs" / cell.run_name(arm)
        for cell in cells
        for arm in ("all_valid", "is_scored_weighted")
    ]
    receipts = [output_root(assets.root) / "audit" / f"prefit_{cell.fold}.json" for cell in cells]
    return child_dirs, receipts


def run_fit(assets_path: Path) -> None:
    """Run exactly four cleanly paired, fixed-epoch canonical child fits in order."""
    assets = run_preflight(assets_path)
    cells = recipe_cells(load_json(RECIPE_PATH))
    from b300_profile import profile_receipt_path, require_profile_receipts

    profiles = require_profile_receipts(assets, cells)
    run_paths, receipt_paths = _fit_outputs(assets, cells)
    after_receipts = [
        output_root(assets.root) / "audit" / f"after_{cell.fold}_{arm}.json"
        for cell in cells
        for arm in ("all_valid", "is_scored_weighted")
    ]
    ensure_unwritten(run_paths + receipt_paths + after_receipts)
    environment = command_environment(assets.root)
    for index, cell in enumerate(cells):
        subprocess.run(
            audit_command(
                assets.manifest_path,
                "prefit",
                cell,
                receipt=receipt_paths[index],
                profile_receipt=profile_receipt_path(assets.root, cell),
            ),
            check=True,
            env=environment,
        )
    for cell in cells:
        for arm in ("all_valid", "is_scored_weighted"):
            from audit import require_b300_runtime, require_same_device

            current_hardware = require_b300_runtime()
            expected_runtime = profiles[cell.fold]["execution_runtime"]["process_end"]
            require_same_device(expected_runtime, current_hardware)
            subprocess.run(
                fit_command(cell, arm, assets.split_paths[cell.fold]),
                check=True,
                env=environment,
            )
            receipt = output_root(assets.root) / "audit" / f"after_{cell.fold}_{arm}.json"
            subprocess.run(
                audit_command(assets.manifest_path, "after", cell, arm, receipt),
                check=True,
                env=environment,
            )


def prediction_command(cell: Cell, role: str, run: str, split_path: Path) -> list[str]:
    """Build a canonical feature-only held-out prediction command in FP32."""
    epoch = "3" if role == "parent" else "0"
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "predict_run.py"),
        run,
        "valid",
        "--epoch",
        epoch,
        "--campaign-split-manifest",
        str(split_path),
        "--campaign-evaluation-only",
        "--inference-precision",
        "fp32",
        "--output-name",
        prediction_name(cell, role),
        "--device",
        "cuda",
    ]


def prediction_outputs(assets: ReceiverAssets, cells: tuple[Cell, ...]) -> list[Path]:
    """List all prediction and legacy sidecar paths the canonical exporter can replace."""
    paths: list[Path] = []
    for cell in cells:
        for role in ("parent", "control", "treatment"):
            name = prediction_name(cell, role)
            paths.extend(
                (
                    assets.root / "cache" / f"pred_{name}_valid.npy",
                    assets.root / "cache" / f"pred_{name}_valid.json",
                    assets.root / "cache" / f"pred_{name}_valid.npy.partial",
                    assets.root / "cache" / f"seqsums_{name}.npz",
                )
            )
    return paths


def run_predict(assets_path: Path) -> None:
    """Export both parent and four child checkpoints on complete held-out folds."""
    assets = run_preflight(assets_path)
    cells = recipe_cells(load_json(RECIPE_PATH))
    audit_dir = output_root(assets.root) / "audit"
    from score_audit import check_audit_receipts

    for cell in cells:
        check_audit_receipts(assets, cell, ("all_valid", "is_scored_weighted"))
        names = (
            f"prefit_{cell.fold}.json",
            f"after_{cell.fold}_all_valid.json",
            f"after_{cell.fold}_is_scored_weighted.json",
        )
        for name in names:
            if not (audit_dir / name).is_file():
                raise FileNotFoundError(f"required model audit is missing: {audit_dir / name}")
    ensure_unwritten(prediction_outputs(assets, cells))
    runs = [
        (cell, role, run)
        for cell in cells
        for role, run in (
            ("parent", cell.parent_run),
            ("control", cell.run_name("all_valid")),
            ("treatment", cell.run_name("is_scored_weighted")),
        )
    ]
    environment = command_environment(assets.root)
    for cell, role, run in runs:
        subprocess.run(
            prediction_command(cell, role, run, assets.split_paths[cell.fold]),
            check=True,
            env=environment,
        )


def build_parser() -> argparse.ArgumentParser:
    """Create a small receiver CLI without importing ML packages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "profile", "fit", "predict"))
    parser.add_argument("--assets", type=Path, default=os.environ.get("WNN_MASK_READOUT_ASSETS"))
    return parser


def main() -> int:
    """Validate assets, fit the panel, or export the fixed checkpoints."""
    args = build_parser().parse_args()
    if args.assets is None:
        raise SystemExit("pass --assets or set WNN_MASK_READOUT_ASSETS")
    if args.command == "preflight":
        assets = run_preflight(args.assets)
        print(json.dumps({"status": "preflight_passed", "project_root": str(assets.root)}))
    elif args.command == "profile":
        assets = run_preflight(args.assets)
        from b300_profile import run_profiles

        run_profiles(assets)
    elif args.command == "fit":
        run_fit(args.assets)
    else:
        run_predict(args.assets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
