from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wnn.multistage_receiver import build_r2_admission, dynamic_reserved_gpu_keys
from wnn.multistage_search import atomic_write_json, file_sha256, read_json
from wnn.multistage_search_adopt_fit import adopt_completed_fit
from wnn.multistage_search_controller import status_summary
from wnn.multistage_search_prepare import prepare_run


def main() -> None:
    """Prepare the receipt-bound R2 study and adopt completed fits without relaunching them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bootstrap", "adopt"))
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--operator-root", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--eval-source", type=Path, required=True)
    parser.add_argument("--recipe-review", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--skip-adoption", action="store_true")
    args = parser.parse_args()
    root = args.runtime_root.resolve(strict=True)
    operator = args.operator_root.resolve(strict=True)
    private_dependencies = root / "python-deps"
    if private_dependencies.is_dir():
        sys.path.insert(0, str(private_dependencies))
    run_dir = args.run_dir or root / "campaign-r2"
    admission_path = root / "receipts/ADMISSION_R2.json"
    if args.action == "bootstrap":
        review_path = args.recipe_review or _default_review_path(root, operator)
        admission, _ = build_r2_admission(
            root, operator, args.training_source, args.eval_source, review_path
        )
        _write_once(admission_path, admission)
        if not run_dir.exists():
            prepare_run(
                operator
                / "research/experiment_drafts/multistage-optuna-b300-20261031/CONTRACT.json",
                admission_path,
                run_dir,
                training_source=args.training_source,
                training_source_revision="source5b-core:2661a89285bd528e1bca690a340e71a21816a7b54e3ba1635d14925f3a9c1e4e",
                evaluation_source=args.eval_source,
                evaluation_source_revision="source6-eval",
                stage_training_sources={
                    "1": (
                        root / "operator-native4",
                        "operator-native4:c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81",
                    ),
                    "2": (
                        args.training_source,
                        "source5b-core:2661a89285bd528e1bca690a340e71a21816a7b54e3ba1635d14925f3a9c1e4e",
                    ),
                    "3": (
                        args.training_source,
                        "source5b-core:2661a89285bd528e1bca690a340e71a21816a7b54e3ba1635d14925f3a9c1e4e",
                    ),
                },
            )
        else:
            _verify_existing_run(run_dir, admission_path)
        if not args.skip_adoption:
            _adopt_r5_fits(root, run_dir, admission)
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "run_dir": str(run_dir),
                    "admission_sha256": file_sha256(admission_path),
                }
            )
        )
        return
    admission = read_json(admission_path)
    _verify_existing_run(run_dir, admission_path)
    _adopt_r5_fits(root, run_dir, admission)
    print(json.dumps({"status": "adoption_checked", "run_dir": str(run_dir)}))


def _default_review_path(runtime_root: Path, operator_root: Path) -> Path:
    """Find the one expected independent continuation-review receipt without guessing revisions."""
    candidates = (
        runtime_root / "receipts/INDEPENDENT_R2_RECIPE_REVIEW.json",
        operator_root
        / "research/experiment_drafts/multistage-optuna-b300-20261031/INDEPENDENT_R2_RECIPE_REVIEW.json",
    )
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    return candidates[0]


def _write_once(path: Path, value: dict[str, object]) -> None:
    """Publish the generated admission only once; never mutate an existing campaign input."""
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f"existing R2 admission differs: {path}")
        return
    atomic_write_json(path, value, replace=False)


def _verify_existing_run(run_dir: Path, admission_path: Path) -> None:
    """Require an existing run manifest to bind the exact current admission bytes."""
    run_path = run_dir / "run.json"
    if not run_path.is_file() or run_path.is_symlink():
        raise FileNotFoundError(f"prepared R2 campaign is missing: {run_path}")
    manifest = read_json(run_path)
    admission = manifest.get("admission")
    if not isinstance(admission, dict) or admission.get("sha256") != file_sha256(admission_path):
        raise ValueError("prepared R2 run is bound to a different admission")


def _adopt_r5_fits(runtime_root: Path, run_dir: Path, admission: dict[str, object]) -> None:
    """Score and import only the two complete source3 fits when a released slot is available."""
    rows = admission.get("preexisting_fit_attempts")
    if not isinstance(rows, list):
        raise ValueError("R2 admission lost its preexisting fit ledger")
    target_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("stage") == 1
        and isinstance(row.get("attempt_id"), str)
        and row["attempt_id"].endswith(("gru-b256-r5", "lstm-b256-r5"))
    ]
    if len(target_rows) != 2:
        raise ValueError("R2 receiver must find both original source3 B256 attempts")
    reserved = set(dynamic_reserved_gpu_keys(admission))
    status = status_summary(run_dir)
    internal_busy = {
        str(item.get("gpu_key"))
        for item in status["owned_workers"]
        if isinstance(item, dict) and item.get("process_alive") is True
    }
    policy = admission.get("profile_policy")
    devices = policy.get("devices") if isinstance(policy, dict) else None
    if not isinstance(devices, dict):
        raise ValueError("R2 profile policy has no measured device map")
    released = [key for key in sorted(devices, key=int) if key not in reserved | internal_busy]
    receipt_dir = runtime_root / "receipts/multistage-r2-fit-observations"
    queue_path = receipt_dir / "r5-adoption-queue.json"
    pending = []
    for row in target_rows:
        terminal_path = row.get("terminal_receipt_path")
        if row.get("status") != "artifacts_complete_exit_unknown" or not isinstance(
            terminal_path, str
        ):
            pending.append(row["attempt_id"])
    if pending or not released:
        queue = {
            "schema": "wnn.multistage-adoption-queue.v1",
            "status": "pending",
            "attempt_ids": pending or [row["attempt_id"] for row in target_rows],
            "released_gpu_keys_observed": released,
            "reserved_gpu_keys_observed": sorted(reserved, key=int),
            "no_fit_relaunch": True,
        }
        _write_once(queue_path, queue)
        return
    for row in target_rows:
        adoption_path = run_dir / "stage1/preexisting" / str(row["attempt_id"]) / "adoption.json"
        if adoption_path.is_file():
            continue
        owner_path = Path(str(row["owner_receipt_path"]))
        terminal_receipt_path = Path(str(row["terminal_receipt_path"]))
        approved_ref = row.get("approved_profile_receipt")
        if not isinstance(approved_ref, dict):
            raise ValueError("source3 fit registration has no approved profile receipt")
        adopt_completed_fit(
            run_dir,
            str(row["attempt_id"]),
            config_path=Path(str(row["config_path"])),
            approved_profile_path=Path(str(approved_ref["path"])),
            owner_receipt_path=owner_path,
            terminal_receipt_path=terminal_receipt_path,
            gpu_key=released[0],
        )


if __name__ == "__main__":
    main()
