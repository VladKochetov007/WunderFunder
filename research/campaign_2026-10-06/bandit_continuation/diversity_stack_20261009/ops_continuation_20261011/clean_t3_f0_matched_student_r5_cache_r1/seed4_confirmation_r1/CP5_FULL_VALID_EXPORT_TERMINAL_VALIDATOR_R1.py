from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import resource
import subprocess
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)
WORKING_TARGET_BYTES = 17_000_000_000
SUPPORT_BOUND_BYTES = 500_000_000
EXPECTED_SHAPE = (1873, 20_000, 2)
EXPECTED_NPY_BYTES = 299_680_128
EXPECTED_GROUP_ORDER_SHA256 = "008f12e284e0f72ff423fee68f34c37054759bc4a345e9581620c1fe9b2acdf7"


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so artifact validation uses bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(checks: list[dict[str, object]], name: str, passed: bool, observed: object) -> None:
    """Append one named validation result to the receipt."""
    checks.append({"name": name, "passed": bool(passed), "observed": observed})


def campaign_cgroup() -> Path:
    """Return the active campaign cgroup path from systemd."""
    result = subprocess.run(
        ["systemctl", "--user", "show", "wnn-campaign.slice", "-p", "ControlGroup"],
        check=True,
        capture_output=True,
        text=True,
    )
    control_group = result.stdout.strip().split("=", 1)[1]
    return Path("/sys/fs/cgroup") / control_group.lstrip("/")


def main() -> int:
    """Validate one terminal full-valid feature export before official grading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--p3-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checks: list[dict[str, object]] = []
    receipt: dict[str, object] = {
        "schema": "wnn.control-seed4-cp5-full-valid-export-terminal-validation.r1",
        "recorded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "scope": "Terminal export artifact validation only; no metric computed.",
        "status": "FAIL",
        "checks": checks,
    }

    try:
        attempt = json.loads(args.attempt.read_text(encoding="utf-8"))
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        p3 = json.loads(args.p3_validation.read_text(encoding="utf-8"))
        metadata_path = Path(contract["export"]["expected_metadata"])
        npy_path = Path(contract["export"]["expected_npy"])
        partial_path = npy_path.with_suffix(npy_path.suffix + ".partial")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        split_path = args.root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        target_path = args.root / "cache/valid_targets.npz"

        cgroup = campaign_cgroup()
        memory_before = int((cgroup / "memory.current").read_text())
        swap_before = int((cgroup / "memory.swap.current").read_text())
        memory_events_before = {
            key: int(value)
            for key, value in (
                line.split() for line in (cgroup / "memory.events.local").read_text().splitlines()
            )
        }
        receipt["support_admission"] = {
            "parent_memory_current_before_bytes": memory_before,
            "support_memory_bound_bytes": SUPPORT_BOUND_BYTES,
            "sum_bytes": memory_before + SUPPORT_BOUND_BYTES,
            "working_target_bytes": WORKING_TARGET_BYTES,
            "within_target": memory_before + SUPPORT_BOUND_BYTES <= WORKING_TARGET_BYTES,
            "parent_swap_current_bytes": swap_before,
            "support_cpu_cores": 1,
            "support_runtime_limit_seconds": 120,
        }
        check(
            checks,
            "bounded_support_admission",
            memory_before + SUPPORT_BOUND_BYTES <= WORKING_TARGET_BYTES and swap_before == 0,
            receipt["support_admission"],
        )

        contract_sha = sha256_file(args.contract)
        p3_sha = sha256_file(args.p3_validation)
        checkpoint_path = Path(contract["parent"]["model_ep5_path"])
        checkpoint_sha = sha256_file(checkpoint_path)
        provenance_path = Path(contract["parent"]["provenance_path"])
        provenance_sha = sha256_file(provenance_path)
        attempt_sha = sha256_file(args.attempt)
        metadata_sha = sha256_file(metadata_path)
        prediction_sha = sha256_file(npy_path)
        split_sha = sha256_file(split_path)

        expected_export = contract["export"]
        check(
            checks,
            "attempt_finished_exit0_bound_contract",
            attempt.get("status") == "finished"
            and attempt.get("phase") == "terminal"
            and attempt.get("exit_status") == 0
            and attempt.get("contract_sha256") == contract_sha,
            {
                "status": attempt.get("status"),
                "phase": attempt.get("phase"),
                "exit_status": attempt.get("exit_status"),
                "attempt_contract_sha256": attempt.get("contract_sha256"),
                "bound_contract_sha256": contract_sha,
            },
        )
        check(
            checks,
            "p3_terminal_validation_and_actual_cp5_hash",
            p3.get("status") == "PASS"
            and all(item.get("passed") is True for item in p3.get("checks", []))
            and checkpoint_sha == expected_export["expected_checkpoint_sha256"]
            and checkpoint_sha == contract["parent"]["model_ep5_sha256"]
            and provenance_sha == contract["parent"]["provenance_sha256"],
            {
                "p3_validation_sha256": p3_sha,
                "p3_validation_status": p3.get("status"),
                "checkpoint_sha256": checkpoint_sha,
                "provenance_sha256": provenance_sha,
            },
        )
        check(
            checks,
            "frozen_predictor_and_split_source_pins",
            sha256_file(Path(contract["source_identity"]["predictor_path"]))
            == contract["source_identity"]["predictor_sha256"]
            and split_sha == contract["source_identity"]["split_manifest_sha256"]
            and split_manifest["counts"]["evaluation_valid"] == 468
            and split_manifest["counts"]["fit_valid"] == 1405
            and split_manifest["counts"]["fit_train"] == 10535,
            {
                "predictor_sha256": sha256_file(
                    Path(contract["source_identity"]["predictor_path"])
                ),
                "split_manifest_sha256": split_sha,
                "split_counts": split_manifest["counts"],
            },
        )
        check(
            checks,
            "metadata_run_epoch_checkpoint_split_precision",
            metadata.get("run") == expected_export["run_name"]
            and metadata.get("split") == "valid"
            and metadata.get("epoch") == 5
            and metadata.get("checkpoint_sha256") == checkpoint_sha
            and metadata.get("campaign_split_manifest_sha256") == split_sha
            and metadata.get("inference_precision") == "fp32"
            and metadata.get("prediction_sha256") == prediction_sha,
            {
                "run": metadata.get("run"),
                "split": metadata.get("split"),
                "epoch": metadata.get("epoch"),
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                "campaign_split_manifest_sha256": metadata.get("campaign_split_manifest_sha256"),
                "inference_precision": metadata.get("inference_precision"),
                "metadata_prediction_sha256": metadata.get("prediction_sha256"),
                "npy_sha256": prediction_sha,
            },
        )

        prediction = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        observed_shape = tuple(int(value) for value in prediction.shape)
        observed_dtype = str(prediction.dtype)
        file_bytes = npy_path.stat().st_size
        finite = True
        finite_groups = 0
        for start in range(0, observed_shape[0], 16):
            block = prediction[start : start + 16]
            if not bool(np.isfinite(block).all()):
                finite = False
                break
            finite_groups += int(block.shape[0])
        check(
            checks,
            "complete_npy_shape_dtype_bytes_and_finite",
            observed_shape == EXPECTED_SHAPE
            and observed_dtype == "float32"
            and file_bytes == EXPECTED_NPY_BYTES
            and finite
            and finite_groups == EXPECTED_SHAPE[0]
            and not partial_path.exists(),
            {
                "shape": observed_shape,
                "dtype": observed_dtype,
                "bytes_including_header": file_bytes,
                "finite_groups_scanned": finite_groups,
                "all_values_finite": finite,
                "partial_absent": not partial_path.exists(),
            },
        )

        with np.load(target_path, allow_pickle=False) as targets:
            target_groups = np.asarray(targets["row_group"], dtype="<i8")
        metadata_groups = np.asarray(metadata.get("row_groups", []), dtype="<i8")
        group_order_sha = hashlib.sha256(target_groups.tobytes()).hexdigest()
        check(
            checks,
            "full_valid_source_group_order_matches_actual_target_index",
            target_groups.shape == (EXPECTED_SHAPE[0],)
            and np.array_equal(metadata_groups, target_groups)
            and group_order_sha == EXPECTED_GROUP_ORDER_SHA256,
            {
                "source": "valid.parquet",
                "target_group_count": int(target_groups.size),
                "metadata_group_count": int(metadata_groups.size),
                "ordered_group_sha256": group_order_sha,
                "target_index_archive_bytes": target_path.stat().st_size,
                "target_index_hash_from_split_manifest": split_manifest["sources"][
                    "cache/valid_targets.npz"
                ]["index_sha256"],
                "accessed_target_members": ["row_group"],
            },
        )

        fit_membership = p3["actual_provenance"]["fit_membership_identity"]
        overlap = p3["actual_provenance"]["fit_valid_eval_overlap"]
        evaluation_count = p3["actual_provenance"]["evaluation_group_count"]
        check(
            checks,
            "source_qualified_fit_membership_eval_excluded",
            fit_membership["train"]["count"] == 10535
            and fit_membership["valid"]["count"] == 1405
            and evaluation_count == 468
            and overlap == 0,
            {
                "train_source": "train.parquet",
                "train": fit_membership["train"],
                "fit_valid_source": "valid.parquet",
                "fit_valid": fit_membership["valid"],
                "evaluation_source": "valid.parquet",
                "evaluation_groups_excluded_from_fit": evaluation_count,
                "fit_valid_eval_overlap": overlap,
            },
        )

        memory_after = int((cgroup / "memory.current").read_text())
        memory_events_after = {
            key: int(value)
            for key, value in (
                line.split() for line in (cgroup / "memory.events.local").read_text().splitlines()
            )
        }
        receipt.update(
            {
                "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
                "attempt": {
                    "id": attempt.get("id"),
                    "unit": attempt.get("unit"),
                    "contract_sha256": contract_sha,
                    "attempt_sha256": attempt_sha,
                    "started_utc": attempt.get("started_utc"),
                    "finished_utc": attempt.get("finished_utc"),
                    "elapsed_seconds": attempt.get("elapsed_seconds"),
                    "exit_status": attempt.get("exit_status"),
                    "peak_worker_ram_bytes": attempt.get("peak_ram_bytes"),
                    "peak_observed_vram_bytes": attempt.get("peak_observed_vram_bytes"),
                    "worker_memory_event_deltas": attempt.get("final_memory_event_deltas"),
                    "parent_local_memory_event_deltas": attempt.get(
                        "final_parent_memory_event_deltas"
                    ),
                    "parent_hierarchical_memory_event_deltas": attempt.get(
                        "final_parent_hierarchical_memory_event_deltas"
                    ),
                },
                "artifacts": {
                    "model_ep5_path": str(checkpoint_path),
                    "model_ep5_sha256": checkpoint_sha,
                    "training_state_path": contract["parent"]["training_state_path"],
                    "training_state_sha256": contract["parent"]["training_state_sha256"],
                    "p3_validation_path": str(args.p3_validation),
                    "p3_validation_sha256": p3_sha,
                    "provenance_path": str(provenance_path),
                    "provenance_sha256": provenance_sha,
                    "metadata_path": str(metadata_path),
                    "metadata_bytes": metadata_path.stat().st_size,
                    "metadata_sha256": metadata_sha,
                    "npy_path": str(npy_path),
                    "npy_bytes": file_bytes,
                    "npy_sha256": prediction_sha,
                    "split_manifest_sha256": split_sha,
                    "export_contract_sha256": contract_sha,
                    "validator_path": str(Path(__file__).resolve()),
                    "validator_sha256": sha256_file(Path(__file__).resolve()),
                },
                "full_valid_membership": {
                    "count": EXPECTED_SHAPE[0],
                    "ordered_valid_source_row_group_sha256": group_order_sha,
                    "evaluation_groups_exported_feature_only": True,
                    "evaluation_groups_excluded_from_fit": 468,
                    "official_metric_grade": "pending",
                },
                "source_qualified_fit_membership": fit_membership,
                "evaluation_group_count": evaluation_count,
                "same_valid_source_fit_eval_overlap": overlap,
                "teacher_ancestry": contract["parent"]["teacher_parents"],
                "target_access_audit": {
                    "prediction_worker_command_feature_only": "--campaign-full-valid-feature-only",
                    "export_worker_read_label_values": False,
                    "validator_accessed_valid_targets_members": ["row_group"],
                    "validator_read_label_values": False,
                    "pre_grade_metadata_probe_disclosure": (
                        "A prior bounded metadata probe accessed the NPZ y member only to print its array shape; "
                        "it did not inspect/use target values, predictions, or metrics."
                    ),
                },
                "no_score_computed": True,
                "no_refit_or_provider_action": True,
                "support_resources": {
                    "parent_memory_current_after_bytes": memory_after,
                    "parent_memory_events_local_before_counts": memory_events_before,
                    "parent_memory_events_local_after_counts": memory_events_after,
                    "validator_peak_rss_bytes": int(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    )
                    * 1024,
                    "support_memory_bound_bytes": SUPPORT_BOUND_BYTES,
                    "support_cpu_cores": 1,
                    "support_runtime_limit_seconds": 120,
                },
            }
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        ImportError,
        subprocess.CalledProcessError,
    ) as error:
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        receipt["status"] = "FAIL"

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("terminal export validation: %s", receipt["status"])
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
