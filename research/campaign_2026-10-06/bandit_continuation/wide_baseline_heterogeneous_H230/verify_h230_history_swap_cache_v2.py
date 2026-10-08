"""Validate the two new all-data history-swap arrays and their provenance."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any  # JSON metadata has a dynamic external schema.

from verify_h230_cache_outputs_v1 import (
    EXPECTED_FAMILIES,
    EXPECTED_SOURCES,
    _ordered_fit_groups,
    file_sha256,
    inspect_npy_file,
    row_groups_sha256,
)

EXPECTED_BASE_FILES = {
    "main_source_train",
    "main_source_valid",
    "families_train",
    "families_valid",
}
EXPECTED_OUTPUTS = {
    "main_history_train": ("train", 2, 10606),
    "main_history_valid": ("valid", 2, 1873),
}
EXPECTED_ROW_COUNTS = {"train": 10606, "valid": 1873}
EXPECTED_ROW_DIGESTS = {
    "train": "658498ef9fb3ad7f4dab9dea249da809168be595171099406795f72b63554fbe",
    "valid": "008f12e284e0f72ff423fee68f34c37054759bc4a345e9581620c1fe9b2acdf7",
}
EXPECTED_FORMULA = "main_source - 0.15*strong_seed3 - 0.15*strong_seed4 + 0.30*parent_ep3"


def _read_json(path: Path) -> dict[str, Any]:
    """Read one small JSON receipt or metadata document."""
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_receipt(path: Path, record: dict[str, Any]) -> None:
    """Atomically persist progress and terminal validation state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_controller_attempt(
    attempt: dict[str, Any], expected_id: str, expected_contract_sha256: str
) -> dict[str, Any]:
    """Require a clean terminal cache-build attempt and retain scoped OOM deltas."""
    if (
        attempt.get("id") != expected_id
        or attempt.get("status") != "finished"
        or attempt.get("phase") != "terminal"
        or attempt.get("exit_status") != 0
        or attempt.get("contract_sha256") != expected_contract_sha256
    ):
        raise ValueError("controller attempt does not bind the successful V3 build")

    worker_events = attempt.get("final_memory_event_deltas")
    parent_local_events = attempt.get("final_parent_memory_event_deltas")
    parent_hierarchical_events = attempt.get("final_parent_hierarchical_memory_event_deltas")
    if not all(
        isinstance(events, dict)
        and all(events.get(key) == 0 for key in ("oom", "oom_group_kill", "oom_kill"))
        for events in (worker_events, parent_local_events, parent_hierarchical_events)
    ):
        raise ValueError("cache build has a nonzero worker or parent OOM delta")
    return {
        "attempt_id": expected_id,
        "unit": attempt.get("unit"),
        "status": attempt.get("status"),
        "phase": attempt.get("phase"),
        "exit_status": attempt.get("exit_status"),
        "elapsed_seconds": attempt.get("elapsed_seconds"),
        "reservation": attempt.get("reservation"),
        "peak_worker_ram_bytes": attempt.get("peak_ram_bytes"),
        "peak_parent_ram_bytes": attempt.get("peak_parent_ram_bytes"),
        "worker_memory_events_scope": attempt.get("attempt_memory_events_scope"),
        "parent_local_memory_events_scope": attempt.get("parent_memory_events_scope"),
        "worker_memory_event_deltas": attempt.get("final_memory_event_deltas"),
        "parent_local_memory_event_deltas": attempt.get("final_parent_memory_event_deltas"),
        "parent_hierarchical_memory_event_deltas": attempt.get(
            "final_parent_hierarchical_memory_event_deltas"
        ),
        "observer_status": attempt.get("observer_status"),
    }


def verify_cache(
    project_root: Path,
    cache_dir: Path,
    expected_metadata_sha256: str,
    expected_base_metadata_sha256: str,
    expected_split_sha256: str,
    expected_parent_checkpoint_sha256: str,
    expected_contract_sha256: str,
    attempt_id: str,
    attempt_path: Path,
    root_formula_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Validate actual history outputs without re-reading unchanged base arrays."""
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema": "h230-history-swap-cache-output-validation-v2",
        "status": "running",
        "cache_dir": str(cache_dir.resolve()),
        "expected_metadata_sha256": expected_metadata_sha256,
        "expected_base_metadata_sha256": expected_base_metadata_sha256,
        "expected_split_sha256": expected_split_sha256,
        "attempt_path": str(attempt_path),
        "root_formula_check_path": str(root_formula_path),
        "arrays": {},
        "failure": None,
        "started_utc": datetime.now(UTC).isoformat(),
    }
    _write_receipt(receipt_path, receipt)

    try:
        metadata_path = cache_dir / "metadata.json"
        base_metadata_path = (
            project_root / "cache/bd26_h230_fold0_teacher_family_all_rows_v1/metadata.json"
        )
        split_path = project_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
        receipt["observed_metadata_sha256"] = file_sha256(metadata_path)
        receipt["observed_base_metadata_sha256"] = file_sha256(base_metadata_path)
        receipt["observed_split_sha256"] = file_sha256(split_path)
        for label, expected, observed in (
            (
                "cache metadata",
                expected_metadata_sha256,
                receipt["observed_metadata_sha256"],
            ),
            (
                "base metadata",
                expected_base_metadata_sha256,
                receipt["observed_base_metadata_sha256"],
            ),
            (
                "split manifest",
                expected_split_sha256,
                receipt["observed_split_sha256"],
            ),
        ):
            if observed != expected:
                raise ValueError(f"{label} digest differs from the frozen identity")

        metadata = _read_json(metadata_path)
        base_metadata = _read_json(base_metadata_path)
        if (
            metadata.get("format") != "wnn.heterogeneous-distillation-cache"
            or metadata.get("version") != 1
            or metadata.get("variant") != "history"
            or metadata.get("membership_mode") != "all_supplied_train_valid"
            or metadata.get("heldout_claim") is not False
            or metadata.get("evaluation_valid_groups") != []
        ):
            raise ValueError("cache does not declare the all-data no-heldout scope")
        if metadata.get("split_manifest_sha256") != expected_split_sha256:
            raise ValueError("cache split identity differs from the frozen manifest")
        if metadata.get("fit_groups") != base_metadata.get("fit_groups"):
            raise ValueError("cache membership differs from the all-data base cache")
        if metadata.get("fit_groups") != {
            "train": base_metadata["fit_groups"]["train"],
            "valid": base_metadata["fit_groups"]["valid"],
        }:
            raise ValueError("all-data membership changed while validating the cache")
        if (
            metadata.get("family_order") != EXPECTED_FAMILIES
            or metadata.get("family_order") != base_metadata.get("family_order")
            or metadata.get("source_coefficients")
            != [
                {
                    "family": family,
                    "run": run,
                    "epoch": epoch,
                    "coefficient": coefficient,
                }
                for family, run, epoch, coefficient in EXPECTED_SOURCES
            ]
            or metadata.get("source_coefficients") != base_metadata.get("source_coefficients")
            or metadata.get("coefficient_sum") != 0.8125
            or metadata.get("homogeneous_family") != "strong"
        ):
            raise ValueError("unchanged teacher-family recipe or weights differ")

        history_swap = metadata.get("history_swap")
        if not isinstance(history_swap, dict):
            raise ValueError("history-swap provenance is absent")
        base_cache = history_swap.get("base_cache", {})
        if (
            base_cache.get("path") != "cache/bd26_h230_fold0_teacher_family_all_rows_v1"
            or base_cache.get("metadata_sha256") != expected_base_metadata_sha256
        ):
            raise ValueError("cache does not bind the exact frozen base cache")
        if history_swap.get("target_formula", {}).get("expression") != EXPECTED_FORMULA:
            raise ValueError("history-swap target formula differs")
        if (
            history_swap.get("target_formula", {}).get("output_cast")
            != "one float32 cast after float64 accumulation"
        ):
            raise ValueError("history target accumulation/cast differs")

        base_verified = history_swap.get("base_verified_files")
        base_files = base_metadata.get("files")
        if (
            not isinstance(base_verified, dict)
            or set(base_verified) != EXPECTED_BASE_FILES
            or not isinstance(base_files, dict)
        ):
            raise ValueError("base-cache references do not list the four reused arrays")
        for name, digest in base_verified.items():
            if digest != base_files.get(name, {}).get("sha256"):
                raise ValueError(f"reused base array reference differs: {name}")

        parent = history_swap.get("parent_verified", {})
        if (
            parent.get("checkpoint_sha256") != expected_parent_checkpoint_sha256
            or parent.get("checkpoint_epoch_zero_based") != 3
            or parent.get("fit_scope") != "all_supplied_train_valid"
        ):
            raise ValueError("history parent identity or scope differs")
        parent_exports = history_swap.get("parent_exports")
        if not isinstance(parent_exports, dict) or set(parent_exports) != {"train", "valid"}:
            raise ValueError("the two parent FP32 exports are not both declared")
        for split_name, width, rows in (
            ("train", 2, EXPECTED_ROW_COUNTS["train"]),
            ("valid", 2, EXPECTED_ROW_COUNTS["valid"]),
        ):
            export = parent_exports[split_name]
            if (
                export.get("dtype") != "float32"
                or export.get("inference_precision") != "fp32"
                or export.get("shape") != [rows, 20_000, width]
                or export.get("row_groups_sha256") != EXPECTED_ROW_DIGESTS[split_name]
            ):
                raise ValueError(f"parent export is not exact FP32 {split_name} coverage")

        output_files = metadata.get("files")
        if not isinstance(output_files, dict) or set(output_files) != set(EXPECTED_OUTPUTS):
            raise ValueError("history cache must contain exactly its two new arrays")
        expected_names = {"metadata.json"} | {
            str(entry.get("path")) for entry in output_files.values()
        }
        observed_names = {path.name for path in cache_dir.iterdir() if path.is_file()}
        if observed_names != expected_names:
            raise ValueError("cache output directory contains unexpected or missing files")

        for name, (split_name, width, rows) in EXPECTED_OUTPUTS.items():
            entry = output_files[name]
            expected_rows = _ordered_fit_groups(
                project_root, split_name, metadata["fit_groups"][split_name]
            )
            if len(expected_rows) != rows or len(set(expected_rows)) != rows:
                raise ValueError(f"all-data row-group count differs: {split_name}")
            if entry.get("row_groups") != expected_rows:
                raise ValueError(f"cache row-group order differs: {split_name}")
            if entry.get("row_groups_sha256") != row_groups_sha256(expected_rows):
                raise ValueError(f"cache row-group digest differs: {split_name}")
            if entry.get("row_groups_sha256") != EXPECTED_ROW_DIGESTS[split_name]:
                raise ValueError(f"cache membership digest differs: {split_name}")
            if entry.get("shape") != [rows, 20_000, width] or entry.get("dtype") != "float32":
                raise ValueError(f"cache array shape/dtype differs: {name}")
            relative_path = Path(str(entry.get("path", "")))
            if relative_path.is_absolute() or relative_path.name != relative_path.as_posix():
                raise ValueError(f"cache output path is not a single filename: {name}")
            path = cache_dir / relative_path
            observed = inspect_npy_file(path, (rows, 20_000, width))
            if observed["sha256"] != entry.get("sha256"):
                raise ValueError(f"cache array digest differs: {name}")
            if observed["bytes"] != entry.get("bytes"):
                raise ValueError(f"cache array byte count differs: {name}")
            receipt["arrays"][name] = {
                **observed,
                "row_groups_sha256": entry["row_groups_sha256"],
                "status": "verified_all_finite",
            }
            _write_receipt(receipt_path, receipt)

        formula_check = _read_json(root_formula_path)
        if (
            formula_check.get("status") != "pass_sampled_formula_and_membership"
            or formula_check.get("metadata_sha256") != expected_metadata_sha256
            or formula_check.get("no_heldout_claim") is not True
        ):
            raise ValueError("independent root formula check does not bind this cache")

        attempt = _read_json(attempt_path)
        receipt["controller_attempt_sha256"] = file_sha256(attempt_path)
        receipt["controller_terminal"] = _validate_controller_attempt(
            attempt, attempt_id, expected_contract_sha256
        )
        receipt["root_formula_check_sha256"] = file_sha256(root_formula_path)
        receipt["root_formula_check"] = {
            "status": formula_check["status"],
            "exact_values_checked": sum(
                check.get("exact_float32_values_checked", 0)
                for check in formula_check.get("checks", [])
            ),
            "split_checks": [check.get("split") for check in formula_check["checks"]],
            "row_group_order_matches_sealed_parent": all(
                check.get("row_group_order_matches_sealed_parent") is True
                for check in formula_check.get("checks", [])
            ),
        }
        if (
            set(receipt["root_formula_check"]["split_checks"]) != {"train", "valid"}
            or receipt["root_formula_check"]["exact_values_checked"] <= 0
            or not receipt["root_formula_check"]["row_group_order_matches_sealed_parent"]
        ):
            raise ValueError("root formula check has no split evidence")
        receipt["status"] = "passed"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        receipt["status"] = "failed"
        receipt["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["finished_utc"] = datetime.now(UTC).isoformat()
        _write_receipt(receipt_path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse frozen cache and attempt identities for the output validator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--base-metadata-sha256", required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--root-formula-check", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate the actual builder output and emit one immutable receipt."""
    args = parse_args(argv)
    try:
        receipt = verify_cache(
            args.project_root,
            args.cache_dir,
            args.metadata_sha256,
            args.base_metadata_sha256,
            args.split_sha256,
            args.parent_checkpoint_sha256,
            args.contract_sha256,
            args.attempt_id,
            args.attempt,
            args.root_formula_check,
            args.receipt,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"cache validation failed: {type(error).__name__}: {error}")
        return 1
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(args.receipt),
                "verified_arrays": len(receipt["arrays"]),
                "elapsed_seconds": receipt["elapsed_seconds"],
                "max_rss_kib": receipt["max_rss_kib"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
