"""Create an exact same-row scored control array from a sealed full-valid export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
from collection_support import (
    compare_bundle_arrays,
    derive_strong_control_inputs,
    resolve_create_only_output,
    validate_export,
    write_create_only_json,
)
from frozen_inputs import (
    load_bundle_arrays,
    validate_diagnostic_manifest,
    validate_parquet_source,
    validate_split_membership,
)
from pins import (
    CANONICAL_SPLIT_SHA256,
    CHECKPOINT_SHA256,
    CONTROL_RUN,
    DIAGNOSTIC_INPUTS_SHA256,
    EVALUATION_COMPONENTS,
    EVALUATION_GROUPS,
    MEMBERSHIP_COMPARISON_SHA256,
    METRIC_ID,
    OLD_SPLIT_SHA256,
    Q8_FULL_GRADE_SHA256,
    ROW_GROUPS,
    ROWS_PER_GROUP,
    SELECTED_ROWS,
)
from source_alignment import (
    SOURCE_COLUMNS,
    collect_evaluation_rows,
    require_sha256,
    sha256_array,
    sha256_file,
    stat_identity,
    write_npy_create_only,
)

ALIGNED_FILE = "strong_control_scored.npy"
ALIGNMENT_FILE = "ALIGNMENT_RECEIPT.json"
DERIVED_INPUTS_FILE = "DERIVED_INPUTS.json"


def collect(args: argparse.Namespace) -> dict[str, object]:
    """Validate future export inputs, prove same-row alignment and publish compact artifacts."""
    output = resolve_create_only_output(args.output)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent must already exist: {output.parent}")

    repo_root = args.repo_root.resolve(strict=True)
    (
        identities,
        evaluation_groups,
        old_fold,
        membership_receipt,
        old_path,
        canonical_path,
        membership_path,
    ) = validate_split_membership(repo_root)
    diagnostic_path = (
        repo_root / "research/campaign_2026-10-18/quality_gap_r1/diagnostics/INPUTS.json"
    )
    diagnostic, bundle_spec, bundle_path, q8_receipt, q8_receipt_path = (
        validate_diagnostic_manifest(diagnostic_path)
    )
    expected_bundle = load_bundle_arrays(bundle_spec, bundle_path)
    bundle_sha_before = sha256_file(bundle_path)
    parquet_path = args.valid_parquet.resolve(strict=True)
    parquet_identity = stat_identity(parquet_path)
    parquet = validate_parquet_source(parquet_path, old_fold, q8_receipt)
    prediction, sidecar, export_receipt, export_bindings = validate_export(
        args.prediction,
        args.sidecar,
        args.export_contract,
        args.expected_export_contract_sha256,
        args.export_receipt,
        args.expected_sidecar_sha256,
        args.expected_export_receipt_sha256,
    )
    row_map = cast(dict[int, int], export_bindings["row_map"])
    extracted, group_audit = collect_evaluation_rows(
        parquet, evaluation_groups, identities, prediction, row_map
    )
    if extracted["prediction"].shape != (SELECTED_ROWS, 2):
        raise ValueError("official mask selected-row count differs from the pinned bundle")
    if not np.isfinite(extracted["prediction"]).all():
        raise ValueError("selected control predictions contain nonfinite values")
    bundle_hashes = cast(dict[str, str], bundle_spec["array_sha256"])
    compare_bundle_arrays(extracted, expected_bundle, bundle_hashes)
    if sha256_file(bundle_path) != bundle_sha_before:
        raise ValueError("compact scored bundle changed while its arrays were aligned")
    if stat_identity(parquet_path) != parquet_identity:
        raise ValueError("validation parquet changed during selected-row extraction")
    for path, key, label in (
        (args.export_contract, "export_contract_sha256", "export contract"),
        (args.sidecar, "sidecar_sha256", "prediction sidecar"),
        (args.export_receipt, "export_receipt_sha256", "export receipt"),
        (
            Path(str(export_bindings["raw_guard_attempt_path"])),
            "raw_guard_attempt_sha256",
            "captured raw guard attempt",
        ),
        (
            Path(str(export_bindings["captured_sidecar_path"])),
            "captured_sidecar_sha256",
            "captured raw prediction sidecar",
        ),
        (
            Path(str(export_bindings["captured_preinference_path"])),
            "raw_preinference_sha256",
            "captured strict-FP32 pre-inference evidence",
        ),
    ):
        require_sha256(
            sha256_file(path, 16 * 1024 * 1024),
            str(export_bindings[key]),
            label,
        )
    if (
        list(stat_identity(args.prediction.resolve(strict=True)))
        != export_bindings["prediction_stat_identity"]
    ):
        raise ValueError("full prediction changed during selected-row extraction")
    if (
        len(group_audit) != EVALUATION_GROUPS
        or len(np.unique(extracted["row_group"])) != EVALUATION_GROUPS
        or len(np.unique(extracted["component"])) != EVALUATION_COMPONENTS
    ):
        raise ValueError("selected evaluation group/component coverage differs from pinned bundle")

    output.mkdir(exist_ok=False)
    prediction_path = output / ALIGNED_FILE
    write_npy_create_only(prediction_path, extracted["prediction"])
    compact_prediction_sha = sha256_file(prediction_path)
    q8_panel_sha256 = str(q8_receipt["panel_sha256"])
    module_paths = {
        "collector": Path(__file__).resolve(),
        "source_alignment": Path(__file__).resolve().with_name("source_alignment.py"),
        "frozen_inputs": Path(__file__).resolve().with_name("frozen_inputs.py"),
        "collection_support": Path(__file__).resolve().with_name("collection_support.py"),
        "export_contract": Path(__file__).resolve().with_name("export_contract.py"),
        "export_receipt": Path(__file__).resolve().with_name("export_receipt.py"),
        "h390_sidecar": Path(__file__).resolve().with_name("h390_sidecar.py"),
        "pins": Path(__file__).resolve().with_name("pins.py"),
    }
    receipt: dict[str, object] = {
        "schema": "wnn.quality_gap.strong_control_alignment.v1",
        "status": "passed",
        "interpretation": "same-row no-training historical strong-control comparison; not the exact public incumbent",
        "metric_id": METRIC_ID,
        "selected_rows": SELECTED_ROWS,
        "row_groups": EVALUATION_GROUPS,
        "components": EVALUATION_COMPONENTS,
        "paired_bundle_sha256": str(bundle_spec["sha256"]),
        "paired_bundle_bytes": int(bundle_spec["bytes"]),
        "prediction_sha256": compact_prediction_sha,
        "prediction_bytes": prediction_path.stat().st_size,
        "prediction_shape": [SELECTED_ROWS, 2],
        "prediction_dtype": "float32",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_run": CONTROL_RUN,
        "checkpoint_epoch": 5,
        "old_split_sha256": OLD_SPLIT_SHA256,
        "canonical_split_sha256": CANONICAL_SPLIT_SHA256,
        "membership_comparison": {
            "path": str(membership_path),
            "sha256": MEMBERSHIP_COMPARISON_SHA256,
            "status": membership_receipt["status"],
            "ordered_evaluation_groups_equal": True,
            "valid_sequence_and_component_map_equal": True,
        },
        "split_files": {
            "old_checkpoint_split": {"path": str(old_path), "sha256": OLD_SPLIT_SHA256},
            "canonical_comparison_split": {
                "path": str(canonical_path),
                "sha256": CANONICAL_SPLIT_SHA256,
            },
        },
        "export": {
            "contract_path": export_bindings["export_contract_path"],
            "contract_sha256": export_bindings["export_contract_sha256"],
            "contract_source_pins": export_receipt["source_pins"],
            "attempt_id": export_bindings["attempt_id"],
            "raw_guard_attempt_sha256": export_bindings["raw_guard_attempt_sha256"],
            "captured_raw_evidence": {
                "guard_attempt_path": export_bindings["raw_guard_attempt_path"],
                "guard_attempt_sha256": export_bindings["raw_guard_attempt_sha256"],
                "sidecar_path": export_bindings["captured_sidecar_path"],
                "sidecar_sha256": export_bindings["captured_sidecar_sha256"],
                "pre_inference_runtime_path": export_bindings["captured_preinference_path"],
                "pre_inference_runtime_sha256": export_bindings["raw_preinference_sha256"],
            },
            "execution": export_receipt["execution"],
            "pre_inference_runtime": export_receipt["pre_inference_runtime"],
            "receipt_path": str(args.export_receipt.resolve()),
            "receipt_sha256": export_bindings["export_receipt_sha256"],
            "receipt_bytes": args.export_receipt.stat().st_size,
            "receipt_schema_field": export_receipt.get("schema"),
            "sidecar_path": str(args.sidecar.resolve()),
            "sidecar_sha256": export_bindings["sidecar_sha256"],
            "sidecar_bytes": args.sidecar.stat().st_size,
            "prediction_path": str(args.prediction.resolve()),
            "prediction_sha256": export_bindings["prediction_sha256"],
            "prediction_bytes": args.prediction.stat().st_size,
            "prediction_shape": list(prediction.shape),
            "prediction_dtype": str(prediction.dtype),
            "campaign_fit_scope": sidecar["campaign_fit_scope"],
            "evaluation_membership_sha256": sidecar["evaluation_membership_sha256"],
            "campaign_evaluation_membership_sha256": sidecar[
                "campaign_evaluation_membership_sha256"
            ],
            "sidecar_row_groups_sha256": hashlib.sha256(
                np.asarray(sidecar["row_groups"], dtype="<i8").tobytes()
            ).hexdigest(),
            "captured_ordered_row_groups_sha256": export_bindings["ordered_row_groups_sha256"],
            "source_row_group_mapping": "physical validation row_group -> explicit prediction sidecar row index",
        },
        "source_parquet": {
            "path": str(parquet_path),
            "bytes": parquet_path.stat().st_size,
            "mtime_ns": parquet_path.stat().st_mtime_ns,
            "full_file_sha256": q8_panel_sha256,
            "full_file_sha256_scope": "inherited from exact pinned full-panel Q8 grade receipt; not rehashed by this bounded collector",
            "inherited_receipt_path": str(q8_receipt_path),
            "inherited_receipt_sha256": Q8_FULL_GRADE_SHA256,
            "footer_bytes": 10_686_926,
            "footer_sha256": "4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b",
            "row_groups": ROW_GROUPS,
            "rows_per_group": ROWS_PER_GROUP,
            "selected_columns_only": SOURCE_COLUMNS,
            "mask_predicate": "need_prediction AND is_scored",
        },
        "row_alignment": {
            "evaluation_order": evaluation_groups,
            "target_array_sha256": bundle_hashes["target"],
            "row_group_array_sha256": bundle_hashes["row_group"],
            "component_array_sha256": bundle_hashes["component"],
            "selected_position_array_sha256": sha256_array(extracted["position"]),
            "groups": group_audit,
            "bundle_arrays_bitwise_equal": True,
        },
        "diagnostic_inputs_sha256": DIAGNOSTIC_INPUTS_SHA256,
        "helper_sha256": {name: sha256_file(path) for name, path in module_paths.items()},
        "derived_inputs_file": DERIVED_INPUTS_FILE,
        "comparisons": [
            "strong_control vs wide",
            "strong_control vs fp32_mix",
            "strong_control vs q8",
        ],
        "search_or_tuning": False,
    }
    alignment_path = output / ALIGNMENT_FILE
    write_create_only_json(alignment_path, receipt)
    alignment_sha256 = sha256_file(alignment_path)
    derived = derive_strong_control_inputs(
        diagnostic,
        diagnostic_path.resolve().parent,
        prediction_path,
        compact_prediction_sha,
        alignment_path,
        alignment_sha256,
    )
    derived_path = output / DERIVED_INPUTS_FILE
    write_create_only_json(derived_path, derived)
    return {
        "status": "passed",
        "output": str(output),
        "alignment_receipt_sha256": alignment_sha256,
        "derived_inputs_sha256": sha256_file(derived_path),
        "compact_prediction_sha256": compact_prediction_sha,
        "selected_rows": SELECTED_ROWS,
        "row_groups": EVALUATION_GROUPS,
        "components": EVALUATION_COMPONENTS,
    }


def main() -> None:
    """Run the create-only, source-bound alignment once a full-valid export arrives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--valid-parquet", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--export-contract", type=Path, required=True)
    parser.add_argument("--expected-export-contract-sha256", required=True)
    parser.add_argument("--expected-sidecar-sha256", required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--expected-export-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args), sort_keys=True))


if __name__ == "__main__":
    main()
