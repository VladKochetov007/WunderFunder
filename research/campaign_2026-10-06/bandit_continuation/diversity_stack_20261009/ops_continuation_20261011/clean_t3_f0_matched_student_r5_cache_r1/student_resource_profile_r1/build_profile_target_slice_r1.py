"""Build the actual 16-group FIT_VALID target slice for a geometry-only profile."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ADAPTER_DIR = HERE.parent / "cache_adapter_r1"
sys.path.insert(0, str(ADAPTER_DIR))

from build_clean_t3_f0_homogeneous_cache_r1 import (  # noqa: E402
    homogeneous_float32_chunk,
    inspect_inputs,
    load_contract,
    load_split_membership,
    load_target_orders,
    ordered_groups_sha256,
    row_positions,
    selected_target_order,
    sha256_file,
)

PROFILE_SEQUENCE_COUNT = 16


def build_profile_target_slice(
    project_root: Path,
    adapter_contract_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Stream-verify producer inputs, then save only 16 non-evaluation FIT_VALID targets."""
    root = project_root.resolve(strict=True)
    contract_path = adapter_contract_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite profile fixture: {output_dir}")

    contract = load_contract(contract_path)
    inspected = inspect_inputs(root, contract_path, require_train=False, hash_arrays=True)
    fit, evaluation, split_record = load_split_membership(root, contract)
    target_orders = load_target_orders(root)
    fit_valid_order = selected_target_order(
        target_orders["valid"], fit["valid"], "valid.parquet FIT_VALID"
    )
    profile_groups = fit_valid_order[:PROFILE_SEQUENCE_COUNT]
    if len(profile_groups) != PROFILE_SEQUENCE_COUNT:
        raise ValueError("fewer than 16 actual F0 FIT_VALID groups are available")
    if set(profile_groups) & set(evaluation):
        raise ValueError("profile fixture would contain an evaluation group")

    records = {item["provenance"]["seed"]: item for item in inspected["teacher_records"]}
    if set(records) != {3, 4, 5}:
        raise ValueError("actual source inspection did not validate all three teacher seeds")
    inputs: list[np.ndarray] = []
    source_receipts: list[dict[str, Any]] = []
    for seed in (3, 4, 5):
        record = records[seed]
        provenance = record["provenance"]
        binding = record["predictions"]["valid"]
        valid_source_membership = set(fit["valid"]) | set(evaluation)
        rows = row_positions(
            binding["row_groups"],
            profile_groups,
            valid_source_membership,
            "valid.parquet full-valid profile projection",
        )
        source = Path(binding["path"])
        array = np.load(source, mmap_mode="r", allow_pickle=False)
        if array.shape != (1873, 20_000, 2) or array.dtype != np.float32:
            raise ValueError(f"seed {seed} full-valid teacher array header differs")
        selected = np.asarray(array[rows], dtype=np.float32)
        if selected.shape != (PROFILE_SEQUENCE_COUNT, 20_000, 2):
            raise ValueError(f"seed {seed} did not yield exactly 16 full sequences")
        inputs.append(selected)
        source_receipts.append(
            {
                "seed": seed,
                "run": provenance["run"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "provenance_sha256": provenance["provenance_sha256"],
                "prediction_path": str(source),
                "prediction_sha256": binding["prediction_sha256"],
                "sidecar_path": binding["sidecar_path"],
                "sidecar_sha256": binding["sidecar_sha256"],
                "full_valid_array_shape": list(array.shape),
                "profile_row_positions": rows.tolist(),
            }
        )
    target = homogeneous_float32_chunk(inputs[0], inputs[1], inputs[2])

    partial_dir.mkdir(parents=True)
    try:
        target_path = partial_dir / "teacher_target_fit_valid16.npy"
        mapped = np.lib.format.open_memmap(
            target_path,
            mode="w+",
            dtype=np.float32,
            shape=(PROFILE_SEQUENCE_COUNT, 20_000, 2),
        )
        mapped[:] = target
        mapped.flush()
        del mapped
        receipt = {
            "schema": "wnn.clean-t3-f0-student-profile-target-slice.r1",
            "status": "PASS_PROFILE_ONLY_16_FIT_VALID_GROUPS",
            "profile_only": True,
            "not_a_complete_teacher_cache": True,
            "not_a_student_fit_or_scientific_grade": True,
            "project_root": str(root),
            "adapter_contract_path": str(contract_path),
            "adapter_contract_sha256": sha256_file(contract_path),
            "split_manifest_path": split_record["path"],
            "split_manifest_sha256": split_record["sha256"],
            "valid_target_index_path": str(root / "cache" / "valid_targets.npz"),
            "valid_target_index_member_read": "row_group only; target labels were not read by this builder",
            "fit_valid_source": "datasets/valid.parquet",
            "fit_valid_count": len(fit["valid"]),
            "evaluation_valid_count": len(evaluation),
            "evaluation_groups_selected": 0,
            "profile_sequence_count": PROFILE_SEQUENCE_COUNT,
            "profile_sequence_rows": 20_000,
            "profile_group_order": "first 16 FIT_VALID IDs in valid_targets.npz row_group order",
            "profile_row_groups": profile_groups,
            "profile_row_groups_ordered_int64_le_sha256": ordered_groups_sha256(profile_groups),
            "target_formula": "float32(0.8125 * float32_mean([seed3, seed4, seed5]))",
            "operation_order": "float32 add seed3 then seed4 then seed5; divide by float32(3); multiply by float32(0.8125); no clipping",
            "teachers": source_receipts,
            "output": {
                "path": str(output_dir / target_path.name),
                "shape": list(target.shape),
                "dtype": "float32",
                "bytes": target_path.stat().st_size,
                "sha256": sha256_file(target_path),
                "finite": bool(np.isfinite(target).all()),
            },
            "source_inspection_status": inspected["status"],
            "fit_train_exports_ready": inspected["fit_train_predictions_ready"],
            "fit_train_exports_pending": inspected["fit_train_predictions_pending"],
            "evaluation_projection_policy": "full-valid source arrays were row-ID-joined; only 16 FIT_VALID rows were selected into this target; no evaluation rows enter the profile fixture",
        }
        (partial_dir / "PROFILE_TARGET_SLICE_RECEIPT_R1.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        os.replace(partial_dir, output_dir)
    except BaseException:
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--adapter-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_profile_target_slice(
        args.project_root, args.adapter_contract, args.output_dir
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "teachers"}, indent=2))


if __name__ == "__main__":
    main()
