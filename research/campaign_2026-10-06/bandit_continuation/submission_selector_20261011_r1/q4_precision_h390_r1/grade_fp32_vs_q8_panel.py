"""Stream official rows to compare the preserved R1 FP32 and R2 Q8 pair panels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from grade_q4_variant import sha256_array, sha256_file, weighted_correlation  # noqa: E402

from wnn.gmetric import GlobalAccumulator  # noqa: E402

EXPECTED_R1_ZIP_SHA256 = "6ff0d6c98a8cbec8ccd26866a74c39c69c2f6f07f1d5850735eb24634d09bae5"
EXPECTED_R1_PREDICTIONS_SHA256 = "31a8b2d426a516ff0df107e12a39d237a90da2cd806c18262c814b3140f11dc1"
EXPECTED_R2_ZIP_SHA256 = "3420ea5e6318e15b6340582125959ed7489983b17338f399c748965cb42c4a4a"
EXPECTED_R2_PREDICTIONS_SHA256 = "682376c1f056efcd4a011d5e5edd5b3de585f1337e5f93313c3f1bf89a2e04c5"
EXPECTED_PANEL_SHA256 = "e5e6472cedfcb20ecf01195777ab986b3560379c5e81caea71973632dde0d6a3"
EXPECTED_PANEL_GROUPS = [
    256678,
    751005,
    1200028,
    1892371,
    1990332,
    2308749,
    4490992,
    6746551,
    8348957,
    9147005,
]
EXPECTED_NATIVE_SHA256 = "ce179a68b7cd321368cc9ab73cdf1b79973afcdb8266ef8f6d7e613bb6f9874e"
EXPECTED_R2_CALLBACK_RECEIPT_SHA256 = (
    "489acfce54cbd0cd95c21a982f8d93b12185d7affd8b5626d08c24fc4917ce20"
)
EXPECTED_Q4_GRADE_RECEIPT_SHA256 = (
    "2751abfc7a68c33895be0a32d2da37bbc229dbd4de61c4bc9c71330f79a8f479"
)
EXPECTED_R1_RETRACTION_SHA256 = "8d4f3cae2f786fde26f2d437eb04ec6ee68493cda6aad70d4bd0a4d0b12c7485"


def validate_archive_paths(names: list[str]) -> None:
    """Reject duplicate or unsafe package paths before temporary extraction."""
    if len(names) != len(set(names)):
        raise ValueError("R1 package contains duplicate paths")
    for name in names:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"unsafe R1 package path: {name!r}")


def instantiate_r1_fp32(r1_zip: Path) -> dict[str, object]:
    """Load the exact R1 ZIP and prove its native members default to float32."""
    instantiation_start = time.perf_counter()
    if sha256_file(r1_zip) != EXPECTED_R1_ZIP_SHA256:
        raise ValueError("R1 ZIP digest mismatch")
    with zipfile.ZipFile(r1_zip) as archive:
        names = archive.namelist()
        validate_archive_paths(names)
        package = {name: archive.read(name) for name in names}
    if hashlib.sha256(package["v3/wnn_native.abi3.so"]).hexdigest() != EXPECTED_NATIVE_SHA256:
        raise ValueError("R1 native v3 library digest mismatch")
    if hashlib.sha256(package["weights_0.npz"]).hexdigest() != (
        "62b11349b4372beda7abbd35f7e006d2c967237070cbd5318c7a2f15bc0d1ec6"
    ):
        raise ValueError("R1 member 0 weight payload mismatch")
    if hashlib.sha256(package["weights_1.npz"]).hexdigest() != (
        "9f019d54a147296037b1bb2c6ad8b15b87070bde0402ff5426ce9c4804f52eaf"
    ):
        raise ValueError("R1 member 1 weight payload mismatch")

    r1_members = json.loads(package["members.json"])
    expected_members = [
        {
            "weights": "weights_0.npz",
            "coef": 0.5,
            "source": "vt_all_h390_ds_homogeneous_s3_r3@ep5",
        },
        {
            "weights": "weights_1.npz",
            "coef": 0.5,
            "source": "vt_all_h390_ds_homogeneous_s4_r3@ep5",
        },
    ]
    if r1_members != expected_members:
        raise ValueError("R1 member names, order, coefficients, or sources changed")

    with tempfile.TemporaryDirectory(prefix="wnn-r1-f32-package-") as temporary:
        package_dir = Path(temporary)
        for name, payload in package.items():
            relative = PurePosixPath(name)
            destination = package_dir.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        source_path = package_dir / "solution.py"
        source_bytes = source_path.read_bytes()
        source_text = source_bytes.decode("utf-8")
        if "wnn_native.Predictor(" not in source_text:
            raise ValueError("R1 solution does not use the expected Predictor constructor")
        if "precision.txt" in source_text or "precision=" in source_text:
            raise ValueError("R1 wrapper unexpectedly reads or sets explicit precision")

        os.environ["WNN_NATIVE_LEVEL"] = "v3"
        module_spec = importlib.util.spec_from_file_location(
            "wnn_r1_solution_actual_precision", source_path
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError("could not import the solution from the exact R1 ZIP")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model = module.PredictionModel()
        actual_precisions = [member.precision for member in model.members]
        if actual_precisions != ["f32", "f32"]:
            raise ValueError(f"R1 native precision assertion failed: {actual_precisions}")
        return {
            "r1_zip_sha256": EXPECTED_R1_ZIP_SHA256,
            "solution_py_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "native_v3_sha256": EXPECTED_NATIVE_SHA256,
            "precision_txt": package["precision.txt"].decode("utf-8").strip(),
            "wrapper_reads_precision_txt": False,
            "wrapper_constructor": "wnn_native.Predictor(weights, fast_activations=True)",
            "actual_member_precisions": actual_precisions,
            "actual_precision_assertion_passed": True,
            "instantiation_seconds": time.perf_counter() - instantiation_start,
            "member_weight_npz_sha256": [
                hashlib.sha256(package["weights_0.npz"]).hexdigest(),
                hashlib.sha256(package["weights_1.npz"]).hexdigest(),
            ],
        }


def grade(
    r1_zip: Path,
    r1_predictions_path: Path,
    r1_callback_receipt: Path,
    r1_retraction_receipt: Path,
    r2_zip: Path,
    r2_predictions_path: Path,
    r2_runtime_receipt: Path,
    r2_callback_receipt: Path,
    panel_path: Path,
    valid_parquet_path: Path,
    q4_grade_receipt: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Score both saved precision panels against the same streamed valid rows."""
    inputs = (
        r1_zip,
        r1_predictions_path,
        r1_callback_receipt,
        r1_retraction_receipt,
        r2_zip,
        r2_predictions_path,
        r2_runtime_receipt,
        r2_callback_receipt,
        panel_path,
        valid_parquet_path,
        q4_grade_receipt,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(r1_retraction_receipt) != EXPECTED_R1_RETRACTION_SHA256:
        raise ValueError("R1 precision-retraction receipt digest mismatch")
    if sha256_file(r1_predictions_path) != EXPECTED_R1_PREDICTIONS_SHA256:
        raise ValueError("preserved R1 prediction panel digest mismatch")
    if sha256_file(r2_predictions_path) != EXPECTED_R2_PREDICTIONS_SHA256:
        raise ValueError("R2 Q8 prediction panel digest mismatch")
    if sha256_file(panel_path) != EXPECTED_PANEL_SHA256:
        raise ValueError("shared feature panel digest mismatch")
    if sha256_file(r2_zip) != EXPECTED_R2_ZIP_SHA256:
        raise ValueError("R2 Q8 ZIP digest mismatch")
    if sha256_file(q4_grade_receipt) != EXPECTED_Q4_GRADE_RECEIPT_SHA256:
        raise ValueError("Q4 grade receipt digest mismatch")
    if sha256_file(r2_callback_receipt) != EXPECTED_R2_CALLBACK_RECEIPT_SHA256:
        raise ValueError("R2 callback receipt digest mismatch")
    with r1_callback_receipt.open() as source:
        r1_callback = json.load(source)
    with r2_runtime_receipt.open() as source:
        r2_runtime = json.load(source)
    with r2_callback_receipt.open() as source:
        r2_callback = json.load(source)
    with q4_grade_receipt.open() as source:
        q4_grade = json.load(source)
    with r1_retraction_receipt.open() as source:
        r1_retraction = json.load(source)
    if r1_retraction["actual_precision"] != "f32":
        raise ValueError("R1 retraction receipt does not declare actual FP32 precision")
    if r1_retraction["failed_route"] != "submission/solution_multi.py":
        raise ValueError("R1 precision-retraction receipt route mismatch")
    if r1_callback["predictions_sha256"] != EXPECTED_R1_PREDICTIONS_SHA256:
        raise ValueError("R1 callback receipt does not bind the preserved prediction payload")
    if r1_callback["zip_sha256"] != EXPECTED_R1_ZIP_SHA256:
        raise ValueError("R1 callback receipt package digest mismatch")
    if r1_callback["data_sha256"] != EXPECTED_PANEL_SHA256:
        raise ValueError("R1 callback receipt panel binding mismatch")
    if r1_callback["native_sha256"] != EXPECTED_NATIVE_SHA256:
        raise ValueError("R1 callback receipt native-library binding mismatch")
    if r2_runtime["predictions_sha256"] != EXPECTED_R2_PREDICTIONS_SHA256:
        raise ValueError("R2 runtime receipt does not bind the Q8 prediction payload")
    if r2_runtime["runtime_precision_assertion"]["passed"] is not True:
        raise ValueError("R2 Q8 runtime precision assertion did not pass")
    if r2_runtime["runtime_precision_assertion"]["expected"] != "q8vnni":
        raise ValueError("R2 runtime receipt is not for the Q8 VNNI precision")
    if (
        "members=2 precision=q8vnni"
        not in r2_runtime["runtime_precision_assertion"]["actual_captured_startup_line"]
    ):
        raise ValueError("R2 runtime receipt lacks the actual two-member Q8 startup line")
    if r2_runtime["zip_sha256"] != EXPECTED_R2_ZIP_SHA256:
        raise ValueError("R2 runtime receipt ZIP digest mismatch")
    if r2_runtime["native_library_sha256"] != EXPECTED_NATIVE_SHA256:
        raise ValueError("R2 runtime receipt native library digest mismatch")
    if r2_callback["zip_sha256"] != EXPECTED_R2_ZIP_SHA256:
        raise ValueError("R2 callback receipt package binding mismatch")
    if r2_callback["predictions_sha256"] != EXPECTED_R2_PREDICTIONS_SHA256:
        raise ValueError("R2 callback receipt prediction binding mismatch")
    if r2_callback["data_sha256"] != EXPECTED_PANEL_SHA256:
        raise ValueError("R2 callback receipt feature-panel binding mismatch")
    if r2_callback["native_sha256"] != EXPECTED_NATIVE_SHA256:
        raise ValueError("R2 callback receipt native-library binding mismatch")
    if sha256_file(r1_zip) != EXPECTED_R1_ZIP_SHA256:
        raise ValueError("R1 ZIP changed since the runtime check")
    if [row["sequence_id"] for row in r1_callback["panel"]] != EXPECTED_PANEL_GROUPS:
        raise ValueError("R1 sequence identity mismatch")
    if [row["sequence_id"] for row in r2_callback["panel"]] != EXPECTED_PANEL_GROUPS:
        raise ValueError("R2 sequence identity mismatch")
    if len(q4_grade["groups"]) != len(EXPECTED_PANEL_GROUPS):
        raise ValueError("Q4 receipt lacks the ten shared target/mask group bindings")

    runtime_assertion = instantiate_r1_fp32(r1_zip)
    with zipfile.ZipFile(r2_zip) as archive:
        r2_weight_hashes = [
            hashlib.sha256(archive.read(name)).hexdigest()
            for name in ("weights_0.npz", "weights_1.npz")
        ]
        r2_members = json.loads(archive.read("members.json"))
    if r2_weight_hashes != runtime_assertion["member_weight_npz_sha256"]:
        raise ValueError("R1 and R2 member NPZ payloads are not byte-identical")
    if r2_members != [
        {
            "weights": "weights_0.npz",
            "coef": 0.5,
            "source": "vt_all_h390_ds_homogeneous_s3_r3@ep5",
        },
        {
            "weights": "weights_1.npz",
            "coef": 0.5,
            "source": "vt_all_h390_ds_homogeneous_s4_r3@ep5",
        },
    ]:
        raise ValueError("R1 and R2 member specifications differ")
    r1_predictions = np.load(r1_predictions_path, mmap_mode="r", allow_pickle=False)
    r2_predictions = np.load(r2_predictions_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (10, 20_000, 2)
    if r1_predictions.shape != expected_shape or r2_predictions.shape != expected_shape:
        raise ValueError("precision prediction panel shape mismatch")
    if r1_predictions.dtype != np.float32 or r2_predictions.dtype != np.float32:
        raise ValueError("precision prediction panel dtype mismatch")

    valid = pq.ParquetFile(valid_parquet_path)
    if valid.metadata.num_row_groups < 10:
        raise ValueError("valid parquet has fewer than ten rowgroups")
    r1_accumulator = GlobalAccumulator()
    r2_accumulator = GlobalAccumulator()
    r1_selected: list[np.ndarray] = []
    r2_selected: list[np.ndarray] = []
    target_selected: list[np.ndarray] = []
    group_records: list[dict[str, object]] = []
    started = time.perf_counter()

    for group_index, expected_sequence in enumerate(EXPECTED_PANEL_GROUPS):
        labels = valid.read_row_group(
            group_index,
            columns=["t0", "t1", "is_scored", "need_prediction", "seq_ix"],
        )
        if labels.num_rows != 20_000:
            raise ValueError(f"valid rowgroup {group_index} does not contain 20,000 rows")
        sequence_ids = labels["seq_ix"].to_numpy(zero_copy_only=False)
        if not np.all(sequence_ids == expected_sequence):
            raise ValueError(f"valid sequence mismatch at rowgroup {group_index}")
        targets = np.column_stack(
            [labels[name].to_numpy(zero_copy_only=False) for name in ("t0", "t1")]
        ).astype(np.float32, copy=False)
        need = labels["need_prediction"].to_numpy(zero_copy_only=False)
        is_scored = labels["is_scored"].to_numpy(zero_copy_only=False)
        mask = np.asarray(need & is_scored, dtype=bool)
        if mask[:99].any():
            raise ValueError(f"official mask includes warmup rows at rowgroup {group_index}")
        target_hash = sha256_array(targets)
        mask_hash = sha256_array(mask)
        prior_group = q4_grade["groups"][group_index]
        if (target_hash, mask_hash) != (
            prior_group["targets_sha256"],
            prior_group["official_mask_sha256"],
        ):
            raise ValueError(f"target or official-mask digest changed at rowgroup {group_index}")

        r1 = np.asarray(r1_predictions[group_index])
        r2 = np.asarray(r2_predictions[group_index])
        r1_accumulator.add(targets, r1, mask)
        r2_accumulator.add(targets, r2, mask)
        r1_selected.append(r1[mask].copy())
        r2_selected.append(r2[mask].copy())
        target_selected.append(targets[mask].copy())
        r1_group = GlobalAccumulator()
        r2_group = GlobalAccumulator()
        r1_group.add(targets, r1, mask)
        r2_group.add(targets, r2, mask)
        group_records.append(
            {
                "sequence_id": expected_sequence,
                "selected_rows": int(mask.sum()),
                "targets_sha256": target_hash,
                "official_mask_sha256": mask_hash,
                "r1_fp32_wp": r1_group.result()["wp"],
                "r2_q8_wp": r2_group.result()["wp"],
                "delta_q8_minus_fp32_wp": r2_group.result()["wp"] - r1_group.result()["wp"],
            }
        )

    r1_all = np.concatenate(r1_selected, axis=0).astype(np.float64)
    r2_all = np.concatenate(r2_selected, axis=0).astype(np.float64)
    targets_all = np.concatenate(target_selected, axis=0).astype(np.float64)
    fidelity: list[dict[str, float | str]] = []
    for target_index, target_name in enumerate(("t0", "t1")):
        f32_values = r1_all[:, target_index]
        q8_values = r2_all[:, target_index]
        y = np.clip(targets_all[:, target_index], -2.0, 2.0)
        weights = np.abs(y)
        fidelity.append(
            {
                "target": target_name,
                "prediction_pearson_raw": float(np.corrcoef(f32_values, q8_values)[0, 1]),
                "prediction_pearson_official_clip_weighted": weighted_correlation(
                    np.clip(f32_values, -2.0, 2.0),
                    np.clip(q8_values, -2.0, 2.0),
                    weights,
                ),
                "prediction_mae": float(np.mean(np.abs(q8_values - f32_values))),
                "prediction_rmse": float(np.sqrt(np.mean(np.square(q8_values - f32_values)))),
                "prediction_max_abs_delta": float(np.max(np.abs(q8_values - f32_values))),
            }
        )

    fp32_result = r1_accumulator.result()
    q8_result = r2_accumulator.result()
    q4_q8_result = q4_grade["scores"]["q4_rounded_then_q8"]
    q4_q8_delta = {name: q4_q8_result[name] - q8_result[name] for name in ("t0", "t1", "wp")}
    for name in ("t0", "t1", "wp"):
        if abs(q4_grade["scores"]["q8"][name] - q8_result[name]) > 1e-12:
            raise AssertionError(f"Q4 and FP32/Q8 graders disagree on Q8 {name}")
    if abs(q4_q8_delta["wp"] - q4_grade["scores"]["delta_q4_minus_q8"]["wp"]) > 1e-12:
        raise AssertionError("Q4/Q8 score link does not match the sealed Q4 grade")

    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "wnn.h390-fp32-versus-q8-official-panel.v1",
        "claim_scope": (
            "Precision-fidelity comparison on an in-sample full-data H390 pair; this panel is not "
            "held-out generalization evidence. R1's precision.txt said q8vnni, but its actual "
            "solution ignored that file and the exact ZIP instantiated as native f32."
        ),
        "precision_retraction": {
            "path": str(r1_retraction_receipt.resolve()),
            "sha256": sha256_file(r1_retraction_receipt),
            "r1_precision_txt_was_stale": True,
        },
        "r1_actual_runtime_assertion": runtime_assertion,
        "r1_panel": {
            "zip_path": str(r1_zip.resolve()),
            "zip_sha256": sha256_file(r1_zip),
            "predictions_path": str(r1_predictions_path.resolve()),
            "predictions_sha256": sha256_file(r1_predictions_path),
            "callback_receipt_sha256": sha256_file(r1_callback_receipt),
            "precision_label_in_old_callback_receipt_retracted": True,
        },
        "r2_panel": {
            "zip_path": str(r2_zip.resolve()),
            "zip_sha256": sha256_file(r2_zip),
            "predictions_path": str(r2_predictions_path.resolve()),
            "predictions_sha256": sha256_file(r2_predictions_path),
            "runtime_receipt_sha256": sha256_file(r2_runtime_receipt),
            "callback_receipt_sha256": sha256_file(r2_callback_receipt),
            "actual_precision": "q8vnni",
        },
        "evaluation": {
            "metric_id": "global_wp_clip2_v1",
            "official_mask": "need_prediction AND is_scored",
            "official_clip": [-2.0, 2.0],
            "weights": "absolute clipped target magnitude",
            "rows_selected": fp32_result["selected_rows"],
            "group_ids": EXPECTED_PANEL_GROUPS,
            "feature_panel_path": str(panel_path.resolve()),
            "feature_panel_sha256": sha256_file(panel_path),
            "full_valid_parquet_sha256": None,
            "valid_parquet_bytes": valid_parquet_path.stat().st_size,
            "valid_parquet_mtime_ns": valid_parquet_path.stat().st_mtime_ns,
            "read_columns_only": ["t0", "t1", "is_scored", "need_prediction", "seq_ix"],
            "target_and_mask_hashes_match_q4_grade": True,
        },
        "scores": {
            "r1_fp32": fp32_result,
            "r2_q8": q8_result,
            "delta_q8_minus_fp32": {
                key: q8_result[key] - fp32_result[key] for key in ("t0", "t1", "wp")
            },
            "q4_rounded_then_q8": q4_q8_result,
            "delta_q4_rounded_q8_minus_q8": q4_q8_delta,
        },
        "prediction_fidelity": {
            "selected_rows": int(r1_all.shape[0]),
            "per_target": fidelity,
        },
        "groups": group_records,
        "execution": {
            "streamed_grade_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
        },
    }
    output_path = output_dir / "FP32_Q8_GRADE_RECEIPT_R1.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    """Run the small actual-runtime check and streamed official-mask comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-zip", type=Path, required=True)
    parser.add_argument("--r1-predictions", type=Path, required=True)
    parser.add_argument("--r1-callback-receipt", type=Path, required=True)
    parser.add_argument("--r1-retraction-receipt", type=Path, required=True)
    parser.add_argument("--r2-zip", type=Path, required=True)
    parser.add_argument("--r2-predictions", type=Path, required=True)
    parser.add_argument("--r2-runtime-receipt", type=Path, required=True)
    parser.add_argument("--r2-callback-receipt", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--valid-parquet", type=Path, required=True)
    parser.add_argument("--q4-grade-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = grade(
        args.r1_zip,
        args.r1_predictions,
        args.r1_callback_receipt,
        args.r1_retraction_receipt,
        args.r2_zip,
        args.r2_predictions,
        args.r2_runtime_receipt,
        args.r2_callback_receipt,
        args.panel,
        args.valid_parquet,
        args.q4_grade_receipt,
        args.output_dir,
    )
    print(
        json.dumps(
            {"scores": result["scores"], "prediction_fidelity": result["prediction_fidelity"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
