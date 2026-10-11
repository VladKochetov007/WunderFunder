"""Compare Q4-grid rounded weights and Q8 weights through the same Q8 runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
from utils import FEATURE_COLUMNS  # noqa: E402
from wnn.gmetric import GlobalAccumulator  # noqa: E402

EXPECTED_Q8_ZIP_SHA256 = "3420ea5e6318e15b6340582125959ed7489983b17338f399c748965cb42c4a4a"
EXPECTED_Q8_PREDICTIONS_SHA256 = "682376c1f056efcd4a011d5e5edd5b3de585f1337e5f93313c3f1bf89a2e04c5"
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
EXPECTED_FEATURE_HASHES = [
    "bfec7e22df30437429f794156f75ce6f6843c882149ec6e6e7d0c57ead588f3e",
    "da1552d094eee4cb347ee88c14613885802e54c3f7052e758834b36f7ad03a5f",
    "016afe4a6d9bcf3af2ec7961295bd3106530f29a760d297d5a6fbb1390063c58",
    "4558b05652d6f9054170613d556ecd8e12fdb35b10d9027a9456e5b11b161c26",
    "05dbac3d81d7241a5761e009d0c5a11b4b152f1cf7890df75a6eac6776051f4b",
    "09e108511c35a5266f967bad67151321d7e8f16d2cee5cece69886392fd3216e",
    "a42ccf8b2e8ead95c55b6d8f785d5e25285ec2b65f56571021eca0f2ff9e3e63",
    "5b587d321a28146b43a5400cdc2450372af08041c21c7763086168b7c9966b21",
    "20947296e63f58d5210c4c6ae88442f04688bf79d66ff7bc624c283493f41175",
    "466d7cc953803912710c0e731aedd9de12446ea3f5e455502393461e242e9419",
]
EXPECTED_NEED_HASH = "838c9b675ba71c3e713fb0279acad0eb1b09e89a84efa3bc6f84ecf91dafb76b"
EXPECTED_NATIVE_SHA256 = "ce179a68b7cd321368cc9ab73cdf1b79973afcdb8266ef8f6d7e613bb6f9874e"


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash a C-order tensor payload."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def validate_archive_paths(names: list[str]) -> None:
    """Fail closed on duplicate or traversal paths before temporary extraction."""
    if len(names) != len(set(names)):
        raise ValueError("Q4 archive contains duplicate paths")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"unsafe package path: {name!r}")


def extract_package(archive_path: Path, destination: Path) -> Path:
    """Extract a validated ZIP to a fresh temporary package directory."""
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        validate_archive_paths(names)
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            output = destination.joinpath(*relative.parts)
            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(archive.read(info.filename))
    return destination


def load_weights(package_dir: Path, member_names: list[str]) -> list[dict[str, np.ndarray]]:
    """Load the exact NPZ members as contiguous float32 arrays for the Rust API."""
    models = []
    for name in member_names:
        with np.load(package_dir / name, allow_pickle=False) as values:
            models.append(
                {key: np.ascontiguousarray(values[key], dtype=np.float32) for key in values.files}
            )
    return models


def load_runtime(package_dir: Path, member_names: list[str]) -> object:
    """Create the native Q8 model from the package's exact v3 library."""
    native_path = package_dir / "v3" / "wnn_native.abi3.so"
    native_sha = sha256_file(native_path)
    if native_sha != EXPECTED_NATIVE_SHA256:
        raise ValueError(f"unexpected native library digest: {native_sha}")
    sys.path.insert(0, str(native_path.parent))
    native = importlib.import_module("wnn_native")
    with (package_dir / "members.json").open() as source:
        members = json.load(source)
    weights = load_weights(package_dir, member_names)
    model = native.EnsemblePredictor(
        weights,
        [np.float32(member["coef"]) for member in members],
        fast_activations=True,
        precision="q8vnni",
        budget_s=None,
    )
    if model.precision != "q8vnni" or model.active_members != 2:
        raise ValueError(
            f"runtime mismatch: precision={model.precision}, active={model.active_members}"
        )
    return model


def make_feature_matrix(table: object) -> np.ndarray:
    """Build the scorer's fixed 112-feature float32 row matrix from one group."""
    matrix = np.empty((table.num_rows, len(FEATURE_COLUMNS)), dtype=np.float32)
    for column_index, name in enumerate(FEATURE_COLUMNS):
        matrix[:, column_index] = table[name].to_numpy(zero_copy_only=False)
    return matrix


def weighted_correlation(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted Pearson correlation in float64 for the selected rows."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    if total <= 0:
        return 0.0
    mean_x = float(np.sum(w * x) / total)
    mean_y = float(np.sum(w * y) / total)
    dx = x - mean_x
    dy = y - mean_y
    covariance = float(np.sum(w * dx * dy) / total)
    variance_x = float(np.sum(w * dx * dx) / total)
    variance_y = float(np.sum(w * dy * dy) / total)
    if variance_x <= 1e-16 or variance_y <= 1e-16:
        return 0.0
    return covariance / np.sqrt(variance_x * variance_y)


def compare_predictions(
    q8_parts: list[np.ndarray], q4_parts: list[np.ndarray], target_parts: list[np.ndarray]
) -> dict[str, object]:
    """Summarize precision fidelity on the official selected rows."""
    q8 = np.concatenate(q8_parts, axis=0).astype(np.float64)
    q4 = np.concatenate(q4_parts, axis=0).astype(np.float64)
    targets = np.concatenate(target_parts, axis=0).astype(np.float64)
    records: list[dict[str, float]] = []
    for target_index, target_name in enumerate(("t0", "t1")):
        p8 = q8[:, target_index]
        p4 = q4[:, target_index]
        clipped8 = np.clip(p8, -2.0, 2.0)
        clipped4 = np.clip(p4, -2.0, 2.0)
        y = np.clip(targets[:, target_index], -2.0, 2.0)
        weights = np.abs(y)
        raw_corr = float(np.corrcoef(p8, p4)[0, 1])
        records.append(
            {
                "target": target_name,
                "prediction_pearson_raw": raw_corr,
                "prediction_pearson_official_clip_weighted": weighted_correlation(
                    clipped8, clipped4, weights
                ),
                "prediction_mae": float(np.mean(np.abs(p4 - p8))),
                "prediction_rmse": float(np.sqrt(np.mean(np.square(p4 - p8)))),
                "prediction_max_abs_delta": float(np.max(np.abs(p4 - p8))),
            }
        )
    return {
        "selected_rows": int(targets.shape[0]),
        "per_target": records,
    }


def grade(
    q4_zip: Path,
    q4_build_receipt: Path,
    q8_predictions_path: Path,
    q8_callback_receipt: Path,
    panel_path: Path,
    valid_parquet_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Replay the Q4-round pair and score against the reused Q8 output."""
    for path in (
        q4_zip,
        q4_build_receipt,
        q8_predictions_path,
        q8_callback_receipt,
        panel_path,
        valid_parquet_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(panel_path) != EXPECTED_PANEL_SHA256:
        raise ValueError("feature panel digest mismatch")
    if sha256_file(q8_predictions_path) != EXPECTED_Q8_PREDICTIONS_SHA256:
        raise ValueError("Q8 prediction digest mismatch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with q4_build_receipt.open() as source:
        build_receipt = json.load(source)
    if build_receipt["source"]["zip_sha256"] != EXPECTED_Q8_ZIP_SHA256:
        raise ValueError("Q4 build receipt is not bound to the pinned Q8 pair ZIP")
    if build_receipt["variant"]["zip_sha256"] != sha256_file(q4_zip):
        raise ValueError("Q4 build receipt does not match variant ZIP")
    with q8_callback_receipt.open() as source:
        callback_receipt = json.load(source)
    if callback_receipt["zip_sha256"] != EXPECTED_Q8_ZIP_SHA256:
        raise ValueError("Q8 callback receipt ZIP binding mismatch")
    if callback_receipt["predictions_sha256"] != EXPECTED_Q8_PREDICTIONS_SHA256:
        raise ValueError("Q8 callback receipt prediction binding mismatch")
    if callback_receipt["data_sha256"] != EXPECTED_PANEL_SHA256:
        raise ValueError("Q8 callback receipt feature panel binding mismatch")
    if callback_receipt["native_sha256"] != EXPECTED_NATIVE_SHA256:
        raise ValueError("Q8 callback receipt native library binding mismatch")
    callback_groups = callback_receipt["panel"]
    if [group["sequence_id"] for group in callback_groups] != EXPECTED_PANEL_GROUPS:
        raise ValueError("Q8 callback receipt sequence IDs mismatch")
    if [group["features_sha256"] for group in callback_groups] != EXPECTED_FEATURE_HASHES:
        raise ValueError("Q8 callback receipt feature digests mismatch")
    if any(group["need_sha256"] != EXPECTED_NEED_HASH for group in callback_groups):
        raise ValueError("Q8 callback receipt need-prediction digests mismatch")

    q8_predictions = np.load(q8_predictions_path, mmap_mode="r", allow_pickle=False)
    if q8_predictions.shape != (10, 20_000, 2) or q8_predictions.dtype != np.float32:
        raise ValueError("Q8 prediction panel shape/dtype mismatch")
    panel = pq.ParquetFile(panel_path)
    valid = pq.ParquetFile(valid_parquet_path)
    if panel.metadata.num_row_groups != 10 or valid.metadata.num_row_groups < 10:
        raise ValueError("unexpected panel or valid parquet row-group count")

    output_dir.mkdir(parents=True, exist_ok=True)
    q4_prediction_path = output_dir / "q4rounded_q8backend_predictions.npy"
    if q4_prediction_path.exists():
        raise FileExistsError(q4_prediction_path)
    q4_predictions = np.lib.format.open_memmap(
        q4_prediction_path, mode="w+", dtype=np.float32, shape=(10, 20_000, 2)
    )
    q8_accumulator = GlobalAccumulator()
    q4_accumulator = GlobalAccumulator()
    q8_selected: list[np.ndarray] = []
    q4_selected: list[np.ndarray] = []
    target_selected: list[np.ndarray] = []
    group_records: list[dict[str, object]] = []
    panel_group_records: list[dict[str, object]] = []
    start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="wnn-q4-pkg-") as temporary:
        with zipfile.ZipFile(q4_zip) as archive:
            validate_archive_paths(archive.namelist())
            package_dir = Path(temporary) / "pkg"
            package_dir.mkdir()
            extract_package(q4_zip, package_dir)
        model = load_runtime(package_dir, ["weights_0.npz", "weights_1.npz"])
        for group_index, expected_sequence in enumerate(EXPECTED_PANEL_GROUPS):
            panel_table = panel.read_row_group(
                group_index, columns=["seq_ix", "need_prediction", *FEATURE_COLUMNS]
            )
            feature_matrix = make_feature_matrix(panel_table)
            feature_digest = sha256_array(feature_matrix)
            if feature_digest != EXPECTED_FEATURE_HASHES[group_index]:
                raise ValueError(f"panel feature digest mismatch at group {group_index}")
            panel_need = panel_table["need_prediction"].to_numpy(zero_copy_only=False)
            need_digest = sha256_array(panel_need)
            if need_digest != EXPECTED_NEED_HASH:
                raise ValueError(f"panel need mask digest mismatch at group {group_index}")
            panel_seq = panel_table["seq_ix"].to_numpy(zero_copy_only=False)
            if panel_table.num_rows != 20_000 or not np.all(panel_seq == expected_sequence):
                raise ValueError(f"panel sequence identity mismatch at group {group_index}")
            panel_group_records.append(
                {
                    "panel_group": group_index,
                    "sequence_id": expected_sequence,
                    "feature_sha256": feature_digest,
                    "need_prediction_sha256": need_digest,
                }
            )

            model.reset()
            for row_index, feature_row in enumerate(feature_matrix):
                q4_predictions[group_index, row_index] = model.step(feature_row)

            labels_table = valid.read_row_group(
                group_index,
                columns=["t0", "t1", "is_scored", "need_prediction", "seq_ix"],
            )
            if labels_table.num_rows != 20_000:
                raise ValueError(f"valid row-group length mismatch at {group_index}")
            label_seq = labels_table["seq_ix"].to_numpy(zero_copy_only=False)
            if not np.all(label_seq == expected_sequence):
                raise ValueError(f"valid/panel sequence mismatch at group {group_index}")
            label_need = labels_table["need_prediction"].to_numpy(zero_copy_only=False)
            is_scored = labels_table["is_scored"].to_numpy(zero_copy_only=False)
            if not np.array_equal(label_need, panel_need):
                raise ValueError(f"valid/panel need mask mismatch at group {group_index}")
            targets = np.column_stack(
                [
                    labels_table["t0"].to_numpy(zero_copy_only=False),
                    labels_table["t1"].to_numpy(zero_copy_only=False),
                ]
            ).astype(np.float32, copy=False)
            official_mask = np.asarray(label_need & is_scored, dtype=bool)
            if official_mask[:99].any():
                raise ValueError(f"official mask includes warmup rows at group {group_index}")

            q8_values = np.asarray(q8_predictions[group_index])
            q4_values = np.asarray(q4_predictions[group_index])
            q8_accumulator.add(targets, q8_values, official_mask)
            q4_accumulator.add(targets, q4_values, official_mask)
            q8_selected.append(q8_values[official_mask].copy())
            q4_selected.append(q4_values[official_mask].copy())
            target_selected.append(targets[official_mask].copy())
            q8_group = GlobalAccumulator()
            q4_group = GlobalAccumulator()
            q8_group.add(targets, q8_values, official_mask)
            q4_group.add(targets, q4_values, official_mask)
            group_records.append(
                {
                    "sequence_id": expected_sequence,
                    "selected_rows": int(official_mask.sum()),
                    "targets_sha256": sha256_array(targets),
                    "official_mask_sha256": sha256_array(official_mask),
                    "q8_wp": q8_group.result()["wp"],
                    "q4_rounded_q8_wp": q4_group.result()["wp"],
                    "delta_wp": q4_group.result()["wp"] - q8_group.result()["wp"],
                }
            )
    q4_predictions.flush()
    del q4_predictions

    q8_result = q8_accumulator.result()
    q4_result = q4_accumulator.result()
    compare = compare_predictions(q8_selected, q4_selected, target_selected)
    result: dict[str, object] = {
        "schema": "wnn.q4-rounded-versus-q8-official-panel.v1",
        "claim_scope": (
            "Precision-fidelity diagnostic on an in-sample full-data H390 pair. The valid rows "
            "are not held-out generalization evidence for these all-data checkpoints."
        ),
        "metric_id": "global_wp_clip2_v1",
        "evaluation": {
            "official_mask": "need_prediction AND is_scored",
            "official_clip": [-2.0, 2.0],
            "weights": "absolute clipped target magnitude",
            "valid_parquet_path": str(valid_parquet_path.resolve()),
            "valid_parquet_bytes": valid_parquet_path.stat().st_size,
            "valid_parquet_mtime_ns": valid_parquet_path.stat().st_mtime_ns,
            "full_valid_parquet_sha256": None,
            "data_identity_note": (
                "The 5.6 GB parquet was not fully hashed; physical rowgroups 0..9, sequence IDs, "
                "need masks, and target/mask rowgroup digests are recorded below."
            ),
            "panel_path": str(panel_path.resolve()),
            "panel_sha256": sha256_file(panel_path),
            "panel_group_sequence_ids": EXPECTED_PANEL_GROUPS,
            "panel_feature_columns": len(FEATURE_COLUMNS),
            "panel_callback_receipt_sha256": sha256_file(q8_callback_receipt),
            "q8_predictions_path": str(q8_predictions_path.resolve()),
            "q8_predictions_sha256": sha256_file(q8_predictions_path),
            "q4_predictions_path": q4_prediction_path.name,
            "q4_predictions_sha256": sha256_file(q4_prediction_path),
            "q8_variant_zip_sha256": EXPECTED_Q8_ZIP_SHA256,
            "q4_variant_zip_sha256": sha256_file(q4_zip),
            "q4_build_receipt_sha256": sha256_file(q4_build_receipt),
            "native_backend": "v3/wnn_native.abi3.so",
            "native_backend_sha256": EXPECTED_NATIVE_SHA256,
            "runtime_precision_asserted": "q8vnni",
            "runtime_members_asserted": 2,
        },
        "scores": {
            "q8": q8_result,
            "q4_rounded_then_q8": q4_result,
            "delta_q4_minus_q8": {
                name: q4_result[name] - q8_result[name] for name in ("t0", "t1", "wp")
            },
        },
        "prediction_fidelity": compare,
        "groups": group_records,
        "panel_groups": panel_group_records,
        "execution": {
            "elapsed_seconds": time.perf_counter() - start,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "prediction_output_bytes": q4_prediction_path.stat().st_size,
        },
    }
    receipt_path = output_dir / "Q4_GRADE_RECEIPT_R1.json"
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    """Run the bounded streamed label/mask grade."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q4-zip", type=Path, required=True)
    parser.add_argument("--q4-build-receipt", type=Path, required=True)
    parser.add_argument("--q8-predictions", type=Path, required=True)
    parser.add_argument("--q8-callback-receipt", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--valid-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = grade(
        arguments.q4_zip,
        arguments.q4_build_receipt,
        arguments.q8_predictions,
        arguments.q8_callback_receipt,
        arguments.panel,
        arguments.valid_parquet,
        arguments.output_dir,
    )
    print(
        json.dumps(
            {"scores": result["scores"], "prediction_fidelity": result["prediction_fidelity"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
