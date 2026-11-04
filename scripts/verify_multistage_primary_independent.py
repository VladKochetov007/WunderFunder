from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

PRIMARY = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
ROWS = 3_528_273
MASK_KEYS = (
    "membership_sha256",
    "mask_sha256",
    "selected_row_index_sha256",
    "sequence_order",
    "row_order",
    "selected_rows",
    "per_sequence_selected_rows",
    "mask",
    "mask_packed_bitorder",
)


def sha256(path: Path) -> str:
    """Hash a file with bounded working memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, limit: int = 32_000_000) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("verification input must be a bounded regular JSON file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("verification JSON must contain an object")
    return value


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def independent_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Compute two-pass weighted centered covariance directly, without project scorers."""
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("metric inputs must have equal (rows,2) shapes")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("metric inputs must be finite")
    y = np.clip(target.astype(np.float64), -2.0, 2.0)
    raw = prediction.astype(np.float64)
    p = np.clip(raw, -2.0, 2.0)
    w = np.abs(y)
    total = w.sum(axis=0)
    correlations, errors = [], []
    for column in range(2):
        weight = float(total[column])
        if weight < 1e-8:
            correlations.append(0.0)
            errors.append(0.0)
            continue
        mean_y = float(np.sum(w[:, column] * y[:, column]) / weight)
        mean_p = float(np.sum(w[:, column] * p[:, column]) / weight)
        dy, dp = y[:, column] - mean_y, p[:, column] - mean_p
        variance_y = float(np.sum(w[:, column] * dy * dy))
        variance_p = float(np.sum(w[:, column] * dp * dp))
        covariance = float(np.sum(w[:, column] * dy * dp))
        correlation = (
            0.0
            if math.sqrt(max(variance_y, 0.0) / weight) <= 1e-8
            or math.sqrt(max(variance_p, 0.0) / weight) <= 1e-8
            else covariance / math.sqrt(variance_y * variance_p)
        )
        correlations.append(float(np.clip(correlation, -1.0, 1.0)))
        errors.append(float(np.sum(w[:, column] * np.square(raw[:, column] - y[:, column]))))
    overall_weight = float(total.sum())
    return {
        "t0": correlations[0],
        "t1": correlations[1],
        "wp": sum(correlations) / 2,
        "wmse_t0": errors[0] / total[0] if total[0] else 0.0,
        "wmse_t1": errors[1] / total[1] if total[1] else 0.0,
        "wmse": sum(errors) / overall_weight if overall_weight else 0.0,
        "selected_rows": len(target),
    }


def reconstruct_block(members: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Accumulate raw members in declared order using FP64, then cast once to FP32."""
    if (
        not members
        or len(members) != len(weights)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or abs(float(weights.sum()) - 1.0) > 1e-10
    ):
        raise ValueError("invalid fixed simplex or member count")
    result = np.zeros(members[0].shape, dtype=np.float64)
    for member, weight in zip(members, weights, strict=True):
        if member.shape != result.shape or not np.isfinite(member).all():
            raise ValueError("member shapes or finiteness differ")
        result += member.astype(np.float64) * float(weight)
    return result.astype(np.float32)


def _array(path: Path, digest: object) -> np.ndarray:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 32_000_000
        or sha256(path) != digest
    ):
        raise ValueError("prediction bytes exceed bounds or differ from their receipt")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (ROWS, 2) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("prediction dtype, shape or finiteness is invalid")
    return array


def _footer(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        trailer = stream.read(8)
        size = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or size > 64_000_000 or size > path.stat().st_size - 8:
            raise ValueError("invalid or oversized Parquet footer")
        stream.seek(-8 - size, os.SEEK_END)
        digest = hashlib.sha256(stream.read(size + 8)).hexdigest()
    return {"bytes": path.stat().st_size, "footer_bytes": size, "footer_sha256": digest}


def read_primary_labels(
    data_root: Path, manifest: dict[str, object]
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    """Read only labels/mask in the frozen official row-group order after warmup99."""
    import pyarrow.parquet as parquet

    groups = manifest["evaluation"]["valid"]
    if (
        manifest.get("schema") != "wnn.chronological-primary.v1"
        or len(manifest["fit"]["train"]) != 10325
        or manifest["fit"]["valid"] != []
        or not isinstance(groups, list)
        or len(groups) != 1873
        or len(set(groups)) != 1873
    ):
        raise ValueError("manifest does not preserve the official10325/0/1873 boundary")
    path = data_root / "datasets/valid.parquet"
    identity = _footer(path)
    if any(
        manifest["sources"]["datasets/valid.parquet"].get(key) != value
        for key, value in identity.items()
    ):
        raise ValueError("actual validation Parquet footer/size differs from frozen manifest")
    source = parquet.ParquetFile(path)
    if set(groups) != set(range(source.num_row_groups)):
        raise ValueError("primary manifest does not cover all validation row groups")
    targets = np.empty((ROWS, 2), dtype=np.float32)
    masks, indices, labels = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    offset, counts = 0, []
    for group in groups:
        table = source.read_row_group(group, columns=["t0", "t1", "is_scored"], use_threads=False)
        if table.num_rows != 20_000 or any(
            table.column(name).null_count for name in ("t0", "t1", "is_scored")
        ):
            raise ValueError("official sequence shape or nullability changed")
        mask = table.column("is_scored").to_numpy(zero_copy_only=False).astype(bool)
        mask[:99] = False
        selected_indices = np.flatnonzero(mask).astype("<u4")
        selected = np.column_stack(
            [table.column(name).to_numpy(zero_copy_only=False) for name in ("t0", "t1")]
        ).astype(np.float32)[mask]
        stop = offset + len(selected)
        if stop > ROWS or not np.isfinite(selected).all():
            raise ValueError("selected validation labels exceed count or contain nonfinite values")
        targets[offset:stop] = selected
        masks.update(np.packbits(mask, bitorder="little").tobytes())
        indices.update(np.asarray([group], dtype="<i8").tobytes())
        indices.update(selected_indices.tobytes())
        labels.update(selected.astype("<f4", copy=False).tobytes())
        counts.append({"row_group": group, "selected_rows": len(selected)})
        offset = stop
    if offset != ROWS:
        raise ValueError("official scored mask does not select3,528,273rows")
    members = [["valid", group] for group in groups]
    mask_receipt = {
        "membership_sha256": hashlib.sha256(
            json.dumps(members, separators=(",", ":")).encode()
        ).hexdigest(),
        "mask_sha256": masks.hexdigest(),
        "selected_row_index_sha256": indices.hexdigest(),
        "sequence_order": "official primary evaluation membership order; source then row_group",
        "row_order": "ascending row_index among is_scored rows after warmup row 99",
        "selected_rows": offset,
        "per_sequence_selected_rows": counts,
        "mask": "is_scored AND row_index>=99",
        "mask_packed_bitorder": "little",
    }
    return (
        targets,
        mask_receipt,
        identity
        | {
            "selected_labels_sha256": labels.hexdigest(),
            "columns_read": ["t0", "t1", "is_scored"],
            "full_file_sha256_recomputed": False,
        },
    )


def _compare(actual: dict[str, float | int], expected: dict[str, object]) -> float:
    if expected.get("selected_rows", ROWS) != actual["selected_rows"]:
        raise ValueError("metric row counts differ")
    maximum = 0.0
    for key, value in actual.items():
        if key == "selected_rows":
            continue
        reference = expected[key]
        if not isinstance(reference, (int, float)) or not math.isfinite(reference):
            raise ValueError("reported metric is not finite numeric data")
        delta = abs(float(value) - reference)
        maximum = max(maximum, delta)
        if delta > 1e-10:
            raise ValueError(f"independent metric differs beyond1e-10 for {key}: {delta}")
    return maximum


def verify(report_path: Path, data_root: Path) -> dict[str, object]:
    """Verify nine frozen prediction outputs and their arithmetic, not training or transfer."""
    report = _read(report_path)
    request_path = Path(report["score_request_path"])
    request = _read(request_path)
    if (
        report.get("status") != "complete"
        or sha256(request_path) != report["score_request_sha256"]
        or report["primary_split_sha256"] != PRIMARY
    ):
        raise ValueError("primary report/request binding changed")
    split_path = Path(report["primary_split_path"])
    if sha256(split_path) != PRIMARY:
        raise ValueError("official primary manifest changed")
    manifest = _read(split_path)
    registry_path = Path(report["portable_data_registry"]["path"])
    registry = _read(registry_path)
    if sha256(registry_path) != report["portable_data_registry"]["sha256"]:
        raise ValueError("portable registry changed")
    source_root = Path(report["source6"]["root"])
    source_files = {
        path.relative_to(source_root).as_posix(): sha256(path)
        for path in sorted((source_root / "src").rglob("*.py"))
    }
    source_files["scripts/predict_sequence.py"] = sha256(
        source_root / "scripts/predict_sequence.py"
    )
    if (
        _canonical(source_files) != report["source6"]["source_sha256"]
        or sha256(source_root / "scripts/score_sequence.py")
        != report["source6"]["score_script_sha256"]
    ):
        raise ValueError("frozen Source6 files changed")
    targets, mask, source_identity = read_primary_labels(data_root, manifest)
    source_identity["declared_full_file_sha256"] = registry["sources"]["datasets/valid.parquet"][
        "sha256"
    ]
    if report["evaluation"]["mask_and_order"] != mask or request["primary_split"][
        "evaluation_members"
    ] != [["valid", group] for group in manifest["evaluation"]["valid"]]:
        raise ValueError("report membership/mask differs from independent raw-label extraction")
    blend_path = report_path.parent / "blend_request.json"
    blend = _read(blend_path)
    if blend["score_request_sha256"] != sha256(request_path):
        raise ValueError("member receipt index differs from the report request")
    candidates, rows = request["candidates"], report["members"]
    if len(candidates) != 8 or len(rows) != 8 or len(blend["members"]) != 8:
        raise ValueError("independent verification requires exactly8frozen members")
    weights = np.asarray([candidate["weight"] for candidate in candidates], dtype=np.float64)
    if (
        weights.tolist() != report["stage4_weights"]
        or _canonical(weights.tolist()) != request["weight_vector_sha256"]
    ):
        raise ValueError("frozen weights changed")
    arrays, results = [], []
    file_bindings = {
        str(path): sha256(path)
        for path in (
            report_path,
            request_path,
            split_path,
            registry_path,
            blend_path,
            Path(__file__),
        )
    }
    for candidate, row, member in zip(candidates, rows, blend["members"], strict=True):
        receipt_path = Path(candidate["receipt_path"])
        receipt = _read(receipt_path)
        prediction = receipt["predictions"]
        if (
            sha256(receipt_path) != member["receipt_sha256"]
            or receipt_path != Path(member["receipt_path"])
            or receipt["name"] != candidate["name"]
            or row["name"] != candidate["name"]
            or {key: prediction[key] for key in MASK_KEYS} != mask
            or prediction["sha256"] != row["scored_predictions_sha256"]
            or prediction["sha256"] != member["prediction_sha256"]
            or prediction["path"] != candidate["predictions_path"]
        ):
            raise ValueError("member identity, mask or prediction binding changed")
        path = Path(prediction["path"])
        array = _array(path, prediction["sha256"])
        metrics = independent_metrics(targets, array)
        delta = _compare(metrics, receipt["metrics"])
        row_metrics = {
            key: row[f"primary_valid_{key}"] for key in metrics if key != "selected_rows"
        }
        delta = max(delta, _compare(metrics, row_metrics))
        results.append(
            {"name": candidate["name"], "metrics": metrics, "maximum_metric_absolute_error": delta}
        )
        arrays.append(array)
        file_bindings[str(path)] = prediction["sha256"]
        file_bindings[str(receipt_path)] = member["receipt_sha256"]
    ensemble_path = Path(report["ensemble_predictions"]["path"])
    ensemble = _array(ensemble_path, report["ensemble_predictions"]["sha256"])
    for start in range(0, ROWS, 65_536):
        stop = min(start + 65_536, ROWS)
        if not np.array_equal(
            reconstruct_block([array[start:stop] for array in arrays], weights),
            ensemble[start:stop],
        ):
            raise ValueError("independent FP64-to-FP32 fixed blend differs from stored ensemble")
    metrics = independent_metrics(targets, ensemble)
    delta = _compare(metrics, report["evaluation"]["metrics"])
    results.append(
        {
            "name": "fixed_stage4_ensemble",
            "metrics": metrics,
            "maximum_metric_absolute_error": delta,
        }
    )
    file_bindings[str(ensemble_path)] = report["ensemble_predictions"]["sha256"]
    if sha256(report_path) != file_bindings[str(report_path)]:
        raise ValueError("report changed during independent verification")
    return {
        "schema": "wnn.independent-fixed-primary-arithmetic.v1",
        "status": "verified",
        "claim_scope": "fixed-prediction arithmetic and raw-label/mask verification only; not training replication or public-transfer confirmation",
        "rows": ROWS,
        "sequences": 1873,
        "tolerance": 1e-10,
        "raw_validation_source": source_identity,
        "mask_and_order": mask,
        "source6_files": source_files,
        "file_sha256": file_bindings,
        "weights": weights.tolist(),
        "ensemble_fp32_reconstruction_bit_exact": True,
        "outputs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("independent result must be fresh")
    result = verify(args.report.resolve(strict=True), args.data_root.resolve(strict=True))
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"status": result["status"], "output": str(args.output), "sha256": sha256(args.output)}
        )
    )


if __name__ == "__main__":
    main()
