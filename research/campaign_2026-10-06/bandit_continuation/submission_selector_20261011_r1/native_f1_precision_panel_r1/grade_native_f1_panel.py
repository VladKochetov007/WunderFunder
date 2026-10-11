"""Stream official labels for exact native predictions on the frozen F1 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

EXPECTED_SPLIT_SHA256 = "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae"


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash an array's contiguous C-order bytes."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def cgroup_snapshot() -> dict[str, str | None]:
    """Capture CPU and memory counters for this command's cgroup."""
    membership = Path("/proc/self/cgroup").read_text().splitlines()
    relative = next((line.split("::", 1)[1] for line in membership if "::" in line), None)
    result: dict[str, str | None] = {"path": relative}
    if relative is None:
        return result
    directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
    for name in ("cpu.max", "cpu.stat", "memory.current", "memory.peak", "memory.events"):
        path = directory / name
        result[name] = path.read_text().strip() if path.is_file() else None
    return result


def _load_candidate_arrays(
    catalog: dict[str, object], requested: set[str], panel_sha256: str
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    candidates = catalog.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidate catalog has no candidate list")
    catalog_ids = [candidate.get("candidate_id") for candidate in candidates]
    if len(catalog_ids) != len(set(catalog_ids)) or requested != set(catalog_ids):
        raise ValueError("requested candidates differ from the frozen candidate pool")
    records: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        if candidate_id not in requested:
            continue
        package = Path(candidate["package_path"]).resolve(strict=True)
        package_receipt_path = Path(candidate["package_receipt_path"]).resolve(strict=True)
        predictions_path = Path(candidate["predictions_path"]).resolve(strict=True)
        inference_receipt_path = Path(candidate["inference_receipt_path"]).resolve(strict=True)
        package_receipt = json.loads(package_receipt_path.read_text())
        inference_receipt = json.loads(inference_receipt_path.read_text())
        if sha256_file(package) != candidate["package_sha256"]:
            raise ValueError(f"{candidate_id}: package hash differs from frozen catalog")
        if sha256_file(package_receipt_path) != candidate["package_receipt_sha256"]:
            raise ValueError(f"{candidate_id}: package-receipt hash differs from catalog")
        if sha256_file(predictions_path) != inference_receipt["predictions"]["sha256"]:
            raise ValueError(f"{candidate_id}: predictions differ from inference receipt")
        if inference_receipt["panel"]["sha256"] != panel_sha256:
            raise ValueError(f"{candidate_id}: inference used a different feature panel")
        if inference_receipt["package"]["sha256"] != candidate["package_sha256"]:
            raise ValueError(f"{candidate_id}: inference used a different package")
        prediction_array = np.load(predictions_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (41, 20_000, 2)
        if prediction_array.shape != expected_shape or prediction_array.dtype != np.float32:
            raise ValueError(f"{candidate_id}: prediction panel shape or dtype mismatch")
        if package_receipt.get("precision") != "q8vnni":
            raise ValueError(f"{candidate_id}: package is not Q8VNNI")
        if inference_receipt["package"]["precision_asserted"] != "q8vnni":
            raise ValueError(f"{candidate_id}: actual runtime did not assert Q8VNNI")
        records.append(candidate)
        arrays[candidate_id] = prediction_array
    return records, arrays


def grade_panel(
    project_root: Path,
    catalog_path: Path,
    feature_manifest_path: Path,
    split_path: Path,
    valid_parquet_path: Path,
    requested_ids: set[str],
    output_path: Path,
) -> dict[str, object]:
    """Compute exact pooled official scores while streaming target row groups."""
    for path in (catalog_path, feature_manifest_path, split_path, valid_parquet_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if sha256_file(split_path) != EXPECTED_SPLIT_SHA256:
        raise ValueError("fold1 split manifest digest mismatch")
    feature_manifest = json.loads(feature_manifest_path.read_text())
    fixture = feature_manifest.get("fixture")
    selection = feature_manifest.get("selection")
    if not isinstance(fixture, dict) or not isinstance(selection, dict):
        raise ValueError("feature manifest is incomplete")
    feature_path = Path(fixture["path"]).resolve(strict=True)
    if sha256_file(feature_path) != fixture["sha256"]:
        raise ValueError("feature fixture no longer matches its receipt")
    if selection.get("split_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("feature panel and grader splits differ")
    if selection.get("evaluation_exposure", "").find("not freshCV") < 0:
        raise ValueError("feature panel exposure qualification is missing")

    sys.path.insert(0, str(project_root.resolve(strict=True)))
    from wnn.gmetric import GlobalAccumulator  # noqa: PLC0415

    catalog = json.loads(catalog_path.read_text())
    candidates, prediction_arrays = _load_candidate_arrays(
        catalog, requested_ids, str(fixture["sha256"])
    )
    split = json.loads(split_path.read_text())
    valid_sequences = split["sequences"]["valid"]
    evaluation_set = set(split["evaluation"]["valid"])
    group_records = fixture["group_records"]
    if len(group_records) != 41 or len(candidates) != len(requested_ids):
        raise ValueError("panel or candidate count differs from the frozen contract")

    source_stat = valid_parquet_path.stat()
    parquet = pq.ParquetFile(valid_parquet_path)
    if parquet.metadata.num_row_groups != 1873:
        raise ValueError("valid.parquet row-group count differs from fold1 mapping")
    accumulators = {candidate["candidate_id"]: GlobalAccumulator() for candidate in candidates}
    component_accumulators: dict[int, dict[str, object]] = {}
    scored_rows = 0
    records_by_group: list[dict[str, object]] = []
    cgroup_before = cgroup_snapshot()
    started = time.perf_counter()

    for panel_index, expected in enumerate(group_records):
        row_group = expected["row_group"]
        if row_group not in evaluation_set:
            raise ValueError(f"row group {row_group} is outside the fold1 evaluation partition")
        split_row = valid_sequences[row_group]
        if (
            split_row["row_group"] != row_group
            or split_row["seq_ix"] != expected["sequence_id"]
            or split_row["component"] != expected["component"]
        ):
            raise ValueError(f"fold1 split mapping differs at row group {row_group}")
        labels = parquet.read_row_group(
            row_group,
            columns=["t0", "t1", "is_scored", "need_prediction", "seq_ix"],
            use_threads=False,
        )
        if labels.num_rows != 20_000:
            raise ValueError(f"label row group {row_group} is not one complete sequence")
        sequence_ids = labels["seq_ix"].to_numpy(zero_copy_only=False)
        if not np.all(sequence_ids == expected["sequence_id"]):
            raise ValueError(f"label sequence identity mismatch at row group {row_group}")
        targets = np.column_stack(
            [labels[name].to_numpy(zero_copy_only=False) for name in ("t0", "t1")]
        ).astype(np.float32, copy=False)
        need = labels["need_prediction"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
        scored = labels["is_scored"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
        if sha256_array(need) != expected["need_prediction_sha256"]:
            raise ValueError(f"label/feature need-mask mismatch at row group {row_group}")
        mask = need & scored
        if mask[:99].any():
            raise ValueError(f"official mask includes warmup rows at row group {row_group}")
        if not np.isfinite(targets).all():
            raise ValueError(f"non-finite targets at row group {row_group}")
        scored_rows += int(mask.sum())
        component_id = int(expected["component"])
        if component_id not in component_accumulators:
            component_accumulators[component_id] = {
                candidate["candidate_id"]: GlobalAccumulator() for candidate in candidates
            }
        group_scores: dict[str, object] = {}
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            predictions = np.asarray(prediction_arrays[candidate_id][panel_index])
            accumulators[candidate_id].add(targets, predictions, mask)
            component_accumulators[component_id][candidate_id].add(targets, predictions, mask)
            group_accumulator = GlobalAccumulator()
            group_accumulator.add(targets, predictions, mask)
            group_scores[candidate_id] = group_accumulator.result()
        records_by_group.append(
            {
                "panel_index": panel_index,
                "row_group": row_group,
                "sequence_id": expected["sequence_id"],
                "component": component_id,
                "selected_rows": int(mask.sum()),
                "targets_sha256": sha256_array(targets),
                "need_prediction_sha256": sha256_array(need),
                "is_scored_sha256": sha256_array(scored),
                "official_mask_sha256": sha256_array(mask),
                "candidate_group_scores": group_scores,
            }
        )

    scores = {
        candidate_id: accumulator.result() for candidate_id, accumulator in accumulators.items()
    }
    component_scores = {
        str(component_id): {
            candidate_id: accumulator.result()
            for candidate_id, accumulator in component_map.items()
        }
        for component_id, component_map in sorted(component_accumulators.items())
    }
    reference_id = catalog.get("reference_candidate_id")
    if reference_id not in scores:
        reference_id = None
    del prediction_arrays
    receipt: dict[str, object] = {
        "schema": "wnn.native_f1_precision_official_grade.r1",
        "claim_scope": (
            "Official-metric comparison on a prespecified 41-group subset of historically exposed "
            "fold1. F1 model-fitting groups exclude these units, but prior fold1 scores/model "
            "selection make this exposed discovery rather than fresh confirmation."
        ),
        "metric": {
            "metric_id": "global_wp_clip2_v1",
            "implementation_path": str((project_root / "src/wnn/gmetric.py").resolve()),
            "implementation_sha256": sha256_file(project_root / "src/wnn/gmetric.py"),
            "mask": "need_prediction AND is_scored",
            "clip": [-2.0, 2.0],
            "weights": "absolute clipped target magnitude",
            "aggregation": "pooled weighted Pearson per target, then equal mean t0/t1",
        },
        "panel": {
            "feature_manifest_path": str(feature_manifest_path.resolve()),
            "feature_manifest_sha256": sha256_file(feature_manifest_path),
            "feature_fixture_path": str(feature_path),
            "feature_fixture_sha256": fixture["sha256"],
            "selection_seed": selection["seed"],
            "row_groups": [row["row_group"] for row in group_records],
            "sequence_ids": [row["sequence_id"] for row in group_records],
            "components": sorted({row["component"] for row in group_records}),
            "evaluation_exposure": selection["evaluation_exposure"],
            "split_path": str(split_path.resolve()),
            "split_sha256": sha256_file(split_path),
            "selected_rows": scored_rows,
            "full_valid_parquet_sha256": None,
            "source_data": {
                "path": str(valid_parquet_path.resolve()),
                "bytes": source_stat.st_size,
                "mtime_ns": source_stat.st_mtime_ns,
                "read_columns_only": ["t0", "t1", "is_scored", "need_prediction", "seq_ix"],
            },
        },
        "candidate_scores": scores,
        "deltas_vs_reference": (
            {
                candidate_id: {
                    field: scores[candidate_id][field] - scores[reference_id][field]
                    for field in ("t0", "t1", "wp")
                }
                for candidate_id in scores
                if candidate_id != reference_id
            }
            if reference_id is not None
            else None
        ),
        "reference_candidate_id": reference_id,
        "component_scores": component_scores,
        "groups": records_by_group,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "cgroup_before": cgroup_before,
            "cgroup_after": cgroup_snapshot(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    """Stream-grade the exact frozen candidate pool on the F1 precision panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--candidate-catalog", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--valid-parquet", type=Path, required=True)
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = grade_panel(
        args.project_root,
        args.candidate_catalog,
        args.feature_manifest,
        args.split,
        args.valid_parquet,
        set(args.candidate_id),
        args.output,
    )
    print(json.dumps({"candidate_scores": result["candidate_scores"]}, sort_keys=True))


if __name__ == "__main__":
    main()
