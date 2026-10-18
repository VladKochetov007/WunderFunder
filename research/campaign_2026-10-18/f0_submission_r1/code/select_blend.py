#!/usr/bin/env python3
"""Score the preregistered five-way F0 wide/deep prediction blend screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wnn.gmetric import METRIC_ID, GlobalAccumulator  # noqa: E402

PACKET = REPO_ROOT / "research/campaign_2026-10-18/priority25_execution_r4"
F0_PACKET = REPO_ROOT / "research/campaign_2026-10-18/f0_submission_r1"
DEFAULT_GRADE = PACKET / "receipts/RANK01_F0_S1001_GRADE.json"
DEFAULT_COLLECTION = PACKET / "receipts/RANK01_F0_S1001_RESULT_COLLECTION.json"
DEFAULT_EXTRACTION = F0_PACKET / "receipts/EXTRACTION.json"
DEFAULT_EXTRACTOR = F0_PACKET / "code/extract_evidence.py"
ROW_COUNT, GROUP_COUNT, COMPONENT_COUNT = 808_863, 468, 141
WIDE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
ENDPOINT_TOLERANCE = 1e-10
INPUT_BINDINGS = (
    "wide_prediction_sha256",
    "deep_prediction_sha256",
    "split_manifest_sha256",
    "grade_sha256",
)


def sha256_file(path: Path) -> str:
    """Hash a file with bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash an array in contiguous little-endian representation."""
    value = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def json_object(path: Path) -> dict[str, object]:
    """Read a JSON object, rejecting other top-level types."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, object], value)


def field(value: object, name: str) -> dict[str, object]:
    """Require a nested JSON object."""
    if not isinstance(value, dict):
        raise ValueError(f"expected object at {name}")
    return cast(dict[str, object], value)


def input_string(archive: np.lib.npyio.NpzFile, key: str) -> str:
    """Read a non-object string scalar from the compact NPZ."""
    if key not in archive.files:
        raise ValueError(f"compact NPZ lacks source binding {key!r}")
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError(f"compact NPZ {key!r} must be a string scalar")
    item = value.item()
    return item.decode("ascii") if isinstance(item, bytes) else str(item)


def load_input(
    path: Path, grade_path: Path, collection_path: Path, extraction_path: Path, extractor_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Load compact selected rows and verify all frozen receipt bindings."""
    grade, collection, extraction = map(json_object, (grade_path, collection_path, extraction_path))
    if grade.get("schema") != "priority25.endpoint_grade.v1" or grade.get("metric_id") != METRIC_ID:
        raise ValueError("wrong F0 endpoint grade schema or metric")
    if (
        grade.get("fold") != 0
        or grade.get("n_folds") != 4
        or grade.get("evaluation_components") != COMPONENT_COUNT
    ):
        raise ValueError("grade does not identify the expected F0 evaluation")
    if collection.get("status") != "passed" or extraction.get("status") != "extracted":
        raise ValueError("result collection or compact extraction did not pass")

    wide_grade, deep_grade = (
        field(grade.get("endpoint"), "grade.endpoint"),
        field(grade.get("reference"), "grade.reference"),
    )
    predictions = field(collection.get("predictions"), "collection.predictions")
    sidecars = field(collection.get("sidecars"), "collection.sidecars")
    wide_record, deep_record = (
        field(predictions.get("wide"), "predictions.wide"),
        field(predictions.get("deep"), "predictions.deep"),
    )
    wide_sidecar, deep_sidecar = (
        field(sidecars.get("wide"), "sidecars.wide"),
        field(sidecars.get("deep"), "sidecars.deep"),
    )
    pins = field(extraction.get("source_pins"), "extraction.source_pins")
    files = field(extraction.get("files"), "extraction.files")
    expected = {
        "wide_prediction_sha256": str(wide_record["sha256"]),
        "deep_prediction_sha256": str(deep_record["sha256"]),
        "split_manifest_sha256": str(grade["split_manifest_sha256"]),
        "grade_sha256": sha256_file(grade_path),
    }
    if field(collection.get("grade"), "collection.grade").get("sha256") != expected["grade_sha256"]:
        raise ValueError("collection receipt does not bind the supplied grade")
    if (
        wide_grade.get("prediction_sha256") != expected["wide_prediction_sha256"]
        or deep_grade.get("prediction_sha256") != expected["deep_prediction_sha256"]
    ):
        raise ValueError("endpoint prediction hashes differ between grade and collection")
    if wide_grade.get("sidecar_sha256") != wide_sidecar.get("sha256") or deep_grade.get(
        "sidecar_sha256"
    ) != deep_sidecar.get("sha256"):
        raise ValueError("endpoint sidecar hashes differ between grade and collection")
    if (
        field(collection.get("fold0"), "collection.fold0").get("sha256")
        != expected["split_manifest_sha256"]
    ):
        raise ValueError("collection fold hash differs from grade")
    if sha256_file(extractor_path) != extraction.get("script_sha256"):
        raise ValueError("extraction script hash differs from extraction receipt")
    if (
        extraction.get("rows") != ROW_COUNT
        or extraction.get("groups") != GROUP_COUNT
        or extraction.get("components") != COMPONENT_COUNT
    ):
        raise ValueError("extraction receipt row/group/component counts differ")
    extracted_npz = field(files.get("scored.npz"), "extraction.files.scored.npz")
    if (
        extracted_npz.get("sha256") != sha256_file(path)
        or extracted_npz.get("bytes") != path.stat().st_size
    ):
        raise ValueError("compact NPZ bytes/size differ from extraction receipt")
    if (
        pins.get("fold") != expected["split_manifest_sha256"]
        or pins.get("wide") != expected["wide_prediction_sha256"]
        or pins.get("deep") != expected["deep_prediction_sha256"]
    ):
        raise ValueError("extraction source pins differ from frozen artifacts")
    extracted_fold = field(files.get("fold0.json"), "extraction.files.fold0.json")
    extracted_wide = field(files.get("wide_sidecar.json"), "extraction.files.wide_sidecar.json")
    extracted_deep = field(files.get("deep_sidecar.json"), "extraction.files.deep_sidecar.json")
    if (
        extracted_fold.get("sha256") != expected["split_manifest_sha256"]
        or extracted_wide.get("sha256") != wide_sidecar.get("sha256")
        or extracted_deep.get("sha256") != deep_sidecar.get("sha256")
    ):
        raise ValueError("extracted fold/sidecar hashes differ from source receipts")
    expected["collection_receipt_sha256"] = sha256_file(collection_path)
    expected["extraction_receipt_sha256"] = sha256_file(extraction_path)

    with np.load(path, allow_pickle=False) as archive:
        required = {"y", "wide", "deep", "row_group", "component", *INPUT_BINDINGS}
        if missing := required - set(archive.files):
            raise ValueError(f"compact NPZ missing fields: {sorted(missing)}")
        for key, value in expected.items():
            if key in INPUT_BINDINGS and input_string(archive, key) != value:
                raise ValueError(f"compact NPZ {key} differs from frozen receipts")
        targets, wide, deep = (np.asarray(archive[key]) for key in ("y", "wide", "deep"))
        groups, components = (np.asarray(archive[key]) for key in ("row_group", "component"))

    shape = (ROW_COUNT, 2)
    for name, array in (("y", targets), ("wide", wide), ("deep", deep)):
        if (
            array.shape != shape
            or array.dtype != np.dtype(np.float32)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"{name} must be finite float32 with shape {shape}")
    if (
        groups.shape != (ROW_COUNT,)
        or groups.dtype.kind not in "iu"
        or components.shape != (ROW_COUNT,)
        or components.dtype.kind not in "iu"
    ):
        raise ValueError("row_group/component must be integer vectors aligned to selected rows")
    expected_groups = np.asarray(grade["evaluation_row_groups"], dtype=np.int64)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    if len(unique_groups) != GROUP_COUNT or not np.array_equal(
        unique_groups, np.sort(expected_groups)
    ):
        raise ValueError("compact rows do not cover the frozen F0 groups")
    comp64 = components.astype(np.int64, copy=False)
    comp_min, comp_max = (
        np.full(GROUP_COUNT, np.iinfo(np.int64).max),
        np.full(GROUP_COUNT, np.iinfo(np.int64).min),
    )
    np.minimum.at(comp_min, inverse, comp64)
    np.maximum.at(comp_max, inverse, comp64)
    if not np.array_equal(comp_min, comp_max) or len(np.unique(components)) != COMPONENT_COUNT:
        raise ValueError("compact component mapping/count differs from the frozen F0 structure")
    return (
        targets,
        wide,
        deep,
        groups,
        components,
        {**grade, "_source_hashes": expected, "_collection": collection, "_extraction": extraction},
    )


def blend(wide: np.ndarray, deep: np.ndarray, weight: float) -> np.ndarray:
    """Apply the preregistered float32 multiply-then-add order."""
    return np.add(
        np.multiply(wide, np.float32(weight)),
        np.multiply(deep, np.float32(1.0 - weight)),
        dtype=np.float32,
    )


def score(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float | int | str]:
    """Compute official pooled clipped weighted Pearson over selected rows."""
    accumulator = GlobalAccumulator()
    accumulator.add(targets[None], predictions[None], np.ones((1, len(targets)), dtype=bool))
    return accumulator.result()


def array_hash(array: np.ndarray) -> str:
    """Hash canonical selected prediction bytes."""
    value = np.ascontiguousarray(array, dtype="<f4")
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def score_grid(targets: np.ndarray, wide: np.ndarray, deep: np.ndarray) -> list[dict[str, object]]:
    """Score only the five fixed, same-weight-for-both-target mixtures."""
    records = []
    for weight in WIDE_WEIGHTS:
        values = blend(wide, deep, weight)
        records.append(
            {
                "wide_weight": weight,
                "deep_weight": 1.0 - weight,
                "score": score(targets, values),
                "prediction_sha256": array_hash(values),
            }
        )
    return records


def report(
    path: Path,
    targets: np.ndarray,
    wide: np.ndarray,
    deep: np.ndarray,
    groups: np.ndarray,
    components: np.ndarray,
    grade: dict[str, object],
) -> dict[str, object]:
    """Build the source-bound selection receipt and independently check endpoints."""
    wide_grade, deep_grade = (
        field(grade["endpoint"], "grade.endpoint"),
        field(grade["reference"], "grade.reference"),
    )
    candidates = score_grid(targets, wide, deep)
    recomputed = {"wide": score(targets, wide), "deep": score(targets, deep)}
    for label, frozen, actual in (
        ("wide", wide_grade, recomputed["wide"]),
        ("deep", deep_grade, recomputed["deep"]),
    ):
        expected = field(frozen["score"], f"grade.{label}.score")
        if expected.get("metric_id") != METRIC_ID or expected.get("selected_rows") != ROW_COUNT:
            raise ValueError(f"frozen {label} endpoint has unexpected metric/row count")
        if any(
            abs(float(expected[key]) - float(actual[key])) > ENDPOINT_TOLERANCE
            for key in ("t0", "t1", "wp")
        ):
            raise ValueError(f"recomputed {label} endpoint scores differ from frozen grade")
    selected = max(
        candidates,
        key=lambda row: (
            float(field(row["score"], "candidate.score")["wp"]),
            float(row["wide_weight"]),
        ),
    )
    hashes = cast(dict[str, str], grade["_source_hashes"])
    collection, extraction = (
        field(grade["_collection"], "collection"),
        field(grade["_extraction"], "extraction"),
    )
    return {
        "schema": "f0_submission.fixed_blend_selection.v1",
        "classification": grade["classification"],
        "selection_note": "five fixed mixtures selected on previously exposed F0; no held-out or public-transfer claim",
        "metric_id": METRIC_ID,
        "arithmetic": "float32(wide*w) + float32(deep*(1-w)); same w for t0 and t1",
        "coefficient_grid_wide": list(WIDE_WEIGHTS),
        "selected_rows": len(targets),
        "row_groups": len(np.unique(groups)),
        "components": len(np.unique(components)),
        "input": {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "y_sha256": sha256_array(targets),
            "wide_sha256": sha256_array(wide),
            "deep_sha256": sha256_array(deep),
            "row_group_sha256": sha256_array(groups),
            "component_sha256": sha256_array(components),
            "extraction_script_sha256": extraction["script_sha256"],
            **hashes,
        },
        "target_lineage": {
            "source": extraction["target_source"],
            "parquet_path": extraction["source_parquet"],
            "parquet_size": extraction["parquet_size"],
            "parquet_mtime_ns": extraction["parquet_mtime_ns"],
            "full_hash_status": extraction["parquet_full_hash"],
            "grade_recorded_cache_sha256": grade["target_cache_sha256"],
            "cache_role": "original grade reference only; labels here came directly from the original parquet",
        },
        "fold0": {
            "fold": grade["fold"],
            "manifest_sha256": grade["split_manifest_sha256"],
            "evaluation_row_groups_sha256": sha256_array(
                np.asarray(grade["evaluation_row_groups"], dtype="<i8")
            ),
            "collection_plan_sha256": field(collection["plan"], "collection.plan")["sha256"],
        },
        "endpoints": {
            "wide": {**wide_grade, "recomputed_score": recomputed["wide"]},
            "deep": {**deep_grade, "recomputed_score": recomputed["deep"]},
        },
        "wide_vs_deep_original_pair": grade["paired_component_bootstrap"],
        "endpoint_score_abs_tolerance": ENDPOINT_TOLERANCE,
        "candidates": candidates,
        "selection": {
            "wide_weight": selected["wide_weight"],
            "deep_weight": selected["deep_weight"],
            "score": selected["score"],
            "tie_break": "larger wide weight for exactly equal computed WP",
        },
        "limits": [
            "F0 was previously exposed; these scores are screening evidence.",
            "No significance or public-transfer claim is made for the selected mixture.",
            "Submission still must pass causal serving, exact normalization, sequence-reset, ABI, Python 3.11, ZIP-size and CPU-runtime gates.",
        ],
    }


def self_check() -> None:
    """Check synthetic endpoints and mixture scoring using the official metric."""
    y = np.random.default_rng(271828).uniform(-0.8, 0.8, size=(128, 2)).astype(np.float32)
    candidates = score_grid(y, y.copy(), -y)
    if abs(float(field(candidates[-1]["score"], "wide.score")["wp"]) - 1.0) > 1e-12:
        raise AssertionError("wide endpoint did not score +1 on synthetic identity data")
    if abs(float(field(candidates[0]["score"], "deep.score")["wp"]) + 1.0) > 1e-12:
        raise AssertionError("deep endpoint did not score -1 on synthetic negated data")
    if not np.array_equal(blend(y, -y, 1.0), y) or not np.array_equal(blend(y, -y, 0.0), -y):
        raise AssertionError("float32 endpoint mixtures changed the source predictions")


def main() -> int:
    """Run the five-way screen or its deterministic synthetic self-check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--grade", type=Path, default=DEFAULT_GRADE)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--extraction", type=Path, default=DEFAULT_EXTRACTION)
    parser.add_argument("--extractor", type=Path, default=DEFAULT_EXTRACTOR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print(json.dumps({"passed": True, "scope": "synthetic endpoints and blend metric"}))
        return 0
    if args.input is None or args.output is None:
        parser.error("input and --output are required unless --self-check is selected")
    input_path = args.input.resolve(strict=True)
    loaded = load_input(
        input_path,
        args.grade.resolve(strict=True),
        args.collection.resolve(strict=True),
        args.extraction.resolve(strict=True),
        args.extractor.resolve(strict=True),
    )
    targets, wide, deep, groups, components, grade = loaded
    rendered = report(input_path, targets, wide, deep, groups, components, grade)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(rendered, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps(rendered, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
