"""Run the bounded fit-side, feature-only innovation-clock diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
# Split manifests and JSON receipts have heterogeneous schemas defined outside this module.
from typing import Any

import numpy as np
from innovation_features import FEATURE_GROUPS, WARMUP_ROWS, innovation_clock

from wnn.data import SequenceFile

ROOT = Path(__file__).resolve().parents[4]
EXPECTED_DATA = {
    "bytes": 29_240_219_011,
    "mtime_ns": 1_577_826_000_000_000_000,
    "footer_bytes": 60_437_285,
    "footer_sha256": "071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb",
}
EXPECTED_SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
SAMPLE_COMPONENTS = 32
SEED = 20261018
TIME_BINS = ((99, 500), (500, 2_000), (2_000, 10_000), (10_000, 20_000))


def sha256_file(path: Path) -> str:
    """Hash a small pinned source file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_train_identity(path: Path) -> dict[str, int | str]:
    """Verify immutable file metadata and footer before PyArrow opens it."""
    stat_before = path.stat()
    observed: dict[str, int | str] = {
        "bytes": stat_before.st_size,
        "mtime_ns": stat_before.st_mtime_ns,
        "footer_bytes": EXPECTED_DATA["footer_bytes"],
    }
    if (
        observed["bytes"] != EXPECTED_DATA["bytes"]
        or observed["mtime_ns"] != EXPECTED_DATA["mtime_ns"]
    ):
        raise ValueError(f"train file size/mtime differs from frozen source: {observed}")

    digest = hashlib.sha256()
    with path.open("rb") as source:
        source.seek(-8, 2)
        trailer = source.read(8)
        footer_bytes = int.from_bytes(trailer[:4], "little")
        if trailer[4:] != b"PAR1" or footer_bytes != EXPECTED_DATA["footer_bytes"]:
            raise ValueError("train Parquet footer trailer differs from the pinned identity")
        source.seek(-(footer_bytes + 8), 2)
        remaining = footer_bytes + 8
        while remaining:
            chunk = source.read(min(1 << 20, remaining))
            if not chunk:
                raise ValueError("train Parquet ended before the pinned footer was read")
            digest.update(chunk)
            remaining -= len(chunk)
    observed["footer_sha256"] = digest.hexdigest()
    stat_after = path.stat()
    if (stat_after.st_size, stat_after.st_mtime_ns) != (
        stat_before.st_size,
        stat_before.st_mtime_ns,
    ):
        raise RuntimeError("train Parquet changed while its bounded footer identity was checked")
    if observed != EXPECTED_DATA:
        raise ValueError(f"train Parquet footer hash differs from frozen source: {observed}")
    return observed


def verify_sources(
    root: Path, recipe_path: Path, split_path: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    """Verify the recipe's source pins and load the fixed split manifest."""
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    split_bytes = split_path.read_bytes()
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    if split_sha256 != EXPECTED_SPLIT_SHA256:
        raise ValueError("fold0 split differs from the feature-only recipe")
    source_hashes = {str(path): sha256_file(root / path) for path in recipe["source_hashes"]}
    if source_hashes != recipe["source_hashes"]:
        raise ValueError(f"source file hashes differ from recipe: {source_hashes}")
    return {
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "split_sha256": split_sha256,
        **source_hashes,
    }, json.loads(split_bytes)


def choose_fit_sequences(split: dict[str, Any]) -> list[dict[str, int]]:
    """Select one complete sequence from each of 32 fit-only components."""
    fit_groups = set(map(int, split["fit"]["train"]))
    by_component: dict[int, list[dict[str, int]]] = {}
    for record in split["sequences"]["train"]:
        group = int(record["row_group"])
        if group in fit_groups and group != 528:
            by_component.setdefault(int(record["component"]), []).append(record)
    components = np.asarray(sorted(by_component), dtype=np.int64)
    if len(components) < SAMPLE_COMPONENTS:
        raise ValueError("fewer than 32 source-qualified fitting components are available")
    rng = np.random.default_rng(SEED)
    selected_components = rng.choice(components, SAMPLE_COMPONENTS, replace=False)
    selected: list[dict[str, int]] = []
    for component in selected_components:
        records = sorted(by_component[int(component)], key=lambda record: int(record["row_group"]))
        record = records[int(rng.integers(len(records)))]
        selected.append({key: int(record[key]) for key in ("component", "row_group", "seq_ix")})
    return selected


def summarize_bin(values: np.ndarray, row_start: int, row_end: int) -> dict[str, float | int]:
    """Summarize one sequence-local time bin without retaining row data."""
    start, end = row_start - WARMUP_ROWS, row_end - WARMUP_ROWS
    if start < 0 or end > len(values) or end <= start:
        raise ValueError(f"invalid feature bin [{row_start}, {row_end})")
    block = values[start:end].astype(np.float64, copy=False)
    return {
        "row_start": row_start,
        "row_end": row_end,
        "mean": float(block.mean()),
        "median": float(np.median(block)),
        "p95": float(np.quantile(block, 0.95)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write a small summary table to physical scratch."""
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, recipe_path: Path, split_path: Path, output_root: Path) -> dict[str, Any]:
    """Verify sources, read only fit-side features, and write compact summaries."""
    scratch = Path("/home/vlad/development/wunder/research_tmp").resolve()
    output_root = output_root.resolve()
    if not output_root.is_relative_to(scratch):
        raise ValueError("all outputs must stay on the approved physical scratch filesystem")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_hashes, split = verify_sources(root, recipe_path, split_path)
    train_path = root / "datasets/train.parquet"
    data_identity = verify_train_identity(train_path)
    selected = choose_fit_sequences(split)

    # This constructor parses metadata only after the pinned footer check above.
    dataset = SequenceFile(train_path)
    if len(dataset) != 10_607:
        raise ValueError(f"unexpected train row-group count: {len(dataset)}")

    component_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    first_features: np.ndarray | None = None
    first_output: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    first_prefix: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    first_group: dict[str, int] | None = None

    for index, member in enumerate(selected):
        sequence = dataset.read(member["row_group"], targets=False, mask=False)
        if sequence.seq_ix != member["seq_ix"]:
            raise ValueError(f"sequence id mismatch at row group {member['row_group']}")
        result = innovation_clock(sequence.features)
        if index == 0:
            prefix_length = 2_000
            first_prefix = innovation_clock(sequence.features[:prefix_length])
            future_perturbation = sequence.features.copy()
            future_perturbation[prefix_length:] += 17.0
            perturbed_output = innovation_clock(future_perturbation)
            prefix_rows = prefix_length - WARMUP_ROWS
            if not all(
                np.array_equal(first_prefix[field], result[field][:prefix_rows])
                and np.array_equal(
                    result[field][:prefix_rows], perturbed_output[field][:prefix_rows]
                )
                for field in range(3)
            ):
                raise AssertionError("future-perturbation prefix invariance failed")
            first_features = sequence.features.copy()
            first_output = tuple(value.copy() for value in result)
            first_group = member

        for row_start, row_end in TIME_BINS:
            global_stats = summarize_bin(result[0], row_start, row_end)
            clock_stats = summarize_bin(result[2], row_start, row_end)
            component_rows.append(
                {
                    **member,
                    **global_stats,
                    "phase": f"{row_start}_{row_end}",
                    "innovation_p95": global_stats["p95"],
                    "clock_mean": clock_stats["mean"],
                    "clock_p95": clock_stats["p95"],
                    "clock_fraction_gt_1": float(
                        np.mean(result[2][row_start - WARMUP_ROWS : row_end - WARMUP_ROWS] > 1.0)
                    ),
                }
            )
            group_start, group_end = row_start - WARMUP_ROWS, row_end - WARMUP_ROWS
            for group_index, (name, bounds) in enumerate(FEATURE_GROUPS.items()):
                group_rows.append(
                    {
                        **member,
                        "phase": f"{row_start}_{row_end}",
                        "semantic_group": name,
                        "channel_start": bounds[0],
                        "channel_end": bounds[1],
                        "innovation_rms_mean": float(
                            result[1][group_start:group_end, group_index].mean()
                        ),
                        "innovation_rms_p95": float(
                            np.quantile(result[1][group_start:group_end, group_index], 0.95)
                        ),
                    }
                )
        del sequence, result

        if index == 1:
            if first_features is None or first_output is None or first_group is None:
                raise RuntimeError("A-B-A reset diagnostic lacks its first sequence")
            after_b = innovation_clock(first_features)
            if not all(
                np.array_equal(before, after)
                for before, after in zip(first_output, after_b, strict=True)
            ):
                raise AssertionError("actual sequence A-B-A state reset failed")

    if len(selected) != SAMPLE_COMPONENTS or len(component_rows) != SAMPLE_COMPONENTS * len(
        TIME_BINS
    ):
        raise RuntimeError("the frozen sample did not produce the expected compact summaries")
    component_csv = output_root / "component_phase_summary.csv"
    group_csv = output_root / "semantic_group_phase_summary.csv"
    write_csv(
        component_csv,
        component_rows,
        [
            "component",
            "row_group",
            "seq_ix",
            "phase",
            "row_start",
            "row_end",
            "mean",
            "median",
            "p95",
            "innovation_p95",
            "clock_mean",
            "clock_p95",
            "clock_fraction_gt_1",
        ],
    )
    write_csv(
        group_csv,
        group_rows,
        [
            "component",
            "row_group",
            "seq_ix",
            "phase",
            "semantic_group",
            "channel_start",
            "channel_end",
            "innovation_rms_mean",
            "innovation_rms_p95",
        ],
    )
    receipt = {
        "id": "E-ONLINE-INNOVATION-CLOCK-FEATURE-20261018-R1",
        "status": "feature_only_screen_completed_no_quality_fit",
        "observed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample": selected,
        "sample_components": len(selected),
        "sampled_feature_rows": len(selected) * 20_000,
        "source_hashes": source_hashes,
        "data_identity": data_identity,
        "feature_recipe": {
            "warmup_rows": WARMUP_ROWS,
            "q": 0.001,
            "r": 1.0,
            "clock_decay": 0.95,
            "clock_baseline": 1.0,
            "clock_cap": 20.0,
            "labels_read": False,
            "scoring_masks_read": False,
            "gpu_used": False,
            "torch_imported": "torch" in sys.modules,
        },
        "invariants": {
            "prefix_truncation_and_future_perturbation_invariance": True,
            "actual_sequence_a_b_a_reset": True,
            "all_outputs_finite": True,
        },
        "summary_paths": [str(component_csv), str(group_csv)],
        "summary_sha256": {
            component_csv.name: sha256_file(component_csv),
            group_csv.name: sha256_file(group_csv),
        },
        "elapsed_wall_seconds": time.perf_counter() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "official_metric": None,
        "quality_claim": None,
    }
    receipt_path = output_root / "feature_only_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"status": receipt["status"], "receipt": str(receipt_path), "groups": len(selected)}
        )
    )
    return receipt


def parse_args() -> argparse.Namespace:
    """Parse the fixed local screen paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=ROOT
        / "research/campaign_2026-10-18/online_features_r1/innovation_clock_r1/recipe.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Execute one bounded, label-free feature pass."""
    arguments = parse_args()
    run(
        arguments.root.resolve(),
        arguments.recipe.resolve(),
        arguments.split.resolve(),
        arguments.output,
    )


if __name__ == "__main__":
    main()
