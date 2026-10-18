"""Independently grade the frozen, prediction-only Kalman replay caches."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
PACKET = Path(__file__).resolve().parent
SCRATCH_OUTPUT = Path(
    "/home/vlad/development/wunder/research_tmp/kalman-online-20261018/kalman_gru_probe_r2"
)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PACKET))

from probe_contract import ARMS, CHECKPOINT_SHA256, DATA, GROUPS, SEQUENCE_IDS, SPLIT  # noqa: E402

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402
from wnn.gmetric import METRIC_ID, GlobalAccumulator  # noqa: E402

SCORED_FROM_ROW = 99
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261018
TEMPORAL_WINDOWS = ((99, 5_000), (5_000, 10_000), (10_000, 15_000), (15_000, 20_000))


def sha256(path: Path) -> str:
    """Hash one artifact in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(replay_dir: Path) -> dict[str, np.ndarray]:
    """Load and validate the three small prediction-only NPZ files."""
    result: dict[str, np.ndarray] = {}
    for arm in ARMS:
        cache = replay_dir / f"{arm}.npz"
        sidecar_path = replay_dir / f"{arm}.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("cache_sha256") != sha256(cache):
            raise ValueError(f"{arm} cache bytes differ from its receipt")
        if sidecar.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            raise ValueError(f"{arm} receipt names a different parent checkpoint")
        if sidecar.get("group_ids") != list(GROUPS) or sidecar.get("sequence_ids") != list(
            SEQUENCE_IDS
        ):
            raise ValueError(f"{arm} row-group or sequence identity differs")
        if sidecar.get("targets_or_masks_loaded") is not False:
            raise ValueError(f"{arm} replay did not certify a feature-only read")
        if sidecar.get("official_metric_called") is not False:
            raise ValueError(f"{arm} replay attempted to score before independent grading")
        with np.load(cache, allow_pickle=False) as saved:
            if not np.array_equal(saved["group_ids"], np.asarray(GROUPS, dtype=np.int64)):
                raise ValueError(f"{arm} cache group order differs")
            if not np.array_equal(saved["sequence_ids"], np.asarray(SEQUENCE_IDS, dtype=np.int64)):
                raise ValueError(f"{arm} cache sequence order differs")
            values = np.asarray(saved["predictions"], dtype=np.float32)
        if values.shape != (len(GROUPS), 20_000, 2) or not np.isfinite(values).all():
            raise ValueError(f"{arm} prediction shape or values are invalid")
        result[arm] = values
    return result


def load_targets_and_mask() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only labels/mask for the frozen 16 groups after prediction export."""
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    source = split["sources"]["datasets/valid.parquet"]
    stat = DATA.stat()
    if stat.st_size != int(source["bytes"]) or stat.st_mtime_ns != int(source["mtime_ns"]):
        raise ValueError("valid parquet identity differs from the frozen split")
    component_for_group = {
        int(row["row_group"]): int(row["component"]) for row in split["sequences"]["valid"]
    }
    components = np.asarray([component_for_group[group] for group in GROUPS], dtype=np.int64)
    targets = np.empty((len(GROUPS), 20_000, 2), dtype=np.float32)
    mask = np.empty((len(GROUPS), 20_000), dtype=bool)
    parquet = pq.ParquetFile(DATA)
    for index, (group, expected_sequence_id) in enumerate(zip(GROUPS, SEQUENCE_IDS, strict=True)):
        table = parquet.read_row_group(
            group, columns=["seq_ix", "t0", "t1", "is_scored"], use_threads=False
        )
        sequence_ids = table.column("seq_ix").to_numpy()
        if table.num_rows != 20_000 or not np.all(sequence_ids == expected_sequence_id):
            raise ValueError(f"label row-group identity differs at group {group}")
        targets[index] = np.column_stack(
            [table.column("t0").to_numpy(), table.column("t1").to_numpy()]
        ).astype(np.float32, copy=False)
        mask[index] = table.column("is_scored").to_numpy().astype(bool, copy=False)
        del table
    mask &= np.arange(20_000)[None, :] >= SCORED_FROM_ROW
    if not mask.any():
        raise ValueError("the selected panel has no officially scored rows")
    return targets, mask, components


def official_score(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> dict:
    """Apply the official pooled metric to the fixed panel."""
    accumulator = GlobalAccumulator()
    for index in range(len(targets)):
        accumulator.add(
            targets[index : index + 1], predictions[index : index + 1], mask[index : index + 1]
        )
    return accumulator.result()


def main() -> int:
    replay_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRATCH_OUTPUT
    if len(sys.argv) > 2:
        raise SystemExit("usage: grade_replay_r2.py [replay-output-directory]")
    predictions = load_predictions(replay_dir)
    targets, mask, components = load_targets_and_mask()
    scores = {arm: official_score(targets, predictions[arm], mask) for arm in ARMS}
    temporal_diagnostics = {
        f"rows_{start}_{stop}": {
            arm: official_score(
                targets[:, start:stop],
                predictions[arm][:, start:stop],
                mask[:, start:stop],
            )
            for arm in ARMS
        }
        for start, stop in TEMPORAL_WINDOWS
    }
    component_diagnostics = {}
    for component in np.unique(components):
        selected = components == component
        component_diagnostics[str(int(component))] = {
            arm: official_score(targets[selected], predictions[arm][selected], mask[selected])
            for arm in ARMS
        }
    comparisons = {}
    for arm in ARMS[1:]:
        candidate = predictions[arm]
        reference = predictions["raw"]
        candidate_moments, reference_moments = paired_component_moments(
            targets, candidate, reference, mask
        )
        uncertainty = paired_component_bootstrap(
            candidate_moments,
            reference_moments,
            components,
            n_boot=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
            centered=True,
        )
        delta = candidate - reference
        comparisons[arm] = {
            "versus": "raw",
            "official_wp_delta": float(scores[arm]["wp"] - scores["raw"]["wp"]),
            "component_bootstrap": uncertainty,
            "prediction_shift": {
                "mean_absolute": float(np.mean(np.abs(delta))),
                "root_mean_square": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
                "p99_absolute": float(np.quantile(np.abs(delta), 0.99)),
            },
            "per_group_wp": [
                {
                    "row_group": group,
                    "component": int(component),
                    "wp": float(
                        official_score(
                            targets[index : index + 1],
                            candidate[index : index + 1],
                            mask[index : index + 1],
                        )["wp"]
                    ),
                    "raw_wp": float(
                        official_score(
                            targets[index : index + 1],
                            reference[index : index + 1],
                            mask[index : index + 1],
                        )["wp"]
                    ),
                }
                for index, (group, component) in enumerate(zip(GROUPS, components, strict=True))
            ],
        }
    report = {
        "schema": "wnn.kalman-frozen-gru-replay-grade.v1",
        "metric_id": METRIC_ID,
        "replay_dir": str(replay_dir),
        "replay_cache_sha256": {arm: sha256(replay_dir / f"{arm}.npz") for arm in ARMS},
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "split_manifest_sha256": sha256(SPLIT),
        "evaluation_groups": list(GROUPS),
        "evaluation_components": list(map(int, components)),
        "n_unique_components": int(np.unique(components).size),
        "scored_row_start": SCORED_FROM_ROW,
        "selected_rows": int(mask.sum()),
        "exposure": "Exposed F0; first 16 groups were frozen before prediction grading. Conditional component bootstrap describes fixed endpoints on this selected panel only.",
        "bootstrap": {
            "unit": "source component, paired across arms",
            "method": "10,000 percentile replicates of paired pooled-WP differences using centered per-sequence moments",
            "seed": BOOTSTRAP_SEED,
            "ci": "marginal 2.5th and 97.5th percentiles; conditional on fixed predictions and selected F0 components",
        },
        "scores": scores,
        "post_hoc_temporal_diagnostics": {
            "windows": list(TEMPORAL_WINDOWS),
            "scores": temporal_diagnostics,
            "interpretation": "Exploratory timing slices added after the whole-panel diagnostic; selected-panel fixed-prediction descriptions only.",
        },
        "component_diagnostics": component_diagnostics,
        "comparisons": comparisons,
        "grader_sha256": sha256(Path(__file__).resolve()),
        "expected_public_score": None,
    }
    output = PACKET / "grade_r2.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "scores": scores,
                "comparisons": {
                    arm: {
                        "official_wp_delta": value["official_wp_delta"],
                        "ci95": value["component_bootstrap"]["ci95"],
                        "p_le_0": value["component_bootstrap"]["p_le_0"],
                    }
                    for arm, value in comparisons.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
