"""Independently grade the expanded causal Kalman input replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
PACKET = Path(__file__).resolve().parent
RECIPE = PACKET / "recipe_r1.json"
PREFLIGHT = PACKET / "preflight_r1.json"
GRADE_PREFLIGHT = PACKET / "grade_preflight_r1.json"
RESULT = PACKET / "result_r1.json"
DEFAULT_REPLAY = Path(
    "/home/vlad/development/wunder/research_tmp/online_features_r1/kalman_input_panel_r2"
)
DATA = ROOT / "datasets/valid.parquet"
SPLIT = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
ARMS = ("raw", "local_trend_posterior")
ROWS = 20_000
WARMUP = 99
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261020
TIME_WINDOWS = ((99, 5_000), (5_000, 10_000), (10_000, 15_000), (15_000, 20_000))


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fail_unless_preflight(replay_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Revalidate recipe, sources, identity and prediction caches before labels."""
    sys.path.insert(0, str(PACKET))
    import replay_panel

    frozen = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    current = replay_panel.static_contract()
    keys = (
        "recipe_sha256",
        "source_hashes",
        "resource_profile",
        "prior_preflight_sha256",
        "split_manifest_sha256",
        "data",
        "parent",
        "membership",
        "identity_and_step_order",
        "arms",
    )
    if frozen.get("status") != "pass_identity_step_order_targets_masks_blind" or any(
        frozen.get(key) != current[key] for key in keys
    ):
        raise ValueError("preflight no longer matches the frozen sources or membership")

    execution_path = PACKET / "execution_r1.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    expected_execution = {
        "schema": "wnn.kalman_input_panel_execution.v1",
        "status": "prediction_only_complete",
        "experiment_id": "E-KALMAN-INPUT-EXPANDED-F0-20261018-R1",
        "recipe_sha256": sha256_file(RECIPE),
        "preflight_sha256": sha256_file(PREFLIGHT),
        "source_hashes": frozen["source_hashes"],
        "parent_checkpoint_sha256": frozen["parent"]["checkpoint_sha256"],
        "groups": len(frozen["membership"]["groups"]),
        "unique_components": frozen["membership"]["unique_components"],
        "group_order_sha256": frozen["membership"]["group_order_sha256"],
        "sequence_id_order_sha256": frozen["membership"]["sequence_id_order_sha256"],
        "component_order_sha256": frozen["membership"]["component_order_sha256"],
        "identity_and_step_order_sha256": frozen["identity_and_step_order"][
            "identity_and_step_order_sha256"
        ],
        "targets_or_masks_loaded": False,
        "official_metric_called": False,
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise ValueError(f"execution receipt field {key!r} differs from the frozen contract")
    if [receipt.get("arm") for receipt in execution.get("arms", [])] != list(ARMS):
        raise ValueError("execution receipt does not contain the exact frozen arms in order")

    predictions: dict[str, np.ndarray] = {}
    for arm, execution_arm in zip(ARMS, execution["arms"], strict=True):
        cache = replay_dir / f"{arm}.npy"
        sidecar_path = replay_dir / f"{arm}.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        expected_shape = [len(frozen["membership"]["groups"]), ROWS, 2]
        expected_sidecar = {
            "schema": "wnn.kalman_input_panel_prediction.v1",
            "arm": arm,
            "cache_path": str(cache.resolve()),
            "cache_bytes": cache.stat().st_size,
            "prediction_shape": expected_shape,
            "prediction_dtype": "float32",
            "group_order_sha256": frozen["membership"]["group_order_sha256"],
            "sequence_id_order_sha256": frozen["membership"]["sequence_id_order_sha256"],
            "groups": frozen["membership"]["groups"],
            "sequence_ids": frozen["membership"]["sequence_ids"],
            "checkpoint_sha256": frozen["parent"]["checkpoint_sha256"],
            "normalization_sha256": frozen["parent"]["normalization_sha256"],
            "preflight_sha256": sha256_file(PREFLIGHT),
            "source_hashes": frozen["source_hashes"],
            "targets_or_masks_loaded": False,
            "official_metric_called": False,
        }
        for key, expected in expected_sidecar.items():
            actual = sidecar.get(key)
            if key == "cache_path" and actual is not None:
                actual = str(Path(actual).resolve())
            if actual != expected:
                raise ValueError(f"{arm} sidecar field {key!r} differs from the frozen contract")
        if execution_arm != sidecar:
            raise ValueError(f"execution receipt and {arm} sidecar do not match exactly")
        if sidecar.get("cache_sha256") != sha256_file(cache):
            raise ValueError(f"{arm} cache hash differs from its receipt")
        if sidecar.get("preflight_sha256") != sha256_file(PREFLIGHT):
            raise ValueError(f"{arm} cache names another expanded-panel preflight")
        if sidecar.get("source_hashes") != frozen["source_hashes"]:
            raise ValueError(f"{arm} source hashes differ from frozen preflight")
        if (
            sidecar.get("groups") != frozen["membership"]["groups"]
            or sidecar.get("sequence_ids") != frozen["membership"]["sequence_ids"]
        ):
            raise ValueError(f"{arm} cache membership differs from frozen preflight")
        if (
            sidecar.get("targets_or_masks_loaded") is not False
            or sidecar.get("official_metric_called") is not False
        ):
            raise ValueError(f"{arm} cache was not generated label-blind")
        values = np.load(cache, mmap_mode="r", allow_pickle=False)
        expected_array_shape = tuple(expected_shape)
        if values.shape != expected_array_shape or values.dtype != np.float32:
            raise ValueError(f"{arm} shape or dtype differs from the frozen prediction contract")
        if not np.isfinite(values[:, WARMUP:, :]).all():
            raise ValueError(f"{arm} has nonfinite scored predictions")
        predictions[arm] = values

    grading_preflight = {
        "schema": "wnn.kalman_input_panel_grade_preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "pass_predictions_and_identity; targets_and_masks_not_read",
        "recipe_sha256": sha256_file(RECIPE),
        "preflight_sha256": sha256_file(PREFLIGHT),
        "execution_receipt_sha256": sha256_file(execution_path),
        "source_hashes": frozen["source_hashes"],
        "cache_sha256": {arm: sha256_file(replay_dir / f"{arm}.npy") for arm in ARMS},
        "groups": frozen["membership"]["groups"],
        "sequence_ids": frozen["membership"]["sequence_ids"],
        "components": frozen["membership"]["components"],
        "identity_and_step_order_sha256": frozen["identity_and_step_order"][
            "identity_and_step_order_sha256"
        ],
        "target_values_read": False,
        "need_prediction_or_is_scored_values_read": False,
    }
    with GRADE_PREFLIGHT.open("x", encoding="utf-8") as stream:
        json.dump(grading_preflight, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return frozen, predictions


def read_targets_and_official_mask(
    frozen: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Read targets only after cache and source preflight, validating both masks."""
    groups = frozen["membership"]["groups"]
    sequence_ids = frozen["membership"]["sequence_ids"]
    parquet = pq.ParquetFile(DATA)
    targets = np.empty((len(groups), ROWS, 2), dtype=np.float32)
    mask = np.empty((len(groups), ROWS), dtype=bool)
    required = ["seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"]
    expected_steps = np.arange(ROWS, dtype=np.int64)
    for index, (group, expected_sequence) in enumerate(zip(groups, sequence_ids, strict=True)):
        table = parquet.read_row_group(group, columns=required, use_threads=False)
        ids = table["seq_ix"].to_numpy()
        steps = table["step_in_seq"].to_numpy()
        need = table["need_prediction"].to_numpy().astype(bool, copy=False)
        scored = table["is_scored"].to_numpy().astype(bool, copy=False)
        if table.num_rows != ROWS or not np.all(ids == expected_sequence):
            raise ValueError(f"target group {group} does not match frozen sequence identity")
        if not np.array_equal(steps, expected_steps):
            raise ValueError(f"step_in_seq is not in exact 0..19999 row order at group {group}")
        scored_by_step = steps >= WARMUP
        if not np.array_equal(need, scored_by_step):
            raise ValueError(
                f"need_prediction is not exactly the row>=99 contract at group {group}"
            )
        targets[index] = np.column_stack((table["t0"].to_numpy(), table["t1"].to_numpy()))
        mask[index] = need & scored
        del table
    if not mask.any() or mask[:, :WARMUP].any():
        raise ValueError("official need_prediction AND is_scored mask is empty or includes warmup")
    return targets, mask


def score_panel(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | str]:
    """Use the canonical centered, streamed global weighted-Pearson metric."""
    from wnn.gmetric import GlobalAccumulator

    accumulator = GlobalAccumulator()
    for index in range(len(targets)):
        accumulator.add(
            targets[index : index + 1],
            predictions[index : index + 1],
            mask[index : index + 1],
        )
    return accumulator.result()


def main() -> None:
    """Verify caches, freeze a grading preflight, then score the exposed panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", nargs="?", type=Path, default=DEFAULT_REPLAY)
    args = parser.parse_args()
    replay_dir = args.replay_dir.resolve(strict=True)
    frozen, predictions = fail_unless_preflight(replay_dir)

    sys.path.insert(0, str(ROOT / "src"))
    from wnn.evaluation import paired_component_bootstrap, paired_component_moments

    targets, mask = read_targets_and_official_mask(frozen)
    components = np.asarray(frozen["membership"]["components"], dtype=np.int64)
    scores = {arm: score_panel(targets, predictions[arm], mask) for arm in ARMS}
    candidate_moments, raw_moments = paired_component_moments(
        targets, predictions["local_trend_posterior"], predictions["raw"], mask
    )
    uncertainty = paired_component_bootstrap(
        candidate_moments,
        raw_moments,
        components,
        n_boot=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        centered=True,
    )
    temporal = {}
    for start, stop in TIME_WINDOWS:
        temporal[f"rows_{start}_{stop}"] = {
            arm: score_panel(
                targets[:, start:stop], predictions[arm][:, start:stop], mask[:, start:stop]
            )
            for arm in ARMS
        }
    result = {
        "schema": "wnn.kalman_input_panel_result.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "experiment_id": "E-KALMAN-INPUT-EXPANDED-F0-20261018-R1",
        "hypothesis_id": "H-KALMAN-CAUSAL-INPUT-20261018-R1",
        "metric_id": "global_wp_clip2_v1",
        "recipe_sha256": sha256_file(RECIPE),
        "preflight_sha256": sha256_file(PREFLIGHT),
        "grade_preflight_sha256": sha256_file(GRADE_PREFLIGHT),
        "execution_receipt_sha256": sha256_file(PACKET / "execution_r1.json"),
        "parent_checkpoint_sha256": frozen["parent"]["checkpoint_sha256"],
        "prediction_cache_sha256": {arm: sha256_file(replay_dir / f"{arm}.npy") for arm in ARMS},
        "fold": 0,
        "groups": len(frozen["membership"]["groups"]),
        "components": frozen["membership"]["unique_components"],
        "group_order_sha256": frozen["membership"]["group_order_sha256"],
        "sequence_id_order_sha256": frozen["membership"]["sequence_id_order_sha256"],
        "component_order_sha256": frozen["membership"]["component_order_sha256"],
        "selected_rows": int(mask.sum()),
        "mask": "need_prediction AND is_scored; need_prediction verified equal to row >= 99",
        "scores": scores,
        "local_trend_delta_vs_raw": float(
            scores["local_trend_posterior"]["wp"] - scores["raw"]["wp"]
        ),
        "paired_component_bootstrap": uncertainty,
        "post_hoc_temporal_diagnostics": {
            "windows": [list(window) for window in TIME_WINDOWS],
            "scores": temporal,
            "interpretation": "Descriptive bins on this exposed, after-first-16 follow-up panel; they do not establish online adaptation or a causal long-horizon effect.",
        },
        "exposure_and_selection": (
            "Previously exposed F0. The deterministic 141-group prefix expands the first-16 panel only after its small positive point was seen; the larger estimate is discovery evidence, not independent or clean-fold confirmation."
        ),
        "claim_limits": [
            "One fixed checkpoint; no training-seed variation or learned feature path was tested.",
            "The frozen GRU was trained on raw inputs; replacing four channels creates a train/inference representation mismatch.",
            "A positive or negative result here does not settle the retained-raw learned Kalman feature teacher/student direction in draft PR #17.",
            "No public score or competition transfer is inferred.",
        ],
    }
    with RESULT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "scores": scores,
                "delta": result["local_trend_delta_vs_raw"],
                "component_bootstrap": uncertainty,
                "selected_rows": result["selected_rows"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
