"""Read-only target error and position diagnostics for the pinned TL3 caches.

This score-only successor reuses R4's role pins and validation. It loads no
model and writes only a create-only JSON receipt at the requested output path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grade_tl3_roles_r4 as r4
import numpy as np
import pyarrow.parquet as pq

from wnn.data import SEQUENCE_LENGTH, WARMUP
from wnn.gmetric import METRIC_CLIP, GlobalAccumulator
from wnn.splits import file_sha256, load_campaign_split

POSITION_BINS = (
    (WARMUP, 256, "pos_0099_0255"),
    (256, 1_000, "pos_0256_0999"),
    (1_000, 2_000, "pos_1000_1999"),
    (2_000, 5_000, "pos_2000_4999"),
    (5_000, 10_000, "pos_5000_9999"),
    (10_000, SEQUENCE_LENGTH, "pos_10000_19999"),
)
CHUNK_SIZES = (256, 512)
BOUNDARY_RADIUS = 8
BLEND_WEIGHTS = (0.34, 0.33, 0.33)
ROLE_NAMES = tuple(pin.run for pin in r4.ROLE_PINS)
MODEL_NAMES = (*ROLE_NAMES, "fixed_tl3_blend")
ROLE_PAIRS = tuple(
    (ROLE_NAMES[left], ROLE_NAMES[right])
    for left in range(len(ROLE_NAMES))
    for right in range(left + 1, len(ROLE_NAMES))
)


class _WeightedMSE:
    """Accumulate raw-prediction WMSE against clipped targets, by target."""

    def __init__(self) -> None:
        self.numerator = np.zeros(2, dtype=np.float64)
        self.weight_mass = np.zeros(2, dtype=np.float64)

    def add(self, targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> None:
        """Accumulate masked weighted squared errors without clipping predictions."""
        y_raw = np.asarray(targets, dtype=np.float32)
        p_raw = np.asarray(predictions, dtype=np.float32)
        selected = np.asarray(mask, dtype=bool)
        if y_raw.shape != p_raw.shape or y_raw.ndim != 2 or y_raw.shape[-1] != 2:
            raise ValueError("WMSE requires equal [rows, 2] target and prediction arrays")
        if selected.shape != y_raw.shape[:1]:
            raise ValueError("WMSE row mask must match the row axis")
        if not np.isfinite(y_raw[selected]).all() or not np.isfinite(p_raw[selected]).all():
            raise ValueError("nonfinite selected WMSE target or raw prediction")
        y = np.where(selected[:, None], y_raw, 0.0).astype(np.float64)
        prediction = np.where(selected[:, None], p_raw, 0.0).astype(np.float64)
        y = np.clip(y, -METRIC_CLIP, METRIC_CLIP)
        weights = np.abs(y) * selected[:, None]
        error = prediction - y
        self.numerator += np.sum(weights * np.square(error), axis=0)
        self.weight_mass += np.sum(weights, axis=0)

    def result(self) -> dict[str, Any]:
        """Return per-target WMSE and the pooled numerator/weight ratio."""
        target_scores: list[float | None] = [
            float(self.numerator[index] / self.weight_mass[index])
            if self.weight_mass[index] > 0
            else None
            for index in range(2)
        ]
        total_weight = float(self.weight_mass.sum())
        pooled = float(self.numerator.sum() / total_weight) if total_weight > 0 else None
        return {
            "t0": target_scores[0],
            "t1": target_scores[1],
            "pooled": pooled,
            "numerator_by_target": self.numerator.tolist(),
            "weight_mass_by_target": self.weight_mass.tolist(),
        }


class _WeightedCorrelation:
    """Stable weighted bivariate moments for paired residual streams."""

    def __init__(self) -> None:
        self.weight = 0.0
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.m2_x = 0.0
        self.m2_y = 0.0
        self.co_moment = 0.0
        self.selected_rows = 0
        self.positive_weight_rows = 0

    def add(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> None:
        """Merge one masked chunk of equally weighted row pairs."""
        x_values = np.asarray(x, dtype=np.float64)
        y_values = np.asarray(y, dtype=np.float64)
        weight_values = np.asarray(weights, dtype=np.float64)
        selected = np.asarray(mask, dtype=bool)
        if not (x_values.shape == y_values.shape == weight_values.shape == selected.shape):
            raise ValueError("residuals, weights and mask must have equal one-dimensional shapes")
        if not np.isfinite(x_values[selected]).all() or not np.isfinite(y_values[selected]).all():
            raise ValueError("nonfinite selected residual")
        if not np.isfinite(weight_values[selected]).all() or np.any(weight_values[selected] < 0):
            raise ValueError("selected residual weights must be finite and nonnegative")
        self.selected_rows += int(selected.sum())
        active = selected & (weight_values > 0)
        if not active.any():
            return
        x_chunk, y_chunk, w_chunk = (
            x_values[active],
            y_values[active],
            weight_values[active],
        )
        chunk_weight = float(w_chunk.sum())
        if chunk_weight <= 0:
            return
        chunk_mean_x = float(np.sum(w_chunk * x_chunk) / chunk_weight)
        chunk_mean_y = float(np.sum(w_chunk * y_chunk) / chunk_weight)
        centered_x, centered_y = x_chunk - chunk_mean_x, y_chunk - chunk_mean_y
        chunk_m2_x = float(np.sum(w_chunk * centered_x * centered_x))
        chunk_m2_y = float(np.sum(w_chunk * centered_y * centered_y))
        chunk_co = float(np.sum(w_chunk * centered_x * centered_y))
        if self.weight == 0:
            self.weight = chunk_weight
            self.mean_x, self.mean_y = chunk_mean_x, chunk_mean_y
            self.m2_x, self.m2_y, self.co_moment = chunk_m2_x, chunk_m2_y, chunk_co
        else:
            total = self.weight + chunk_weight
            delta_x, delta_y = chunk_mean_x - self.mean_x, chunk_mean_y - self.mean_y
            cross = self.weight * chunk_weight / total
            self.m2_x += chunk_m2_x + delta_x * delta_x * cross
            self.m2_y += chunk_m2_y + delta_y * delta_y * cross
            self.co_moment += chunk_co + delta_x * delta_y * cross
            self.mean_x += delta_x * chunk_weight / total
            self.mean_y += delta_y * chunk_weight / total
            self.weight = total
        self.positive_weight_rows += int(active.sum())

    def result(self) -> dict[str, float | int | None]:
        """Return weighted Pearson, using gmetric's zero-variance convention."""
        if self.selected_rows == 0:
            return {
                "pearson": None,
                "selected_rows": 0,
                "positive_weight_rows": 0,
                "weight_mass": 0.0,
            }
        scale = max(self.weight, 1e-8)
        std_x = np.sqrt(max(self.m2_x / scale, 0.0))
        std_y = np.sqrt(max(self.m2_y / scale, 0.0))
        corr = 0.0
        if self.weight >= 1e-8 and std_x > 1e-8 and std_y > 1e-8:
            corr = float(np.clip(self.co_moment / (self.weight * std_x * std_y), -1.0, 1.0))
        return {
            "pearson": corr,
            "selected_rows": self.selected_rows,
            "positive_weight_rows": self.positive_weight_rows,
            "weight_mass": self.weight,
        }


@dataclass
class _MetricPair:
    """Official WP and raw-prediction WMSE for one model and region."""

    wp: GlobalAccumulator
    wmse: _WeightedMSE
    selected_rows: int = 0

    @classmethod
    def create(cls) -> _MetricPair:
        """Create empty official-metric accumulators."""
        return cls(GlobalAccumulator(), _WeightedMSE())

    def add(self, targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> None:
        """Add the same row subset to official WP and WMSE accumulators."""
        self.wp.add(targets, predictions, mask)
        self.wmse.add(targets, predictions, mask)
        self.selected_rows += int(np.asarray(mask, dtype=bool).sum())

    def result(self) -> dict[str, Any]:
        """Return finite metric values, using null for genuinely empty slices."""
        wmse = self.wmse.result()
        if self.selected_rows == 0:
            wp: dict[str, Any] = {
                "t0": None,
                "t1": None,
                "wp": None,
                "selected_rows": 0,
                "metric_id": "global_wp_clip2_v1",
            }
        else:
            wp = self.wp.result()
        if int(wp["selected_rows"]) != self.selected_rows:
            raise ValueError("WP and WMSE selected-row counts differ")
        return {"wp": wp, "wmse": wmse}


def _region_masks(scored_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Partition scored rows into fixed position and chunk-boundary regions."""
    mask = np.asarray(scored_mask, dtype=bool)
    if mask.shape != (SEQUENCE_LENGTH,):
        raise ValueError(f"expected one {SEQUENCE_LENGTH}-row sequence mask")
    if mask[:WARMUP].any():
        raise ValueError("position regions must exclude the official warmup rows")
    positions = np.arange(SEQUENCE_LENGTH)
    regions: dict[str, np.ndarray] = {"overall": mask.copy()}
    position_masks = []
    for start, end, name in POSITION_BINS:
        region = mask & (positions >= start) & (positions < end)
        regions[name] = region
        position_masks.append(region)
    coverage = np.sum(position_masks, axis=0, dtype=np.int8)
    if not np.array_equal(coverage, mask.astype(np.int8)):
        raise ValueError("predeclared sequence-position bins do not partition scored rows")
    for chunk_size in CHUNK_SIZES:
        near = np.zeros(SEQUENCE_LENGTH, dtype=bool)
        for boundary in range(chunk_size, SEQUENCE_LENGTH, chunk_size):
            start = max(0, boundary - BOUNDARY_RADIUS)
            end = min(SEQUENCE_LENGTH, boundary + BOUNDARY_RADIUS)
            near[start:end] = True
        prefix = f"chunk{chunk_size}"
        regions[f"{prefix}_boundary_near"] = mask & near
        regions[f"{prefix}_interior"] = mask & ~near
        if not np.array_equal(
            (regions[f"{prefix}_boundary_near"] | regions[f"{prefix}_interior"]), mask
        ) or np.any(regions[f"{prefix}_boundary_near"] & regions[f"{prefix}_interior"]):
            raise ValueError(f"{prefix} boundary regions do not partition scored rows")
    return regions


def _fixed_blend(predictions: list[np.ndarray]) -> np.ndarray:
    """Reproduce R4's fixed FP32 left-to-right blend without clipping."""
    blend = np.float32(BLEND_WEIGHTS[0]) * np.asarray(predictions[0], dtype=np.float32)
    blend += np.float32(BLEND_WEIGHTS[1]) * np.asarray(predictions[1], dtype=np.float32)
    blend += np.float32(BLEND_WEIGHTS[2]) * np.asarray(predictions[2], dtype=np.float32)
    return blend


def _new_residual_accumulators() -> dict[str, dict[str, _WeightedCorrelation]]:
    """Create pair-by-target accumulators for both raw and official-clipped errors."""
    return {
        residual_kind: {
            f"{left}|{right}|t{target}": _WeightedCorrelation()
            for left, right in ROLE_PAIRS
            for target in range(2)
        }
        for residual_kind in ("raw_prediction_minus_clipped_target", "official_clipped_residual")
    }


def _load_targets(table: Any, group: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract one complete validation sequence and apply the official scored mask."""
    if table.num_rows != SEQUENCE_LENGTH:
        raise ValueError(f"validation row group {group} is not one complete sequence")
    targets = np.column_stack([table[name].to_numpy() for name in ("t0", "t1")]).astype(
        np.float32, copy=False
    )
    mask = table["is_scored"].to_numpy().astype(bool, copy=True)
    mask[:WARMUP] = False
    return targets, mask


def _score_regions(
    project_root: Path, groups: list[int], arrays: list[np.ndarray]
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Stream valid targets and score all fixed regions and pairwise residuals."""
    parquet_path = project_root / "datasets" / "valid.parquet"
    if parquet_path.is_symlink() or not parquet_path.is_file():
        raise ValueError("official valid.parquet must be a regular non-symlink file")
    parquet = pq.ParquetFile(parquet_path)
    if not {"t0", "t1", "is_scored"}.issubset(parquet.schema_arrow.names):
        raise ValueError("valid.parquet lacks targets or its official is_scored mask")
    region_names = ["overall", *(name for _start, _end, name in POSITION_BINS)]
    for chunk_size in CHUNK_SIZES:
        region_names.extend((f"chunk{chunk_size}_boundary_near", f"chunk{chunk_size}_interior"))
    metrics = {
        region: {model: _MetricPair.create() for model in MODEL_NAMES} for region in region_names
    }
    region_rows = dict.fromkeys(region_names, 0)
    region_sequences = dict.fromkeys(region_names, 0)
    residuals = _new_residual_accumulators()
    selected_rows = 0
    for sequence_position, group in enumerate(groups):
        table = parquet.read_row_group(group, columns=["t0", "t1", "is_scored"], use_threads=False)
        targets, mask = _load_targets(table, group)
        selected_rows += int(mask.sum())
        predictions = [np.asarray(array[sequence_position], dtype=np.float32) for array in arrays]
        for run, prediction in zip(ROLE_NAMES, predictions, strict=True):
            if not np.isfinite(prediction[mask]).all():
                raise ValueError(f"nonfinite selected prediction in {run}, group {group}")
        blend = _fixed_blend(predictions)
        all_predictions = [*predictions, blend]
        clipped_targets = np.clip(targets.astype(np.float64), -METRIC_CLIP, METRIC_CLIP)
        weights = np.abs(clipped_targets)
        region_masks = _region_masks(mask)
        for region in region_names:
            region_mask = region_masks[region]
            count = int(region_mask.sum())
            region_rows[region] += count
            if count:
                region_sequences[region] += 1
            for model_name, prediction in zip(MODEL_NAMES, all_predictions, strict=True):
                metrics[region][model_name].add(targets, prediction, region_mask)
        raw_residuals = [
            prediction.astype(np.float64) - clipped_targets for prediction in predictions
        ]
        clipped_residuals = [
            np.clip(prediction.astype(np.float64), -METRIC_CLIP, METRIC_CLIP) - clipped_targets
            for prediction in predictions
        ]
        for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
            left, right = ROLE_NAMES[left_index], ROLE_NAMES[right_index]
            for target in range(2):
                key = f"{left}|{right}|t{target}"
                residuals["raw_prediction_minus_clipped_target"][key].add(
                    raw_residuals[left_index][:, target],
                    raw_residuals[right_index][:, target],
                    weights[:, target],
                    mask,
                )
                residuals["official_clipped_residual"][key].add(
                    clipped_residuals[left_index][:, target],
                    clipped_residuals[right_index][:, target],
                    weights[:, target],
                    mask,
                )
    if selected_rows != r4.EXPECTED_VALID_ROWS:
        raise ValueError("official scored-row count differs from R4's pinned full-valid total")
    if sum(region_rows[name] for _start, _end, name in POSITION_BINS) != selected_rows:
        raise ValueError("position-bin rows do not cover the official scored rows")
    for chunk_size in CHUNK_SIZES:
        if (
            region_rows[f"chunk{chunk_size}_boundary_near"]
            + region_rows[f"chunk{chunk_size}_interior"]
            != selected_rows
        ):
            raise ValueError(f"chunk{chunk_size} row counts do not cover the official mask")
    region_results: dict[str, Any] = {}
    for region in region_names:
        one_model = metrics[region][MODEL_NAMES[0]]
        wmse = one_model.wmse.result()
        region_results[region] = {
            "selected_rows": region_rows[region],
            "sequences_with_selected_rows": region_sequences[region],
            "target_weight_mass": wmse["weight_mass_by_target"],
            "models": {name: pair.result() for name, pair in metrics[region].items()},
        }
    residual_results = {
        residual_kind: {key: accumulator.result() for key, accumulator in result.items()}
        for residual_kind, result in residuals.items()
    }
    return region_results, residual_results, selected_rows


def diagnose(project_root: Path, split_manifest: Path) -> dict[str, Any]:
    """Validate all R4 pins, stream targets, and return source-bound diagnostics."""
    project_root = project_root.resolve(strict=True)
    split_manifest = split_manifest.resolve(strict=True)
    split = load_campaign_split(split_manifest, project_root)
    groups = r4._validate_split(split)
    split_raw_sha = file_sha256(split.path)
    parquet_path = project_root / "datasets" / "valid.parquet"
    parquet_sha = file_sha256(parquet_path)
    loaded = [r4._load_role(project_root, split, groups, pin) for pin in r4.ROLE_PINS]
    arrays = [item[0] for item in loaded]
    role_records = [item[1] for item in loaded]
    region_results, residual_results, selected_rows = _score_regions(project_root, groups, arrays)
    for pin, record in zip(r4.ROLE_PINS, role_records, strict=True):
        if file_sha256(Path(record["sidecar_path"])) != record["sidecar_file_sha256"]:
            raise ValueError(f"role sidecar changed during diagnostics: {pin.run}")
        if file_sha256(Path(record["run_provenance_path"])) != record["run_provenance_file_sha256"]:
            raise ValueError(f"role provenance changed during diagnostics: {pin.run}")
        if file_sha256(Path(record["prediction_path"])) != pin.prediction_sha256:
            raise ValueError(f"role prediction changed during diagnostics: {pin.run}")
        checkpoint = project_root / "artifacts" / "runs" / pin.run / record["checkpoint_file"]
        if file_sha256(checkpoint) != pin.checkpoint_sha256:
            raise ValueError(f"role checkpoint changed during diagnostics: {pin.run}")
    if file_sha256(split.path) != split_raw_sha or file_sha256(parquet_path) != parquet_sha:
        raise ValueError("split manifest or valid.parquet changed during diagnostics")
    return {
        "schema": "wnn.tl3-residual-diagnostics.v1",
        "source": {
            "project_root": str(project_root),
            "split_manifest_path": str(split.path),
            "split_manifest_file_sha256": split_raw_sha,
            "split_manifest_identity_sha256": split.sha256,
            "valid_parquet_path": str(parquet_path),
            "valid_parquet_sha256": parquet_sha,
            "diagnostic_script_path": str(Path(__file__).resolve()),
            "diagnostic_script_sha256": file_sha256(Path(__file__).resolve()),
            "r4_validator_sha256": file_sha256(Path(r4.__file__).resolve()),
            "roles": role_records,
            "inputs_unchanged_after_scoring": True,
        },
        "coverage": {
            "valid_sequences": len(groups),
            "selected_rows": selected_rows,
            "official_expected_selected_rows": r4.EXPECTED_VALID_ROWS,
            "row_mask": "valid.parquet is_scored AND zero_based_position >= 99",
            "warmup_excluded_positions": [0, WARMUP],
            "position_bins_zero_based_half_open": [
                {"start": start, "end": end, "name": name} for start, end, name in POSITION_BINS
            ],
            "chunk_boundary_definition": {
                "chunk_sizes": list(CHUNK_SIZES),
                "radius_rows": BOUNDARY_RADIUS,
                "near_interval": "[boundary-8, boundary+8), 8 rows on each side of each interior boundary",
                "interior": "scored rows outside all boundary-near intervals",
                "interpretation": "predeclared positional grid aligned to upcoming Transformer horizons; not verified TL3 inference chunk or reset boundaries",
                "causality_limit": "near/interior score differences are descriptive and cannot diagnose reset bugs or causality because row populations differ",
            },
        },
        "metric_definitions": {
            "official_wp": "wnn.gmetric.GlobalAccumulator: clip targets and predictions to [-2,2], weight each target row by abs(clipped_target), pool all selected rows per target, then WP is the mean of t0 and t1; zero-weight or near-constant target/prediction variance returns 0",
            "wmse": "raw prediction minus clip(target,-2,2), squared and weighted by abs(clipped_target), divided by the target's total weight; pooled WMSE is total two-target numerator divided by total two-target weight",
            "residual_association": "weighted Pearson between same-row role residuals, separately per target, over identical official scored rows, with weights abs(clip(target,-2,2)); raw residual is prediction_raw - clipped_target; official-clipped residual is clip(prediction_raw,-2,2) - clipped_target; residuals themselves are not clipped",
        },
        "regions": region_results,
        "pairwise_weighted_residual_pearson": residual_results,
        "fixed_blend": {
            "weights": list(BLEND_WEIGHTS),
            "arithmetic": "FP32 left-to-right multiply/add; no clipping before WMSE; GlobalAccumulator clips blend predictions for WP",
        },
        "interpretation_limits": [
            "Exposed official validation is development evidence only, not an independent or public score.",
            "Position and chunk-boundary slices are predeclared descriptive diagnostics; they were not fitted or used to choose bins.",
            "No model was loaded or run, no blend search occurred, and no cache, dataset, checkpoint, sidecar, or provenance file was written.",
            "The existing mixture ancestry limitation remains: published legacy consumed-cache bytes are not reconstructed.",
        ],
    }


def main() -> None:
    """Run score-only diagnostics and write a create-only JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = diagnose(args.project_root, args.split_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
