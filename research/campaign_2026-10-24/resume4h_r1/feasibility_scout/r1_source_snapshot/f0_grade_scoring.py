"""Score the admitted F0 endpoints with disk-backed labels and predictions."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from f0_grade_preflight import PreparedF0Grade
from online_relationship_grade_core import npy_coverage, sha256_file

from wnn.evaluation import paired_component_bootstrap, paired_component_moments
from wnn.gmetric import GlobalAccumulator

ROWS = 20_000
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 271828
TIME_BINS = {
    "overall": (99, ROWS),
    "rows_99_500": (99, 500),
    "rows_500_2000": (500, 2_000),
    "rows_2000_10000": (2_000, 10_000),
    "rows_10000_20000": (10_000, ROWS),
}
MAX_NPY_MEMBER_BYTES = 1_000_000_000


def build_contrasts() -> list[dict[str, str]]:
    """Freeze the requested primary pairs and the original descriptive pairs."""
    contrasts: list[dict[str, str]] = []
    for lag in (1, 16):
        for candidate, reference in (("C", "B"), ("B", "A")):
            contrasts.append(
                {
                    "name": f"{candidate}-minus-{reference}_lag{lag}",
                    "candidate": f"{candidate}_lag{lag}",
                    "reference": f"{reference}_lag{lag}",
                    "role": "primary",
                }
            )
        contrasts.append(
            {
                "name": f"C-minus-A_lag{lag}",
                "candidate": f"C_lag{lag}",
                "reference": f"A_lag{lag}",
                "role": "secondary",
            }
        )
        for arm in ("A", "B", "C"):
            contrasts.append(
                {
                    "name": f"{arm}-minus-parent_lag{lag}",
                    "candidate": f"{arm}_lag{lag}",
                    "reference": "parent",
                    "role": "secondary",
                }
            )
    for arm in ("A", "B", "C"):
        contrasts.append(
            {
                "name": f"{arm}_lag16-minus-lag1",
                "candidate": f"{arm}_lag16",
                "reference": f"{arm}_lag1",
                "role": "secondary",
            }
        )
    return contrasts


def _extract_npy_member(archive_path: Path, member_name: str, destination: Path) -> dict[str, Any]:
    """Copy one NPY ZIP member to physical scratch in bounded chunks."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or member_name not in names:
            raise ValueError(f"target archive lacks a unique {member_name} member")
        member = archive.getinfo(member_name)
        if member.file_size > MAX_NPY_MEMBER_BYTES:
            raise ValueError(f"target member {member_name} exceeds the bounded disk allowance")
        written = 0
        with archive.open(member) as source, destination.open("xb") as output:
            while block := source.read(1 << 20):
                written += len(block)
                if written > member.file_size:
                    raise ValueError(f"target member {member_name} exceeds its ZIP size")
                output.write(block)
    if written != member.file_size:
        raise ValueError(f"target member {member_name} ended before its ZIP size")
    coverage = npy_coverage(destination)
    if coverage["file_bytes"] != member.file_size:
        raise ValueError(f"target member {member_name} is not a complete NPY file")
    return coverage


def _official_mask(mask: np.ndarray, start: int, stop: int) -> np.ndarray:
    rows = np.arange(ROWS)
    return np.asarray(mask, dtype=bool) & (rows >= start) & (rows < stop)


def stream_global_score(
    targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    """Accumulate the official metric one evaluation group at a time."""
    if targets.ndim != 3 or predictions.shape != targets.shape or mask.shape != targets.shape[:2]:
        raise ValueError("streaming score arrays must be aligned by group, row, and target")
    accumulator = GlobalAccumulator()
    for index in range(len(targets)):
        accumulator.add(targets[index], predictions[index], mask[index])
    return accumulator.result()


def _materialize_target_maps(
    prepared: PreparedF0Grade, scratch: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int], list[Path]]:
    target_path = prepared.target_cache_path
    with np.load(target_path, allow_pickle=False) as archive:
        expected_members = {"row_group", "y", "mask"}
        if not expected_members.issubset(archive.files):
            raise ValueError("official target cache lacks row_group, y, or mask")
        row_groups = np.asarray(archive["row_group"])
    if row_groups.ndim != 1 or row_groups.dtype.kind not in "iu":
        raise ValueError("official target row-group vector is malformed")
    if len(np.unique(row_groups)) != len(row_groups):
        raise ValueError("official target row-group vector contains duplicates")
    positions = {int(group): index for index, group in enumerate(row_groups)}
    if any(group not in positions for group in prepared.evaluation_groups):
        raise ValueError("official target cache omits source-qualified F0 evaluation groups")

    y_path = scratch / "valid_targets_y.npy"
    mask_path = scratch / "valid_targets_mask.npy"
    y_coverage = _extract_npy_member(target_path, "y.npy", y_path)
    mask_coverage = _extract_npy_member(target_path, "mask.npy", mask_path)
    expected_y_shape = [len(row_groups), ROWS, 2]
    expected_mask_shape = [len(row_groups), ROWS]
    if y_coverage["shape"] != expected_y_shape or y_coverage["dtype"] != "<f4":
        raise ValueError("official target member shape or dtype differs from the source contract")
    if mask_coverage["shape"] != expected_mask_shape or mask_coverage["dtype"] != "|b1":
        raise ValueError("official mask member shape or dtype differs from the source contract")
    targets = np.load(y_path, mmap_mode="r", allow_pickle=False)
    masks = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    return targets, masks, row_groups, positions, [y_path, mask_path]


def score_prepared_f0(prepared: PreparedF0Grade) -> dict[str, Any]:
    """Grade the seven endpoints while keeping full targets disk-backed."""
    expected_target_sha = prepared.parent_recipe["target_cache"]["sha256"]
    observed_target_sha = sha256_file(prepared.target_cache_path)
    if observed_target_sha != expected_target_sha:
        raise ValueError("official target archive differs from the frozen source hash")
    scratch = Path(prepared.parent_recipe["resource_contract"]["scratch"]).resolve()
    if not scratch.is_dir():
        raise ValueError("physical scratch root is unavailable")
    with tempfile.TemporaryDirectory(prefix="resume4h-f0-targets-", dir=scratch) as temp_name:
        targets, masks, row_groups, positions, temp_files = _materialize_target_maps(
            prepared, Path(temp_name)
        )
        del row_groups
        parent = np.load(prepared.parent_prediction_path, mmap_mode="r", allow_pickle=False)
        predictions = {"parent": parent}
        for name, path in prepared.candidate_prediction_paths.items():
            predictions[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        endpoint_names = ["parent", *sorted(prepared.candidate_prediction_paths)]
        accumulators = {
            name: {band: GlobalAccumulator() for band in TIME_BINS} for name in endpoint_names
        }
        moments = {
            name: np.zeros((len(prepared.evaluation_groups), 2, 6), dtype=np.float64)
            for name in endpoint_names
        }
        rows = np.arange(ROWS)
        for group_index, group in enumerate(prepared.evaluation_groups):
            target_index = positions[group]
            target_row = targets[target_index]
            mask_row = masks[target_index]
            for name in endpoint_names:
                prediction_row = predictions[name][group_index]
                for band, (start, stop) in TIME_BINS.items():
                    selected = np.asarray(mask_row, dtype=bool) & (rows >= start) & (rows < stop)
                    accumulators[name][band].add(target_row, prediction_row, selected)
                scored = _official_mask(mask_row, 99, ROWS)
                candidate_moments, _ = paired_component_moments(
                    target_row[None, ...],
                    prediction_row[None, ...],
                    prediction_row[None, ...],
                    scored[None, ...],
                )
                moments[name][group_index] = candidate_moments[0]

        endpoint_scores = {
            name: {band: accumulator.result() for band, accumulator in bands.items()}
            for name, bands in accumulators.items()
        }
        contrasts = []
        for contrast in build_contrasts():
            candidate = contrast["candidate"]
            reference = contrast["reference"]
            interval = paired_component_bootstrap(
                moments[candidate],
                moments[reference],
                prepared.components,
                n_boot=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED,
                centered=True,
            )
            time_bin_deltas = {}
            for band in TIME_BINS:
                candidate_score = endpoint_scores[candidate][band]
                reference_score = endpoint_scores[reference][band]
                if candidate_score["selected_rows"] != reference_score["selected_rows"]:
                    raise ValueError(f"contrast {contrast['name']} uses mismatched scored rows")
                time_bin_deltas[band] = {
                    "wp": candidate_score["wp"] - reference_score["wp"],
                    "t0": candidate_score["t0"] - reference_score["t0"],
                    "t1": candidate_score["t1"] - reference_score["t1"],
                    "selected_rows": candidate_score["selected_rows"],
                }
            contrasts.append(
                {
                    **contrast,
                    "point_wp_delta": (
                        endpoint_scores[candidate]["overall"]["wp"]
                        - endpoint_scores[reference]["overall"]["wp"]
                    ),
                    "point_t0_delta": (
                        endpoint_scores[candidate]["overall"]["t0"]
                        - endpoint_scores[reference]["overall"]["t0"]
                    ),
                    "point_t1_delta": (
                        endpoint_scores[candidate]["overall"]["t1"]
                        - endpoint_scores[reference]["overall"]["t1"]
                    ),
                    "paired_component_bootstrap": interval,
                    "time_bin_deltas": time_bin_deltas,
                    "uncertainty_scope": "conditional on these fixed fitted endpoints",
                }
            )
        result = {
            "schema": "wnn.online-relationships-f0-grade.v1",
            "status": "graded_exploratory_f0_only",
            "preflight": prepared.report,
            "metric": "global_wp_clip2_v1",
            "metric_source": "src/wnn/gmetric.py",
            "target_cache_identity": {
                "path": str(prepared.target_cache_path),
                "expected_sha256": expected_target_sha,
                "observed_sha256": observed_target_sha,
                "verified_before_label_reads": True,
            },
            "score_mask": "official mask AND row >= 99",
            "clip": [-2.0, 2.0],
            "weight": "absolute clipped target magnitude",
            "fold": 0,
            "seed": 1001,
            "evaluation_groups": len(prepared.evaluation_groups),
            "evaluation_components": int(len(np.unique(prepared.components))),
            "endpoint_scores": endpoint_scores,
            "contrasts": contrasts,
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
            "new_fit_count": 0,
            "exposure": "F0 was already exposed by prior research; exploratory only",
            "limits": [
                "F0/seed1001 is one exposed cell; fold and fitting seed effects are not separated.",
                "Component intervals condition on fixed predictions and do not measure fit or transfer uncertainty.",
                "No independent checkpoint inference parity or public score is established.",
                "The original F0/F1 12-fit panel remains incomplete.",
            ],
            "scratch_extraction": {
                "root": str(scratch),
                "members": [path.name for path in temp_files],
                "persistent_target_copy": False,
            },
        }
        del predictions, parent, targets, masks
        return result
