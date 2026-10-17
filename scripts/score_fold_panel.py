"""Grade a run or fixed blend on an explicit source-qualified evaluation fold.

Example::

    python scripts/score_fold_panel.py candidate_s3_f3@ep2 \
        --split research/campaign_2026-09-29/artifacts/splits/fold3.json \
        --expected-fold 3 --reference parent_s3_f3@ep2 --out grade.json

The scorer validates fitting membership, checkpoint identity and prediction
sidecars before applying the unchanged official metric. The campaign folds are
development evidence; their repeated inspection does not make them fresh holdouts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_bandit_continuation import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    SEQUENCE_ROWS,
    _official_mask,
    _read_json,
    _validate_prediction_artifact,
    compare_arrays,
)

from wnn.gmetric import METRIC_ID, global_wp  # noqa: E402
from wnn.splits import CampaignSplit, file_sha256, load_campaign_split  # noqa: E402

RUN_AT_EPOCH = re.compile(r"([A-Za-z0-9_.-]{1,96})@ep([0-9]+)\Z")
RUN_NAME = re.compile(r"[A-Za-z0-9_.-]{1,96}\Z")


def _parse_member(run_at: str, default_epoch: int | None = None) -> tuple[str, int]:
    """Parse a safe run name and fixed checkpoint epoch."""
    match = RUN_AT_EPOCH.fullmatch(run_at)
    if match is not None:
        return match.group(1), int(match.group(2))
    if default_epoch is not None and RUN_NAME.fullmatch(run_at):
        return run_at, default_epoch
    raise ValueError(f"malformed run reference {run_at!r}; expected RUN@epN")


def _member(
    run_at: str,
    split: CampaignSplit,
    row_groups: list[int],
    root: Path,
    *,
    default_epoch: int | None = None,
) -> tuple[np.memmap, dict[str, object]]:
    """Load a prediction artifact bound to the selected split and checkpoint."""
    run, epoch = _parse_member(run_at, default_epoch)
    run_root = root / "artifacts" / "runs" / run
    provenance = _read_json(run_root / "provenance.json")
    fit = provenance.get("fit_row_groups")
    if not isinstance(fit, dict):
        raise ValueError(f"{run}: provenance lacks fit-row membership")
    train_groups = fit.get("train")
    valid_groups = fit.get("valid")
    if not isinstance(train_groups, list) or not isinstance(valid_groups, list):
        raise ValueError(f"{run}: provenance fit membership must list train and valid groups")
    raw_groups = [*train_groups, *valid_groups]
    if any(not isinstance(group, int) or isinstance(group, bool) for group in raw_groups):
        raise ValueError(f"{run}: provenance fit membership contains non-integer groups")
    train = set(train_groups)
    valid = set(valid_groups)
    if len(train) != len(train_groups) or len(valid) != len(valid_groups):
        raise ValueError(f"{run}: provenance fit membership contains duplicate groups")
    if valid & set(split.evaluation_groups):
        raise ValueError(f"{run}: evaluation groups occur in direct fit membership")
    split.assert_fit_groups(train, valid)

    prediction_path = root / "cache" / f"pred_{run}_valid.npy"
    sidecar_path = prediction_path.with_suffix(".json")
    if not prediction_path.is_file():
        raise FileNotFoundError(f"{run}: expected prediction array is missing: {prediction_path}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"{run}: prediction sidecar is missing: {sidecar_path}")
    prediction, binding = _validate_prediction_artifact(
        prediction_path,
        _read_json(sidecar_path),
        run=run,
        epoch=epoch,
        split_sha256=split.sha256,
        checkpoint_sha256=file_sha256(run_root / f"model_ep{epoch}.pt"),
        row_groups=row_groups,
        shape=(len(row_groups), SEQUENCE_ROWS, 2),
    )
    return prediction, binding


def _parse_blend_member(part: str, default_epoch: int | None = None) -> tuple[str, float]:
    """Parse one RUN@epN[*weight] blend component."""
    run_at, separator, raw_weight = part.partition("*")
    if separator and (not raw_weight or "*" in raw_weight):
        raise ValueError(f"malformed blend member {part!r}")
    _parse_member(run_at, default_epoch)
    try:
        weight = float(raw_weight) if separator else 1.0
    except ValueError as error:
        raise ValueError(f"malformed blend weight in {part!r}") from error
    if not math.isfinite(weight):
        raise ValueError(f"blend weight must be finite in {part!r}")
    return run_at, weight


def _blend(
    spec: str,
    split: CampaignSplit,
    row_groups: list[int],
    root: Path,
    indices: np.ndarray,
    *,
    default_epoch: int | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Combine raw fixed-weight predictions before official metric clipping."""
    parts = spec.split(",")
    if not parts or any(not part for part in parts):
        raise ValueError(f"malformed blend specification {spec!r}")
    total = np.zeros((len(indices), SEQUENCE_ROWS, 2), dtype=np.float64)
    bindings: list[dict[str, object]] = []
    for part in parts:
        run_at, weight = _parse_blend_member(part, default_epoch)
        prediction, binding = _member(run_at, split, row_groups, root, default_epoch=default_epoch)
        total += weight * np.asarray(prediction[indices], dtype=np.float64)
        bindings.append(binding | {"weight": weight})
    return total.astype(np.float32), bindings


def _load_targets(
    target_path: Path, evaluation_groups: tuple[int, ...]
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Read targets in manifest order and reject ambiguous or missing row groups."""
    with np.load(target_path, allow_pickle=False) as cache:
        row_groups = list(map(int, cache["row_group"]))
        if len(set(row_groups)) != len(row_groups):
            raise ValueError("valid target cache contains duplicate row groups")
        position = {group: index for index, group in enumerate(row_groups)}
        missing = [int(group) for group in evaluation_groups if int(group) not in position]
        if missing:
            raise ValueError(f"valid target cache lacks evaluation groups: {missing[:5]}")
        indices = np.asarray([position[int(group)] for group in evaluation_groups], dtype=np.int64)
        targets = np.asarray(cache["y"][indices], dtype=np.float32)
        is_scored = np.asarray(cache["mask"][indices], dtype=bool)
    expected_target_shape = (len(evaluation_groups), SEQUENCE_ROWS, 2)
    if targets.shape != expected_target_shape:
        raise ValueError("valid target cache has an incomplete target shape")
    if is_scored.shape != (len(evaluation_groups), SEQUENCE_ROWS):
        raise ValueError("valid target cache has an incomplete mask shape")
    return row_groups, indices, targets, is_scored


def score_panel(
    candidate_spec: str,
    *,
    references: Sequence[str],
    split_path: Path,
    expected_fold: int,
    output_path: Path,
    root: Path = ROOT,
    n_boot: int = BOOTSTRAP_REPLICATES,
    candidate_epoch: int | None = None,
    expected_group_count: int | None = None,
    exposure_description: str | None = None,
    include_fold_metadata: bool = True,
) -> dict[str, object]:
    """Validate and score one candidate against fixed references on one fold.

    Args:
        candidate_spec: A run name or comma-separated fixed-weight blend.
        references: Fixed run or blend specifications for paired comparisons.
        split_path: Campaign split manifest used by fitting and evaluation.
        expected_fold: Fold number declared by the caller.
        output_path: Grade JSON path to write.
        root: Project root holding source data, runs, targets and predictions.
        n_boot: Number of paired component-bootstrap draws.
        candidate_epoch: Default epoch for an unqualified candidate run name.
        expected_group_count: Optional fixed group-count contract supplied by a wrapper.
        exposure_description: Optional wrapper-specific exposure statement.
        include_fold_metadata: Whether to include generic fold and split-path fields.

    Returns:
        The grade report written to ``output_path``.
    """
    if expected_fold < 0:
        raise ValueError("expected fold must be non-negative")
    split = load_campaign_split(split_path, root)
    if split.fold != expected_fold:
        raise ValueError(
            f"split manifest declares fold {split.fold}, expected fold {expected_fold}"
        )
    if expected_group_count is not None and len(split.evaluation_groups) != expected_group_count:
        raise ValueError(f"split requires {expected_group_count} evaluation groups")

    target_path = root / "cache" / "valid_targets.npz"
    row_groups, indices, targets, is_scored = _load_targets(target_path, split.evaluation_groups)
    mask = _official_mask(is_scored)
    component_of = {
        int(group): int(component)
        for source, group, component in split.components
        if source == "valid"
    }
    missing_components = [
        int(group) for group in split.evaluation_groups if int(group) not in component_of
    ]
    if missing_components:
        raise ValueError(f"split manifest lacks valid component IDs: {missing_components[:5]}")
    components = np.asarray(
        [component_of[int(group)] for group in split.evaluation_groups], dtype=np.int64
    )

    candidate, candidate_bindings = _blend(
        candidate_spec,
        split,
        row_groups,
        root,
        indices,
        default_epoch=candidate_epoch,
    )
    report: dict[str, object] = {
        "metric_id": METRIC_ID,
        "created_utc": datetime.now(UTC).isoformat(),
        "candidate": candidate_spec,
        "candidate_members": candidate_bindings,
        "split_manifest_sha256": split.sha256,
        "valid_targets_sha256": file_sha256(target_path),
        "evaluation_groups": len(indices),
        "components": int(len(set(components.tolist()))),
        "official": global_wp(targets, candidate, mask),
        "exposure": exposure_description
        or (
            f"fold{expected_fold} source-qualified evaluation panel; repeatedly inspected "
            "development evidence, not a fresh holdout or public forecast"
        ),
        "contrasts": [],
    }
    if include_fold_metadata:
        report["fold"] = expected_fold
        report["expected_fold"] = expected_fold
        report["split_path"] = str(split_path)
    for reference_spec in references:
        reference, reference_bindings = _blend(reference_spec, split, row_groups, root, indices)
        contrast = compare_arrays(
            targets,
            candidate,
            reference,
            is_scored,
            components,
            n_boot=n_boot,
            seed=BOOTSTRAP_SEED,
        )
        report["contrasts"].append(
            {
                "reference": reference_spec,
                "reference_members": reference_bindings,
                "reference_official": global_wp(targets, reference, mask),
                "paired": contrast,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=1, default=float))
    return report


def main() -> None:
    """Command-line entry point requiring an explicit split and fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="RUN@epN or a fixed blend RUN@epN*W,...")
    parser.add_argument("--reference", action="append", default=[], help="fixed RUN@epN blend")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--expected-fold", type=int, required=True)
    parser.add_argument("--epoch", type=int, help="default epoch for an unqualified candidate")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="project/data root holding runs, targets, and predictions (defaults to this checkout)",
    )
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    report = score_panel(
        args.candidate,
        references=args.reference,
        split_path=args.split,
        expected_fold=args.expected_fold,
        output_path=args.out,
        root=args.root,
        n_boot=args.n_boot,
        candidate_epoch=args.epoch,
    )
    print(json.dumps({"candidate": args.candidate, "wp": report["official"]["wp"]}))


if __name__ == "__main__":
    main()
