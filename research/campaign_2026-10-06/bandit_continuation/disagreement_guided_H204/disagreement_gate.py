from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from disagreement_screen import E514Inputs

GATE_QUANTILE = 0.75
WARMUP_ROWS = 99
CLIP = 2.0
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 3_772_181_504
EXPECTED_MANIFEST_SHA256 = "8de4dc6c1e2a9023510d3edbb3a46771579470bfb859ac50f451f7baa269d6fd"
EXPECTED_PARTITION_SHA256 = "e9e0cb9391789f54e34c1d02836d02c336c5d3a13bde6f9473196bbbc71da1c2"
EXPECTED_ROW_GROUPS_SHA256 = "e9584232498a4ac40ca0ad2bebeb1b136862e2de242c2ec4378ea425357aa4be"
SOURCE_MANIFEST = (
    "research/campaign_2026-10-06/bandit_continuation/"
    "conditional_weak_portfolio_E514/SOURCE_MANIFEST_E514_V3.json"
)
CHAIN_PARTITION = (
    "research/campaign_2026-10-06/bandit_continuation/"
    "conditional_weak_portfolio_E514/E514_CHAIN_PARTITION.json"
)
E517_RELATIVE_DIR = Path(
    "research/campaign_2026-10-06/bandit_continuation/disagreement_guided_H204"
)


@dataclass(frozen=True)
class GateFit:
    """Frozen global and disagreement-gated convex blend for one target."""

    threshold: float
    gated_beta: float
    global_beta: float
    calibration_rows: int
    calibration_gated_rows: int
    calibration_gate_fraction: float
    calibration_weighted_gate_fraction: float


def _beta(
    target: NDArray[np.float64],
    anchor: NDArray[np.float64],
    delta: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """Fit the clipped convex coefficient for one fixed prediction direction."""
    denominator = float(np.sum(weights * delta * delta, dtype=np.float64))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    numerator = float(np.sum(weights * delta * (target - anchor), dtype=np.float64))
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def fit_row_gate(
    targets: NDArray[np.float32],
    anchor: NDArray[np.float32],
    candidate: NDArray[np.float32],
    mask: NDArray[np.bool_],
    *,
    quantile: float = GATE_QUANTILE,
) -> GateFit:
    """Fit one causal row-disagreement gate using calibration rows only.

    The gate threshold depends only on candidate-anchor magnitude. The blend
    coefficient minimizes the official weighted squared-error surrogate inside
    that gate, constrained to a convex mixture. The caller supplies the
    preregistered scored-row mask.
    """
    if targets.ndim != 2 or targets.shape != anchor.shape or targets.shape != candidate.shape:
        raise ValueError("targets, anchor, and candidate must have one equal 2D shape")
    if mask.shape != targets.shape:
        raise ValueError("mask must match the group-by-time arrays")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    if not mask.any():
        raise ValueError("calibration mask selects no rows")

    target = np.clip(targets.astype(np.float64), -CLIP, CLIP)
    anchor_values = np.clip(anchor.astype(np.float64), -CLIP, CLIP)
    candidate_values = np.clip(candidate.astype(np.float64), -CLIP, CLIP)
    selected = mask
    if not (
        np.isfinite(target[selected]).all()
        and np.isfinite(anchor_values[selected]).all()
        and np.isfinite(candidate_values[selected]).all()
    ):
        raise ValueError("selected calibration values must be finite")

    delta = candidate_values - anchor_values
    threshold = float(np.quantile(np.abs(delta[selected]), quantile, method="linear"))
    gate = np.abs(delta) >= threshold
    weights = np.abs(target)
    selected_gate = selected & gate
    gated_beta = _beta(
        target[selected_gate],
        anchor_values[selected_gate],
        delta[selected_gate],
        weights[selected_gate],
    )
    global_beta = _beta(
        target[selected],
        anchor_values[selected],
        delta[selected],
        weights[selected],
    )
    selected_count = int(selected.sum())
    gated_count = int(selected_gate.sum())
    total_metric_weight = float(weights[selected].sum(dtype=np.float64))
    gated_metric_weight = float(weights[selected_gate].sum(dtype=np.float64))
    return GateFit(
        threshold=threshold,
        gated_beta=gated_beta,
        global_beta=global_beta,
        calibration_rows=selected_count,
        calibration_gated_rows=gated_count,
        calibration_gate_fraction=gated_count / selected_count,
        calibration_weighted_gate_fraction=(
            gated_metric_weight / total_metric_weight if total_metric_weight > 0 else 0.0
        ),
    )


def apply_row_gate(
    anchor: NDArray[np.float32],
    candidate: NDArray[np.float32],
    fit: GateFit,
    *,
    gated: bool,
) -> NDArray[np.float32]:
    """Apply a frozen global or disagreement-gated blend to any row matrix."""
    if anchor.ndim != 2 or anchor.shape != candidate.shape:
        raise ValueError("anchor and candidate must have one equal 2D shape")
    anchor_values = np.clip(anchor.astype(np.float64), -CLIP, CLIP)
    candidate_values = np.clip(candidate.astype(np.float64), -CLIP, CLIP)
    delta = candidate_values - anchor_values
    beta = fit.gated_beta if gated else fit.global_beta
    gate = np.abs(delta) >= fit.threshold if gated else np.ones(delta.shape, dtype=bool)
    return np.clip(anchor_values + beta * gate * delta, -CLIP, CLIP).astype(np.float32)


def _merge_pair(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    """Merge target-wise centered moments for a chain bootstrap replicate."""
    from wnn.gmetric import _merge_moments

    return np.stack([_merge_moments(left[index], right[index]) for index in (0, 1)])


def _chain_moments(
    inputs: E514Inputs,
    predictions: NDArray[np.float32],
    metric_mask: NDArray[np.bool_],
    chain_ids: NDArray[np.int64],
) -> dict[int, NDArray[np.float64]]:
    """Return official stable centered moments separately for each chain."""
    from wnn.gmetric import GlobalAccumulator

    output: dict[int, NDArray[np.float64]] = {}
    for chain in chain_ids:
        rows = np.flatnonzero(inputs.chain_by_sequence == chain)
        accumulator = GlobalAccumulator()
        accumulator.add(inputs.targets[rows], predictions[rows], metric_mask[rows])
        output[int(chain)] = accumulator._moments.copy()
    return output


def _paired_bootstrap(
    inputs: E514Inputs,
    predictions: tuple[NDArray[np.float32], ...],
    metric_mask: NDArray[np.bool_],
    chain_ids: NDArray[np.int64],
    seed: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """Compute paired, fold-stratified complete-chain intervals conditionally."""
    from wnn.gmetric import _wp_from_moments

    moments = [
        _chain_moments(inputs, prediction, metric_mask, chain_ids) for prediction in predictions
    ]
    outer_folds = sorted(set(inputs.chain_to_outer_fold.values()))
    fold_chains = {
        fold: np.asarray(
            [chain for chain in chain_ids if inputs.chain_to_outer_fold[int(chain)] == fold],
            dtype=np.int64,
        )
        for fold in outer_folds
    }
    rng = np.random.default_rng(seed)
    deltas_vs_anchor = np.empty((BOOTSTRAP_REPLICATES, len(predictions) - 1), dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        totals = [np.zeros((2, 6), dtype=np.float64) for _ in predictions]
        for fold in outer_folds:
            fold_members = fold_chains[fold]
            sampled = rng.choice(fold_members, size=fold_members.size, replace=True)
            for chain in sampled:
                for model_index in range(len(predictions)):
                    totals[model_index] = _merge_pair(
                        totals[model_index], moments[model_index][int(chain)]
                    )
        scores = [
            0.5 * sum(_wp_from_moments(target) for target in model_moments)
            for model_moments in totals
        ]
        for model_index in range(1, len(predictions)):
            deltas_vs_anchor[replicate, model_index - 1] = scores[model_index] - scores[0]

    intervals = np.quantile(deltas_vs_anchor, [0.025, 0.975], axis=0).T.tolist()
    # Columns are [anchor, gated blend, global blend].
    gated_vs_global = deltas_vs_anchor[:, 0] - deltas_vs_anchor[:, 1]
    gated_global_interval = np.quantile(gated_vs_global, [0.025, 0.975]).tolist()
    return intervals, [gated_global_interval]


def fit_and_crossfit(
    inputs: E514Inputs,
    *,
    seed: int,
) -> tuple[dict[str, object], NDArray[np.float32], NDArray[np.float32]]:
    """Cross-fit one gated and one global ab12 blend on frozen whole chains."""
    from wnn.gmetric import global_wp

    if inputs.anchor.shape != inputs.targets.shape:
        raise ValueError("anchor and target cache shapes differ")
    if inputs.candidate_predictions["ab12"].shape != inputs.targets.shape:
        raise ValueError("ab12 prediction shape differs from target cache")
    chain_ids = np.unique(inputs.chain_by_sequence).astype(np.int64)
    outer_folds = sorted(set(inputs.chain_to_outer_fold.values()))
    if outer_folds != [0, 1, 2]:
        raise ValueError("expected the frozen three-fold chain partition")

    warmup = np.arange(inputs.targets.shape[1]) >= WARMUP_ROWS
    metric_mask = inputs.is_scored & warmup[None, :]
    candidate_t0 = inputs.candidate_predictions["ab12"][:, :, 0]
    anchor_t0 = inputs.anchor[:, :, 0]
    target_t0 = inputs.targets[:, :, 0]
    gated_oof = inputs.anchor.copy()
    global_oof = inputs.anchor.copy()
    fold_records: list[dict[str, object]] = []

    for outer_fold in outer_folds:
        evaluation_chains = np.asarray(
            [chain for chain in chain_ids if inputs.chain_to_outer_fold[int(chain)] == outer_fold]
        )
        calibration_chains = np.asarray(
            [chain for chain in chain_ids if inputs.chain_to_outer_fold[int(chain)] != outer_fold]
        )
        evaluation_rows = np.flatnonzero(np.isin(inputs.chain_by_sequence, evaluation_chains))
        calibration_rows = np.flatnonzero(np.isin(inputs.chain_by_sequence, calibration_chains))
        fit = fit_row_gate(
            target_t0[calibration_rows],
            anchor_t0[calibration_rows],
            candidate_t0[calibration_rows],
            metric_mask[calibration_rows],
        )
        gated_oof[evaluation_rows, :, 0] = apply_row_gate(
            anchor_t0[evaluation_rows], candidate_t0[evaluation_rows], fit, gated=True
        )
        global_oof[evaluation_rows, :, 0] = apply_row_gate(
            anchor_t0[evaluation_rows], candidate_t0[evaluation_rows], fit, gated=False
        )
        evaluation_mask = metric_mask[evaluation_rows]
        evaluation_target = np.clip(target_t0[evaluation_rows].astype(np.float64), -CLIP, CLIP)
        evaluation_delta = np.clip(
            candidate_t0[evaluation_rows].astype(np.float64), -CLIP, CLIP
        ) - np.clip(anchor_t0[evaluation_rows].astype(np.float64), -CLIP, CLIP)
        evaluation_gate = np.abs(evaluation_delta) >= fit.threshold
        selected_gate = evaluation_mask & evaluation_gate
        evaluation_weight = np.abs(evaluation_target) * evaluation_mask
        evaluation_weight_total = float(evaluation_weight.sum(dtype=np.float64))
        evaluation_weight_gated = float(
            (np.abs(evaluation_target) * selected_gate).sum(dtype=np.float64)
        )
        evaluation_rows_scored = int(evaluation_mask.sum())
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "calibration_chain_count": int(calibration_chains.size),
                "evaluation_chain_count": int(evaluation_chains.size),
                "fit": asdict(fit),
                "evaluation_gate": {
                    "scored_row_count": evaluation_rows_scored,
                    "gated_scored_row_count": int(selected_gate.sum()),
                    "gated_scored_row_fraction": (
                        int(selected_gate.sum()) / evaluation_rows_scored
                        if evaluation_rows_scored
                        else 0.0
                    ),
                    "gated_official_weight_fraction": (
                        evaluation_weight_gated / evaluation_weight_total
                        if evaluation_weight_total > 0
                        else 0.0
                    ),
                },
                "anchor": global_wp(
                    inputs.targets[evaluation_rows],
                    inputs.anchor[evaluation_rows],
                    metric_mask[evaluation_rows],
                ),
                "gated_ab12": global_wp(
                    inputs.targets[evaluation_rows],
                    gated_oof[evaluation_rows],
                    metric_mask[evaluation_rows],
                ),
                "global_ab12": global_wp(
                    inputs.targets[evaluation_rows],
                    global_oof[evaluation_rows],
                    metric_mask[evaluation_rows],
                ),
            }
        )

    endpoints = [
        global_wp(inputs.targets, prediction, metric_mask)
        for prediction in (inputs.anchor, gated_oof, global_oof)
    ]
    bootstrap_intervals, gated_vs_global_interval = _paired_bootstrap(
        inputs,
        (inputs.anchor, gated_oof, global_oof),
        metric_mask,
        chain_ids,
        seed,
    )
    gated_anchor_interval = bootstrap_intervals[0]
    gated_global_interval = gated_vs_global_interval[0]
    positive_gate_folds = sum(float(fold["fit"]["gated_beta"]) > 0.0 for fold in fold_records)
    development_gate_passed = bool(
        endpoints[1]["wp"] > endpoints[0]["wp"]
        and endpoints[1]["wp"] > endpoints[2]["wp"]
        and gated_anchor_interval[0] > 0.0
        and gated_global_interval[0] > 0.0
        and positive_gate_folds >= 2
    )
    report: dict[str, object] = {
        "experiment_id": "E-517",
        "hypothesis_id": "H-207",
        "bandit_arm": "arm:disagreement-gated-weak-blend",
        "metric": "global_wp_clip2_v1",
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "chain_partition_sha256": inputs.chain_partition_sha256,
        "data_exposure": "E-514 pool and fold-0 groups previously inspected; exploratory only",
        "candidate": "ab12",
        "target_with_candidate_gate": 0,
        "target_left_at_anchor": 1,
        "gate_feature": "rowwise absolute difference between clipped ab12 and anchor predictions",
        "gate_quantile": GATE_QUANTILE,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": seed,
        "bootstrap_conditional_on_outer_fits": True,
        "anchor": endpoints[0],
        "gated_ab12": endpoints[1],
        "global_ab12": endpoints[2],
        "delta_gated_vs_anchor": float(endpoints[1]["wp"] - endpoints[0]["wp"]),
        "delta_global_vs_anchor": float(endpoints[2]["wp"] - endpoints[0]["wp"]),
        "delta_gated_vs_global": float(endpoints[1]["wp"] - endpoints[2]["wp"]),
        "paired_chain_bootstrap_delta_gated_vs_anchor_ci95": bootstrap_intervals[0],
        "paired_chain_bootstrap_delta_global_vs_anchor_ci95": bootstrap_intervals[1],
        "paired_chain_bootstrap_delta_gated_vs_global_ci95": gated_vs_global_interval[0],
        "positive_gate_folds": positive_gate_folds,
        "development_gate_passed": development_gate_passed,
        "folds": fold_records,
        "new_public_gain_verified": False,
        "official_metric_score": None,
        "expected_public_gain": None,
        "public_forecast": None,
        "numerical_prior_update": None,
        "oof_gated_sha256": hashlib.sha256(memoryview(gated_oof).cast("B")).hexdigest(),
        "oof_global_sha256": hashlib.sha256(memoryview(global_oof).cast("B")).hexdigest(),
    }
    return report, gated_oof, global_oof


def validate_registered_inputs(inputs: E514Inputs) -> dict[str, str]:
    """Require the exact audited E-514 source pool and row-group population."""
    if inputs.source_manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("E-517 source-manifest identity does not match registration")
    if inputs.chain_partition_sha256 != EXPECTED_PARTITION_SHA256:
        raise ValueError("E-517 chain-partition identity does not match registration")
    row_groups = np.asarray(inputs.row_groups, dtype="<i8")
    row_group_digest = hashlib.sha256(row_groups.tobytes()).hexdigest()
    if row_groups.shape != (432,) or row_group_digest != EXPECTED_ROW_GROUPS_SHA256:
        raise ValueError("E-517 row-group identity does not match registration")
    if inputs.chain_by_sequence.shape != (432,):
        raise ValueError("E-517 sequence-to-chain map differs from the frozen population")
    if inputs.targets.shape != (432, 20_000, 2):
        raise ValueError("E-517 target shape differs from the frozen population")
    if inputs.anchor.shape != inputs.targets.shape:
        raise ValueError("E-517 anchor shape differs from targets")
    if inputs.candidate_predictions["ab12"].shape != inputs.targets.shape:
        raise ValueError("E-517 ab12 prediction shape differs from targets")
    chain_ids = np.unique(inputs.chain_by_sequence)
    fold_counts = {
        fold: sum(inputs.chain_to_outer_fold[int(chain)] == fold for chain in chain_ids)
        for fold in (0, 1, 2)
    }
    if chain_ids.size != 123 or fold_counts != {0: 41, 1: 41, 2: 41}:
        raise ValueError("E-517 chain population differs from the frozen partition")
    return {
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "chain_partition_sha256": inputs.chain_partition_sha256,
        "row_groups_sha256": row_group_digest,
    }


def run_registered_e517(
    root: Path, inputs: E514Inputs | None = None
) -> tuple[
    E514Inputs,
    dict[str, object],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """Run E-517 on frozen verified inputs and persist its registered outputs.

    The live notebook passes the shared input object produced by its upstream
    ``load_e514_inputs`` cell to avoid duplicating several hundred megabytes.
    Standalone callers omit it and load through the same frozen verifier.
    """
    loaded_internally = inputs is None
    if loaded_internally:
        from disagreement_screen import load_e514_inputs

        inputs = load_e514_inputs(root)
    identity = validate_registered_inputs(inputs)
    result, gated_oof, global_oof = fit_and_crossfit(inputs, seed=BOOTSTRAP_SEED)
    result["source_manifest"] = SOURCE_MANIFEST
    result["chain_partition"] = CHAIN_PARTITION
    result["input_load_mode"] = (
        "registered_loader" if loaded_internally else "shared_verified_marimo_input_cell"
    )
    result["row_groups_sha256"] = identity["row_groups_sha256"]
    result["row_groups_order"] = "E514 manifest order after pure fold-0 selection"
    result["bootstrap_seed_source_sha256"] = (
        "e0d6ec0015e849a93ffaeca8df29ef9e21ceaf36575d4156fbd7e94810036161"
    )
    result["implementation_sha256"] = _sha256(Path(__file__))
    result["input_loader_sha256"] = _sha256(Path(__file__).with_name("disagreement_screen.py"))
    result["preregistration_sha256"] = _sha256(
        root / E517_RELATIVE_DIR / "GATED_BLEND_PREREGISTRATION_E517.md"
    )

    output_dir = root / E517_RELATIVE_DIR
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    gated_path = output_dir / "E517_GATED_OOF_PREDICTIONS.npy"
    global_path = output_dir / "E517_GLOBAL_OOF_PREDICTIONS.npy"
    np.save(gated_path, gated_oof, allow_pickle=False)
    np.save(global_path, global_oof, allow_pickle=False)
    result["oof_gated_file_sha256"] = _sha256(gated_path)
    result["oof_global_file_sha256"] = _sha256(global_path)
    chain_summary_path = plot_dir / "E517_CHAIN_SUMMARY.csv"
    save_chain_summary(inputs, inputs.anchor, gated_oof, global_oof, chain_summary_path)
    result["chain_summary_path"] = str(chain_summary_path.relative_to(root))
    result_path = output_dir / "E517_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return inputs, result, gated_oof, global_oof


def _sha256(path: Path) -> str:
    """Hash a persisted result artifact in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_chain_summary(
    inputs: E514Inputs,
    anchor: NDArray[np.float32],
    gated: NDArray[np.float32],
    global_blend: NDArray[np.float32],
    output_path: Path,
) -> None:
    """Save descriptive per-chain official WP values for the frozen endpoints."""
    from wnn.gmetric import global_wp

    warmup = np.arange(inputs.targets.shape[1]) >= WARMUP_ROWS
    mask = inputs.is_scored & warmup[None, :]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "chain_id",
        "outer_fold",
        "anchor_wp",
        "gated_wp",
        "global_wp",
        "gated_delta_vs_anchor",
        "gated_delta_vs_global",
    ]
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for chain in np.unique(inputs.chain_by_sequence):
            rows = np.flatnonzero(inputs.chain_by_sequence == chain)
            scores = [
                global_wp(inputs.targets[rows], prediction[rows], mask[rows])["wp"]
                for prediction in (anchor, gated, global_blend)
            ]
            writer.writerow(
                {
                    "chain_id": int(chain),
                    "outer_fold": int(inputs.chain_to_outer_fold[int(chain)]),
                    "anchor_wp": scores[0],
                    "gated_wp": scores[1],
                    "global_wp": scores[2],
                    "gated_delta_vs_anchor": scores[1] - scores[0],
                    "gated_delta_vs_global": scores[1] - scores[2],
                }
            )
