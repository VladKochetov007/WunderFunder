"""Score only the new combo endpoint and audit unchanged evaluation labels."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
from idea_screens_r3.probe_helpers import metric_without, raw_sums_by_group
from idea_screens_r15.representation import (
    DictionaryAutoencoder,
    R15Predictor,
    build_sequence_features,
)

from wnn.gmetric import GlobalAccumulator

EVAL_ROWS = 59_436


def _predict(model: R15Predictor, features: np.ndarray) -> np.ndarray:
    """Run one final endpoint in bounded CPU batches."""
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), 2048):
            batch = torch.from_numpy(np.ascontiguousarray(features[start : start + 2048]))
            mean, logvar = model.forward_all(batch)
            if not torch.isfinite(mean).all().item() or not torch.isfinite(logvar).all().item():
                raise FloatingPointError("nonfinite combo endpoint output")
            outputs.append(mean.cpu().numpy())
    if not outputs:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(outputs)


def _reference_arm(result: Mapping[str, object], arm: str) -> Mapping[str, object]:
    try:
        return result["evaluation"]["arms"][arm]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"stored R15 result is missing its {arm} endpoint") from exc


def _label_sums_match(
    candidate: Mapping[str, list[list[float]]], reference: Mapping[str, object]
) -> None:
    """Verify component membership and target-only W/WY/WYY sums for a stored arm."""
    if set(candidate) != set(reference):
        raise ValueError("combo endpoint component set differs from stored R15 endpoint")
    for component, actual in candidate.items():
        expected = np.asarray(reference[component], dtype=np.float64)
        observed = np.asarray(actual, dtype=np.float64)
        if expected.shape != (2, 6) or observed.shape != (2, 6):
            raise ValueError("stored or candidate component sufficient statistics have bad shape")
        if not np.allclose(observed[:, (0, 1, 3)], expected[:, (0, 1, 3)], rtol=0.0, atol=1e-10):
            raise ValueError(f"combo evaluation labels/mask differ for component {component}")


def score_candidate(
    sequences: Sequence[dict[str, object]],
    model: R15Predictor,
    dictionary: DictionaryAutoencoder,
    raw_mean: np.ndarray,
    raw_scale: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    reference_result: Mapping[str, object],
    memory_guard: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    """Score new mean predictions with canonical WP and verify R15 label sums."""
    pooled = GlobalAccumulator()
    per_component: dict[int, GlobalAccumulator] = {}
    component_rows: dict[str, int] = {}
    rows_seen = 0
    eval_sequences = [item for item in sequences if item.get("role") == "eval"]
    if len(eval_sequences) != 32:
        raise ValueError("combo scoring requires exactly 32 evaluation components")
    for index, sequence in enumerate(eval_sequences):
        memory_guard(f"before_combo_eval_component_{index}")
        x = np.asarray(sequence["x"], dtype=np.float32)
        y = np.asarray(sequence["y"], dtype=np.float32)
        mask = np.asarray(sequence["scored"], dtype=bool)
        component = int(sequence["component"])
        if x.shape != (20_000, 112) or y.shape != (20_000, 2) or mask.shape != (20_000,):
            raise ValueError("combo evaluation sequence dimensions differ from the accepted panel")
        if component in per_component:
            raise ValueError("combo evaluation component is duplicated")
        selected_y = np.asarray(y[mask], dtype=np.float32)
        if not np.isfinite(selected_y).all():
            raise ValueError("nonfinite selected combo evaluation target")
        full, _ = build_sequence_features(x, raw_mean, raw_scale, dictionary)
        selected = full[mask]
        selected[:] = ((selected.astype(np.float64) - feature_mean) / feature_scale).astype(
            np.float32
        )
        prediction = _predict(model, selected)
        valid = np.ones(len(selected_y), dtype=bool)
        part = GlobalAccumulator()
        if len(selected_y):
            part.add(selected_y, prediction, valid)
            pooled.add(selected_y, prediction, valid, group=component)
        per_component[component] = part
        component_rows[str(component)] = len(selected_y)
        rows_seen += len(selected_y)
        del full, selected
        memory_guard(f"after_combo_eval_component_{index}")
    if rows_seen != EVAL_ROWS:
        raise ValueError(f"combo expected {EVAL_ROWS} selected evaluation rows, got {rows_seen}")
    stored_arms = {arm: _reference_arm(reference_result, arm) for arm in ("sparse", "dense")}
    for arm, stored in stored_arms.items():
        if component_rows != stored.get("component_rows"):
            raise ValueError(f"combo component row counts differ from stored R15 {arm} endpoint")
    sums = raw_sums_by_group(pooled)
    for component in component_rows:
        sums.setdefault(component, np.zeros((2, 6), dtype=np.float64).tolist())
    for _arm, stored in stored_arms.items():
        _label_sums_match(
            sums, stored["official_clipped_sufficient_statistics_w_wy_wp_wyy_wpp_wyp"]
        )
    candidate_metric = pooled.result()
    drops = {str(key): metric_without(per_component, key) for key in sorted(per_component)}
    old_sparse_metric = stored_arms["sparse"]["official_metric"]
    old_dense_metric = stored_arms["dense"]["official_metric"]
    component_deltas: dict[str, dict[str, dict[str, float]]] = {}
    drop_one_deltas: dict[str, dict[str, dict[str, float]]] = {}
    for arm, stored in stored_arms.items():
        component_deltas[arm] = {}
        for component, score in ((str(key), value) for key, value in sorted(per_component.items())):
            candidate_score = score.result() if score.selected_rows else None
            reference_score = stored["component_scores"][component]
            component_deltas[arm][component] = (
                {
                    name: float(candidate_score[name] - reference_score[name])
                    for name in ("t0", "t1", "wp")
                }
                if candidate_score is not None and reference_score is not None
                else {}
            )
        drop_one_deltas[arm] = {
            component: {
                name: float(
                    drops[component][name] - stored["component_drop_one_scores"][component][name]
                )
                for name in ("t0", "t1", "wp")
            }
            for component in drops
        }
    return {
        "metric_id": candidate_metric["metric_id"],
        "selected_rows": rows_seen,
        "evaluation_components": len(per_component),
        "component_rows": component_rows,
        "official_metric": candidate_metric,
        "component_scores": {
            str(key): value.result() if value.selected_rows else None
            for key, value in sorted(per_component.items())
        },
        "component_drop_one_scores": drops,
        "component_deltas_vs_stored_wmse": component_deltas,
        "fixed_prediction_drop_one_deltas_vs_stored_wmse": drop_one_deltas,
        "official_clipped_sufficient_statistics_w_wy_wp_wyy_wpp_wyp": sums,
        "label_mask_parity_against_stored_sparse_and_dense": True,
        "primary_delta_wp_vs_stored_sparse_wmse": float(
            candidate_metric["wp"] - old_sparse_metric["wp"]
        ),
        "secondary_delta_wp_vs_stored_dense_wmse": float(
            candidate_metric["wp"] - old_dense_metric["wp"]
        ),
        "stored_sparse_metric": dict(old_sparse_metric),
        "stored_dense_metric": dict(old_dense_metric),
        "component_sensitivities_are_not_independent_replications": True,
    }
