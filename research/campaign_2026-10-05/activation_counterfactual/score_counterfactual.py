"""CPU-only E486 crossed inference-shape scorer; real test scoring is opt-in."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

# E483 receipts and scorer outputs have heterogeneous JSON schemas.
from typing import Any

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "research/campaign_2026-10-05/activation_counterfactual"
E483_DIR = ROOT / "research/campaign_2026-10-05/activation_momentum"
E483_RUN = E483_DIR / "runs/eighteen_v1"
E483_SPEC = E483_DIR / "e483_eighteen_fit_v1.sealed.json"
E483_FIT_ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e483-isolated-muon-momentum-eighteen-fit-v1.json"
)
E483_GRADE_SPEC = E483_DIR / "e483_cpu_grade_v3.sealed.json"
E483_GRADE_ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-04/runner/attempts/e483-momentum-independent-cpu-grade-v3.json"
)
E483_GRADE_DIR = E483_DIR / "eda/eighteen_v3"
REPRODUCTION_RECEIPT = (
    ROOT / "research/campaign_2026-10-05/checks/momentum_activation_export_reproduction.json"
)
ADDENDUM_PATH = HERE / "INDEPENDENT_SCORE_ADDENDUM.md"
DESIGN_PATH = HERE / "E486_DESIGN.md"
CRITIC_PATH = HERE / "SPEC_CRITIC.md"
EXPECTED_E483_SCORER_SHA256 = "1be6039a6a11553470d93736e2bb20470f1516d603fdc902cbf61f5c7ea2cb58"
EXPECTED_REPRODUCER_SHA256 = "ee460dd0acb191f9ec0d562961386b34c5f709c36f92e4f94d79d6584eedc49a"
EXPECTED_REPRODUCER_METHOD_SHA256 = (
    "a7ba0ceaa435ec9b2b6e7b935f0796c896c4fbc8b6f80774b8d3575e084e3121"
)
EXPECTED_E483_RESULT_SHA256 = "0ab4f57de99d2ab34d53e9fedfe17528a9bec72306a970d6adc72aa588452555"
EXPECTED_E483_EXPORT_SHA256 = "6449434be4be9c65462957416842f2c0b13cd611a75cba5c5d624bbc5310d66e"
EXPECTED_E483_GRADE_ATTEMPT_ID = "e483-momentum-independent-cpu-grade-v3"
EXPECTED_E483_GRADE_SPEC_NAME = "e483_independent_momentum_cpu_grade_v3"
EXPECTED_E483_RESULT_REL = (
    "research/campaign_2026-10-05/activation_momentum/eda/eighteen_v3/scores.json"
)
EXPECTED_E483_EXPORT_REL = (
    "research/campaign_2026-10-05/activation_momentum/eda/eighteen_v3/selected_exports.npz"
)
SCORE_ABS_TOL = 2e-12

sys.path.insert(0, str(E483_DIR))
import score_momentum_v3 as e483  # noqa: E402


def sha256_file(path: Path) -> str:
    """Hash a file in bounded-size reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_file_identity(path: Path, expected_sha256: str, label: str) -> None:
    """Fail closed when an upstream byte identity differs."""
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"E486 required {label} digest differs: {path}")


def verify_grade_attempt() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify E483's completed grade and arithmetic reproduction before test loading."""
    assert_file_identity(
        E483_DIR / "score_momentum_v3.py", EXPECTED_E483_SCORER_SHA256, "E483 scorer"
    )
    spec = e483.validate_spec(json.loads(E483_GRADE_SPEC.read_text()))
    expected_argv = [
        "{python}",
        "research/campaign_2026-10-05/activation_momentum/score_momentum_v3.py",
        "--device",
        "cpu",
        "--score-exposed-test",
        "--output",
        "research/campaign_2026-10-05/activation_momentum/eda/eighteen_v3",
    ]
    if (
        spec.get("name") != EXPECTED_E483_GRADE_SPEC_NAME
        or spec.get("kind") != "cpu"
        or spec.get("limits")
        != {
            "cpus": 1,
            "gpu_memory_bytes": 0,
            "gpu_stop_bytes": 0,
            "ram_bytes": 6 * 1024**3,
            "timeout_seconds": 2400,
        }
        or spec.get("argv") != expected_argv
        or spec.get("artifacts") != [EXPECTED_E483_RESULT_REL, EXPECTED_E483_EXPORT_REL]
    ):
        raise ValueError("E483 grade sealed spec identity, command, or limits differ")
    for item in [*spec["code"], *spec["inputs"]]:
        source = ROOT / item["path"]
        if (
            not source.is_file()
            or source.stat().st_size != item["size"]
            or sha256_file(source) != item["sha256"]
        ):
            raise ValueError(f"E483 grade sealed dependency changed: {item['path']}")
    record = json.loads(E483_GRADE_ATTEMPT.read_text())
    if (
        record.get("id") != EXPECTED_E483_GRADE_ATTEMPT_ID
        or record.get("kind") != "cpu"
        or record.get("spec") != spec
        or record.get("spec_sha256") != e483.spec_digest(spec)
    ):
        raise ValueError("E483 grade attempt does not bind its sealed spec")
    argv = record.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not Path(argv[0]).is_absolute()
        or argv[1:] != spec["argv"][1:]
    ):
        raise ValueError("E483 grade attempt argv differs from the sealed command")
    if record.get("runtime_sha256") != e483.runtime_identity(ROOT):
        raise ValueError("E483 grade attempt runtime receipt changed")
    if (
        record.get("status") != "finished"
        or record.get("exit_code") != 0
        or record.get("signals") != []
        or record.get("teardown_verified") is not True
        or any(
            record.get(key) is not None for key in ("failure", "failure_scope", "failure_category")
        )
    ):
        raise ValueError("E483 official grade attempt did not finish cleanly")
    unit = record.get("unit_properties", {})
    if unit.get("Result") != "success" or unit.get("ExecMainStatus") not in (0, "0"):
        raise ValueError("E483 grade systemd unit reports failure")
    for before_name, after_name in (
        ("initial_memory_events", "final_memory_events"),
        ("initial_parent_memory_events", "final_parent_memory_events"),
    ):
        before, after = record.get(before_name), record.get(after_name)
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError(f"E483 grade lacks memory-event snapshots: {before_name}")
        for counter in ("oom", "oom_kill", "oom_group_kill"):
            if type(before.get(counter)) is not int or before[counter] != after.get(counter):
                raise ValueError(f"E483 grade OOM counter changed: {before_name}/{counter}")
    artifacts = []
    for relative in spec["artifacts"]:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"E483 grade artifact is missing: {relative}")
        artifacts.append(
            {"path": relative, "sha256": sha256_file(path), "size": path.stat().st_size}
        )
    if (
        record.get("artifacts") != artifacts
        or artifacts[0]["sha256"] != EXPECTED_E483_RESULT_SHA256
        or artifacts[1]["sha256"] != EXPECTED_E483_EXPORT_SHA256
    ):
        raise ValueError("E483 grade artifact inventory or registered bytes differ")
    assert_file_identity(
        ROOT / "research/campaign_2026-10-05/reproduce_momentum_exports.py",
        EXPECTED_REPRODUCER_SHA256,
        "root E483 arithmetic reproducer",
    )
    assert_file_identity(
        ROOT / "research/campaign_2026-10-05/reproduce_activation_exports.py",
        EXPECTED_REPRODUCER_METHOD_SHA256,
        "root E483 arithmetic method",
    )
    receipt = json.loads(REPRODUCTION_RECEIPT.read_text())
    if (
        receipt.get("study") != "E483"
        or receipt.get("status") != "all_checked_values_match"
        or receipt.get("arms") != 18
        or receipt.get("global_contrasts") != 27
        or receipt.get("maximum_absolute_error", math.inf) > SCORE_ABS_TOL
        or receipt.get("source_sha256") != EXPECTED_REPRODUCER_SHA256
        or receipt.get("method_source_sha256") != EXPECTED_REPRODUCER_METHOD_SHA256
        or receipt.get("result_sha256") != EXPECTED_E483_RESULT_SHA256
        or receipt.get("export_sha256") != EXPECTED_E483_EXPORT_SHA256
    ):
        raise ValueError("root E483 raw-export arithmetic receipt is invalid")
    return spec, record, receipt


def load_e483_reference() -> dict[str, Any]:
    """Load the previously completed exports after validating their runner proof."""
    scores = json.loads((E483_GRADE_DIR / "scores.json").read_text())
    if (
        scores.get("arm_count") != 18
        or scores.get("contrast_count") != 27
        or scores.get("metric_id") != e483.METRIC_ID
    ):
        raise ValueError("E483 reference score matrix is incomplete")
    with np.load(E483_GRADE_DIR / "selected_exports.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arm_names = [e483.arm_key(seed, shape, beta) for seed, shape, beta in e483.ARMS]
    component_ids = np.asarray(scores.get("component_ids"), dtype=np.int64)
    row_groups = np.asarray(scores.get("row_groups"), dtype=np.int64)
    selected_masks = arrays.get("selected_masks_by_component")
    need_masks = arrays.get("need_masks_by_component")
    scored_masks = arrays.get("scored_masks_by_component")
    if (
        arrays.get("arm_names", np.asarray([])).tolist() != arm_names
        or arrays.get("predictions_raw_selected_by_arm", np.empty((0,))).shape[0] != 18
        or arrays.get("targets_raw_selected", np.empty((0,))).shape[-1:] != (2,)
        or arrays.get("arm_bootstrap_wp", np.empty((0,))).shape != (18, 2_000)
        or arrays.get("contrast_bootstrap_diffs", np.empty((0,))).shape != (27, 2_000)
        or arrays.get("contrast_names", np.empty((0,))).shape != (27,)
        or set(scores.get("arm_scores", {})) != set(arm_names)
        or component_ids.shape != (16,)
        or row_groups.shape != (16,)
        or len(np.unique(component_ids)) != 16
        or len(np.unique(row_groups)) != 16
        or not np.array_equal(arrays.get("component_ids"), component_ids)
        or not np.array_equal(arrays.get("row_groups"), row_groups)
        or selected_masks is None
        or need_masks is None
        or scored_masks is None
        or selected_masks.shape != (16, e483.ROWS)
        or need_masks.shape != selected_masks.shape
        or scored_masks.shape != selected_masks.shape
        or not np.array_equal(selected_masks, need_masks & scored_masks)
        or not np.array_equal(arrays.get("selected_row_counts"), selected_masks.sum(axis=1))
        or arrays["targets_raw_selected"].shape != (int(selected_masks.sum()), 2)
        or arrays["predictions_raw_selected_by_arm"].shape != (18, int(selected_masks.sum()), 2)
        or arrays.get("selected_component_index", np.empty((0,))).shape
        != (int(selected_masks.sum()),)
        or arrays.get("selected_row_index_within_component", np.empty((0,))).shape
        != (int(selected_masks.sum()),)
        or not np.isfinite(arrays["targets_raw_selected"]).all()
        or not np.isfinite(arrays["predictions_raw_selected_by_arm"]).all()
    ):
        raise ValueError("E483 reference export layout, masks, or raw values are invalid")
    if not np.array_equal(
        arrays.get("bootstrap_multiplicities"),
        e483.bootstrap_multiplicities(component_ids),
    ):
        raise ValueError("E483 reference bootstrap draw identity differs")
    if not np.array_equal(component_ids, np.sort(component_ids)):
        raise ValueError("E483 reference component IDs are not sorted")
    if not np.array_equal(row_groups, np.asarray(scores["row_groups"], dtype=np.int64)):
        raise ValueError("E483 reference row-group map differs")
    expected_contrast_names = sorted(
        f"seed{seed}__{name}"
        for seed, values in scores["contrasts_by_seed"].items()
        for name in values
    )
    if arrays["contrast_names"].tolist() != expected_contrast_names:
        raise ValueError("E483 contrast bootstrap name/order identity differs")
    expected_component_index = np.repeat(np.arange(16), selected_masks.sum(axis=1))
    expected_row_index = np.concatenate([np.flatnonzero(mask) for mask in selected_masks])
    if not np.array_equal(
        arrays["selected_component_index"], expected_component_index
    ) or not np.array_equal(arrays["selected_row_index_within_component"], expected_row_index):
        raise ValueError("E483 selected-export row identity arrays differ")
    return {"scores": scores, "arrays": arrays, "arm_names": arm_names}


def verify_grade_data_alignment(
    test_nodes: list[dict[str, Any]],
    targets: np.ndarray,
    need: np.ndarray,
    scored: np.ndarray,
    selected: np.ndarray,
    reference: dict[str, Any],
) -> None:
    """Require the just-read panel to match E483 IDs, masks, and raw targets."""
    arrays = reference["arrays"]
    ids = np.asarray([int(node["component"]) for node in test_nodes], dtype=np.int64)
    groups = np.asarray([int(node["row_group"]) for node in test_nodes], dtype=np.int64)
    if not np.array_equal(ids, arrays["component_ids"]) or not np.array_equal(
        groups, arrays["row_groups"]
    ):
        raise ValueError("E486 test component order/row-group identity differs from E483")
    if (
        not np.array_equal(need, arrays["need_masks_by_component"])
        or not np.array_equal(scored, arrays["scored_masks_by_component"])
        or not np.array_equal(selected, arrays["selected_masks_by_component"])
        or not np.array_equal(selected, need & scored)
    ):
        raise ValueError("E486 full target mask differs from E483 exported masks")
    raw_selected_targets = targets[selected].astype(np.float64)
    if not np.array_equal(raw_selected_targets, arrays["targets_raw_selected"]):
        raise ValueError("E486 selected raw targets differ from E483 exports")


def forward_with_state_extrema(
    model: e483.IndependentMomentumModel,
    x: Tensor,
    state: Tensor | None,
) -> tuple[Tensor, Tensor, float]:
    """Run the E483 recurrence and capture hidden-state extrema for each layer."""
    encoded = torch.nn.functional.gelu(model.inp((x - model.mean) / model.std))
    batch = encoded.shape[0]
    hidden = encoded.new_zeros((2, batch, e483.HIDDEN)) if state is None else state
    if hidden.shape != (2, batch, e483.HIDDEN):
        raise ValueError("E486 carried state shape differs")
    layer_input = encoded
    finals: list[Tensor] = []
    max_abs_state = 0.0
    for layer in range(2):
        layer_input, final = e483.reset_after_scan(
            layer_input,
            hidden[layer],
            getattr(model.gru, f"weight_ih_l{layer}"),
            getattr(model.gru, f"weight_hh_l{layer}"),
            getattr(model.gru, f"bias_ih_l{layer}"),
            getattr(model.gru, f"bias_hh_l{layer}"),
            model.sine,
        )
        max_abs_state = max(max_abs_state, float(layer_input.detach().abs().max().cpu()))
        finals.append(final)
    prediction = model.head(torch.cat((layer_input, encoded), dim=-1))
    return prediction, torch.stack(finals), max_abs_state


def predict_counterfactual(
    model: e483.IndependentMomentumModel, x: np.ndarray, device: torch.device
) -> tuple[np.ndarray, float]:
    """Predict complete sequences with zero initial state and fixed chunk carry."""
    if x.ndim != 3 or x.shape[-1] != e483.FEATURES or not np.isfinite(x).all():
        raise ValueError("E486 inference features must be finite [component,time,42]")
    model = model.to(device=device, dtype=torch.float32).eval()
    predictions = np.empty((len(x), x.shape[1], 2), dtype=np.float32)
    max_abs_state = 0.0
    state = None
    with e483.full_ieee_float32(device), torch.no_grad():
        for start in range(0, x.shape[1], e483.CHUNK):
            chunk = torch.from_numpy(x[:, start : start + e483.CHUNK]).to(
                device=device, dtype=torch.float32
            )
            output, state, chunk_max = forward_with_state_extrema(model, chunk, state)
            if not torch.isfinite(output).all() or not torch.isfinite(state).all():
                raise ValueError(f"E486 nonfinite prediction/state at chunk {start}")
            predictions[:, start : start + output.shape[1]] = output.cpu().numpy()
            max_abs_state = max(max_abs_state, chunk_max)
            state = state.detach()
    if not np.isfinite(predictions).all():
        raise ValueError("E486 produced nonfinite raw predictions")
    return predictions, max_abs_state


def cell_key(seed: int, train_shape: str, eval_shape: str, beta: float) -> str:
    """Return the canonical key for one existing checkpoint/inference pair."""
    if (
        seed not in e483.SEEDS
        or train_shape not in e483.SHAPES
        or eval_shape not in e483.SHAPES
        or beta not in e483.MOMENTA
    ):
        raise ValueError("unregistered E486 training/inference cell")
    return f"seed{seed}__train_{train_shape}__eval_{eval_shape}__beta{beta}"


def cell_alias(train_shape: str, eval_shape: str) -> str:
    return f"{train_shape}_{eval_shape}"


def contrast_summary(
    draws: np.ndarray, point: float, candidate: str, reference: str
) -> dict[str, Any]:
    """Summarize one contrast from its shared component bootstrap draws."""
    if draws.shape != (2_000,) or not np.isfinite(draws).all() or not math.isfinite(point):
        raise ValueError("invalid E486 paired contrast values")
    return {
        "candidate": candidate,
        "reference": reference,
        "mean_diff": float(point),
        "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "p_le_0": float(np.mean(draws <= 0)),
        "n_components": 16,
        "n_boot": 2_000,
        "bootstrap_seed": e483.BOOTSTRAP_SEED,
    }


def registered_cross_contrasts(
    score: dict[str, float], draws: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """Derive all six preregistered contrasts from the same four-cell draws."""
    expressions = {
        "D_minus_A_matched_diagonal": ("sine_sine", "baseline_baseline"),
        "B_minus_A_inference_sine_on_base_fit": ("baseline_sine", "baseline_baseline"),
        "C_minus_D_remove_sine_on_sine_fit": ("sine_baseline", "sine_sine"),
        "C_minus_A_train_shape_under_base_eval": ("sine_baseline", "baseline_baseline"),
        "D_minus_B_train_shape_under_sine_eval": ("sine_sine", "baseline_sine"),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, (candidate, reference) in expressions.items():
        results[name] = {
            "family": "crossed_cell_difference",
            **contrast_summary(
                draws[candidate] - draws[reference],
                score[candidate] - score[reference],
                candidate,
                reference,
            ),
        }
    interaction_draws = (draws["sine_sine"] - draws["baseline_sine"]) - (
        draws["sine_baseline"] - draws["baseline_baseline"]
    )
    interaction_point = (score["sine_sine"] - score["baseline_sine"]) - (
        score["sine_baseline"] - score["baseline_baseline"]
    )
    results["training_by_inference_interaction"] = {
        "family": "training_by_inference_difference_in_differences",
        **contrast_summary(interaction_draws, interaction_point, "(D-B)", "(C-A)"),
    }
    if len(results) != 6:
        raise AssertionError("E486 requires six contrasts per seed/momentum")
    return results


def _inflate_selected(values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Rebuild full row-position predictions for the official metric helper."""
    full = np.zeros((*selected.shape, 2), dtype=np.float32)
    offset = 0
    for index, mask in enumerate(selected):
        size = int(mask.sum())
        full[index, mask] = values[offset : offset + size]
        offset += size
    if offset != len(values):
        raise ValueError("E486 selected prediction stream does not match full mask")
    return full


def verify_diagonal_prediction_gate(
    predictions: dict[str, np.ndarray], reference: dict[str, Any]
) -> None:
    """Require exact selected-row prediction equality for all 18 E483 diagonal cells."""
    previous = reference["arrays"]["predictions_raw_selected_by_arm"]
    arm_names = reference["arm_names"]
    indices = {name: index for index, name in enumerate(arm_names)}
    for seed, shape, beta in e483.ARMS:
        key = cell_key(seed, shape, shape, beta)
        old_key = e483.arm_key(seed, shape, beta)
        current = predictions.get(key)
        if current is None or not np.array_equal(current, previous[indices[old_key]]):
            raise ValueError(f"E486 diagonal raw predictions differ from E483: {key}")


def run_crossed_matrix(
    cells: dict[tuple[int, str, float], dict[str, Any]],
    test_nodes: list[dict[str, Any]],
    x: np.ndarray,
    targets: np.ndarray,
    selected: np.ndarray,
    run: Path,
    reference: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score all 36 fixed-checkpoint cells, then accept contrasts after diagonal parity."""
    if device.type != "cpu":
        raise ValueError("E486 is registered as CPU-only")
    component_ids = np.asarray([int(node["component"]) for node in test_nodes], dtype=np.int64)
    if not np.array_equal(component_ids, np.sort(component_ids)):
        raise ValueError("E486 components must be numeric-sorted")
    if x.shape[:2] != selected.shape or targets.shape != (*selected.shape, 2):
        raise ValueError("E486 feature, target, and mask shapes differ")
    if x.shape[0] != 16 or not selected.any(axis=1).all():
        raise ValueError("E486 requires all 16 nonempty exposed test components")
    if not np.isfinite(targets).all():
        raise ValueError("E486 raw labels are nonfinite")
    counts = e483.bootstrap_multiplicities(component_ids)
    arm_names = reference["arm_names"]
    old_boot = reference["arrays"]["arm_bootstrap_wp"]
    old_scores = reference["scores"]["arm_scores"]
    export_indices = {name: index for index, name in enumerate(arm_names)}
    predictions_by_cell: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, float]] = {}

    for seed, train_shape, beta in e483.ARMS:
        arm = cells[(seed, train_shape, beta)]
        checkpoint = run / arm["checkpoint"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = e483.IndependentMomentumModel(sine=train_shape == "sine")
        model.load_state_dict(state, strict=True)
        for eval_shape in e483.SHAPES:
            model.sine = eval_shape == "sine"
            key = cell_key(seed, train_shape, eval_shape, beta)
            prediction, state_abs_max = predict_counterfactual(model, x, device)
            selected_raw = prediction[selected]
            if not np.isfinite(selected_raw).all():
                raise ValueError(f"nonfinite selected raw prediction in {key}")
            predictions_by_cell[key] = selected_raw.copy()
            diagnostics[key] = {
                "selected_raw_prediction_min": float(selected_raw.min()),
                "selected_raw_prediction_max": float(selected_raw.max()),
                "selected_raw_prediction_abs_max": float(np.abs(selected_raw).max()),
                "hidden_state_abs_max_all_layers_all_chunks": state_abs_max,
            }

    # Gate all contrasts on exact raw prediction parity for every diagonal cell.
    verify_diagonal_prediction_gate(predictions_by_cell, reference)

    pair_scores: dict[tuple[int, float], dict[str, float]] = {}
    pair_draws: dict[tuple[int, float], dict[str, np.ndarray]] = {}
    cell_scores: dict[str, dict[str, Any]] = {}
    component_lengths = selected.sum(axis=1).astype(np.int64)
    for seed in e483.SEEDS:
        for beta in e483.MOMENTA:
            pair_scores[(seed, beta)] = {}
            pair_draws[(seed, beta)] = {}
            for train_shape in e483.SHAPES:
                for eval_shape in e483.SHAPES:
                    key = cell_key(seed, train_shape, eval_shape, beta)
                    alias = cell_alias(train_shape, eval_shape)
                    prediction_stream = predictions_by_cell[key]
                    per_component_predictions: list[np.ndarray] = []
                    offset = 0
                    for length in component_lengths:
                        end = offset + int(length)
                        per_component_predictions.append(prediction_stream[offset:end])
                        offset = end
                    if offset != len(prediction_stream):
                        raise ValueError("E486 selected prediction/component partition differs")
                    moments = np.zeros((16, 2, 6), dtype=np.float64)
                    for component_index, length in enumerate(component_lengths):
                        count = int(length)
                        raw_y = targets[component_index, selected[component_index]].astype(
                            np.float64
                        )
                        raw_p = per_component_predictions[component_index].astype(np.float64)
                        y = np.clip(
                            raw_y.astype(np.float32), -e483.METRIC_CLIP, e483.METRIC_CLIP
                        ).astype(np.float64)
                        p = np.clip(
                            raw_p.astype(np.float32), -e483.METRIC_CLIP, e483.METRIC_CLIP
                        ).astype(np.float64)
                        if len(y) != count or len(p) != count:
                            raise ValueError("E486 selected component lengths differ")
                        for target_index in range(2):
                            moments[component_index, target_index] = e483._centered_moments(
                                y[:, target_index], p[:, target_index]
                            )
                    metric = e483.global_wp(
                        targets, _inflate_selected(prediction_stream, selected), selected
                    )
                    if metric.get("metric_id") != e483.METRIC_ID or metric.get(
                        "selected_rows"
                    ) != int(selected.sum()):
                        raise ValueError(f"E486 official metric metadata differs: {key}")
                    arm_draws = np.asarray(
                        [e483.pool_component_moments(moments, draw) for draw in counts],
                        dtype=np.float64,
                    )
                    point = float(metric["wp"])
                    pair_scores[(seed, beta)][alias] = point
                    pair_draws[(seed, beta)][alias] = arm_draws
                    cell_scores[key] = {
                        "seed": seed,
                        "train_shape": train_shape,
                        "inference_shape": eval_shape,
                        "muon_beta1": beta,
                        "global_wp": point,
                        "target0_wp": float(metric["t0"]),
                        "target1_wp": float(metric["t1"]),
                        "selected_rows": int(metric["selected_rows"]),
                        "selected_rows_per_component": component_lengths.tolist(),
                        "checkpoint_sha256": cells[(seed, train_shape, beta)]["checkpoint_sha256"],
                        **diagnostics[key],
                    }

    # Every diagonal score, target metric, and bootstrap draw must reproduce E483.
    for seed, shape, beta in e483.ARMS:
        key = cell_key(seed, shape, shape, beta)
        alias = cell_alias(shape, shape)
        old_key = e483.arm_key(seed, shape, beta)
        prior = old_scores[old_key]
        current = cell_scores[key]
        for current_name, prior_name in (
            ("global_wp", "wp"),
            ("target0_wp", "t0"),
            ("target1_wp", "t1"),
        ):
            if abs(current[current_name] - float(prior[prior_name])) > SCORE_ABS_TOL:
                raise ValueError(f"E486 diagonal score differs from E483: {key}/{current_name}")
        if not np.array_equal(pair_draws[(seed, beta)][alias], old_boot[export_indices[old_key]]):
            raise ValueError(f"E486 diagonal bootstrap draws differ from E483: {key}")

    contrasts: dict[str, dict[str, Any]] = {}
    contrast_draws: dict[str, np.ndarray] = {}
    for seed in e483.SEEDS:
        for beta in e483.MOMENTA:
            pair = (seed, beta)
            pair_name = f"seed{seed}__beta{beta}"
            per_pair = registered_cross_contrasts(pair_scores[pair], pair_draws[pair])
            diagonal_effect = per_pair["D_minus_A_matched_diagonal"]
            old_effect = reference["scores"]["contrasts_by_seed"][str(seed)][
                f"sine_minus_baseline_beta{beta}"
            ]
            if abs(diagonal_effect["mean_diff"] - old_effect["mean_diff"]) > SCORE_ABS_TOL:
                raise ValueError(f"E486 D-A point estimate differs from E483: {pair_name}")
            if not np.allclose(
                diagonal_effect["ci95"], old_effect["ci95"], rtol=0, atol=SCORE_ABS_TOL
            ):
                raise ValueError(f"E486 D-A bootstrap interval differs from E483: {pair_name}")
            contrasts[pair_name] = per_pair
            for name in per_pair:
                if name == "D_minus_A_matched_diagonal":
                    delta = pair_draws[pair]["sine_sine"] - pair_draws[pair]["baseline_baseline"]
                elif name == "B_minus_A_inference_sine_on_base_fit":
                    delta = (
                        pair_draws[pair]["baseline_sine"] - pair_draws[pair]["baseline_baseline"]
                    )
                elif name == "C_minus_D_remove_sine_on_sine_fit":
                    delta = pair_draws[pair]["sine_baseline"] - pair_draws[pair]["sine_sine"]
                elif name == "C_minus_A_train_shape_under_base_eval":
                    delta = (
                        pair_draws[pair]["sine_baseline"] - pair_draws[pair]["baseline_baseline"]
                    )
                elif name == "D_minus_B_train_shape_under_sine_eval":
                    delta = pair_draws[pair]["sine_sine"] - pair_draws[pair]["baseline_sine"]
                else:
                    delta = (pair_draws[pair]["sine_sine"] - pair_draws[pair]["baseline_sine"]) - (
                        pair_draws[pair]["sine_baseline"] - pair_draws[pair]["baseline_baseline"]
                    )
                contrast_draws[f"{pair_name}__{name}"] = delta
    if len(cell_scores) != 36 or sum(map(len, contrasts.values())) != 54:
        raise AssertionError("E486 crossed matrix must contain 36 cells and 54 contrasts")

    result = {
        "kind": "E486 fixed-checkpoint crossed activation inference diagnostic",
        "metric_id": e483.METRIC_ID,
        "scope": "18 previous checkpoints crossed with two inference shapes; same 16 exposed components; descriptive only",
        "new_fits": 0,
        "new_checkpoints": 0,
        "cell_count": 36,
        "factorial_cells": {
            "A": "train baseline, infer baseline",
            "B": "train baseline, infer sine",
            "C": "train sine, infer baseline",
            "D": "train sine, infer sine",
        },
        "contrast_formulas": {
            "D_minus_A_matched_diagonal": "D-A (E483 parity repeat)",
            "B_minus_A_inference_sine_on_base_fit": "B-A",
            "C_minus_D_remove_sine_on_sine_fit": "C-D",
            "C_minus_A_train_shape_under_base_eval": "C-A",
            "D_minus_B_train_shape_under_sine_eval": "D-B",
            "training_by_inference_interaction": "(D-B)-(C-A)",
        },
        "cells": cell_scores,
        "contrast_count": 54,
        "contrasts_by_seed_momentum": contrasts,
        "bootstrap": {
            "method": "E483 official centered weighted-Pearson component moments",
            "count": 2_000,
            "seed": e483.BOOTSTRAP_SEED,
            "resampling_unit": "numeric-sorted component ID",
            "draws_shared_across_all_four_cells_per_seed_momentum": True,
            "no_seed_pooling": True,
            "component_ids": component_ids.tolist(),
            "multiplicity_sha256": hashlib.sha256(counts.astype("<i2").tobytes()).hexdigest(),
        },
        "interpretation_limits": [
            "off-diagonal cells change hidden trajectories and may extrapolate beyond training dynamics",
            "score changes include output-scale effects and official clipping; not pure representation or amplitude isolation",
            "diagonal cells and D-A contrasts repeat E483 as a parity gate, not new evidence",
            "the exposed 16-component panel is not independent public confirmation",
        ],
    }
    export = {
        "predictions_selected": predictions_by_cell,
        "contrast_draws": contrast_draws,
        "cell_draws": {
            cell_key(
                seed,
                train_shape,
                eval_shape,
                beta,
            ): arm_draws[cell_alias(train_shape, eval_shape)]
            for (seed, beta), arm_draws in pair_draws.items()
            for train_shape in e483.SHAPES
            for eval_shape in e483.SHAPES
        },
        "counts": counts,
    }
    return result, export


def write_outputs(
    output: Path,
    result: dict[str, Any],
    export: dict[str, Any],
    targets: np.ndarray,
    selected: np.ndarray,
    need: np.ndarray,
    scored: np.ndarray,
    test_nodes: list[dict[str, Any]],
    proof: dict[str, Any],
) -> None:
    """Write a new result bundle and refuse to replace any prior output."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite E486 output: {output}")
    output.mkdir(parents=True)
    cell_names = list(export["predictions_selected"])
    counts = export["counts"]
    row_indices = np.concatenate([np.flatnonzero(mask) for mask in selected])
    component_index = np.repeat(np.arange(len(test_nodes)), selected.sum(axis=1))
    export_path = output / "selected_exports.npz"
    np.savez_compressed(
        export_path,
        cell_names=np.asarray(cell_names),
        targets_raw_selected=targets[selected].astype(np.float64),
        predictions_raw_selected_by_cell=np.stack(
            [export["predictions_selected"][name] for name in cell_names]
        ).astype(np.float32),
        component_ids=np.asarray([int(node["component"]) for node in test_nodes], dtype=np.int64),
        row_groups=np.asarray([int(node["row_group"]) for node in test_nodes], dtype=np.int64),
        selected_row_counts=selected.sum(axis=1).astype(np.int32),
        selected_component_index=component_index.astype(np.int16),
        selected_row_index_within_component=row_indices.astype(np.int32),
        selected_masks_by_component=selected,
        need_masks_by_component=need,
        scored_masks_by_component=scored,
        bootstrap_multiplicities=counts,
        cell_bootstrap_wp=np.stack([export["cell_draws"][name] for name in cell_names]),
        contrast_names=np.asarray(sorted(export["contrast_draws"])),
        contrast_bootstrap_diffs=np.stack(
            [export["contrast_draws"][name] for name in sorted(export["contrast_draws"])]
        ),
    )
    result["proof"] = proof
    result["component_ids"] = [int(node["component"]) for node in test_nodes]
    result["row_groups"] = [int(node["row_group"]) for node in test_nodes]
    result["selected_rows_total"] = int(selected.sum())
    result["selected_exports_path"] = export_path.name
    result["selected_exports_sha256"] = sha256_file(export_path)
    (output / "scores.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def validate_upstream() -> tuple[
    dict[tuple[int, str, float], dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Verify E483 source, fit, grade, and raw arithmetic before test loading."""
    fit_spec_raw = json.loads(E483_SPEC.read_text())
    fit_attempt = e483.validate_attempt(fit_spec_raw, E483_FIT_ATTEMPT, E483_RUN)
    fit_report = json.loads((E483_RUN / "fit_manifest.json").read_text())
    fit_nodes, test_nodes = e483.canonical_cohort()
    cells, fit_provenance, mean, scale = e483.validate_fit_manifest(E483_RUN, fit_report, fit_nodes)
    preflight = e483.verify_preflight()
    grade_spec, grade_attempt, reproduction = verify_grade_attempt()
    reference = load_e483_reference()
    proof = {
        "e483_scorer_sha256": EXPECTED_E483_SCORER_SHA256,
        "e483_fit_spec_sha256": e483.spec_digest(e483.validate_spec(fit_spec_raw)),
        "e483_fit_attempt_sha256": fit_attempt["attempt_sha256"],
        "e483_fit_manifest_sha256": sha256_file(E483_RUN / "fit_manifest.json"),
        "e483_initialization_hashes_verified": fit_report["initial_state_sha256"],
        "e483_grade_spec_sha256": e483.spec_digest(grade_spec),
        "e483_grade_attempt_sha256": sha256_file(E483_GRADE_ATTEMPT),
        "e483_grade_attempt_id": grade_attempt["id"],
        "e483_result_sha256": EXPECTED_E483_RESULT_SHA256,
        "e483_selected_export_sha256": EXPECTED_E483_EXPORT_SHA256,
        "e483_root_reproduction_sha256": sha256_file(REPRODUCTION_RECEIPT),
        "e483_root_reproduction_status": reproduction["status"],
        "e483_preflight_status": preflight["status"],
        "fit_provenance": fit_provenance,
        "test_rows_loaded": False,
        "test_rows_scored": False,
        "normalizer_fit_mean_sha256": hashlib.sha256(mean.astype("<f4").tobytes()).hexdigest(),
        "normalizer_fit_scale_sha256": hashlib.sha256(scale.astype("<f4").tobytes()).hexdigest(),
    }
    return cells, fit_nodes, test_nodes, mean, scale, {"proof": proof, "reference": reference}


def validate_e486_seal(seal_path: Path, output: Path) -> dict[str, Any]:
    """Require root's sealed one-CPU admission before any E486 test-row load."""
    spec = e483.validate_spec(json.loads(seal_path.read_text()))
    output_rel = output.resolve().relative_to(ROOT.resolve()).as_posix()
    seal_rel = seal_path.resolve().relative_to(ROOT.resolve()).as_posix()
    expected_argv = [
        "{python}",
        "research/campaign_2026-10-05/activation_counterfactual/score_counterfactual.py",
        "--score-exposed-test",
        "--output",
        output_rel,
        "--sealed-spec",
        seal_rel,
    ]
    limits = spec.get("limits", {})
    if (
        spec.get("name") != "e486_activation_counterfactual_cross_v1"
        or spec.get("kind") != "cpu"
        or limits.get("cpus") != 1
        or limits.get("ram_bytes", 0) > 6 * 1024**3
        or limits.get("gpu_memory_bytes") != 0
        or limits.get("gpu_stop_bytes") != 0
        or limits.get("timeout_seconds", 0) > 2_400
        or spec.get("argv") != expected_argv
        or spec.get("artifacts")
        != [f"{output_rel}/scores.json", f"{output_rel}/selected_exports.npz"]
    ):
        raise ValueError("E486 sealed admission identity, CPU limits, argv, or artifacts differ")
    required_code = {
        "research/campaign_2026-10-05/activation_counterfactual/score_counterfactual.py": Path(
            __file__
        ),
        "research/campaign_2026-10-05/activation_counterfactual/test_score_counterfactual.py": HERE
        / "test_score_counterfactual.py",
        "research/campaign_2026-10-05/activation_counterfactual/E486_DESIGN.md": DESIGN_PATH,
        "research/campaign_2026-10-05/activation_counterfactual/SPEC_CRITIC.md": CRITIC_PATH,
        "research/campaign_2026-10-05/activation_counterfactual/INDEPENDENT_SCORE_ADDENDUM.md": ADDENDUM_PATH,
        "research/campaign_2026-10-05/activation_momentum/score_momentum_v3.py": E483_DIR
        / "score_momentum_v3.py",
    }
    code_items = {item["path"]: item for item in spec["code"]}
    for relative, path in required_code.items():
        item = code_items.get(relative)
        if (
            item is None
            or item["sha256"] != sha256_file(path)
            or item["size"] != path.stat().st_size
        ):
            raise ValueError(f"E486 seal does not bind required source: {relative}")
    required_inputs = {
        "datasets/valid.parquet": ROOT / "datasets/valid.parquet",
        "research/campaign_2026-10-05/target_mechanics/cohort.json": ROOT
        / "research/campaign_2026-10-05/target_mechanics/cohort.json",
        "research/campaign_2026-10-05/activation_momentum/runs/eighteen_v1/fit_manifest.json": E483_RUN
        / "fit_manifest.json",
        "research/campaign_2026-10-05/activation_momentum/eda/eighteen_v3/scores.json": E483_GRADE_DIR
        / "scores.json",
        "research/campaign_2026-10-05/activation_momentum/eda/eighteen_v3/selected_exports.npz": E483_GRADE_DIR
        / "selected_exports.npz",
        "research/campaign_2026-10-04/runner/attempts/e483-isolated-muon-momentum-eighteen-fit-v1.json": E483_FIT_ATTEMPT,
        "research/campaign_2026-10-04/runner/attempts/e483-momentum-independent-cpu-grade-v3.json": E483_GRADE_ATTEMPT,
        "research/campaign_2026-10-05/checks/momentum_activation_export_reproduction.json": REPRODUCTION_RECEIPT,
        "research/campaign_2026-10-05/activation_momentum/score_momentum_v3.py": E483_DIR
        / "score_momentum_v3.py",
    }
    input_items = {item["path"]: item for item in spec["inputs"]}
    for relative, path in required_inputs.items():
        item = input_items.get(relative)
        if (
            item is None
            or item["sha256"] != sha256_file(path)
            or item["size"] != path.stat().st_size
        ):
            raise ValueError(f"E486 seal does not bind required input: {relative}")
    for item in [*spec["code"], *spec["inputs"]]:
        path = ROOT / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["size"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"E486 sealed dependency changed: {item['path']}")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "eda/cross_v1")
    parser.add_argument("--score-exposed-test", action="store_true")
    parser.add_argument("--sealed-spec", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.score_exposed_test and output.exists():
        raise FileExistsError(f"refusing to overwrite E486 output: {output}")
    if args.score_exposed_test and args.sealed_spec is None:
        raise ValueError("real E486 scoring requires root's --sealed-spec admission")
    seal = (
        validate_e486_seal(args.sealed_spec.resolve(), output)
        if args.score_exposed_test and args.sealed_spec is not None
        else None
    )
    cells, _fit_nodes, test_nodes, mean, scale, validated = validate_upstream()
    proof = validated["proof"]
    proof["e486_sealed_spec_sha256"] = e483.spec_digest(seal) if seal is not None else None
    reference = validated["reference"]
    if not args.score_exposed_test:
        print(
            json.dumps(
                {"status": "e483_proofs_validated_test_unread", "proof": proof},
                indent=2,
            )
        )
        return
    torch.set_num_threads(1)
    x, targets, need, scored, selected = e483.load_test_data(test_nodes, mean, scale)
    verify_grade_data_alignment(test_nodes, targets, need, scored, selected, reference)
    proof["test_rows_loaded"] = True
    proof["test_rows_scored"] = True
    result, export = run_crossed_matrix(
        cells,
        test_nodes,
        x,
        targets,
        selected,
        E483_RUN,
        reference,
        torch.device("cpu"),
    )
    result["source_sha256"] = {
        "scorer": sha256_file(Path(__file__)),
        "e483_scorer": EXPECTED_E483_SCORER_SHA256,
        "design": sha256_file(DESIGN_PATH),
        "critic": sha256_file(CRITIC_PATH),
        "implementation_addendum": sha256_file(ADDENDUM_PATH),
    }
    write_outputs(output, result, export, targets, selected, need, scored, test_nodes, proof)
    print(json.dumps({"output": str(output), "cells": 36, "contrasts": 54}, indent=2))


if __name__ == "__main__":
    main()
