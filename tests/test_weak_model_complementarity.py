from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

RESEARCH_DIR = (
    Path(__file__).resolve().parents[1]
    / "research/campaign_2026-10-06/bandit_continuation/disagreement_guided_H204"
)
SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(RESEARCH_DIR))
sys.path.insert(0, str(SOURCE_DIR))

import disagreement_gate as _disagreement_gate  # noqa: E402
import disagreement_screen as _disagreement_screen  # noqa: E402
from disagreement_screen import (  # noqa: E402
    _top_disagreement_indices,
    association_screen,
    chain_diagnostics,
    holm_adjust,
    spearman_rho,
)
from simplex_stack import (  # noqa: E402
    E514Inputs,
    experiment_identity,
    fit_target_simplex,
    predict_groups,
)


def _toy_inputs(group_count: int = 9, row_count: int = 128) -> E514Inputs:
    rng = np.random.default_rng(41)
    targets = rng.normal(size=(group_count, row_count, 2)).astype(np.float32)
    anchor = (0.65 * targets + rng.normal(scale=0.9, size=targets.shape)).astype(np.float32)
    ab12 = (0.75 * targets + rng.normal(scale=0.25, size=targets.shape)).astype(np.float32)
    ab48 = (0.2 * targets + rng.normal(scale=1.0, size=targets.shape)).astype(np.float32)
    baseline = (0.35 * targets + rng.normal(scale=0.6, size=targets.shape)).astype(np.float32)
    chain_ids = np.arange(group_count, dtype=np.int64)
    fold_map = {i: i // (group_count // 3) for i in range(group_count)}
    return E514Inputs(
        targets=targets,
        is_scored=np.ones((group_count, row_count), dtype=np.bool_),
        anchor=anchor,
        candidate_predictions={"ab12": ab12, "ab48": ab48, "baseline": baseline},
        chain_by_sequence=chain_ids,
        chain_to_outer_fold=fold_map,
        row_groups=chain_ids,
        source_manifest_sha256="toy",
        chain_partition_sha256="toy",
    )


def test_spearman_ties_and_holm_adjustment() -> None:
    assert np.isclose(
        spearman_rho(np.asarray([1.0, 1.0, 2.0]), np.asarray([2.0, 2.0, 4.0])),
        1.0,
    )
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_disagreement_cutoff_ties_use_chain_id_not_input_order() -> None:
    disagreement = np.asarray([0.0, 1.0, 1.0, 1.0])
    chain_ids = np.asarray([40, 10, 30, 20])
    selected = _top_disagreement_indices(disagreement, chain_ids, 2)
    selected_chain_ids = chain_ids[selected]
    assert set(selected_chain_ids) == {20, 30}

    permutation = np.asarray([3, 0, 2, 1])
    permuted_indices = _top_disagreement_indices(
        disagreement[permutation], chain_ids[permutation], 2
    )
    assert set(chain_ids[permutation][permuted_indices]) == {20, 30}


def test_chain_diagnostics_clip_and_keep_chain_as_unit() -> None:
    inputs = _toy_inputs()
    diagnostics = chain_diagnostics(
        inputs.targets,
        inputs.is_scored,
        inputs.anchor,
        {"ab12": inputs.candidate_predictions["ab12"]},
        {"ab12": 1.0},
        inputs.chain_by_sequence,
        inputs.chain_to_outer_fold,
        np.arange(inputs.targets.shape[0]),
    )
    assert len(diagnostics) == 2 * inputs.targets.shape[0] * 2
    assert {row["candidate"] for row in diagnostics} == {"anchor_error", "ab12"}
    assert all(row["weight_mass"] > 0 for row in diagnostics)


def test_association_screen_is_deterministic_on_fixed_chain_summaries() -> None:
    diagnostics: list[dict[str, float | int | str]] = []
    for fold in range(3):
        for local_chain in range(10):
            chain = fold * 10 + local_chain
            error = float(local_chain)
            diagnostics.extend(
                [
                    {
                        "chain_id": chain,
                        "outer_fold": fold,
                        "target": 0,
                        "candidate": "anchor_error",
                        "disagreement": 0.0,
                        "anchor_weighted_squared_error": error,
                        "weight_mass": 1.0,
                    },
                    {
                        "chain_id": chain,
                        "outer_fold": fold,
                        "target": 0,
                        "candidate": "donor",
                        "disagreement": error,
                        "anchor_weighted_squared_error": error,
                        "weight_mass": 1.0,
                    },
                ]
            )
    first = association_screen(
        diagnostics, seed=77, bootstrap_replicates=40, permutation_replicates=80
    )
    second = association_screen(
        diagnostics, seed=77, bootstrap_replicates=40, permutation_replicates=80
    )
    assert first == second
    assert first[0]["positive_folds"] == 3
    assert first[0]["screen_gate_pass"] is True


def test_target_simplex_recovers_useful_weak_member_and_clips_endpoint() -> None:
    inputs = _toy_inputs(group_count=9)
    fits = [
        fit_target_simplex(inputs, np.arange(6), target, calibration_chain_count=6)
        for target in (0, 1)
    ]
    for fit in fits:
        assert len(fit.weights) == 4
        assert all(weight >= 0 for weight in fit.weights)
        assert np.isclose(sum(fit.weights), 1.0)
        assert fit.weights[1] > fit.weights[0]
    prediction = predict_groups(inputs, np.asarray([6, 7, 8]), fits)
    assert prediction.shape == (3, inputs.targets.shape[1], 2)
    assert np.max(np.abs(prediction)) <= 2.0

    identity = experiment_identity(inputs)
    assert identity["experiment_id"] == "E-516"
    assert identity["hypothesis_id"] == "H-206"
    assert identity["source_manifest_sha256"] == "toy"
    assert identity["chain_partition_sha256"] == "toy"


def test_disagreement_gate_uses_prediction_magnitude_and_convex_calibration() -> None:
    targets = np.asarray([[-1.0, -1.0, -1.0, 1.0, 2.0]], dtype=np.float32)
    anchor = np.zeros_like(targets)
    candidate = np.asarray([[0.1, 0.2, 0.3, 1.0, 2.0]], dtype=np.float32)
    mask = np.ones_like(targets, dtype=np.bool_)

    fit = _disagreement_gate.fit_row_gate(targets, anchor, candidate, mask)
    gated = _disagreement_gate.apply_row_gate(anchor, candidate, fit, gated=True)
    global_blend = _disagreement_gate.apply_row_gate(anchor, candidate, fit, gated=False)

    assert fit.threshold == 1.0
    assert fit.gated_beta == 1.0
    assert 0.0 < fit.global_beta < 1.0
    assert fit.calibration_gate_fraction == 0.4
    assert fit.calibration_weighted_gate_fraction == 0.5
    assert np.array_equal(gated, np.asarray([[0.0, 0.0, 0.0, 1.0, 2.0]], dtype=np.float32))
    assert np.allclose(global_blend, candidate * fit.global_beta)


def test_disagreement_gate_returns_anchor_when_direction_has_no_weighted_support() -> None:
    targets = np.asarray([[1.0, 2.0, 1.0, 2.0]], dtype=np.float32)
    anchor = np.zeros_like(targets)
    candidate = np.zeros_like(targets)
    fit = _disagreement_gate.fit_row_gate(
        targets, anchor, candidate, np.ones_like(targets, dtype=np.bool_)
    )

    assert fit.gated_beta == 0.0
    assert fit.global_beta == 0.0
    assert np.array_equal(
        _disagreement_gate.apply_row_gate(anchor, candidate, fit, gated=True), anchor
    )


def test_registered_gate_rejects_unrecognized_frozen_source_identity() -> None:
    inputs = _toy_inputs(group_count=9)
    corrupted = replace(inputs, source_manifest_sha256="not-the-registered-source")

    try:
        _disagreement_gate.validate_registered_inputs(corrupted)
    except ValueError as error:
        assert "source-manifest identity" in str(error)
    else:
        raise AssertionError("unrecognized E-517 source identity was accepted")


def test_registered_runner_persists_outputs_and_uses_frozen_seed(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _toy_inputs(group_count=9)
    experiment_dir = Path("temporary_e517")
    monkeypatch.setattr(_disagreement_gate, "E517_RELATIVE_DIR", experiment_dir)
    (tmp_path / experiment_dir).mkdir(parents=True)
    (tmp_path / experiment_dir / "GATED_BLEND_PREREGISTRATION_E517.md").write_text(
        "synthetic preregistration"
    )
    loader_calls = {"count": 0}

    def fake_load(root: Path) -> E514Inputs:
        loader_calls["count"] += 1
        return inputs

    monkeypatch.setattr(_disagreement_screen, "load_e514_inputs", fake_load)
    monkeypatch.setattr(
        _disagreement_gate,
        "validate_registered_inputs",
        lambda loaded: {
            "source_manifest_sha256": loaded.source_manifest_sha256,
            "chain_partition_sha256": loaded.chain_partition_sha256,
            "row_groups_sha256": "synthetic-row-groups",
        },
    )
    observed: dict[str, int] = {}

    def fake_fit(loaded, *, seed):
        observed["seed"] = seed
        return {"experiment_id": "E-517", "folds": []}, loaded.anchor.copy(), loaded.anchor.copy()

    monkeypatch.setattr(_disagreement_gate, "fit_and_crossfit", fake_fit)

    _, result, gated, global_blend = _disagreement_gate.run_registered_e517(tmp_path)

    output_dir = tmp_path / experiment_dir
    assert observed["seed"] == _disagreement_gate.BOOTSTRAP_SEED
    assert result["input_load_mode"] == "registered_loader"
    assert loader_calls["count"] == 1
    assert result["row_groups_sha256"] == "synthetic-row-groups"
    assert (output_dir / "E517_RESULT.json").is_file()
    assert np.array_equal(
        np.load(output_dir / "E517_GATED_OOF_PREDICTIONS.npy", allow_pickle=False), gated
    )
    assert np.array_equal(
        np.load(output_dir / "E517_GLOBAL_OOF_PREDICTIONS.npy", allow_pickle=False),
        global_blend,
    )
    assert (output_dir / "plots/E517_CHAIN_SUMMARY.csv").is_file()

    shared_dir = Path("temporary_e517_shared")
    monkeypatch.setattr(_disagreement_gate, "E517_RELATIVE_DIR", shared_dir)
    (tmp_path / shared_dir).mkdir()
    (tmp_path / shared_dir / "GATED_BLEND_PREREGISTRATION_E517.md").write_text(
        "synthetic preregistration"
    )
    _, shared_result, _, _ = _disagreement_gate.run_registered_e517(tmp_path, inputs)
    assert shared_result["input_load_mode"] == "shared_verified_marimo_input_cell"
    assert loader_calls["count"] == 1


def test_gated_crossfit_scores_official_endpoint_and_preserves_target_one_anchor() -> None:
    inputs = _toy_inputs(group_count=9, row_count=128)
    result, gated, global_blend = _disagreement_gate.fit_and_crossfit(inputs, seed=19)

    assert result["experiment_id"] == "E-517"
    assert result["hypothesis_id"] == "H-207"
    assert result["anchor"]["metric_id"] == "global_wp_clip2_v1"
    assert len(result["folds"]) == 3
    assert gated.shape == inputs.targets.shape
    assert global_blend.shape == inputs.targets.shape
    assert np.array_equal(gated[:, :, 1], inputs.anchor[:, :, 1])
    assert np.array_equal(global_blend[:, :, 1], inputs.anchor[:, :, 1])
    assert np.isfinite(result["delta_gated_vs_anchor"])
    for fold in result["folds"]:
        fit = fold["fit"]
        evaluation_gate = fold["evaluation_gate"]
        assert 0.0 <= fit["calibration_gate_fraction"] <= 1.0
        assert 0.0 <= fit["calibration_weighted_gate_fraction"] <= 1.0
        assert evaluation_gate["scored_row_count"] > 0
        assert 0 <= evaluation_gate["gated_scored_row_count"] <= evaluation_gate["scored_row_count"]
        assert 0.0 <= evaluation_gate["gated_scored_row_fraction"] <= 1.0
        assert 0.0 <= evaluation_gate["gated_official_weight_fraction"] <= 1.0
        evaluation_rows = np.flatnonzero(
            np.asarray(
                [
                    inputs.chain_to_outer_fold[int(chain)] == fold["outer_fold"]
                    for chain in inputs.chain_by_sequence
                ]
            )
        )
        scored = inputs.is_scored[evaluation_rows] & (
            np.arange(inputs.targets.shape[1])[None, :] >= _disagreement_gate.WARMUP_ROWS
        )
        clipped_targets = np.clip(
            inputs.targets[evaluation_rows, :, 0].astype(np.float64), -2.0, 2.0
        )
        clipped_delta = np.clip(
            inputs.candidate_predictions["ab12"][evaluation_rows, :, 0].astype(np.float64),
            -2.0,
            2.0,
        ) - np.clip(inputs.anchor[evaluation_rows, :, 0].astype(np.float64), -2.0, 2.0)
        selected = scored & (np.abs(clipped_delta) >= fit["threshold"])
        assert int(selected.sum()) == evaluation_gate["gated_scored_row_count"]
        assert np.isclose(
            evaluation_gate["gated_scored_row_fraction"], selected.sum() / scored.sum()
        )
        expected_weight_fraction = (
            np.abs(clipped_targets)[selected].sum() / np.abs(clipped_targets)[scored].sum()
        )
        assert np.isclose(
            evaluation_gate["gated_official_weight_fraction"], expected_weight_fraction
        )
