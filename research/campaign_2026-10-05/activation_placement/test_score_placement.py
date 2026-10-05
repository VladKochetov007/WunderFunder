"""Synthetic CPU tests for the independent E474 grader; no fit data are read."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import score_placement as grader  # noqa: E402

from wnn.gmetric import _centered_moments  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402


def test_independent_base_layout_and_parity() -> None:
    torch.manual_seed(271828)
    scored = grader.ScoringRNN("gelu", "gelu").eval()
    reference = RNNModel(42, hidden=64, layers=2, dropout=0.0, n_out=2).eval()
    reference.load_state_dict(scored.state_dict(), strict=True)
    assert sum(parameter.numel() for parameter in scored.parameters()) == grader.PARAMETERS
    assert tuple(scored.state_dict()) == tuple(reference.state_dict())
    with torch.inference_mode():
        x = torch.randn(2, 29, 42)
        actual, actual_state = scored(x)
        expected, expected_state = reference(x)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual_state, expected_state, atol=0.0, rtol=0.0)
    assert (
        sum(parameter.numel() for parameter in grader.ScoringActivation("sine").parameters()) == 0
    )


def _synthetic_arm_moments(seed_shift: float) -> np.ndarray:
    rng = np.random.default_rng(500 + int(seed_shift * 100))
    result = np.zeros((16, 2, 6), dtype=np.float64)
    for component in range(16):
        target = rng.normal(loc=0.2 * component, size=(64, 2))
        prediction = 0.8 * target + rng.normal(scale=0.4 + abs(seed_shift) * 0.1, size=(64, 2))
        for target_index in range(2):
            result[component, target_index] = _centered_moments(
                target[:, target_index], prediction[:, target_index] + seed_shift
            )
    return result


def test_joint_component_bootstrap_includes_four_arm_interaction() -> None:
    moments = {arm: _synthetic_arm_moments(index * 0.03) for index, arm in enumerate(grader.ARMS)}
    point_scores = {}
    for arm, arm_moments in moments.items():
        target_scores = grader.score_moments_by_target(arm_moments, np.ones(16, dtype=np.int64))
        point_scores[arm] = {"t0": float(target_scores[0]), "t1": float(target_scores[1])}
    result = grader.joint_component_bootstrap(moments, point_scores, n_boot=128, seed=9)
    assert len(result) == 7
    assert all(value["component_draws_shared_across_all_arms"] for value in result.values())
    interaction = result["sine_both_minus_sine_embed_minus_sine_head_plus_base"]
    expected = (
        point_scores["sine_both"]["t0"]
        + point_scores["sine_both"]["t1"]
        - point_scores["sine_embed"]["t0"]
        - point_scores["sine_embed"]["t1"]
        - point_scores["sine_head"]["t0"]
        - point_scores["sine_head"]["t1"]
        + point_scores["base"]["t0"]
        + point_scores["base"]["t1"]
    ) / 2.0
    assert np.isclose(interaction["point_delta"]["wp"], expected, atol=1e-12, rtol=0)
    assert interaction["joint_component_resampling"] is True


def test_nonfinite_raw_prediction_rejected_before_clip() -> None:
    raw = np.asarray([[2.7, -2.6], [np.nan, 1.0]], dtype=np.float32)
    try:
        grader.finite_prediction(raw, "synthetic")
    except ValueError as error:
        assert "before clipping" in str(error)
    else:
        raise AssertionError("nonfinite raw predictions were accepted")
    finite = grader.finite_prediction(raw[:1], "synthetic")
    assert finite[0, 0] == 2.7 and finite[0, 1] == -2.6


def test_runner_shaped_artifact_receipt_requires_exact_current_bytes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        artifact = root / "fit_manifest.json"
        artifact.write_text('{"status":"finished"}\n')
        relative = "fit_manifest.json"
        receipt = grader.content_file(root, relative)
        assert grader.validate_artifact_records(root, {relative}, [receipt]) == {relative: receipt}
        stale = {**receipt, "sha256": "0" * 64}
        for records in (
            [stale],
            [receipt, receipt],
            [],
            [{"path": relative, "size": 1, "sha256": "f" * 64}],
        ):
            try:
                grader.validate_artifact_records(root, {relative}, records)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    "missing, duplicate, or mismatched attempt artifact was accepted"
                )


def main() -> None:
    torch.set_num_threads(1)
    test_independent_base_layout_and_parity()
    test_joint_component_bootstrap_includes_four_arm_interaction()
    test_nonfinite_raw_prediction_rejected_before_clip()
    test_runner_shaped_artifact_receipt_requires_exact_current_bytes()
    print("E474 independent grader synthetic CPU tests passed")


if __name__ == "__main__":
    main()
