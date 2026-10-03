from __future__ import annotations

import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
F = Fraction


def enumerate_graph(dependent: bool, marginal: bool, unweighted: bool) -> dict:
    """Enumerate a fixed forward DAG and all four joint gate patterns exactly."""
    outcomes = []
    expectation = F(0)
    second = F(0)
    for b1, b2 in itertools.product([0, 1], repeat=2):
        q1 = F(1, 2)
        q2 = F(3, 4) if dependent and b1 else F(1, 4) if dependent else F(1, 2)
        probability = (q1 if b1 else 1 - q1) * (q2 if b2 else 1 - q2)
        d1 = F(1) if unweighted else q1
        d2 = F(1) if unweighted else F(1, 2) if marginal else q2
        gradient = 1 + 3 * F(b2) / d2 + 6 * F(b1) / d1 * F(b2) / d2
        expectation += probability * gradient
        second += probability * gradient**2
        outcomes.append(
            dict(b1=b1, b2=b2, probability=str(probability), q2=str(q2), gradient=str(gradient))
        )
    return dict(
        outcomes=outcomes, expectation=str(expectation), variance=str(second - expectation**2)
    )


def checks() -> dict:
    """Use exact rational or tiny analytic operations, no actor/model/capture."""
    independent = enumerate_graph(False, False, False)
    dependent = enumerate_graph(True, False, False)
    wrong_marginal = enumerate_graph(True, True, False)
    ordinary = enumerate_graph(False, False, True)
    assert independent["expectation"] == dependent["expectation"] == "10"
    assert wrong_marginal["expectation"] == "13"
    assert ordinary["expectation"] == "4"
    assert independent["variance"] == "153" and dependent["variance"] == "87"
    # Clipping an unbiased gradient is NOT an unbiased clipped update.
    raw = [-1, 3]
    mean = F(sum(raw), 2)
    clipped_mean = sum(F(max(-1, min(1, g)), 2) for g in raw)
    assert mean == 1 and clipped_mean == 0
    compensation_cap_mean = F(1, 2) * min(2, 1)
    assert compensation_cap_mean == F(1, 2)
    # Normalization = exact polar direction for a nonzero1x2 row, Muon-LIKE only.
    expected_direction = [F(3, 4) + float(F(1, 4)) / math.sqrt(17), 1 / math.sqrt(17)]
    direction_of_mean = [1 / math.sqrt(2), 1 / math.sqrt(2)]
    assert any(
        abs(float(a) - b) > 1e-2 for a, b in zip(expected_direction, direction_of_mean, strict=True)
    )
    affine_mean = sum(F(3 + 2 * g, 2) for g in raw)
    assert affine_mean == 3 + 2 * mean
    # Gate-dependent coefficients correlate with inverse-probability multiplier.
    coefficient_expectation = F(1, 2)
    compensated_coefficient = F(1, 2) * 2
    assert coefficient_expectation != compensated_coefficient
    # q(theta)=theta: differentiating B*theta/q(theta) gives0; detached q gives1.
    surrogate_gradient = F(0)
    detached_estimator_mean = F(1, 2) / F(1, 2)
    assert surrogate_gradient == 0 and detached_estimator_mean == 1
    theta1, theta2 = 1, 2
    mixed = theta1 + theta2 + 1
    current = 2 * theta2 + 1
    detached_local = theta1 + 1
    assert (mixed, current, detached_local) == (4, 5, 2)
    # Two-state noncommuting fixed Jacobians catch accidentally reversed products.
    a = np.array([[1, 1], [0, 1]], dtype=np.int64)
    b = np.array([[1, 0], [1, 1]], dtype=np.int64)
    u, v, w, readout = (np.array(z, dtype=np.int64) for z in [[1, 0], [1, 1], [1, 1], [1, 0]])
    vector_full = int(readout @ (b @ a @ u + b @ v + w))
    vector_wrong_order = int(readout @ (a @ b @ u + b @ v + w))
    assert (vector_full, vector_wrong_order) == (3, 4)
    # Intentional tiny fp16 overflow, NOT actual GradScaler/backend evidence.
    scale = np.float16(32768)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = [np.float16(g) * scale for g in [0, 2]]
    skipped = [not bool(np.isfinite(g)) for g in scaled]
    executed = [
        0.0 if skip else float(g) / float(scale) for g, skip in zip(scaled, skipped, strict=True)
    ]
    assert skipped == [False, True] and sum(executed) / 2 == 0
    assert F(0 + 2, 2) == 1  # Raw compensated mean before overflow/selective skip.
    return dict(
        passed=True,
        evidence="tiny exhaustive/analytic CPU math ONLY",
        fixed_graph_gradient=10,
        independent=independent,
        dependent=dependent,
        wrong_marginal=wrong_marginal,
        unweighted=ordinary,
        clipping=dict(raw_mean=str(mean), expected_clip=str(clipped_mean), clip_of_mean=1),
        compensation_cap_mean=str(compensation_cap_mean),
        normalization=dict(
            expected=[float(v) for v in expected_direction], of_mean=direction_of_mean
        ),
        affine_positive_control=str(affine_mean),
        gate_correlated_coefficient_bias=str(compensated_coefficient - coefficient_expectation),
        inverse_hazard_autodiff=dict(
            surrogate_gradient=str(surrogate_gradient), detached_mean=str(detached_estimator_mean)
        ),
        mixed_weight_graph=dict(mixed=mixed, current=current, local_detached=detached_local),
        noncommuting_jacobian=dict(correct=vector_full, wrong=vector_wrong_order),
        intentional_toy_fp16_overflow=dict(skipped=skipped, raw_mean=1, executed_mean=0),
        actor_reads=0,
        actual_capture_reads=0,
        training_or_score_claim=False,
    )


def freeze_hashes() -> None:
    """Administrative hash finalization; run only after CPU admission is cleared."""
    artifacts = {}
    for path in sorted(ROOT.iterdir()):
        if path.is_file() and path.name not in {
            "HASHES.json",
            "checks.log",
            "checks_final.log",
            "hashes.log",
            "freeze.log",
        }:
            artifacts[path.name] = dict(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size
            )
    (ROOT / "HASHES.json").write_text(
        json.dumps(dict(artifacts=artifacts, exclusions="self and open execution logs"), indent=2)
    )


if __name__ == "__main__":
    result = checks()
    (ROOT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    freeze_hashes()
