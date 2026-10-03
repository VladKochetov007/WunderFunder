from __future__ import annotations

import itertools
import math
import sys
from fractions import Fraction
from typing import NamedTuple

Q = Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


class Graph(NamedTuple):
    """Immutable local derivatives at one fixed shared-parameter trajectory."""

    jacobians: tuple[Matrix, ...]
    direct: tuple[Matrix, ...]
    losses: tuple[Vector, ...]
    states: tuple[Vector, ...]


def graph(
    weights: Matrix,
    bias: Vector,
    initial: Vector,
    inputs: tuple[Vector, ...],
    losses: tuple[Vector, ...],
    square: bool = False,
) -> Graph:
    """Derive local Jacobians for h=A*h+b+x or h=A*(h*h)+b+x."""
    n = len(initial)
    if not inputs or len(inputs) != len(losses):
        raise ValueError("nonempty graph and local-loss lengths required")
    states = [initial]
    jacobians, direct = [], []
    for value in inputs:
        previous = states[-1]
        feature = tuple(item * item if square else item for item in previous)
        states.append(
            tuple(
                sum(weights[i][j] * feature[j] for j in range(n)) + bias[i] + value[i]
                for i in range(n)
            )
        )
        jacobians.append(
            tuple(
                tuple(weights[i][j] * (2 * previous[j] if square else 1) for j in range(n))
                for i in range(n)
            )
        )
        direct.append(
            tuple(
                tuple(
                    feature[k % n]
                    if k < n * n and k // n == i
                    else Q(1)
                    if k == n * n + i
                    else Q(0)
                    for k in range(n * n + n)
                )
                for i in range(n)
            )
        )
    return Graph(tuple(jacobians), tuple(direct), losses, tuple(states))


def forward_oracle(value: Graph) -> Vector:
    """Full shared-parameter chain rule, using forward sensitivities only."""
    n, m = len(value.losses[0]), len(value.direct[0][0])
    sensitivity = tuple(tuple(Q(0) for _ in range(m)) for _ in range(n))
    total = [Q(0)] * m
    for jacobian, local, loss in zip(value.jacobians, value.direct, value.losses, strict=True):
        sensitivity = tuple(
            tuple(
                local[i][k] + sum(jacobian[i][j] * sensitivity[j][k] for j in range(n))
                for k in range(m)
            )
            for i in range(n)
        )
        for k in range(m):
            total[k] += sum(loss[i] * sensitivity[i][k] for i in range(n))
    return tuple(total)


def reverse_credit(
    value: Graph, keep: tuple[bool, ...], probability: Vector, compensate: bool
) -> tuple[Vector, int, int]:
    """Change backward credit only; return gradient and algebraic work proxies."""
    edges = len(value.losses) - 1
    if len(keep) != edges or len(probability) != edges:
        raise ValueError("one decision and probability per recurrent credit edge required")
    if any(p <= 0 or p > 1 for p in probability):
        raise ValueError("survival probabilities must be in (0,1]")
    n, m = len(value.losses[0]), len(value.direct[0][0])
    total = [Q(0)] * m
    adjoint = value.losses[-1]
    active_local_vjps, active_edge_vjps = 0, 0
    for t in range(len(value.losses) - 1, -1, -1):
        active_local_vjps += int(any(adjoint))
        for k in range(m):
            total[k] += sum(value.direct[t][i][k] * adjoint[i] for i in range(n))
        if t:
            scale = Q(int(keep[t - 1])) / (probability[t - 1] if compensate else Q(1))
            active_edge_vjps += int(bool(scale) and any(adjoint))
            adjoint = tuple(
                value.losses[t - 1][j]
                + scale * sum(value.jacobians[t][i][j] * adjoint[i] for i in range(n))
                for j in range(n)
            )
    return tuple(total), active_local_vjps, active_edge_vjps


def enumerate_masks(probability: Vector) -> tuple[tuple[tuple[bool, ...], Fraction], ...]:
    """Enumerate the complete independent Bernoulli distribution, never sampling."""
    if len(probability) > 10 or any(p <= 0 or p > 1 for p in probability):
        raise ValueError("bounded positive-probability enumeration required")
    outcomes = []
    for keep in itertools.product((False, True), repeat=len(probability)):
        mass = math.prod(
            p if retained else 1 - p for retained, p in zip(keep, probability, strict=True)
        )
        if mass:
            outcomes.append((keep, mass))
    return tuple(outcomes)


def expectation(distribution: tuple[tuple[Vector, Fraction], ...]) -> Vector:
    """Exact probability-weighted vector expectation."""
    return tuple(
        sum(mass * gradient[k] for gradient, mass in distribution)
        for k in range(len(distribution[0][0]))
    )


def tolerance(oracle: Vector) -> float:
    """Fixed analytic-scale machine precision bound, independent of discrepancies."""
    return 256 * sys.float_info.epsilon * max(1.0, float(sum(abs(x) for x in oracle)))


def analyze(value: Graph, probability: Vector, expect_bias: bool) -> dict[str, object]:
    """Check exact gradient unbiasedness, ordinary bias and rounding/variance controls."""
    oracle = forward_oracle(value)
    outcomes = enumerate_masks(probability)
    compensated, ordinary = [], []
    local_work, edge_work = Q(0), Q(0)
    for keep, mass in outcomes:
        gradient, local, edges = reverse_credit(value, keep, probability, True)
        compensated.append((gradient, mass))
        ordinary.append((reverse_credit(value, keep, probability, False)[0], mass))
        local_work += mass * local
        edge_work += mass * edges
    compensated_dist, ordinary_dist = tuple(compensated), tuple(ordinary)
    mean = expectation(compensated_dist)
    ordinary_mean = expectation(ordinary_dist)
    if sum(mass for _, mass in outcomes) != 1 or mean != oracle:
        raise AssertionError("exact compensated expectation/oracle mismatch")
    if (ordinary_mean != oracle) != expect_bias:
        raise AssertionError("ordinary detached planted bias/null control mismatch")
    bound = tolerance(oracle)
    error = max(
        abs(math.fsum(float(mass) * float(g[k]) for g, mass in compensated_dist) - float(oracle[k]))
        for k in range(len(oracle))
    )
    if error > bound:
        raise AssertionError("predeclared machine-precision check failed")
    variance = tuple(
        sum(mass * (g[k] - mean[k]) ** 2 for g, mass in compensated_dist)
        for k in range(len(oracle))
    )
    if (
        reverse_credit(value, (True,) * len(probability), (Q(1),) * len(probability), True)[0]
        != oracle
    ):
        raise AssertionError("all-retained exact full-BPTT null failed")
    return {
        "oracle": oracle,
        "compensated_mean": mean,
        "ordinary_mean": ordinary_mean,
        "ordinary_bias": tuple(x - y for x, y in zip(ordinary_mean, oracle, strict=True)),
        "coordinate_variance": variance,
        "variance_trace": sum(variance),
        "masks_enumerated": 2 ** len(probability),
        "positive_mass_masks": len(outcomes),
        "probabilities": probability,
        "float64_max_error": error,
        "machine_bound": bound,
        "expected_active_local_vjps": local_work,
        "expected_active_edge_vjps": edge_work,
        "expected_retained_edges": sum(probability),
        "full_steps": len(value.losses),
        "max_gradient_abs": max(abs(x) for g, _ in compensated_dist for x in g),
        "minimum_mask_mass": min(mass for _, mass in outcomes),
        "exact_compensation_pass": True,
        "ordinary_bias_expected": expect_bias,
        "all_retained_null_pass": True,
    }


def clipping(gradient: tuple[float, ...], cap: float) -> tuple[float, ...]:
    """Apply scalar or vector global L2 clipping."""
    norm = math.sqrt(math.fsum(x * x for x in gradient))
    scale = min(1.0, cap / norm) if norm else 1.0
    return tuple(scale * x for x in gradient)


def update_witnesses() -> dict[str, object]:
    """Exact scalar and noncollinear vector nonlinear-update counterexamples."""
    scalar = ((Q(1),), (Q(3),))
    vector = ((Q(1), Q(1)), (Q(1), Q(3)))
    results = {}
    for name, draws in (("scalar", scalar), ("vector", vector)):
        distribution = tuple((g, Q(1, 2)) for g in draws)
        mean = expectation(distribution)
        adam_mean = tuple(
            sum(Q(1, 2) * g[k] / (abs(g[k]) + 1) for g in draws) for k in range(len(mean))
        )
        adam_of_mean = tuple(x / (abs(x) + 1) for x in mean)
        if adam_mean == adam_of_mean:
            raise AssertionError("nonlinear Adam-like witness failed")
        clipped = tuple(clipping(tuple(float(x) for x in g), 2.0) for g in draws)
        clip_mean = tuple(math.fsum(g[k] / 2 for g in clipped) for k in range(len(mean)))
        clip_of_mean = clipping(tuple(float(x) for x in mean), 2.0)
        clip_gap = tuple(x - y for x, y in zip(clip_mean, clip_of_mean, strict=True))
        if max(abs(x) for x in clip_gap) <= tolerance(mean):
            raise AssertionError("nonlinear clipping witness failed")
        sgd_mean = expectation(
            tuple((tuple(-Q(1, 8) * x for x in g), mass) for g, mass in distribution)
        )
        if sgd_mean != tuple(-Q(1, 8) * x for x in mean):
            raise AssertionError("linear SGD expectation null failed")
        results[name] = {
            "draws": draws,
            "full_gradient": mean,
            "E_clip": clip_mean,
            "clip_E": clip_of_mean,
            "clip_gap": clip_gap,
            "E_Adam_like": adam_mean,
            "Adam_like_E": adam_of_mean,
            "Adam_like_gap": tuple(x - y for x, y in zip(adam_mean, adam_of_mean, strict=True)),
            "linear_SGD_expectation_exact": True,
        }
    return results
