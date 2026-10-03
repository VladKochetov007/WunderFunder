from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

Row = tuple[int, Fraction, Fraction]
Chunk = tuple[Row, ...]
BinaryRow = tuple[int, int, int]


def distilled_optimum(
    chunks: Sequence[Chunk], *, chunk_normalized: bool, alpha: Fraction
) -> dict[int, Fraction]:
    """Solve finite quadratic coefficients exactly; teacher is fixed at +/-1/4."""
    a: dict[int, Fraction] = defaultdict(Fraction)
    b: dict[int, Fraction] = defaultdict(Fraction)
    all_weight = sum(w for chunk in chunks for _, _, w in chunk)
    for chunk in chunks:
        denominator = sum(w for _, _, w in chunk) * len(chunks) if chunk_normalized else all_weight
        for x, y, w in chunk:
            a[x] += w / denominator
            b[x] += w * y / denominator
    # Independently averaged unweighted teacher loss, one observation of each x
    # per chunk. Its mass is 1/2 per x, irrespective of supervised denominators.
    return {
        x: ((1 - alpha) * b[x] + alpha * Fraction(1, 2) * Fraction(2 * x - 1, 4))
        / ((1 - alpha) * a[x] + alpha * Fraction(1, 2))
        for x in a
    }


def weighted_pearson(y: Sequence[float], p: Sequence[float]) -> float:
    """Synthetic diagnostic implementing METRIC.md clipping/weight convention."""
    if len(y) != len(p) or not y:
        raise ValueError("nonempty aligned rows required")
    if not all(math.isfinite(v) for v in (*y, *p)):
        raise ValueError("raw selected values must be finite")
    yc = [min(2.0, max(-2.0, v)) for v in y]
    pc = [min(2.0, max(-2.0, v)) for v in p]
    weights = [abs(v) for v in yc]
    total = sum(weights)
    if total < 1e-8:
        return 0.0
    ym = sum(w * v for w, v in zip(weights, yc, strict=True)) / total
    pm = sum(w * v for w, v in zip(weights, pc, strict=True)) / total
    yy = sum(w * (v - ym) ** 2 for w, v in zip(weights, yc, strict=True))
    pp = sum(w * (v - pm) ** 2 for w, v in zip(weights, pc, strict=True))
    if math.sqrt(yy / total) <= 1e-8 or math.sqrt(pp / total) <= 1e-8:
        return 0.0
    yp = sum(w * (v - ym) * (u - pm) for w, v, u in zip(weights, yc, pc, strict=True))
    return min(1.0, max(-1.0, yp / math.sqrt(yy * pp)))


def check_normalizers() -> dict[str, object]:
    """Same high CV can coexist with unchanged or materially changed optima."""
    scales = [1] * 14 + [1000] * 2
    residuals = {
        "balanced": [1] * 7 + [-1] * 7 + [1, -1],
        "correlated": [1] * 6 + [-1] * 8 + [1, 1],
    }
    denominator = [2 * s for s in scales]
    mean = sum(denominator) / len(denominator)
    cv = math.sqrt(sum((d - mean) ** 2 for d in denominator) / len(denominator)) / mean
    assert cv > 2
    result: dict[str, object] = {"denominators": denominator, "normalizer_cv": cv}
    for name, residual in residuals.items():
        chunks = [
            tuple((x, Fraction(2 * x - 1, 2) + e, Fraction(s)) for x in (0, 1))
            for s, e in zip(scales, residual, strict=True)
        ]
        normalized = distilled_optimum(chunks, chunk_normalized=True, alpha=Fraction(2, 5))
        pooled = distilled_optimum(chunks, chunk_normalized=False, alpha=Fraction(2, 5))
        differences = [abs(normalized[x] - pooled[x]) for x in (0, 1)]
        if name == "balanced":
            assert normalized == pooled == {0: Fraction(-2, 5), 1: Fraction(2, 5)}
        else:
            assert min(differences) > Fraction(1, 2)
        targets = [float(y) for chunk in chunks for _, y, _ in chunk]
        pn = [float(normalized[x]) for chunk in chunks for x, _, _ in chunk]
        pp = [float(pooled[x]) for chunk in chunks for x, _, _ in chunk]
        wp_normalized, wp_pooled = weighted_pearson(targets, pn), weighted_pearson(targets, pp)
        assert abs(wp_normalized - wp_pooled) < 1e-12
        result[name] = {
            "normalized_optimum_exact": {str(x): str(v) for x, v in normalized.items()},
            "pooled_optimum_exact": {str(x): str(v) for x, v in pooled.items()},
            "max_optimum_difference": float(max(differences)),
            "target_weighted_pearson_normalized": wp_normalized,
            "target_weighted_pearson_pooled": wp_pooled,
        }
    return result


def check_stopping_selection() -> dict[str, object]:
    """Complete-case selection makes two genuinely different laws indistinct."""
    records = [
        (market, path, 3 if market == 0 and path[1] + path[2] == 0 else 1)
        for market in (0, 1)
        for path in itertools.product((-1, 1), repeat=3)
    ]
    laws: dict[str, Callable[[tuple[int, ...]], int]] = {
        "fixed_step3": sum,
        "first_unit_barrier": lambda p: p[0],
        "fixed_step2": lambda p: p[0] + p[1],
    }
    complete = [r for r in records if r[2] == 3]

    def mse(rows: Sequence[tuple[int, tuple[int, ...], int]], name: str) -> Fraction:
        return Fraction(sum((laws[name](p) - sum(p)) ** 2 for _, p, _ in rows), len(rows))

    observed = {name: mse(complete, name) for name in laws}
    full = {name: mse(records, name) for name in laws}
    assert observed["fixed_step3"] == observed["first_unit_barrier"] == 0
    assert observed["fixed_step2"] == 1
    assert full["first_unit_barrier"] == 2
    disagreements = sum(laws["first_unit_barrier"](p) != sum(p) for _, p, _ in records)
    assert disagreements * 2 == len(records)
    unknown = Fraction(len(records) - len(complete), len(records))
    assert unknown == Fraction(3, 4)
    # Prefix [1] admits final displacement 1 or 3: unseen transition mass is
    # not a zero return and cannot be replaced with a non-hit event.
    assert (1, -1, 1)[:1] == (1, 1, 1)[:1]
    assert sum((1, -1, 1)) != sum((1, 1, 1))
    reverse_firsthit_error = Fraction(
        sum((p[-1] - sum(p)) ** 2 for _, p, _ in complete), len(complete)
    )
    assert reverse_firsthit_error == 2
    assert all(sum(p[::-1]) == sum(p) for _, p, _ in records)
    # The short stopping candidate is computable even on the censored rows.
    # Restricting it to the common full-path subset discards a decisive falsifier.
    assert all(coverage >= 1 for _, _, coverage in records)
    return {
        "population_rows": len(records),
        "complete_rows": len(complete),
        "coverage_by_class": {"0": 0.5, "1": 0.0},
        "unknown_transition_mass": float(unknown),
        "candidate_specific_available_rows": {
            "fixed_step3": len(complete),
            "first_unit_barrier": len(records),
            "fixed_step2": len(complete),
        },
        "first_unit_barrier_mse_on_all_available_rows": float(full["first_unit_barrier"]),
        "observed_complete_case_mse": {k: float(v) for k, v in observed.items()},
        "transport_full_population_mse": {k: float(v) for k, v in full.items()},
        "law_disagreement_fraction": disagreements / len(records),
        "reverse_firsthit_complete_mse": float(reverse_firsthit_error),
        "reverse_sum_control": "invariant by construction; not a falsifier of endpoint displacement",
        "unobserved_mse_bound_from_range_only": [0, float(unknown * 36)],
        "label_scale": "raw synthetic terminal displacements; MSE is not official WP",
    }


def risk(rows: Sequence[BinaryRow], predictor: Callable[[int, int], float]) -> float:
    return sum((predictor(c, h) - y) ** 2 for c, h, y in rows) / len(rows)


def conditional_risk(rows: Sequence[BinaryRow], *, use_history: bool) -> Fraction:
    """Enumerate exact minimum squared risk on the specified finite population."""
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for c, h, y in rows:
        groups[(c, h) if use_history else (c,)].append(y)
    squared = Fraction(0)
    for targets in groups.values():
        prediction = Fraction(sum(targets), len(targets))
        squared += sum((prediction - y) ** 2 for y in targets)
    return squared / len(rows)


def check_usage_support() -> dict[str, object]:
    """Response measures function behavior, not conditional information value."""
    redundant = [(c, c, c) for c in (-1, 1)]

    def sensitive(c: int, h: int) -> float:
        return c + 10 * (h - c)

    supported = {(c, h) for c, h, _ in redundant}
    assert risk(redundant, sensitive) == 0
    assert conditional_risk(redundant, use_history=False) == 0
    assert conditional_risk(redundant, use_history=True) == 0
    assert all((c, -h) not in supported for c, h, _ in redundant)
    raw_response = sum(abs(sensitive(c, -h) - sensitive(c, h)) for c, h, _ in redundant) / 2
    assert raw_response == 20
    identity_response = sum(abs(sensitive(c, h) - sensitive(c, h)) for c, h, _ in redundant)
    assert identity_response == 0
    informative = [(c, h, h) for c, h in itertools.product((-1, 1), repeat=2)]

    def ignored(c: int, h: int) -> float:
        return c

    assert all(ignored(c, -h) == ignored(c, h) for c, h, _ in informative)
    assert risk(informative, ignored) == 2
    assert conditional_risk(informative, use_history=False) == 1
    assert conditional_risk(informative, use_history=True) == 0
    assert risk(informative, lambda c, h: h) == 0
    # If current-only has already been fitted optimally, the history value is 1;
    # 2-to-0 also includes fixing the ignored model's avoidable current-only error.
    return {
        "redundant_history": {
            "raw_prediction_response": raw_response,
            "perturbed_ood_mse": risk(redundant, lambda c, h: sensitive(c, -h)),
            "perturbed_out_of_support_fraction": 1.0,
            "conditional_history_risk_improvement": 0.0,
            "identity_and_conditional_donor_response": identity_response,
            "wp_original": weighted_pearson(
                [r[2] for r in redundant], [sensitive(c, h) for c, h, _ in redundant]
            ),
            "wp_ood": weighted_pearson(
                [r[2] for r in redundant], [sensitive(c, -h) for c, h, _ in redundant]
            ),
        },
        "ignored_informative_history": {
            "history_flip_response": 0.0,
            "perturbed_out_of_support_fraction": 0.0,
            "baseline_mse": risk(informative, ignored),
            "optimal_current_only_mse": float(conditional_risk(informative, use_history=False)),
            "optimal_current_and_history_mse": float(
                conditional_risk(informative, use_history=True)
            ),
            "matched_optimal_current_only_history_value": 1.0,
            "feasible_history_predictor_mse": risk(informative, lambda c, h: h),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind finite specification counterexamples")
    parser.add_argument("--fixture", choices=("all", "N1", "S1", "U1"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    own_dir = Path(__file__).resolve().parent
    if args.output.resolve().parent != own_dir:
        parser.error("results must be written directly in this critic's own directory")
    functions = {"N1": check_normalizers, "S1": check_stopping_selection, "U1": check_usage_support}
    results = {
        name: function() for name, function in functions.items() if args.fixture in ("all", name)
    }
    output = {
        "status": "all selected preregistered assertions passed",
        "evidence_scope": "exact finite synthetic populations; no real actor or GRU assessment",
        "sample_ids": [],
        "fixtures": results,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
