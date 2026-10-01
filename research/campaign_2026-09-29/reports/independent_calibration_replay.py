"""Replay small calibration matrices without opening any competition dataset."""

from __future__ import annotations

import ast
import hashlib
import json
import resource
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "research/campaign_2026-09-29/artifacts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from decide import wp
from holdout_compare import blend_sums

from wnn.gmetric import global_wp


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Independently solve centered univariate OLS in extended precision."""
    x, y = x.astype(np.longdouble), y.astype(np.longdouble)
    dx, dy = x - x.mean(), y - y.mean()
    slope = np.sum(dx * dy) / np.sum(dx * dx)
    return float(slope), float(y.mean() - slope * x.mean())


def inner_errors(x: np.ndarray, y: np.ndarray, training: np.ndarray) -> np.ndarray:
    result = []
    for variant in x:
        errors = []
        for held in training:
            fitted = training[training != held]
            slope, intercept = fit(variant[fitted], y[fitted])
            errors.append(intercept + slope * variant[held] - y[held])
        result.append(np.sqrt(np.mean(np.square(errors))))
    return np.asarray(result)


def close(actual: float, expected: float) -> None:
    assert abs(actual - expected) < 2e-11, (actual, expected)


def replay(scenario: str, families: dict[str, str]) -> dict:
    report = json.loads((ARTIFACTS / f"nested_public_calibration_{scenario}.json").read_text())
    family = json.loads((ARTIFACTS / f"nested_public_family_holdout_{scenario}.json").read_text())
    names = report["submission_order"]
    y = np.asarray(report["public_scores_user_reported"])
    x = np.asarray(report["local_scores_by_variant_by_submission"])
    assert x.shape == (21, 11) and y.shape == (11,) and np.isfinite(x).all()
    assert len(set(names)) == 11 and set(names) == set(families)
    variants = report["proxy_variants"]
    expected_variants = [
        (weight, lam)
        for weight in (
            "holdout",
            "train_mix",
            "class3_x2",
            "class3_x0_5",
            "isolated",
            "linked",
            "train_unlike",
        )
        for lam in (0, 0.02, 0.05)
    ]
    assert [(row["weight_id"], row["soft_mask_lambda"]) for row in variants] == expected_variants
    all_rows = np.arange(len(names))
    errors, variant_choices = [], []
    maximum = 0.0
    for held, recorded in enumerate(report["nested_loso"]):
        training = all_rows[all_rows != held]
        inner = inner_errors(x, y, training)
        chosen = int(inner.argmin())
        slope, intercept = fit(x[chosen, training], y[training])
        prediction = intercept + slope * x[chosen, held]
        label = f"{variants[chosen]['weight_id']}_lambda{variants[chosen]['soft_mask_lambda']:g}"
        assert recorded["name"] == names[held] and recorded["chosen_variant"] == label
        close(recorded["public"], y[held])
        close(recorded["inner_rmse"], inner[chosen])
        close(recorded["predicted"], prediction)
        close(recorded["error"], prediction - y[held])
        maximum = max(maximum, abs(prediction - recorded["predicted"]))
        errors.append(prediction - y[held])
        variant_choices.append(label)
    errors = np.asarray(errors)
    close(report["nested_loso_rmse"], np.sqrt(np.mean(errors**2)))
    close(report["nested_loso_mae"], np.mean(np.abs(errors)))
    close(report["nested_loso_bias"], errors.mean())
    assert np.allclose(
        report["nested_loso_error_range"], [errors.min(), errors.max()], atol=2e-11, rtol=0
    )
    full_inner = inner_errors(x, y, all_rows)
    chosen = int(full_inner.argmin())
    slope, intercept = fit(x[chosen], y)
    label = f"{variants[chosen]['weight_id']}_lambda{variants[chosen]['soft_mask_lambda']:g}"
    assert label == report["full_selected_variant"]
    close(report["full_slope"], slope)
    close(report["full_intercept"], intercept)
    close(report["full_variant_loso_rmse"], full_inner[chosen])
    top = np.flatnonzero(y >= 0.6586)
    comparisons = [(i, j) for i in top for j in top if j > i and abs(y[i] - y[j]) >= 0.0004]
    close(
        report["top_pair_sign_accuracy"],
        np.mean(
            [np.sign(x[chosen, i] - x[chosen, j]) == np.sign(y[i] - y[j]) for i, j in comparisons]
        ),
    )

    recorded_family = {row["name"]: row for row in family["family_heldout_rows"]}
    assert set(recorded_family) == set(names)
    grouped_errors = []
    family_of = np.asarray([families[name] for name in names])
    for group in dict.fromkeys(family_of):
        held = all_rows[family_of == group]
        training = all_rows[family_of != group]
        inner = inner_errors(x, y, training)
        chosen = int(inner.argmin())
        slope, intercept = fit(x[chosen, training], y[training])
        for index in held:
            row = recorded_family[names[index]]
            predicted = intercept + slope * x[chosen, index]
            assert row["family"] == group and row["selected_variant"] == variants[chosen]
            close(row["inner_package_loso_rmse"], inner[chosen])
            close(row["public"], y[index])
            close(row["predicted"], predicted)
            close(row["error"], predicted - y[index])
            maximum = max(maximum, abs(predicted - row["predicted"]))
            grouped_errors.append(predicted - y[index])
    grouped_errors = np.asarray(grouped_errors)
    close(family["family_heldout_rmse"], np.sqrt(np.mean(grouped_errors**2)))
    close(family["family_heldout_mae"], np.abs(grouped_errors).mean())
    close(family["family_heldout_bias"], grouped_errors.mean())
    assert family["n_families"] == len(set(family_of))
    assert family["scenario"] == scenario and report["ema_membership_scenario"] == scenario
    assert len(set(report["holdout_row_groups"])) == 368
    assert len(set(report["isolated_row_groups"])) == 48
    assert set(report["isolated_row_groups"]) <= set(report["holdout_row_groups"])
    return {
        "status": "pass",
        "max_abs_replay_error": maximum,
        "nested_loso_rmse": report["nested_loso_rmse"],
        "family_holdout_rmse": family["family_heldout_rmse"],
        "best_reported_exploratory_forecast": max(
            report["candidate_forecasts_exploratory"].values()
        ),
        "full_selected_variant": label,
        "n_top_pairs": len(comparisons),
        "candidate_forecast_scope": "reported values only: candidate local-score matrix is not persisted",
    }


def identity() -> dict:
    reports = {
        scenario: json.loads((ARTIFACTS / f"nested_public_calibration_{scenario}.json").read_text())
        for scenario in ("pair", "single")
    }
    pair, single = reports["pair"], reports["single"]
    for key in (
        "submission_order",
        "public_scores_user_reported",
        "proxy_variants",
        "holdout_row_groups",
        "isolated_row_groups",
        "input_file_sha256",
        "local_archive_sha256",
        "ema_member0_npz_sha256",
        "calibration_script_sha256",
    ):
        assert pair[key] == single[key], key
    a, b = (
        np.asarray(report["local_scores_by_variant_by_submission"]) for report in (pair, single)
    )
    assert np.array_equal(a[:, :9], b[:, :9])
    assert np.array_equal(b[:, -1], b[:, -2])
    for key in set(pair["prediction_file_sha256"]) & set(single["prediction_file_sha256"]):
        assert pair["prediction_file_sha256"][key] == single["prediction_file_sha256"][key]
    entries = {}
    for name, expected in pair["local_archive_sha256"].items():
        path = ROOT / "submissions" / name / f"{name}.zip"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        with zipfile.ZipFile(path) as archive:
            solution = archive.read("solution.py").decode()
            members = json.loads(archive.read("members.json"))
            member0 = hashlib.sha256(archive.read(members[0]["weights"])).hexdigest()
            v2 = (
                next(
                    item
                    for item in archive.namelist()
                    if item.startswith("v2/") and item.endswith(".so")
                )
                if name.endswith("shared")
                else None
            )
            entries[name] = {
                "member0_sha256": member0,
                "v2_native_sha256": hashlib.sha256(archive.read(v2)).hexdigest() if v2 else None,
                "v2_collapses_to_member0": 'members = [{**members[0], "coef": 1.0}]' in solution,
            }
    failed = ("final_ema6_shared", "final_mixed_ema6_shared")
    assert entries[failed[0]] == entries[failed[1]]
    assert entries[failed[0]]["member0_sha256"] == pair["ema_member0_npz_sha256"][failed[0]]
    return {
        "status": "pass",
        "archived_identity": entries,
        "single_scenario_same_proxy_different_public": True,
        "association_conclusion": "same deterministic v2 predictor cannot produce distinct scores on one identical evaluation panel; portal CPU/panel/association is not directly observed",
    }


def legacy_metric_counterexamples() -> dict:
    y = np.zeros((2, 131, 2), dtype=np.float32)
    center = np.float32(0.35)
    y[:, 99:] = np.tile([[np.nextafter(center, np.float32(0))] * 2, [center] * 2], (16, 1))
    mask = np.zeros(y.shape[:-1], dtype=bool)
    mask[:, 99:] = True
    with patch("holdout_compare.np.load", return_value=y.copy()):
        sums = blend_sums({"synthetic": 1.0}, np.arange(2), y, mask)
    official, raw = global_wp(y, y, mask)["wp"], wp(sums)
    assert official == 1 and raw == 0, (official, raw)
    y[:, 99:] = np.tile([[-1, -1], [1, 1]], (16, 1))
    bad = y.copy()
    bad[:, 100] = np.inf
    with patch("holdout_compare.np.load", return_value=bad):
        accepted = float(wp(blend_sums({"synthetic": 1.0}, np.arange(2), y, mask)))
    try:
        global_wp(y, bad, mask)
    except ValueError:
        pass
    else:
        raise AssertionError("official reference accepted nonfinite raw prediction")
    return {
        "nearconstant_official": official,
        "nearconstant_legacy_calibration_helper": raw,
        "selected_infinity_accepted_by_legacy_helper": accepted,
        "actual_artifact_impact": "not determined without dataset scans; no such scans performed",
    }


def main() -> None:
    start = time.perf_counter()
    tree = ast.parse((ROOT / "scripts/campaign_calibration_audit.py").read_text())
    families = ast.literal_eval(
        next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "FAMILY" for target in node.targets
            )
        )
    )
    result = {
        "replays": {scenario: replay(scenario, families) for scenario in ("pair", "single")},
        "identity": identity(),
        "metric_counterexamples": legacy_metric_counterexamples(),
        "family_definition": families,
    }
    paths = [
        *ARTIFACTS.glob("nested_public_calibration_*.json"),
        *ARTIFACTS.glob("nested_public_family_holdout_*.json"),
        ROOT / "scripts/campaign_public_calibration.py",
        ROOT / "scripts/campaign_calibration_audit.py",
        ROOT / "scripts/holdout_compare.py",
        ROOT / "scripts/decide.py",
    ]
    result["sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    result["elapsed_seconds"] = time.perf_counter() - start
    result["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
