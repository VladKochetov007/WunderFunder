from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any  # Executed source namespaces and JSON have mixed schemas.

import numpy as np
import torch


def source_globals_probe(trainer_path: Path) -> dict[str, Any]:
    """Execute original option branches and input builder, with no model or fit."""
    from wnn.data import N_FEATURES
    from wnn.features import N_EVENT_FEATURES, event_features_torch, online_zscore_torch

    tree = ast.parse(trainer_path.read_text())
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_inputs"
    )
    branches = [
        node
        for node in main.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "args"
        and node.test.attr in ("raw_only", "online_norm")
    ]
    assert len(branches) == 2
    option_function = ast.FunctionDef(
        name="apply_original_options",
        args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg="args")], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id="n_in", ctx=ast.Store())],
                value=ast.Constant(value=N_FEATURES + N_EVENT_FEATURES),
            ),
            *copy.deepcopy(branches),
            ast.Return(value=ast.Name(id="n_in", ctx=ast.Load())),
        ],
        decorator_list=[],
    )
    isolated = ast.fix_missing_locations(
        ast.Module(body=[copy.deepcopy(builder), option_function], type_ignores=[])
    )
    namespace: dict[str, Any] = {
        "torch": torch,
        "N_FEATURES": N_FEATURES,
        "event_features_torch": event_features_torch,
        "online_zscore_torch": online_zscore_torch,
        "RAW_ONLY": False,
        "ONLINE_NORM": False,
        "INPUT_PROPENSITY": None,
        "RESEARCH_FEATURES": "none",
        "PRIVILEGED_MODE": "none",
        "SENTINEL_PREPROCESSOR": None,
    }
    exec(compile(isolated, str(trainer_path), "exec"), namespace)
    torch.manual_seed(538)
    raw = torch.randn(1, 5, N_FEATURES)
    namespace["apply_original_options"](SimpleNamespace(raw_only=False, online_norm=False))
    fresh_full = namespace["build_inputs"](raw)
    namespace["apply_original_options"](SimpleNamespace(raw_only=True, online_norm=False))
    namespace["apply_original_options"](SimpleNamespace(raw_only=False, online_norm=False))
    after_raw = namespace["build_inputs"](raw)
    assert torch.count_nonzero(fresh_full[..., N_FEATURES:]) > 0
    assert torch.count_nonzero(after_raw[..., N_FEATURES:]) == 0
    assert namespace["RAW_ONLY"] is True
    namespace["RAW_ONLY"] = False
    namespace["apply_original_options"](SimpleNamespace(raw_only=False, online_norm=True))
    declared_width = namespace["apply_original_options"](
        SimpleNamespace(raw_only=False, online_norm=False)
    )
    after_online = namespace["build_inputs"](raw)
    assert after_online.shape[-1] == 323
    assert declared_width == 211
    return {
        "source_sha256": hashlib.sha256(trainer_path.read_bytes()).hexdigest(),
        "fresh_full_event_nonzero": int(torch.count_nonzero(fresh_full[..., N_FEATURES:])),
        "raw_then_full_event_nonzero": int(torch.count_nonzero(after_raw[..., N_FEATURES:])),
        "online_then_plain_observed_width": after_online.shape[-1],
        "online_then_plain_declared_width": declared_width,
        "scope": "exact AST option branches and input builder; no main data path or fitting executed",
    }


def independently_centered_wp(
    labels: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> float:
    """Separate direct implementation of the frozen metric for adversarial data."""
    result: list[float] = []
    for target in range(2):
        y = np.clip(labels[..., target][mask].astype(np.float64), -2.0, 2.0)
        p = np.clip(prediction[..., target][mask].astype(np.float64), -2.0, 2.0)
        w = np.abs(y)
        if w.sum() < 1e-8:
            result.append(0.0)
            continue
        yc = y - np.average(y, weights=w)
        pc = p - np.average(p, weights=w)
        vy = np.average(yc**2, weights=w)
        vp = np.average(pc**2, weights=w)
        value = (
            0.0
            if np.sqrt(vy) <= 1e-8 or np.sqrt(vp) <= 1e-8
            else float(np.average(yc * pc, weights=w) / np.sqrt(vy * vp))
        )
        result.append(value)
    return float(np.mean(result))


def numerical_probes() -> dict[str, Any]:
    """Check official clipping and grouped aggregation against separate algebra."""
    from wnn.distillation import four_family_weighted_mse
    from wnn.evaluation import paired_component_bootstrap, paired_component_moments
    from wnn.gmetric import global_wp
    from wnn.nn import weighted_mse_loss

    generator = np.random.default_rng(538)
    errors = []
    bootstrap_errors = []
    for _ in range(20):
        labels = generator.standard_t(2, (5, 101, 2)).astype(np.float32)
        candidate = (0.3 * labels + generator.normal(size=labels.shape)).astype(np.float32)
        reference = generator.normal(size=labels.shape).astype(np.float32)
        mask = generator.random((5, 101)) < 0.4
        labels[~mask] = np.nan
        candidate[~mask] = np.nan
        oracle = independently_centered_wp(labels, candidate, mask)
        errors.append(abs(global_wp(labels, candidate, mask)["wp"] - oracle))
        moments = paired_component_moments(labels, candidate, reference, mask)
        result = paired_component_bootstrap(
            *moments, np.array([0, 0, 1, 2, 2]), n_boot=8, seed=538, centered=True
        )
        bootstrap_errors.append(abs(result["candidate_wp"] - oracle))
    assert max(errors) < 1e-12
    assert max(bootstrap_errors) < 1e-12
    labels = torch.tensor([[[9.0, -4.0], [0.0, 0.5], [20.0, 10.0]]])
    prediction = torch.tensor([[[0.1, 0.3], [0.2, 0.1], [30.0, -15.0]]])
    valid = torch.tensor([[0.25, 0.7, 0.0]])
    clipped = np.clip(labels.numpy(), -2, 2)
    weights = np.abs(clipped) * valid.numpy()[..., None]
    expected = float((weights * (prediction.numpy() - clipped) ** 2).sum() / weights.sum())
    observed = float(weighted_mse_loss(prediction, labels, valid))
    assert abs(observed - expected) < 1e-6 * max(1.0, abs(expected))
    families = prediction[..., None, :].expand(-1, -1, 4, -1)
    family_targets = labels[..., None, :].expand_as(families)
    original_weights = labels.clamp(-2, 2).abs() * valid[..., None]
    expected_family = float(
        (original_weights[..., None, :] * (families - family_targets) ** 2).sum()
        / (original_weights.sum() * 4)
    )
    observed_family = float(four_family_weighted_mse(families, family_targets, original_weights))
    assert abs(observed_family - expected_family) < 1e-6 * max(1.0, abs(expected_family))
    return {
        "official_wp_random_cases": 20,
        "maximum_wp_oracle_error": max(errors),
        "maximum_grouped_moments_oracle_error": max(bootstrap_errors),
        "wmse_expected": expected,
        "wmse_observed": observed,
        "four_family_raw_expected": expected_family,
        "four_family_raw_observed": observed_family,
    }


def membership_probe(root: Path) -> dict[str, Any]:
    """Recompute direct fitting overlap from original provenance, without weights."""
    split = json.loads(
        (root / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text()
    )
    evaluated = set(map(int, split["evaluation"]["valid"]))
    result = {}
    correction = (
        root
        / "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_INCUMBENT_MEMBERSHIP_CORRECTION_V2_20261008.json"
    )
    names = [record["run"] for record in json.loads(correction.read_text())["records"]]
    names.extend(("bd26_h230_hetero_s3_f0", "bd26_h230_familyoff_s3_retry1_f0"))
    for name in dict.fromkeys(names):
        path = root / "artifacts/runs" / name / "provenance.json"
        if not path.exists():
            result[name] = {"status": "source_not_present"}
            continue
        provenance = json.loads(path.read_text())
        fit = set(map(int, provenance["fit_row_groups"]["valid"]))
        result[name] = {
            "direct_eval_valid_overlap_count": len(fit & evaluated),
            "provenance_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {"evaluation_group_count": len(evaluated), "runs": result}


def actual_update_clocks(root: Path) -> dict[str, Any]:
    """Extract delivered optimizer counters without interpreting skips as invalidity."""
    result = {}
    for run in (
        "bd26_h230_hetero_s3_f0",
        "bd26_h230_familyoff_s3_retry1_f0",
        "bd26_w384raw_s3_f0",
        "bd26_w384raw_s4_f0",
        "bd26_c1v2_s3_f0",
        "bd26_c1v2_s4_f0",
    ):
        path = root / "artifacts/runs" / run / "provenance.json"
        provenance = json.loads(path.read_text())
        value = provenance["cumulative_updates"]
        result[run] = {
            key: value[key]
            for key in (
                "optimizer_attempts",
                "amp_skipped_updates",
                "optimizer_completed_updates",
            )
        }
        result[run]["provenance_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> None:
    """Run bounded nonfitting code checks and save actual observations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.main_root / "src"))
    torch.set_num_threads(1)
    result = {
        "schema": "e538-bounded-code-audit-v1",
        "warm_module_state": source_globals_probe(args.main_root / "scripts/train_rnn.py"),
        "numerical_checks": numerical_probes(),
        "direct_membership_checks": membership_probe(args.main_root),
        "actual_update_clocks": actual_update_clocks(args.main_root),
        "reviewed_source_sha256": {
            str(path): hashlib.sha256((args.main_root / path).read_bytes()).hexdigest()
            for path in (
                Path("scripts/train_rnn.py"),
                Path("src/wnn/nn.py"),
                Path("src/wnn/distillation.py"),
                Path("src/wnn/gmetric.py"),
                Path("src/wnn/evaluation.py"),
                Path("research/campaign_2026-09-29/artifacts/splits/fold0.json"),
                Path(
                    "research/campaign_2026-10-06/bandit_continuation/"
                    "wide_baseline_heterogeneous_H230/H230_SEED3_INDEPENDENT_GRADE_V1.py"
                ),
                Path(
                    "research/campaign_2026-10-06/bandit_continuation/"
                    "continuous_execution_E529/h230_source_v2/scripts/train_rnn.py"
                ),
                Path(
                    "research/campaign_2026-10-06/bandit_continuation/"
                    "continuous_execution_E529/vast/h217_packet_v2/worker.py"
                ),
            )
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
