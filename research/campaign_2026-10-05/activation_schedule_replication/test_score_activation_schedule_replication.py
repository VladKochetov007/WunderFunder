"""Synthetic CPU checks for the independent E480 scorer; never reads test data."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import score_activation_schedule_replication as scorer  # noqa: E402

from wnn.campaign_spec import digest as spec_digest  # noqa: E402
from wnn.campaign_spec import seal_spec  # noqa: E402
from wnn.evaluation import paired_component_moments  # noqa: E402
from wnn.gmetric import global_wp  # noqa: E402


def test_lr_trace_source_identity() -> None:
    """Check current E480 CPU receipt against frozen E476 schedule hashes."""
    run = HERE / "preflight_cpu_v3"
    traces = scorer.validate_lr_traces(run)
    assert set(traces) == set(scorer.SCHEDULES)
    for name, values in traces.items():
        assert len(values) == scorer.UPDATES
        assert scorer.trace_digest(values) == scorer.trace_digest(
            np.asarray(
                __import__("json").loads((run / "lr_traces.json").read_text())["protocols"][name][
                    "values"
                ],
                dtype=np.float64,
            )
        )


def test_independent_model_formula_and_identity_buffers() -> None:
    """Check standalone model key layout and explicit normalization buffers."""
    base = scorer.make_model("base")
    sine = scorer.make_model("sine_embed")
    assert tuple(base.state_dict()) == tuple(sine.state_dict())
    scorer.validate_identity_buffers(base.state_dict())
    with torch.no_grad():
        x = torch.linspace(-5.0, 5.0, 42).reshape(1, 1, 42)
        projected = sine.inp(x)
        expected = torch.nn.functional.gelu(projected, approximate="none") + 0.1 * torch.sin(
            projected
        )
        torch.testing.assert_close(sine.encode(x), expected, atol=0.0, rtol=0.0)
    invalid = dict(base.state_dict())
    invalid["mean"] = torch.ones(42)
    try:
        scorer.validate_identity_buffers(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("non-identity model normalization buffer was accepted")


def test_external_fit_normalizer_is_shared_and_test_independent() -> None:
    """Reject a test-side normalizer and accept the same fit-only transform."""
    mean = np.arange(42, dtype=np.float32)
    scale = np.full(42, 2.0, dtype=np.float32)
    norm = {"fit_mean": mean.tolist(), "fit_scale": scale.tolist()}
    fit_manifest = {"normalization": norm}
    all_manifest = {"normalization": norm.copy()}
    fit_nodes = [{"row_group": 10}, {"row_group": 11}]
    all_nodes = [*fit_nodes, {"row_group": 20}]
    fit_x = np.zeros((2, 3, 42), dtype=np.float32)
    all_x = np.concatenate((fit_x, np.ones((1, 3, 42), dtype=np.float32)), axis=0)
    got_mean, got_scale = scorer.validate_external_normalizer(
        fit_manifest, fit_x, fit_nodes, all_manifest, all_x, all_nodes
    )
    np.testing.assert_array_equal(got_mean, mean)
    np.testing.assert_array_equal(got_scale, scale)
    changed_test_stats = {"normalization": {**norm, "fit_mean": (mean + 1).tolist()}}
    try:
        scorer.validate_external_normalizer(
            fit_manifest, fit_x, fit_nodes, changed_test_stats, all_x, all_nodes
        )
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent fit/test normalizer was accepted")
    fit_features = np.arange(24 * 5 * 42, dtype=np.float32).reshape(24, 5, 42)
    derived_mean, derived_scale = scorer.normalizer_from_fit_features(fit_features)
    expected_mean = fit_features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    expected_scale = np.maximum(fit_features.std(axis=(0, 1), dtype=np.float64), 0.001).astype(
        np.float32
    )
    np.testing.assert_array_equal(derived_mean, expected_mean)
    np.testing.assert_array_equal(derived_scale, expected_scale)


def test_raw_label_finite_gate_precedes_clipping() -> None:
    """Nonfinite selected labels must fail before clipping could hide them."""
    raw = np.asarray([[0.1, -0.2], [2.5, -3.0]], dtype=np.float64)
    mask = np.asarray([True, True])
    clipped = np.clip(raw, -2.0, 2.0).astype(np.float32)
    np.testing.assert_array_equal(scorer.validate_raw_target_component(raw, mask, clipped, 11), raw)
    broken = raw.copy()
    broken[0, 0] = np.inf
    try:
        scorer.validate_raw_target_component(broken, mask, clipped, 11)
    except ValueError:
        pass
    else:
        raise AssertionError("selected infinity was not rejected before clipping")


def test_synthetic_runner_receipt_is_fail_closed() -> None:
    """Exercise successful attempt proof and representative failure conditions."""
    with tempfile.TemporaryDirectory(prefix="e480-attempt-proof-") as directory:
        root = Path(directory)
        for relative in scorer.EXPECTED_RUN_ARTIFACTS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic sealed output\n")
        code_path = root / "fit_actor.py"
        input_path = root / "fit_input.json"
        code_path.write_text("pass\n")
        input_path.write_text("{}\n")
        raw_spec = {
            "version": 1,
            "name": scorer.EXPECTED_SPEC_NAME,
            "kind": "gpu",
            "argv": ["{python}", "fit_actor.py", "--device", "cuda"],
            "code": ["fit_actor.py"],
            "inputs": ["fit_input.json"],
            "artifacts": list(scorer.EXPECTED_RUN_ARTIFACTS),
            "limits": {
                "cpus": 1,
                "ram_bytes": 2 * 1024**3,
                "timeout_seconds": 300,
                "gpu_memory_bytes": 512 * 1024**2,
                "gpu_stop_bytes": 1024 * 1024**2,
            },
            "environment": {},
        }
        spec = seal_spec(root, raw_spec)
        runtime_files = ("runtime_a.py", "runtime_b.py")
        for name in runtime_files:
            (root / name).write_text("runtime\n")
        runtime = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in runtime_files
        }
        run = root / Path(scorer.EXPECTED_RUN_ARTIFACTS[0]).parent
        artifacts = [
            {
                "path": relative,
                "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                "size": (root / relative).stat().st_size,
            }
            for relative in scorer.EXPECTED_RUN_ARTIFACTS
        ]
        record = {
            "id": scorer.EXPECTED_ATTEMPT_ID,
            "kind": "gpu",
            "spec": spec,
            "spec_sha256": spec_digest(spec),
            "argv": ["/usr/bin/python", *spec["argv"][1:]],
            "runtime_sha256": runtime,
            "status": "finished",
            "exit_code": 0,
            "signals": [],
            "teardown_verified": True,
            "failure": None,
            "failure_scope": None,
            "failure_category": None,
            "unit_properties": {"Result": "success", "ExecMainStatus": 0},
            "initial_memory_events": {"oom": 0, "oom_kill": 0, "oom_group_kill": 0},
            "final_memory_events": {"oom": 0, "oom_kill": 0, "oom_group_kill": 0},
            "initial_parent_memory_events": {"oom": 2, "oom_kill": 0, "oom_group_kill": 0},
            "final_parent_memory_events": {"oom": 2, "oom_kill": 0, "oom_group_kill": 0},
            "artifacts": artifacts,
        }
        attempt_path = root / "attempt.json"
        attempt_path.write_text(json.dumps(record))
        with (
            patch.object(scorer, "ROOT", root),
            patch.object(scorer, "RUNTIME_FILES", runtime_files),
            patch.object(scorer, "runtime_identity", lambda _root: runtime),
        ):
            receipt = scorer.validate_attempt_record(attempt_path, spec, run)
            assert receipt["worker_oom_counters_unchanged"]
            assert receipt["parent_oom_counters_unchanged"]
            for change in (
                {"status": "failed", "exit_code": 1},
                {"teardown_verified": False},
                {"signals": [15]},
            ):
                invalid = {**record, **change}
                attempt_path.write_text(json.dumps(invalid))
                try:
                    scorer.validate_attempt_record(attempt_path, spec, run)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"invalid synthetic attempt accepted: {change}")
            invalid_oom = json.loads(json.dumps(record))
            invalid_oom["final_parent_memory_events"]["oom_group_kill"] += 1
            attempt_path.write_text(json.dumps(invalid_oom))
            try:
                scorer.validate_attempt_record(attempt_path, spec, run)
            except ValueError:
                pass
            else:
                raise AssertionError("parent cgroup OOM increase was accepted")


def test_twenty_pairs_two_interactions_and_shared_component_draws() -> None:
    """Recompute synthetic known curves with common component multiplicities."""
    previous = scorer.BOOTSTRAP_COUNT
    scorer.BOOTSTRAP_COUNT = 29
    try:
        rng = np.random.default_rng(480_829)
        n_components, n_rows = 16, 40
        targets = rng.normal(0, 0.7, size=(n_components, n_rows, 2)).astype(np.float32)
        mask = rng.random((n_components, n_rows)) > 0.2
        mask[:, 0] = True
        component_ids = np.asarray([f"c{index:02}" for index in range(n_components)])
        multiplicities = scorer.bootstrap_multiplicities(component_ids)
        assert multiplicities.shape == (scorer.BOOTSTRAP_COUNT, n_components)
        assert np.all(multiplicities.sum(axis=1) == n_components)

        wps: dict[str, float] = {}
        arm_draws: dict[str, np.ndarray] = {}
        for arm_index, (activation, optimizer, schedule) in enumerate(
            (a, o, s)
            for a in scorer.ACTIVATIONS
            for o in scorer.OPTIMIZERS
            for s in scorer.SCHEDULES
        ):
            name = f"{activation}__{optimizer}__{schedule}"
            act_shift = 0.09 if activation == "sine_embed" else 0.0
            opt_shift = 0.03 if optimizer == "muon_adamw" else 0.0
            schedule_shift = {
                "constant_peak": 0.0,
                "warmcos_47_427": 0.08,
                "constant_mean_warmcos": -0.02,
            }[schedule]
            prediction = (0.72 + act_shift + opt_shift + schedule_shift) * targets
            prediction = prediction + (0.2 + 0.001 * arm_index) * rng.normal(
                size=targets.shape
            ).astype(np.float32)
            wps[name] = float(global_wp(targets, prediction, mask)["wp"])
            moments, _ = paired_component_moments(targets, prediction, prediction, mask)
            arm_draws[name] = np.asarray(
                [scorer.pool_component_moments(moments, counts) for counts in multiplicities]
            )
            np.testing.assert_allclose(
                scorer.pool_component_moments(moments, np.ones(n_components, dtype=np.int64)),
                wps[name],
                atol=1e-7,
                rtol=0,
            )
            if arm_index == 0:
                for draw in range(5):
                    row_ids = np.repeat(np.arange(n_components), multiplicities[draw])
                    direct = float(
                        global_wp(
                            targets[row_ids],
                            prediction[row_ids],
                            mask[row_ids],
                        )["wp"]
                    )
                    np.testing.assert_allclose(
                        scorer.pool_component_moments(moments, multiplicities[draw]),
                        direct,
                        atol=1e-7,
                        rtol=0,
                    )

        pairs, interactions, primary = scorer.compute_seed_contrasts(
            271829, wps, arm_draws, n_components
        )
        assert len(scorer.PAIR_CONTRASTS) == len(pairs) == 20
        assert len(interactions) == 2
        assert len(primary) == 4
        for candidate, reference, _family in scorer.PAIR_CONTRASTS:
            assert np.isclose(
                pairs[f"{candidate}_minus_{reference}"]["mean_diff"],
                wps[candidate] - wps[reference],
                atol=1e-12,
                rtol=0,
            )
        for optimizer in scorer.OPTIMIZERS:
            keys = (
                f"sine_embed__{optimizer}__warmcos_47_427",
                f"base__{optimizer}__warmcos_47_427",
                f"sine_embed__{optimizer}__constant_mean_warmcos",
                f"base__{optimizer}__constant_mean_warmcos",
            )
            expected = (wps[keys[0]] - wps[keys[1]]) - (wps[keys[2]] - wps[keys[3]])
            assert np.isclose(
                interactions[f"sine_activation_x_warmcos_shape__{optimizer}"]["mean_diff"],
                expected,
                atol=1e-12,
                rtol=0,
            )
        identical = dict(arm_draws)
        identical_wps = dict(wps)
        identical["sine_embed__adamw__warmcos_47_427"] = identical["base__adamw__warmcos_47_427"]
        identical_wps["sine_embed__adamw__warmcos_47_427"] = identical_wps[
            "base__adamw__warmcos_47_427"
        ]
        zero_draws = (
            identical["sine_embed__adamw__warmcos_47_427"]
            - identical["base__adamw__warmcos_47_427"]
        )
        assert np.array_equal(zero_draws, np.zeros_like(zero_draws))
        zero_pairs, _, _ = scorer.compute_seed_contrasts(
            271829, identical_wps, identical, n_components
        )
        zero_effect = zero_pairs[
            "sine_embed__adamw__warmcos_47_427_minus_base__adamw__warmcos_47_427"
        ]
        assert zero_effect["mean_diff"] == 0.0 and zero_effect["ci95"] == [0.0, 0.0]
        assert np.isfinite(np.concatenate(list(arm_draws.values()))).all()
    finally:
        scorer.BOOTSTRAP_COUNT = previous


def main() -> None:
    """Run bounded source-only scorer checks."""
    torch.set_num_threads(1)
    for test in (
        test_lr_trace_source_identity,
        test_independent_model_formula_and_identity_buffers,
        test_external_fit_normalizer_is_shared_and_test_independent,
        test_raw_label_finite_gate_precedes_clipping,
        test_synthetic_runner_receipt_is_fail_closed,
        test_twenty_pairs_two_interactions_and_shared_component_draws,
    ):
        test()
        print(f"PASS {test.__name__}", flush=True)


if __name__ == "__main__":
    main()
