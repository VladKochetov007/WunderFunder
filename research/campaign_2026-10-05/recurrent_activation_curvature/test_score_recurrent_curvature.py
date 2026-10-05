"""Synthetic CPU checks for the independent E481 scorer; no test data access."""

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

import score_recurrent_curvature as scorer  # noqa: E402

from wnn.campaign_spec import digest as spec_digest  # noqa: E402


def test_source_and_preflight_receipts() -> None:
    """Check actor source hashes and fit-only preflight seal locally."""
    for name, relative in scorer.SOURCE_PATHS.items():
        assert scorer.file_sha256(ROOT / relative) == scorer.EXPECTED_SOURCE_HASHES[name]
    preflight = json.loads((HERE / "preflight_cpu_v2/fit_manifest.json").read_text())
    assert preflight["status"] == "cpu_fit_only_one_update_per_arm_passed_test_unread"
    assert preflight["test_rows_loaded"] is False
    assert len(preflight["arms"]) == 18
    assert (
        scorer.file_sha256(HERE / "preflight_cpu_v2/fit_manifest.json") == scorer.PREFLIGHT_SHA256
    )


def test_independent_model_formulas_chunk_carry_and_causality() -> None:
    """Check formula maps, checkpoint layout, streaming equivalence, and prefix invariance."""
    u = torch.linspace(-1.0, 1.0, 101)
    baseline = scorer.transform_candidate(u, 0)
    sine = scorer.transform_candidate(u, 1)
    cubic = scorer.transform_candidate(u, 2)
    torch.testing.assert_close(baseline, u, atol=0, rtol=0)
    torch.testing.assert_close(sine, u + 0.1 * torch.sin(torch.pi * u), atol=0, rtol=0)
    torch.testing.assert_close(cubic, u + 0.1 * torch.pi * u * (1 - u.square()), atol=0, rtol=0)
    assert float(sine.abs().max()) <= 1.0 + 1e-6
    assert float(cubic.abs().max()) <= 1.0 + 1e-6
    assert float(1.0 - 0.1 * torch.pi) > 0 and float(1.0 - 0.2 * torch.pi) > 0

    generator = torch.Generator().manual_seed(481)
    x = torch.randn((1, 12, scorer.FEATURES), generator=generator)
    future_changed = x.clone()
    future_changed[:, 7:] += torch.randn((1, 5, scorer.FEATURES), generator=generator)
    for shape in scorer.SHAPES:
        model = scorer.CurvatureModel(shape).eval()
        assert tuple(model.state_dict()) == tuple(scorer.CurvatureModel("baseline").state_dict())
        with torch.no_grad():
            whole, _ = model(x)
            first, state = model(x[:, :5])
            second, _ = model(x[:, 5:], state)
            streamed = torch.cat((first, second), dim=1)
            changed, _ = model(future_changed)
        torch.testing.assert_close(whole, streamed, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(whole[:, :7], changed[:, :7], atol=1e-6, rtol=1e-6)


def test_normalizer_receipt_consistency_is_fail_closed() -> None:
    """Use synthetic normalized inputs to test common fit-derived transform checks."""
    mean = np.arange(42, dtype=np.float32)
    scale = np.full(42, 2.0, dtype=np.float32)
    fit_manifest = {"normalization": {"fit_mean": mean.tolist(), "fit_scale": scale.tolist()}}
    all_manifest = {"normalization": dict(fit_manifest["normalization"])}
    fit_nodes = [{"row_group": i, "role": "fit"} for i in range(24)]
    all_nodes = [*fit_nodes, {"row_group": 100, "role": "test"}]
    fit_x = np.zeros((24, 2, 42), dtype=np.float32)
    all_x = np.concatenate((fit_x, np.ones((1, 2, 42), dtype=np.float32)), axis=0)
    with patch.object(scorer, "derive_fit_normalizer", lambda _nodes: (mean, scale)):
        found_mean, found_scale = scorer.validate_normalizer(
            fit_manifest, fit_x, fit_nodes, all_manifest, all_x, all_nodes
        )
        np.testing.assert_array_equal(found_mean, mean)
        np.testing.assert_array_equal(found_scale, scale)
        malformed = {
            "normalization": {"fit_mean": (mean + 1).tolist(), "fit_scale": scale.tolist()}
        }
        try:
            scorer.validate_normalizer(fit_manifest, fit_x, fit_nodes, malformed, all_x, all_nodes)
        except ValueError:
            pass
        else:
            raise AssertionError("fit/test normalizer mismatch was accepted")


def test_raw_nonfinite_labels_rejected_before_clipping() -> None:
    """Reject nonfinite raw labels even when clipping would hide infinity."""
    raw = np.asarray([[0.1, -0.3], [2.5, -4.0]], dtype=np.float64)
    mask = np.asarray([True, True])
    clipped = np.clip(raw, -2, 2).astype(np.float32)
    np.testing.assert_array_equal(scorer.validate_raw_targets(raw, mask, clipped, 91), raw)
    bad = raw.copy()
    bad[0, 0] = np.inf
    try:
        scorer.validate_raw_targets(bad, mask, clipped, 91)
    except ValueError:
        pass
    else:
        raise AssertionError("infinite raw label was clipped without rejection")


def test_runner_receipt_requires_success_artifact_and_no_oom() -> None:
    """Exercise good proof and representative failed/lost/OOM synthetic receipts."""
    with tempfile.TemporaryDirectory(prefix="e481-attempt-proof-") as directory:
        root = Path(directory)
        relative = "research/campaign_2026-10-05/recurrent_activation_curvature/runs/eighteen_v1/fit_manifest.json"
        artifact = root / relative
        artifact.parent.mkdir(parents=True)
        artifact.write_text("synthetic result\n")
        run = artifact.parent
        checked_spec = {
            "name": scorer.EXPECTED_SPEC_NAME,
            "kind": "gpu",
            "argv": [
                "{python}",
                scorer.SOURCE_PATHS["fit_recurrent_curvature.py"],
                "--device",
                "cuda",
                "--output",
                "research/campaign_2026-10-05/recurrent_activation_curvature/runs/eighteen_v1",
            ],
            "artifacts": [relative],
        }
        runtime = {"runtime.py": "abc123"}
        record = {
            "id": scorer.EXPECTED_ATTEMPT_ID,
            "kind": "gpu",
            "spec": checked_spec,
            "spec_sha256": spec_digest(checked_spec),
            "argv": ["/usr/bin/python", *checked_spec["argv"][1:]],
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
            "artifacts": [
                {
                    "path": relative,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "size": artifact.stat().st_size,
                }
            ],
        }
        attempt_path = root / "attempt.json"
        attempt_path.write_text(json.dumps(record))
        with (
            patch.object(scorer, "ROOT", root),
            patch.object(scorer, "RUNTIME_FILES", ("runtime.py",)),
            patch.object(scorer, "runtime_identity", lambda _root: runtime),
            patch.object(scorer, "validate_spec_identity", lambda _spec: checked_spec),
        ):
            proof = scorer.validate_attempt(attempt_path, checked_spec, run)
            assert proof["worker_oom_counters_unchanged"]
            assert proof["parent_oom_counters_unchanged"]
            for change in (
                {"status": "failed", "exit_code": 1},
                {"signals": [15]},
                {"teardown_verified": False},
            ):
                attempt_path.write_text(json.dumps({**record, **change}))
                try:
                    scorer.validate_attempt(attempt_path, checked_spec, run)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"invalid runner receipt accepted: {change}")
            oom = json.loads(json.dumps(record))
            oom["final_parent_memory_events"]["oom_group_kill"] += 1
            attempt_path.write_text(json.dumps(oom))
            try:
                scorer.validate_attempt(attempt_path, checked_spec, run)
            except ValueError:
                pass
            else:
                raise AssertionError("parent OOM increase was accepted")


def test_registered_30_contrasts_and_joint_interaction() -> None:
    """Check exact effect families and shared numeric draw arithmetic."""
    rng = np.random.default_rng(481_829)
    scores = {}
    draws = {}
    for shape_i, shape in enumerate(scorer.SHAPES):
        for optimizer_i, optimizer in enumerate(scorer.OPTIMIZERS):
            key = f"{optimizer}__{shape}"
            scores[key] = 0.1 * shape_i + 0.03 * optimizer_i
            draws[key] = scores[key] + rng.normal(0, 0.002, scorer.BOOTSTRAP_COUNT)
    got = scorer.registered_contrasts(scores, draws)
    assert len(got) == 10
    assert sum(v["family"] == "shape_within_optimizer" for v in got.values()) == 6
    assert sum(v["family"] == "optimizer_within_shape" for v in got.values()) == 3
    interaction = got["curvature_cubic_minus_sine_x_muon_minus_adamw"]
    expected = (scores["muon_adamw__cubic"] - scores["muon_adamw__sine"]) - (
        scores["adamw__cubic"] - scores["adamw__sine"]
    )
    assert np.isclose(interaction["mean_diff"], expected)
    identical_draws = dict(draws)
    identical_scores = dict(scores)
    identical_draws["muon_adamw__cubic"] = identical_draws["muon_adamw__sine"]
    identical_scores["muon_adamw__cubic"] = identical_scores["muon_adamw__sine"]
    zero = scorer.registered_contrasts(identical_scores, identical_draws)
    assert np.isfinite([value["mean_diff"] for value in zero.values()]).all()
    assert np.isfinite(np.concatenate(list(draws.values()))).all()


def main() -> None:
    """Run bounded synthetic scorer tests only."""
    torch.set_num_threads(1)
    tests = (
        test_source_and_preflight_receipts,
        test_independent_model_formulas_chunk_carry_and_causality,
        test_normalizer_receipt_consistency_is_fail_closed,
        test_raw_nonfinite_labels_rejected_before_clipping,
        test_runner_receipt_requires_success_artifact_and_no_oom,
        test_registered_30_contrasts_and_joint_interaction,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}", flush=True)


if __name__ == "__main__":
    main()
