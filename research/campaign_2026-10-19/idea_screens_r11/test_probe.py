"""Small synthetic checks for R11 math, fixed budgets, and provenance gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from idea_screens_r2.views import build_view_for
from idea_screens_r7.model_math import apply_normalizer, predict_rows, state_fingerprint
from idea_screens_r7.panel import score_pair

from wnn.gmetric import GlobalAccumulator

from .importance_math import (
    analytic_joint_gradient_norms,
    clipped_weight_table,
    correction_factors,
    importance_probabilities,
    sample_indices,
    uncentered_gradient_second_moment,
)
from .importance_training import EXPECTED_UPDATES, initialize_pair
from .probe_execution import scoped_cgroup_memory_path
from .result_payload import build_result, named_sampler_score
from .run_probe import (
    RUNTIME_IMPORT_PATHS,
    _is_exact_cli_main,
    smoke_imports,
    verify_runtime_imports,
)


def test_fixed_mu_and_analytic_joint_norm_match_fp32_autograd_with_tolerance() -> None:
    """The FP64 closed form agrees with the joint two-target FP32 gradient."""
    uniform, _, _ = initialize_pair()
    rng = np.random.default_rng(41)
    features = rng.normal(size=(3, 224)).astype(np.float32)
    targets = rng.normal(size=(3, 2)).astype(np.float32)
    clipped, weights, mu_w = clipped_weight_table(targets)
    norms = analytic_joint_gradient_norms(features, clipped, weights, mu_w, uniform)
    actual: list[float] = []
    parameters = [
        uniform.trunk.weight,
        uniform.trunk.bias,
        uniform.mean_head.weight,
        uniform.mean_head.bias,
    ]
    x_tensor = torch.from_numpy(features)
    y_tensor = torch.from_numpy(clipped)
    w_tensor = torch.from_numpy(weights)
    for index in range(len(features)):
        prediction = uniform.forward_mean(x_tensor[index : index + 1])
        row_loss = (
            w_tensor[index : index + 1] * (prediction - y_tensor[index : index + 1]).square()
        ).sum()
        row_loss = row_loss / mu_w
        gradients = torch.autograd.grad(row_loss, parameters)
        norm = torch.sqrt(sum(gradient.double().square().sum() for gradient in gradients))
        actual.append(float(norm.item()))
    assert np.allclose(norms, actual, rtol=5e-5, atol=5e-6)
    assert mu_w == pytest.approx(float(weights.sum(dtype=np.float64) / len(weights)))


def test_importance_correction_has_the_fixed_global_gradient_expectation() -> None:
    """Exact categorical expectation of corrected per-row gradients is global."""
    uniform, _, _ = initialize_pair()
    rng = np.random.default_rng(42)
    features = torch.from_numpy(rng.normal(size=(4, 224)).astype(np.float32))
    targets_np = rng.normal(size=(4, 2)).astype(np.float32)
    clipped_np, weights_np, mu_w = clipped_weight_table(targets_np)
    targets = torch.from_numpy(clipped_np)
    weights = torch.from_numpy(weights_np)
    parameters = [
        uniform.trunk.weight,
        uniform.trunk.bias,
        uniform.mean_head.weight,
        uniform.mean_head.bias,
    ]
    row_gradients: list[torch.Tensor] = []
    row_losses: list[torch.Tensor] = []
    for index in range(len(features)):
        prediction = uniform.forward_mean(features[index : index + 1])
        loss = (
            weights[index : index + 1] * (prediction - targets[index : index + 1]).square()
        ).sum()
        loss = loss / mu_w
        gradients = torch.autograd.grad(loss, parameters, retain_graph=False)
        row_gradients.append(torch.cat([gradient.reshape(-1) for gradient in gradients]))
        row_losses.append(loss.detach())
    gradient_table = torch.stack(row_gradients)
    row_norms = gradient_table.double().norm(dim=1).numpy()
    probabilities = importance_probabilities(row_norms)
    corrections = correction_factors(probabilities)
    probability_tensor = torch.from_numpy(probabilities)
    correction_tensor = torch.from_numpy(corrections)
    expected_gradient = (
        gradient_table.double() * (probability_tensor * correction_tensor)[:, None]
    ).sum(dim=0)
    full_gradient = gradient_table.double().mean(dim=0)
    assert torch.allclose(expected_gradient, full_gradient, rtol=1e-12, atol=1e-12)
    expected_loss = np.dot(
        probabilities, np.asarray([float(loss) for loss in row_losses]) * corrections
    )
    assert expected_loss == pytest.approx(float(torch.stack(row_losses).mean()), rel=1e-6, abs=1e-6)
    assert float(corrections.max()) <= 2.0 + 1e-12


def test_probability_floor_inverse_cdf_and_second_moment_diagnostic() -> None:
    """The sampler preserves support, correction bounds, and deterministic U mapping."""
    norms = np.array([0.0, 2.0, 1.0, 10.0], dtype=np.float64)
    probabilities = importance_probabilities(norms)
    corrections = correction_factors(probabilities)
    uniforms = np.array([0.0, 0.1, 0.5, 0.999], dtype=np.float64)
    first = sample_indices(uniforms, probabilities)
    assert np.array_equal(first, sample_indices(uniforms, probabilities))
    assert np.array_equal(sample_indices(uniforms, np.full(4, 0.25)), [0, 0, 2, 3])
    assert np.all(probabilities >= 0.5 / len(norms))
    assert corrections.max() <= 2.0 + 1e-12
    assert uncentered_gradient_second_moment(norms, probabilities) > 0.0
    assert np.isclose(probabilities.sum(), 1.0)


def test_initialization_is_identical_and_unused_variance_head_stays_zero() -> None:
    """The paired fixed-seed models begin byte-identical with a zero variance head."""
    uniform, importance, fingerprints = initialize_pair()
    assert fingerprints["initial"] == fingerprints["uniform_initial"]
    assert fingerprints["initial"] == fingerprints["importance_initial"]
    assert state_fingerprint(uniform) == state_fingerprint(importance)
    for model in (uniform, importance):
        assert torch.count_nonzero(model.logvar_head.weight).item() == 0
        assert torch.count_nonzero(model.logvar_head.bias).item() == 0
        assert model.logvar_head.weight.grad is None
        assert model.logvar_head.bias.grad is None
    assert EXPECTED_UPDATES == 756


def test_reused_panel_scorer_is_pooled_and_arm_names_are_explicit() -> None:
    """R11 arm labels preserve canonical pooled scoring and component sensitivity."""
    uniform, importance, _ = initialize_pair()
    with torch.no_grad():
        importance.mean_head.bias.add_(torch.tensor([0.12, -0.08]))
    rng = np.random.default_rng(43)
    mean = np.zeros(224, dtype=np.float32)
    scale = np.ones(224, dtype=np.float32)
    evaluation = []
    for component in (11, 29):
        x = rng.normal(size=(20_000, 112)).astype(np.float32)
        y = rng.normal(size=(20_000, 2)).astype(np.float32)
        mask = np.zeros(20_000, dtype=bool)
        mask[[100, 101, 136, 190, 200]] = True
        evaluation.append(
            {
                "x": x,
                "y": y,
                "scored": mask,
                "component": component,
                "source": "synthetic",
                "group": str(component),
            }
        )
    named = named_sampler_score(score_pair(uniform, importance, evaluation, mean, scale))
    assert set(named["arms"]) == {"uniform", "importance"}
    assert "heteroskedastic" not in named["arms"]
    direct = {name: GlobalAccumulator() for name in ("uniform", "importance")}
    models = {"uniform": uniform, "importance": importance}
    for record in evaluation:
        selected_y = record["y"][record["scored"]]
        view = build_view_for(record["x"], "raw_signed_delta_ema_hl32")
        selected_x = apply_normalizer(view[record["scored"]], mean, scale)
        for name, model in models.items():
            prediction, _ = predict_rows(model, selected_x)
            direct[name].add(selected_y, prediction, np.ones(len(selected_y), dtype=bool))
    for name in direct:
        assert named["arms"][name]["official_metric"]["wp"] == pytest.approx(
            direct[name].result()["wp"], abs=1e-12
        )
    expected_delta = direct["importance"].result()["wp"] - direct["uniform"].result()["wp"]
    assert named["primary_delta_importance_minus_uniform_wp"] == pytest.approx(
        expected_delta, abs=1e-12
    )


def test_cgroup_guard_requires_exact_nonroot_r11_service() -> None:
    """A host-root or unrelated cgroup cannot pass the R11 scoped memory gate."""
    valid = "0::/wnn-campaign.slice/computer-vision-homework-eda19-r11-v2.service/job\n"
    path = scoped_cgroup_memory_path(valid)
    assert str(path).endswith(
        "/wnn-campaign.slice/computer-vision-homework-eda19-r11-v2.service/job/memory.current"
    )
    for invalid in (
        "0::/\n",
        "0::/other.slice/computer-vision-homework-eda19-r11-v1.service\n",
        "0::/wnn-campaign.slice/computer-vision-homework-eda19-r11-v1.service\n",
        "0::/wnn-campaign.slice/computer-vision-homework-eda19-r11-v1.service/computer-vision-homework-eda19-r11-v2.service\n",
        "0::/wnn-campaign.slice/other.service\n",
        "0::/wnn-campaign.slice/../other.service\n",
    ):
        with pytest.raises(RuntimeError):
            scoped_cgroup_memory_path(invalid)


def test_python_m_main_alias_is_exact_and_named_module_stays_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The -m CLI inventory accepts only its exact __main__ spec and pinned file."""
    root = Path(__file__).resolve().parents[1]
    source_root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "idea_screens_r11/PREPARATION_MANIFEST.json").read_bytes())
    pins = manifest["files"]

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an import-inventory check attempted to load data or a checkpoint")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    smoke_imports(root, source_root, pins)

    module_name = "idea_screens_r11.run_probe"
    expected = (root / RUNTIME_IMPORT_PATHS[module_name]).resolve()
    fake_main = ModuleType("__main__")
    fake_main.__spec__ = ModuleSpec(module_name, loader=None)
    fake_main.__file__ = str(expected)
    monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    assert _is_exact_cli_main(fake_main, expected)
    inventory = verify_runtime_imports(root, source_root, pins)
    assert inventory[module_name] == str(expected)
    assert module_name not in sys.modules

    wrong_main = ModuleType("__main__")
    wrong_main.__spec__ = ModuleSpec("idea_screens_r11.not_run_probe", loader=None)
    wrong_main.__file__ = str(expected)
    monkeypatch.setitem(sys.modules, "__main__", wrong_main)
    assert not _is_exact_cli_main(wrong_main, expected)
    with pytest.raises(RuntimeError, match="runtime source was not imported"):
        verify_runtime_imports(root, source_root, pins)


def test_actual_python_m_cli_entrypoint_validates_before_help() -> None:
    """The real package -m entrypoint checks its __main__ module identity first."""
    repo_root = Path(__file__).resolve().parents[3]
    campaign_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(repo_root / "src"), str(campaign_root)))
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "idea_screens_r11.run_probe", "--help"],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Fixed-step R11 CPU importance-sampling comparison" in completed.stdout


def test_r3_sample_fixture_and_synthetic_result_are_bound_and_json_safe() -> None:
    """The accepted metadata fixture and compact report schema retain fixed scope."""
    fixture = (
        Path(__file__).resolve().parents[1]
        / "idea_screens_r3/records/sample_v2/terminal_capture/REMOTE_SAMPLE.json"
    )
    metadata = json.loads(fixture.read_bytes())
    assert metadata["schema"] == "idea_screens_r3.sample.v2"
    records = metadata["records"]
    assert sum(record["role"] == "fit" for record in records) == 64
    assert sum(record["role"] == "eval" for record in records) == 32

    norms = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    probabilities = importance_probabilities(norms)
    corrections = correction_factors(probabilities)
    arm = {
        "arm": "uniform",
        "successful_updates": EXPECTED_UPDATES,
        "attempted_updates": EXPECTED_UPDATES,
        "nonfinite_count": 0,
        "rounds": [],
    }
    score = {
        "metric_id": "global_wp_clip2_v1",
        "components": 32,
        "rows": 59_436,
        "component_sensitivity_is_not_independent_replication": True,
        "arms": {"uniform": {}, "importance": {}},
        "primary_delta_importance_minus_uniform_wp": 0.0,
    }
    result = build_result(
        identities={"sample_sha256": "sample", "membership_sha256": "membership"},
        preparation_sha256="preparation",
        mu_w=1.25,
        initialization={"initial": "a" * 64},
        norms=norms,
        probabilities=probabilities,
        corrections=corrections,
        uniform_second_moment=2.0,
        importance_second_moment=1.5,
        uniform_fit=arm,
        importance_fit={**arm, "arm": "importance"},
        score=score,
        archive={"sha256": "archive", "bytes": 1},
        runtime={"device": "cpu"},
        phase_seconds={"prepass": 0.1},
        started_wall=time.perf_counter() - 0.1,
        started_cpu=time.process_time() - 0.1,
    )
    encoded = json.dumps(result, allow_nan=False)
    assert "12 epochs" not in encoded
    assert "not variance or training-wide variance reduction" in encoded
    assert "predictions_or_model_parameter_arrays_in_json" in result
