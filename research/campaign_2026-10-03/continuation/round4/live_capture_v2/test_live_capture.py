from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from live_capture import (
    Capture,
    cloned,
    compare,
    compiled_chunk,
    replay,
    restore,
    trainer_module,
    validate,
)
from runner import capture_step, synthetic_environment


@pytest.fixture
def captured() -> Capture:
    torch.manual_seed(734019)
    return capture_step(synthetic_environment())


def test_default_actual_replay(captured: Capture) -> None:
    actual = captured.post
    replayed = replay(captured.pre, trainer_module())
    deltas = compare(actual, replayed)
    assert max(deltas.values()) == 0
    parts = replay(captured.pre, trainer_module(), True)
    assert set(parts["component_gradients"]) == {"main", "ema", "struct", "next"}
    compare(parts["losses"], actual["losses"])
    for name, total in actual["preclip"].items():
        combined = sum(
            group[name]
            for group in parts["component_gradients"].values()
            if group[name] is not None
        )
        torch.testing.assert_close(combined, total, atol=1e-7, rtol=1e-6)


def test_disabled_hooks_match_original_statements(captured: Capture) -> None:
    module = trainer_module()
    original = restore(captured.pre, module)
    exec(compiled_chunk(cpu=True, instrumented=False), original)
    first = {key: cloned(original[key].state_dict()) for key in ("model", "opt", "sched", "scaler")}
    rng = torch.get_rng_state().clone()
    instrumented = restore(captured.pre, module)
    exec(compiled_chunk(cpu=True), instrumented)
    second = {
        key: cloned(instrumented[key].state_dict()) for key in ("model", "opt", "sched", "scaler")
    }
    compare(first, second, 0, 0)
    compare(
        original["ema_model"].module.state_dict(),
        instrumented["ema_model"].module.state_dict(),
        0,
        0,
    )
    compare(original["h"], instrumented["h"], 0, 0)
    compare(original["h_ema"], instrumented["h_ema"], 0, 0)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "field", ["h", "h_ema", "optimizer", "scaler", "scheduler", "rng", "batch"]
)
def test_absent_state_rejected(captured: Capture, field: str) -> None:
    invalid = copy.deepcopy(captured.pre)
    del invalid[field]
    with pytest.raises(ValueError, match="schema"):
        validate(invalid)


def test_source_drift_rejected(captured: Capture) -> None:
    invalid = copy.deepcopy(captured.pre)
    invalid["source"]["scripts/train_rnn.py"] = "0" * 64
    with pytest.raises(ValueError, match="source"):
        validate(invalid)


def test_group_order_rejected(captured: Capture) -> None:
    invalid = copy.deepcopy(captured.pre)
    invalid["optimizer_names"][0].reverse()
    with pytest.raises(ValueError, match="order|shape|association"):
        replay(invalid, trainer_module())


def test_missing_moment_rejected(captured: Capture) -> None:
    invalid = copy.deepcopy(captured.pre)
    invalid["optimizer"]["state"].pop(next(iter(invalid["optimizer"]["state"])))
    with pytest.raises(ValueError, match="state missing"):
        validate(invalid)


def test_hidden_and_moments_are_material(captured: Capture) -> None:
    module = trainer_module()
    hidden = copy.deepcopy(captured.pre)
    hidden["h_ema"].zero_()
    with pytest.raises(AssertionError):
        compare(captured.post, replay(hidden, module))
    moments = copy.deepcopy(captured.pre)
    for state in moments["optimizer"]["state"].values():
        for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
            if key in state:
                state[key].zero_()
    changed = replay(moments, module)
    compare(changed["preclip"], captured.post["preclip"])
    with pytest.raises(AssertionError):
        compare(changed["student"], captured.post["student"])


def test_cpu_amp_skip_protocol() -> None:
    torch.manual_seed(734019)
    actual = capture_step(synthetic_environment(scaler_enabled=True), force_skip=True)
    compare(actual.post, replay(actual.pre, trainer_module()))
    compare(actual.pre["student"], actual.post["student"], 0, 0)
    assert int(actual.post["clock"]["skips"]) == 1
    assert actual.post["clock"]["attempts"] == 5
    assert actual.post["clock"]["ema_updates"] == 5
    assert actual.post["scheduler"]["last_epoch"] == 5
    assert actual.post["scaler"]["scale"] == actual.pre["scaler"]["scale"] / 2
    assert actual.post["scaler"]["_growth_tracker"] == 0


def test_partial_capture_survives_failure(tmp_path: Path) -> None:
    directory = tmp_path / "failed"
    with pytest.raises(ValueError, match="enabled scaler"):
        capture_step(synthetic_environment(), directory, True)
    assert (directory / "pre.pt").exists()
    assert not (directory / "post.pt").exists()


@pytest.mark.parametrize(
    "key,value",
    [("arch", "lstm"), ("ema_metric_weight", True), ("td_alpha", 0.1), ("aux_next_k", "2")],
)
def test_unsupported_recipe_rejected(captured: Capture, key: str, value: object) -> None:
    invalid = copy.deepcopy(captured.pre)
    invalid["args"][key] = value
    with pytest.raises(ValueError, match="unsupported"):
        validate(invalid)


def test_unreviewed_gpu_candidate_rejected(tmp_path: Path) -> None:
    import json

    from runner import verify_review

    with pytest.raises(ValueError, match="receipt"):
        verify_review(None)
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps({"status": "pending", "queue_finished": False}))
    with pytest.raises(ValueError, match="review"):
        verify_review(pending)


def test_panel_is_fit_only_and_preregistered() -> None:
    import json

    from fit_panel import PANEL
    from live_capture import ROOT

    panel = json.loads(PANEL.read_text())
    provenance = json.loads(
        (ROOT / "artifacts/runs/c20261003_combo_s3_f0/provenance.json").read_text()
    )
    manifest = json.loads(Path(panel["manifest_path"]).read_text())
    assert len(panel["rows"]) == 32
    assert sum(row["split"] == "train" for row in panel["rows"]) == 28
    assert panel["scored"] is False
    for row in panel["rows"]:
        assert row["row_group"] in provenance["fit_row_groups"][row["split"]]
        if row["split"] == "valid":
            assert row["row_group"] not in manifest["evaluation"]["valid"]


@pytest.mark.parametrize("field", ["h", "h_ema"])
def test_empty_hidden_rejected(captured: Capture, field: str) -> None:
    invalid = copy.deepcopy(captured.pre)
    invalid[field] = None
    with pytest.raises(ValueError, match="hidden"):
        validate(invalid)


def test_partial_rng_rejected(captured: Capture) -> None:
    invalid = copy.deepcopy(captured.pre)
    del invalid["rng"]["numpy"]
    with pytest.raises(ValueError, match="RNG"):
        validate(invalid)


def test_implicit_precision_cast_rejected(captured: Capture) -> None:
    invalid = copy.deepcopy(captured.pre)
    invalid["student"]["inp.weight"] = invalid["student"]["inp.weight"].half()
    with pytest.raises(ValueError, match="fp32"):
        validate(invalid)


@pytest.mark.parametrize("claimed_map_too", [False, True])
def test_same_shape_serialized_optimizer_ids_rejected(
    captured: Capture, claimed_map_too: bool
) -> None:
    invalid = copy.deepcopy(captured.pre)
    ids = invalid["optimizer"]["param_groups"][1]["params"]
    first, second = ids[:2]
    ids[0], ids[1] = second, first
    if claimed_map_too:
        mapping = invalid["optimizer_id_names"]
        mapping[first], mapping[second] = mapping[second], mapping[first]
    with pytest.raises(ValueError, match="serialized|association"):
        validate(invalid)
    with pytest.raises(ValueError, match="serialized|association"):
        replay(invalid, trainer_module())


def test_large_integer_clock_never_inherits_float_tolerance() -> None:
    with pytest.raises(AssertionError, match="discrete"):
        compare(torch.tensor(100000), torch.tensor(100001), 1e-5, 1e-4)
    with pytest.raises(AssertionError):
        compare(torch.tensor(True), torch.tensor(False), 1e-5, 1e-4)
    with pytest.raises(AssertionError):
        compare(1, True)


def test_unmeasured_float_envelope_cannot_admit_delta(captured: Capture) -> None:
    changed = copy.deepcopy(captured.post)
    next(iter(changed["parameter_delta"].values())).flatten()[0].add_(5e-6)
    with pytest.raises(AssertionError, match="bytes"):
        compare(captured.post, changed)
    with pytest.raises(AssertionError, match="uncalibrated"):
        compare(captured.post, changed, 1e-5, 1e-4)


def test_predictors_target_and_actual_clip_norm_persist(captured: Capture) -> None:
    assert set(("out", "out_ema", "target", "clip_norm")) <= set(captured.post)
    assert captured.post["out"].shape == (2, 10, 23)
    assert captured.post["target"].shape == (2, 10, 2)
    assert captured.post["clip_norm"].ndim == 0
    assert not captured.post["out_ema"].requires_grad
    compare(captured.post, replay(captured.pre, trainer_module()))


@pytest.mark.parametrize("field", ["ema_decay", "h", "h_ema"])
def test_decay_and_hidden_dtype_no_implicit_restoration(captured: Capture, field: str) -> None:
    invalid = copy.deepcopy(captured.pre)
    if field == "ema_decay":
        invalid[field] = 0.998
    else:
        invalid[field] = invalid[field].half()
    with pytest.raises(ValueError, match="decay|dtype"):
        validate(invalid)


def test_floor_policy_is_independent_and_fixed() -> None:
    from baselines import compare_policy, derive_policy

    samples = [{"delta": torch.tensor([0.0]), "clock": torch.tensor(100000)} for _ in range(6)]
    policy = derive_policy(samples, "cpu")
    assert policy["limits"]["root.delta"] == 0
    with pytest.raises(AssertionError):
        compare_policy(
            samples[0], {"delta": torch.tensor([5e-6]), "clock": torch.tensor(100000)}, policy
        )
    samples[5]["delta"] += 1e-6
    noisy = derive_policy(samples, "cuda")  # policy arithmetic only, NO CUDA execution
    with pytest.raises(AssertionError, match="discrete"):
        compare_policy(
            samples[0], {"delta": torch.tensor([0.0]), "clock": torch.tensor(100001)}, noisy
        )
    with pytest.raises(AssertionError, match="no relaxation"):
        compare_policy(
            samples[0], {"delta": torch.tensor([3e-6]), "clock": torch.tensor(100000)}, noisy
        )
    samples[5]["delta"] += 1e-3
    with pytest.raises(AssertionError, match="ceiling"):
        derive_policy(samples, "cuda")


def test_nonfinite_patterns_are_exact() -> None:
    from baselines import compare_policy, derive_policy

    samples = [{"gradient": torch.tensor([float("nan"), float("inf"), 0.0])} for _ in range(6)]
    policy = derive_policy(samples, "cpu")
    with pytest.raises(AssertionError, match="nonfinite"):
        compare_policy(samples[0], {"gradient": torch.tensor([0.0, float("inf"), 0.0])}, policy)


def test_independent_baseline_matches_original_observed_step(captured: Capture) -> None:
    from baselines import original_default

    compare(captured.post, original_default(captured.pre))


def test_serialized_tensor_clock_dtype_cannot_be_cast(captured: Capture) -> None:
    invalid = copy.deepcopy(captured.pre)
    invalid["clock"]["skips"] = invalid["clock"]["skips"].float()
    with pytest.raises(ValueError, match="clock types"):
        replay(invalid, trainer_module())
