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


@pytest.mark.parametrize("dtype", ["uint64", "float64", ">u4"])
def test_real_mt19937_numeric_dtype_alias_rejected_before_restore(
    captured: Capture, dtype: str
) -> None:
    import numpy as np
    from live_capture import get_rng, restore

    invalid = copy.deepcopy(captured.pre)
    state = list(invalid["rng"]["numpy"])
    state[1] = state[1].astype(np.dtype(dtype))
    invalid["rng"]["numpy"] = tuple(state)
    before = get_rng("cpu")
    with pytest.raises(ValueError, match="NumPy"):
        validate(invalid)
    with pytest.raises(ValueError, match="NumPy"):
        restore(invalid, trainer_module())
    compare(before, get_rng("cpu"))
    with pytest.raises(AssertionError, match="NumPy"):
        compare(captured.pre["rng"], invalid["rng"])


def test_numpy_exact_class_shape_bytes_and_nan_payload() -> None:
    import numpy as np

    pairs = [
        (np.array([True]), np.array([1], dtype=np.uint32)),
        (np.array([0.0], dtype=np.float32), np.array([-0.0], dtype=np.float32)),
        (np.array([1, 2], dtype=np.uint32), np.array([[1, 2]], dtype=np.uint32)),
        (np.array([1], dtype=np.uint32), [1]),
        (np.array([1], dtype=np.uint32), np.array([1], dtype=np.uint32).view(np.recarray)),
        (
            np.array([0x7FC00001], dtype=np.uint32).view(np.float32),
            np.array([0x7FC00002], dtype=np.uint32).view(np.float32),
        ),
    ]
    for left, right in pairs:
        with pytest.raises(AssertionError):
            compare(left, right)
    same_nan = np.array([0x7FC00001], dtype=np.uint32).view(np.float32)
    compare(same_nan, same_nan.copy())
    with pytest.raises(AssertionError, match="float bytes"):
        compare({"cached": 0.0}, {"cached": -0.0})


@pytest.mark.parametrize(
    "change",
    [
        "outer_list",
        "shape",
        "array_subclass",
        "position_bool",
        "position_float",
        "flag_bool",
        "cached_np_scalar",
        "algorithm",
        "scalar_tuple",
    ],
)
def test_numpy_rng_scalar_tuple_grammar_rejected(captured: Capture, change: str) -> None:
    import numpy as np
    from live_capture import get_rng, set_rng

    invalid = copy.deepcopy(captured.pre["rng"])
    state = list(invalid["numpy"])
    if change == "outer_list":
        invalid["numpy"] = state
    elif change == "scalar_tuple":
        invalid["numpy"] = ("MT19937", 1, 0, 0, 0.0)
    else:
        if change == "shape":
            state[1] = state[1].reshape(312, 2)
        elif change == "array_subclass":
            state[1] = state[1].view(np.recarray)
        elif change == "position_bool":
            state[2] = True
        elif change == "position_float":
            state[2] = float(state[2])
        elif change == "flag_bool":
            state[3] = False
        elif change == "cached_np_scalar":
            state[4] = np.float64(state[4])
        elif change == "algorithm":
            state[0] = "PCG64"
        invalid["numpy"] = tuple(state)
    before = get_rng("cpu")
    with pytest.raises(ValueError, match="NumPy"):
        set_rng(invalid, "cpu")
    compare(before, get_rng("cpu"))


@pytest.mark.parametrize(
    "change",
    [
        "python_outer",
        "python_bool",
        "python_words",
        "torch_dtype",
        "torch_shape",
        "torch_list",
        "cuda_tuple",
        "cuda_nonempty",
    ],
)
def test_all_rng_groups_validated_before_global_restoration(captured: Capture, change: str) -> None:
    from live_capture import get_rng, set_rng

    invalid = copy.deepcopy(captured.pre["rng"])
    if change == "python_outer":
        invalid["python"] = list(invalid["python"])
    elif change == "python_bool":
        invalid["python"] = (True, *invalid["python"][1:])
    elif change == "python_words":
        invalid["python"] = (invalid["python"][0], list(invalid["python"][1]), invalid["python"][2])
    elif change == "torch_dtype":
        invalid["torch_cpu"] = invalid["torch_cpu"].long()
    elif change == "torch_shape":
        invalid["torch_cpu"] = invalid["torch_cpu"][None]
    elif change == "torch_list":
        invalid["torch_cpu"] = invalid["torch_cpu"].tolist()
    elif change == "cuda_tuple":
        invalid["cuda_all"] = ()
    else:
        invalid["cuda_all"] = [torch.zeros(1, dtype=torch.uint8)]
    before = get_rng("cpu")
    with pytest.raises(ValueError, match="RNG|rng"):
        set_rng(invalid, "cpu")
    compare(before, get_rng("cpu"))


def test_legitimate_seed_independent_gaussian_rng_roundtrip(captured: Capture) -> None:
    import random

    import numpy as np
    from live_capture import get_rng, set_rng, validate_rng

    original = get_rng("cpu")
    state = copy.deepcopy(captured.pre["rng"])
    numpy_engine = np.random.RandomState(8675309)  # noqa: NPY002 - exercise actual legacy engine
    numpy_engine.normal(size=1)
    state["numpy"] = numpy_engine.get_state()
    python_engine = random.Random(310)
    python_engine.gauss(0, 1)
    state["python"] = python_engine.getstate()
    state["torch_cpu"] = torch.Generator().manual_seed(771).get_state()
    try:
        validate_rng(state, "cpu")
        set_rng(state, "cpu")
        compare(state, get_rng("cpu"))
    finally:
        set_rng(original, "cpu")


@pytest.mark.parametrize("field", ["momentum_buffer", "exp_avg", "exp_avg_sq"])
def test_warm_moment_float64_alias_rejected_before_load(captured: Capture, field: str) -> None:
    from live_capture import restore

    invalid = copy.deepcopy(captured.pre)
    state = next(entry for entry in invalid["optimizer"]["state"].values() if field in entry)
    state[field] = state[field].double()
    with pytest.raises(ValueError, match="moments.*fp32"):
        validate(invalid)
    with pytest.raises(ValueError, match="moments.*fp32"):
        restore(invalid, trainer_module())


@pytest.mark.parametrize("kind", ["attached", "meta", "nonfinite"])
def test_incompatible_moment_storage_rejected(captured: Capture, kind: str) -> None:
    invalid = copy.deepcopy(captured.pre)
    state = next(
        entry for entry in invalid["optimizer"]["state"].values() if "momentum_buffer" in entry
    )
    if kind == "attached":
        state["momentum_buffer"].requires_grad_(True)
    elif kind == "meta":
        state["momentum_buffer"] = state["momentum_buffer"].to("meta")
    else:
        state["momentum_buffer"].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="moments.*fp32"):
        validate(invalid)


def test_pristine_canonical_named_moment_roundtrip(captured: Capture) -> None:
    from live_capture import cloned, optimizer_identity, restore

    env = restore(captured.pre, trainer_module())
    assert optimizer_identity(env["model"], env["opt"])[2] == captured.pre["optimizer_id_names"]
    compare(captured.pre["optimizer"], cloned(env["opt"].state_dict()))
