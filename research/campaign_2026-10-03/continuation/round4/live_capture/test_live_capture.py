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
    with pytest.raises(ValueError, match="order|shape"):
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
