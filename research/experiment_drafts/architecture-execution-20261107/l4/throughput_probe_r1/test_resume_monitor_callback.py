from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from prepare_gru_resume_source import resume_source
from test_monitor_transition import recipe_fixture
from transition_support_s4 import validate_transition

from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_provenance import canonical_sha256


class StateSink:
    """Capture state restoration without constructing or forwarding a model."""

    def __init__(self) -> None:
        self.loaded: object = None

    def load_state_dict(self, state: object, **_kwargs: object) -> None:
        self.loaded = state


class RankLog:
    """Exercise the existing zero-byte integer return branch."""

    def truncate(self, offset: int, **_kwargs: object) -> int:
        return offset


class Monitor:
    """Represent the real monitor's documented side-effect-only return contract."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[dict[str, object], int]] = []

    def restore_offsets(self, offsets: dict[str, object], step: int) -> None:
        self.calls.append((offsets, step))
        if self.error is not None:
            raise self.error


def source_text() -> str:
    """Read frozen-equivalent S1/S3 resume code; never edit a trainer source file."""
    import wnn.training.sequence_fit_resume as original

    return Path(original.__file__).read_text()


def invoke(
    monkeypatch: pytest.MonkeyPatch,
    patched: bool,
    monitor: Monitor | None,
    *,
    monitoring: bool = True,
    finite: bool = True,
) -> tuple[int, int, list[dict[str, object]]]:
    """Isolate callback semantics while retaining the actual canonical coordinator."""
    text = source_text()
    namespace: dict[str, object] = {}
    exec(
        compile(resume_source(text) if patched else text, "resume-source-under-test", "exec"),
        namespace,
    )
    state = {
        "next_epoch": 8,
        "global_step": 51680,
        "history": [{"epoch": 7}],
        "model": {"model-fixture": 1},
        "optimizer": {"optimizer-fixture": 2},
        "scheduler": {"scheduler-fixture": 3},
        "rank_rng_states": [{"rng-fixture": 4}],
        "rank_log_offsets": [0],
        "metrics_offset": 0,
        "monitor_offsets": {"events": 0, "evaluations": 0},
    }
    monkeypatch.setitem(namespace, "load_training_state", lambda *_: state)
    monkeypatch.setitem(namespace, "restore_rng_state", lambda *_: None)
    monkeypatch.setitem(namespace, "optimizer_states_are_finite_fp32", lambda *_: finite)
    monkeypatch.setitem(namespace, "truncate_with_evidence", lambda *_args, **_kwargs: 0)
    model, optimizer, scheduler = StateSink(), StateSink(), StateSink()
    result = namespace["load_resume"](
        Path("unused-trusted-fixture.pt"),
        {},
        model,
        optimizer,
        scheduler,
        DistributedContext(0, 0, 1, torch.device("cpu"), False),
        RankLog(),
        Path("unused"),
        monitor,
        monitoring,
    )
    assert model.loaded == state["model"]
    assert optimizer.loaded == state["optimizer"]
    assert scheduler.loaded == state["scheduler"]
    return result


def test_original_none_callback_reproduces_actual_native_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = Monitor()
    with pytest.raises(RuntimeError, match="safe-boundary monitor log restore returned no value"):
        invoke(monkeypatch, False, monitor)
    assert monitor.calls == [({"events": 0, "evaluations": 0}, 51680)]


def test_s4_acknowledges_successful_side_effect_without_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = Monitor()
    assert invoke(monkeypatch, True, monitor) == (8, 51680, [{"epoch": 7}])
    assert monitor.calls == [({"events": 0, "evaluations": 0}, 51680)]


def test_s4_propagates_genuine_monitor_restore_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = Monitor(ValueError("checkpointed evaluation prefix is incomplete"))
    with pytest.raises(
        RuntimeError, match="safe-boundary monitor log restore failed.*prefix is incomplete"
    ):
        invoke(monkeypatch, True, monitor)
    assert len(monitor.calls) == 1


def test_s4_does_not_hide_optimizer_finiteness_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = Monitor()
    with pytest.raises(FloatingPointError, match="optimizer state is not finite"):
        invoke(monkeypatch, True, monitor, finite=False)
    assert monitor.calls == []


def test_disabled_monitor_branch_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = Monitor(ValueError("must not be called"))
    assert invoke(monkeypatch, True, monitor, monitoring=False)[0:2] == (8, 51680)
    assert monitor.calls == []


def test_only_monitor_side_effect_adapter_changes_in_ast() -> None:
    old, new = ast.parse(source_text()), ast.parse(resume_source(source_text()))
    old_function = next(
        n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "load_resume"
    )
    new_function = next(
        n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "load_resume"
    )
    assert ast.dump(ast.Module(old_function.body[:-2], [])) == ast.dump(
        ast.Module(new_function.body[:-2], [])
    )
    assert ast.dump(old_function.body[-1]) == ast.dump(new_function.body[-1])
    old.body.remove(old_function)
    new.body.remove(new_function)
    assert ast.dump(old) == ast.dump(new)
    callback = new_function.body[-2].body[1]
    assert isinstance(callback, ast.FunctionDef) and callback.name == "restore_monitor"
    assert isinstance(callback.body[-1], ast.Return) and callback.body[-1].value.value is True


def test_source_anchor_drift_refused() -> None:
    with pytest.raises(ValueError, match="anchor differs"):
        resume_source(source_text().replace('"safe-boundary monitor log restore"', '"changed"'))


def test_s4_admission_requires_exactly_four_source_files() -> None:
    old, new, config, amended = recipe_fixture()
    path = "src/wnn/training/sequence_fit_resume.py"
    old["source_files"][path] = "6" * 64
    new["source_files"][path] = "7" * 64
    for identity in (old, new):
        identity["source_sha256"] = canonical_sha256(identity["source_files"])
    validate_transition(config, amended, old, new)
    new["source_files"][path] = old["source_files"][path]
    new["source_sha256"] = canonical_sha256(new["source_files"])
    with pytest.raises(ValueError, match="diagnostic whitelist"):
        validate_transition(config, amended, old, new)


def test_s4_extra_numerical_source_edit_refused() -> None:
    old, new, config, amended = recipe_fixture()
    old["source_files"]["src/wnn/training/sequence_fit_resume.py"] = "6" * 64
    new["source_files"]["src/wnn/training/sequence_fit_resume.py"] = "7" * 64
    new["source_files"]["scripts/train_sequence.py"] = "8" * 64
    for identity in (old, new):
        identity["source_sha256"] = canonical_sha256(identity["source_files"])
    with pytest.raises(ValueError, match="diagnostic whitelist"):
        validate_transition(config, amended, old, new)
