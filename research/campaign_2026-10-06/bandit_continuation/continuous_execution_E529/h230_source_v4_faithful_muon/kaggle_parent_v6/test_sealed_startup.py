from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
import tempfile
import types
from pathlib import Path

import pytest

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "specs/README.md").is_file()
)
BASE = ROOT / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
PACKET = BASE / "h230_source_v4_faithful_muon/kaggle_parent_v6"


def load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARCHIVE_DIRECTORY = tempfile.TemporaryDirectory(prefix="h230-v6-test-")
ARCHIVE_ROOT = Path(ARCHIVE_DIRECTORY.name)
with tarfile.open(PACKET / "bundle/bundle.wnn", "r:gz") as archive:
    archive.extractall(ARCHIVE_ROOT, filter="data")

sys.path.insert(0, str(ROOT / "scripts"))
HELPER = load_module("h230_v5_sealed_helper", ARCHIVE_ROOT / "scripts/h230_profile_then_fit.py")
CASES = load_module("h230_v4_cases", PACKET.parent / "kaggle_parent_v4/test_profile_gate.py")
CASES.evaluate_profile = HELPER.evaluate_profile


def test_prior_exact_profile_and_rejection_cases() -> None:
    CASES.test_profile_gate_accepts_exact_finite_t4_profile()
    CASES.test_profile_gate_rejects_device_dtype_shape_and_gradient_drift()
    CASES.test_profile_gate_rejects_projection_over_worker_timeout()


def test_actual_t4_emulation_record_keeps_float32_gate() -> None:
    producer = (
        BASE
        / "gru_capacity_v4_terminal_collect_20261008_R1/gru512_768_muon_capacity_v4_manifest_profile_receipts/muon512-profile.log"
    )
    line = producer.read_text().splitlines()[0]
    metadata = json.loads(line.removeprefix("OPTIMIZER_COMPUTE_PRECISION "))["groups"][0]
    assert metadata["torch_cuda_bf16_supported"] is True
    assert metadata["native_bf16_selected"] is False
    plan, records, parameters = CASES._valid_case()
    records["muon_runtime"].update(
        device_capability=metadata["device_capability"],
        cuda_bf16_supported=metadata["torch_cuda_bf16_supported"],
        selected_dtype=metadata["selected_dtype"],
    )
    assert CASES._evaluate(plan, records, parameters)["status"] == "passed"
    records["muon_runtime"]["selected_dtype"] = "bfloat16"
    assert CASES._evaluate(plan, records, parameters)["status"] != "passed"


def test_child_initializes_cuda_before_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    calls: list[str] = []
    monkeypatch.setenv("WNN_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(HELPER, "_verify_raw_inputs", lambda *args: {})
    monkeypatch.setitem(sys.modules, "train_rnn", types.ModuleType("train_rnn"))
    monkeypatch.setattr(torch.cuda, "init", lambda: calls.append("init"))
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(f"device{device}"))

    def reset(device: int) -> None:
        assert device == 0
        assert calls == ["init", "device0"]
        calls.append("reset")

    def stop_after_startup(module: object, argv: list[str], log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("test stops before training\n")
        raise SystemExit(95)

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(HELPER, "_invoke_trainer", stop_after_startup)
    with pytest.raises(SystemExit) as exception:
        HELPER.run(PACKET / "profile_fit_plan_v1.json")
    assert exception.value.code == 95
    assert calls == ["init", "device0", "reset"]


def test_both_full_commands_parse_in_actual_sealed_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import argparse

    monkeypatch.syspath_prepend(str(ARCHIVE_ROOT / "src"))
    monkeypatch.syspath_prepend(str(ARCHIVE_ROOT / "scripts"))
    trainer = load_module("sealed_train_rnn", ARCHIVE_ROOT / "scripts/train_rnn.py")
    original_parse = argparse.ArgumentParser.parse_args
    parsed_seeds: list[int] = []

    def parse_then_stop(parser: argparse.ArgumentParser, *args: object, **kwargs: object) -> None:
        parsed = original_parse(parser, *args, **kwargs)
        parsed_seeds.append(parsed.seed)
        assert parsed.epochs == 4 and parsed.hidden == 256 and parsed.optim == "muon"
        raise SystemExit(95)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    plan = json.loads((PACKET / "profile_fit_plan_v1.json").read_text())
    for kind in ("profile", "fit"):
        with pytest.raises(SystemExit) as exception:
            HELPER._invoke_trainer(trainer, plan[f"{kind}_argv"], tmp_path / f"{kind}.log")
        assert exception.value.code == 95
    assert parsed_seeds == [9, 9]
