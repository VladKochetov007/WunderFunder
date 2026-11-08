from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import wave_execution
from wave_checks import CONFIG_HASHES, wave_limits

from wnn.multistage_search import file_sha256
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import canonical_sha256


def make_wave(root: Path, *, error: bool = False, cancel: bool = False) -> SimpleNamespace:
    (root / "receipts").mkdir()
    (root / "receipts/first-profile-r1.json").write_text(
        json.dumps({"first_profile_epoch": time.time() - 10})
    )
    control = root / "control"
    (control / "owned").mkdir(parents=True)
    event = threading.Event()
    if cancel:
        event.set()

    def check(_stage: str) -> None:
        if error:
            raise ValueError("resource floor reached")

    return SimpleNamespace(
        runtime=root,
        control=control,
        args=SimpleNamespace(python="fixture-python"),
        owner={"pid": 4, "pgid": 4, "birth": "5"},
        plan_sha="pinned-plan",
        cancelled=event,
        guard=SimpleNamespace(check=check),
        status=lambda *_args, **_kwargs: None,
    )


@pytest.mark.parametrize("cancel", [False, True])
def test_resource_and_cancel_callbacks_latch_exact_owned_zero_grace_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    wave = make_wave(tmp_path, error=not cancel, cancel=cancel)
    cell = {"name": "raw112_d0_s101101", "gpu_uuid": "GPU-fixture", "gpu": 0}
    stopped = []
    monkeypatch.setattr(wave_execution, "is_same_process", lambda _: True)
    monkeypatch.setattr(
        wave_execution,
        "terminate_owned_process",
        lambda child, grace_seconds: stopped.append((child, grace_seconds)),
    )

    def owned(command: list[str], **kwargs: object) -> tuple[int, bool, dict[str, object]]:
        assert kwargs["timeout_seconds"] < 10800
        receipt = {
            "owner": kwargs["owner"],
            "command": command,
            "mode": "fit",
            "child": {"pid": 123, "pgid": 123, "birth": "456"},
        }
        kwargs["receipt_path"].write_text(json.dumps(receipt))
        # This must return normally twice; the legacy throwing-callback branch is forbidden.
        kwargs["progress"]()
        kwargs["progress"]()
        return -9, False, receipt

    monkeypatch.setattr(wave_execution, "run_owned_command", owned)
    with pytest.raises(RuntimeError, match="ROOT cancellation|resource floor"):
        wave_execution.execute_owned(wave, cell, "fit", ["fixture-trainer"], tmp_path / "fit")
    assert stopped == [({"pid": 123, "pgid": 123, "birth": "456"}, 0.0)]


def test_wrong_owner_is_never_signalled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wave = make_wave(tmp_path, error=True)
    monkeypatch.setattr(
        wave_execution,
        "terminate_owned_process",
        lambda *_args, **_kwargs: pytest.fail("foreign stop"),
    )

    def owned(command: list[str], **kwargs: object) -> tuple[int, bool, dict[str, object]]:
        kwargs["receipt_path"].write_text(
            json.dumps({"owner": {"foreign": True}, "command": command, "mode": "fit"})
        )
        kwargs["progress"]()
        return 1, False, {}

    monkeypatch.setattr(wave_execution, "run_owned_command", owned)
    with pytest.raises(RuntimeError, match="cleanup refused"):
        wave_execution.execute_owned(
            wave,
            {"name": "fixture", "gpu": 0, "gpu_uuid": "GPU-fixture"},
            "fit",
            ["fixture"],
            tmp_path / "fit",
        )


def test_root_scalar_clock_is_derived_without_mutation_and_extensions_refused(
    tmp_path: Path,
) -> None:
    make_wave(tmp_path)
    path = tmp_path / "receipts/first-profile-r1.json"
    before = path.read_bytes()
    bound = wave_limits(tmp_path)
    assert bound["fit_deadline_epoch"] - bound["first_profile_epoch"] == 10800
    assert bound["wave_deadline_epoch"] - bound["first_profile_epoch"] == 14400
    assert path.read_bytes() == before
    path.write_text(json.dumps(bound | {"wave_deadline_epoch": bound["wave_deadline_epoch"] + 1}))
    with pytest.raises(ValueError, match="extend"):
        wave_limits(tmp_path)


def test_all_six_native_raw_and_canonical_config_pins() -> None:
    root = Path(__file__).parent / "configs"
    assert len(CONFIG_HASHES) == 6
    for name, pins in CONFIG_HASHES.items():
        path = root / f"{name}.json"
        assert file_sha256(path) == pins["raw"]
        assert canonical_sha256(load_config(path)) == pins["normalized"]
