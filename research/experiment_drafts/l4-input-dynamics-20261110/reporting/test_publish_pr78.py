"""Regression: a failed final GitHub publication must not terminate monitoring."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_final_publication_retries_after_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    path = Path(__file__).with_name("publish_pr78.py")
    spec = importlib.util.spec_from_file_location("pr78_delivery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot = {"stages": [{"epochs_expected": 4, "endpoint_ok": True}], "clock": {"now": 1000.0}}
    monkeypatch.setattr(module, "_snapshot", lambda root: snapshot)
    monkeypatch.setattr(module, "_pr77_controls", lambda root: {})
    monkeypatch.setattr(
        module, "_comparison", lambda stages, controls: [{"paired": "fixed-endpoint"}]
    )
    monkeypatch.setattr(module, "_body", lambda *args: "fixed report")
    monkeypatch.setattr(module, "_client", lambda deadline: object())
    monkeypatch.setattr(module.time, "sleep", lambda duration: None)
    attempts = []

    def publish(client: object, path: Path, marker: str, body: str, when: float) -> str:
        if "paired-endpoints" in marker:
            attempts.append(marker)
            if len(attempts) == 1:
                raise ConnectionError("simulated lost response")
        return "posted"

    monkeypatch.setattr(module, "_publish", publish)
    assert module.run(Path("."), Path("unused"), 10, dryrun=False, once=False) == 0
    assert len(attempts) == 2 and attempts[0] == attempts[1]
