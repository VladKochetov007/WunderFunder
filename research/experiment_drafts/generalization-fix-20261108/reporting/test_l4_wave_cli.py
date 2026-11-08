from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import report_l4_wave


def test_dry_run_never_constructs_network_or_rpc_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry run constructed a live client")

    monkeypatch.setattr(report_l4_wave, "ProgressGitHub", refuse)
    monkeypatch.setattr(report_l4_wave, "ReadoutMirror", refuse)
    monkeypatch.setattr(report_l4_wave.os, "write", lambda *args: 0)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"native_root": str(tmp_path)}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--mirror-config",
            str(config),
            "--pr",
            "71",
            "--state",
            str(tmp_path / "state.json"),
            "--readout-root",
            str(tmp_path / "readouts"),
            "--rpc-root",
            str(tmp_path / "rpc"),
            "--native-run-root",
            "/unused/dry-run",
            "--dry-run",
            "--once",
        ],
    )
    assert report_l4_wave.main() == 0
    assert not (tmp_path / "rpc").exists() and not (tmp_path / "state.json").exists()
