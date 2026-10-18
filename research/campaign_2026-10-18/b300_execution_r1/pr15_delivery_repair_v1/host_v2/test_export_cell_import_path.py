"""Verify that the pinned predictor can import its sibling trainer module."""

from __future__ import annotations

import json
import sys

import export_cell


def test_export_entrypoint_adds_predictor_script_parent(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    scripts = source_root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "predict_run.py").write_text("", encoding="utf-8")
    source_src = source_root / "src"
    source_src.mkdir()
    request = tmp_path / "worker_request.json"
    request.write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "mode": "fit",
                "predict_argv": ["--run", "existing_fit"],
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}
    original_path = sys.path.copy()
    original_argv = sys.argv.copy()

    def fake_run_path(path: str, *, run_name: str) -> None:
        observed["path"] = path
        observed["run_name"] = run_name
        observed["sys_path"] = sys.path.copy()

    monkeypatch.setattr(export_cell.runpy, "run_path", fake_run_path)
    monkeypatch.setattr(sys, "argv", ["export_cell.py", "--request", str(request)])
    try:
        export_cell.main()
    finally:
        sys.path[:] = original_path
        sys.argv[:] = original_argv

    assert observed["path"] == str(scripts / "predict_run.py")
    assert observed["run_name"] == "__main__"
    assert observed["sys_path"][:2] == [str(scripts), str(source_src)]
