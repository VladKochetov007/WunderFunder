from __future__ import annotations

import json
from pathlib import Path

import pytest
from execution_host import _append_json_line
from probe_execution import _initialize_progress_stream


def test_progress_log_is_create_only_and_line_delimited_json(tmp_path: Path) -> None:
    """The initial marker and later events share valid JSONL without overwrite."""
    path = tmp_path / "PROGRESS.jsonl"
    initial = {"event": "initialized", "elapsed_s": 0.0}
    _initialize_progress_stream(path, initial)
    first_bytes = path.read_bytes()
    assert [json.loads(line) for line in first_bytes.splitlines()] == [initial]

    _append_json_line(path, {"event": "window"})
    assert [json.loads(line) for line in path.read_bytes().splitlines()] == [
        initial,
        {"event": "window"},
    ]
    with pytest.raises(FileExistsError):
        _initialize_progress_stream(path, {"event": "replacement"})
    assert b'{"event":"replacement"}' not in path.read_bytes()
