"""Focused local regressions for causal-stem progress delivery."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import publish_l4_stem as reporter


def test_terminal_publisher_retries_lost_final_post_until_reconciled(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A lost cohort response remains pending and is reconciled before exit."""

    cells = [
        {
            "cell": name,
            "ok": True,
            "complete": True,
            "events": [],
            "terminal": {"status": "completed", "sha": f"{index:064x}"},
        }
        for index, name in enumerate(reporter.CELLS, start=1)
    ]
    reads = [0]

    def inspect_cell(
        _monitor: Path, _name: str, _admission: dict[str, object]
    ) -> dict[str, object]:
        reads[0] += 1
        return cells[reporter.CELLS.index(_name)]

    class LostFinalClient:
        def __init__(self) -> None:
            self.found: dict[str, dict[str, object]] = {}
            self.posted: list[str] = []

        def find_report(self, number: int, marker: str, _prepared_at: float) -> object:
            assert number == 79
            return self.found.get(marker)

        def post_report(self, number: int, body: str, *, before_post: Any) -> dict[str, object]:
            assert number == 79
            marker = body.splitlines()[-1]
            before_post()
            self.posted.append(marker)
            record = {"id": len(self.posted), "body": body}
            self.found[marker] = record
            if "-cohort-" in marker and self.posted.count(marker) == 1:
                raise RuntimeError("simulated lost final response")
            return record

    client = LostFinalClient()
    monkeypatch.setattr(
        reporter,
        "_admission",
        lambda _path, pinned, _configs: {
            "admission_sha256": pinned,
            "source_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(reporter, "inspect_cell", inspect_cell)
    monkeypatch.setattr(reporter, "_report", lambda *_args: "local synthetic report")
    monkeypatch.setattr(
        reporter.P,
        "_source_clock",
        lambda _root: {"local_now": 1_800_000_000.0, "age_seconds": None},
    )
    monkeypatch.setattr(
        reporter,
        "_module",
        lambda _path, _name: SimpleNamespace(ProgressGitHub=lambda _allow, deadline: client),
    )
    monkeypatch.setattr(reporter.time, "sleep", lambda _seconds: None)

    state_path = tmp_path / "publisher-state.json"
    result = reporter.run(
        Namespace(
            maxseconds=60,
            pr=79,
            profile_admission=tmp_path / "admission.json",
            admission_sha256="a" * 64,
            config_root=tmp_path,
            root=tmp_path,
            monitor_root=tmp_path,
            dryrun=False,
            once=False,
            state_file=state_path,
        )
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    statuses = {value["status"] for value in state["reports"].values()}
    assert result == 0
    assert reads[0] == 2 * len(reporter.CELLS)
    assert len(client.posted) == len(reporter.CELLS) + 2
    assert statuses == {"posted"}
    assert sum("-cohort-" in marker for marker in client.posted) == 1
