from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(
    "research/campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/collect_terminal_r5.py"
)


@pytest.fixture
def collector():
    spec = importlib.util.spec_from_file_location("f3_r5_collector_test", SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_rejects_excess_before_payload(collector, monkeypatch):
    helper = collector.load_helper(Path.cwd())
    monkeypatch.setattr(collector, "declared_length", lambda _: 600_000_000)
    files = [
        SimpleNamespace(file_name=f"results/model{i}.pt", url="https://example.test")
        for i in range(2)
    ]
    with pytest.raises(ValueError, match="one-GB"):
        collector.inventory(files, 10, helper)


@pytest.mark.parametrize("name", ["results/../../private", "results/../x", "results/a\\b"])
def test_inventory_rejects_escaping_paths(collector, monkeypatch, name):
    helper = collector.load_helper(Path.cwd())

    def unexpected_head(_: str) -> int:
        pytest.fail("invalid paths must be rejected before provider access")

    monkeypatch.setattr(collector, "declared_length", unexpected_head)
    with pytest.raises(ValueError):
        collector.inventory(
            [SimpleNamespace(file_name=name, url="https://example.test")], 0, helper
        )


def test_inventory_selects_results_and_rejects_duplicates(collector, monkeypatch):
    helper = collector.load_helper(Path.cwd())
    monkeypatch.setattr(collector, "declared_length", lambda _: 42)
    selected = SimpleNamespace(file_name="results/fit/model_ep2.pt", url="https://example.test")
    files = [SimpleNamespace(file_name="source/bundle.wnn"), selected]
    assert collector.inventory(files, 10, helper) == [
        {"path": selected.file_name, "size_bytes": 42}
    ]
    with pytest.raises(ValueError, match="duplicate"):
        collector.inventory([selected, selected], 10, helper)


def test_prior_collection_prevents_any_network_access(collector, tmp_path, monkeypatch):
    output = tmp_path / collector.PACKET / "collected/attempt_001"
    output.mkdir(parents=True)
    monkeypatch.setattr(
        collector, "load_helper", lambda _: pytest.fail("must not load provider helper")
    )
    with pytest.raises(FileExistsError, match="no automatic retry"):
        collector.collect(tmp_path)
