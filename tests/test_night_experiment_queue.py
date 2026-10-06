from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.night_experiment_queue import inventory


def _draft(
    name: str,
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> dict[str, object]:
    return {
        "version": 1,
        "name": name,
        "kind": "cpu",
        "argv": ["{python}", "worker.py"],
        "code": ["worker.py"],
        "code_sha256": "0" * 64,
        "inputs": inputs or [],
        "artifacts": outputs or [f"{name}.json"],
        "limits": {
            "cpus": 1,
            "ram_bytes": 1_000_000_000,
            "timeout_seconds": 60,
            "gpu_memory_bytes": 0,
            "gpu_stop_bytes": 0,
        },
        "environment": {},
    }


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    directory = root / "research" / "workstream"
    (directory / "specs").mkdir(parents=True)
    (root / "worker.py").write_text("# fixture worker\n")
    return root, directory


def _write(directory: Path, draft: dict[str, object]) -> Path:
    path = directory / "specs" / f"{draft['name']}.draft.json"
    path.write_text(json.dumps(draft, indent=2) + "\n")
    return path


def test_inventory_reports_external_input_without_mutating_draft(tmp_path: Path) -> None:
    root, directory = _workspace(tmp_path)
    path = _write(directory, _draft("missing_input", inputs=["upstream/data.csv"]))
    original = path.read_bytes()

    result = inventory(root, [directory])

    job = result["jobs"][0]
    assert job["draft_sha256"] == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original
    assert job["missing_external_or_ambiguous_inputs"] == ["upstream/data.csv"]
    assert job["preparation_status"] == "blocked_external_dependency"
    assert result["schema_errors"] == []


def test_inventory_reports_cycle_and_cycle_dependent_job(tmp_path: Path) -> None:
    root, directory = _workspace(tmp_path)
    _write(directory, _draft("alpha", inputs=["beta.json"], outputs=["alpha.json"]))
    _write(directory, _draft("beta", inputs=["alpha.json"], outputs=["beta.json"]))
    _write(directory, _draft("downstream", inputs=["alpha.json"]))

    result = inventory(root, [directory])

    assert result["dependency_order"] == []
    assert {Path(item).stem for item in result["cyclic_or_cycle_dependent_specs"]} == {
        "alpha.draft",
        "beta.draft",
        "downstream.draft",
    }


def test_inventory_reports_ambiguous_output_producers_and_blocks_consumer(
    tmp_path: Path,
) -> None:
    root, directory = _workspace(tmp_path)
    _write(directory, _draft("producer_a", outputs=["shared.json"]))
    _write(directory, _draft("producer_b", outputs=["shared.json"]))
    _write(directory, _draft("consumer", inputs=["shared.json"]))

    result = inventory(root, [directory])
    consumer = next(job for job in result["jobs"] if job["name"] == "consumer")
    producers = [job for job in result["jobs"] if job["name"].startswith("producer_")]

    assert len(result["ambiguous_output_producers"]["shared.json"]) == 2
    assert len(producers) == 2
    assert all(job["preparation_status"] == "blocked_ambiguous_output_producer" for job in producers)
    assert consumer["missing_external_or_ambiguous_inputs"] == ["shared.json"]
    assert consumer["preparation_status"] == "blocked_external_dependency"
