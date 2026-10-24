"""Small deterministic fixtures for research-tree preservation and safety."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import research_tree


def test_legacy_experiment_id_preserves_original_record(tmp_path: Path) -> None:
    """Index legacy experiment events without changing their append-only bytes."""
    _fixture(tmp_path)
    ledger = tmp_path / "research/experiments.jsonl"
    legacy = b'{"experiment_id":"E-1","status":"legacy_complete"}\r\n'
    with ledger.open("ab") as stream:
        stream.write(legacy)
    original = ledger.read_bytes()
    assert research_tree.main(_args(tmp_path, "build")) == 0
    assert research_tree.main(_args(tmp_path, "check")) == 0
    node = next(n for n in _tree(tmp_path)["nodes"] if n["id"] == "experiment:E-1")
    assert len(node["raw_rows"]) == 3
    assert node["latest_recorded_status"] == "legacy_complete"
    assert node["raw_rows"][-1]["raw_record"].encode("utf-8") == legacy
    assert node["raw_rows"][-1]["source"]["sha256"] == hashlib.sha256(legacy).hexdigest()
    assert ledger.read_bytes() == original


def test_experiment_id_alias_does_not_replace_a_hypothesis_id(tmp_path: Path) -> None:
    """Keep malformed hypothesis records invalid despite the experiment-only alias."""
    _fixture(tmp_path)
    (tmp_path / "research/hypotheses.jsonl").write_text('{"experiment_id":"H-1"}\n')
    assert research_tree.main(_args(tmp_path, "build")) == 1


def _fixture(root: Path, cyclic: bool = False) -> Path:
    research = root / "research"
    reflect = root / "reflect_05_day"
    config_dir = research / "bandit_tree"
    config_dir.mkdir(parents=True)
    reflect.mkdir(parents=True)
    (research / "hypotheses.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "H-1", "status": "open", "statement": "alpha"}),
                json.dumps(
                    {
                        "id": "H-1",
                        "status": "supported",
                        "statement": "alpha updated",
                        "parent_ids": ["H-2"],
                        "evidence": ["E-1", "research/a/REPORT.md", "since/base 0.98/0.25"],
                    }
                ),
                json.dumps({"id": "H-2", "status": "proposed", "statement": "other"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (research / "experiments.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "E-1", "hypotheses": ["H-1"], "status": "planned"}),
                json.dumps({"id": "E-1", "hypotheses": ["H-1"], "status": "complete"}),
                json.dumps({"id": "E-2", "hypotheses": ["H-1", "H-2"], "result": "recorded"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (research / "bandit").mkdir()
    legacy = {
        "_doc": "historical notes",
        "horizon_h": 12,
        "arms": {"old": {"mu": 0.4, "sd": 0.2, "pulls": ["E-1"]}},
        "continuation_evidence": {"round": {"status": "preserved"}},
    }
    (research / "bandit" / "arms.json").write_text(json.dumps(legacy), encoding="utf-8")
    (config_dir / "journal.jsonl").write_bytes(
        b'{"id":"J-1","event":"first"}\r\n{"id":"J-1","event":"second"}\n'
    )
    (research / "a").mkdir()
    (research / "b").mkdir()
    (research / "a" / "REPORT.md").write_text("a", encoding="utf-8")
    (research / "b" / "REPORT.md").write_text("b", encoding="utf-8")
    nodes = [
        {
            "id": "arm:current",
            "parent_id": "family:alpha",
            "title": "Current arm",
            "kind": "arm",
            "hypothesis_ids": ["H-1"],
            "experiment_ids": ["E-1"],
            "artifacts": ["REPORT.md", "research/absent.md"],
            "status": "not_tested",
            "next_test": "none",
            "related_nodes": [],
            "allocation": {
                "eligible": True,
                "priority": 2,
                "reason": "fixture",
                "launch_authorized": False,
            },
            "belief": {
                "mu": None,
                "sd": None,
                "noise_sd": None,
                "unit": "absolute_global_wp_delta",
                "reviewed": False,
            },
            "cost_h": None,
        },
        {
            "id": "arm:old-stale",
            "parent_id": "family:alpha",
            "title": "Old prior awaiting refresh",
            "kind": "arm",
            "hypothesis_ids": [],
            "experiment_ids": [],
            "artifacts": [],
            "status": "not_tested",
            "next_test": "refresh",
            "related_nodes": [],
            "legacy_id": "old",
            "allocation": {
                "eligible": True,
                "priority": 99,
                "reason": "stale fixture",
                "launch_authorized": False,
            },
            "belief": {
                "mu": 0.4,
                "sd": 0.2,
                "noise_sd": 0.1,
                "unit": "absolute_global_wp_delta",
                "reviewed": True,
            },
            "cost_h": 1,
        },
    ]
    if cyclic:
        nodes = [
            {**nodes[0], "id": "arm:a", "parent_id": "arm:b"},
            {**nodes[1], "id": "arm:b", "parent_id": "arm:a", "legacy_id": None},
        ]
    config = {
        "version": 1,
        "root": {
            "id": "root",
            "title": "Fixture research",
            "objective_mode": "competition",
            "metric_id": "global_wp_clip2_v1",
            "public_best": 0.5,
            "public_milestone": 0.7,
            "resources": {},
            "evaluation_exposure": "exposed fixture data",
        },
        "families": [
            {"id": "family:alpha", "title": "Alpha", "keywords": ["alpha"]},
            {"id": "family:unclassified", "title": "Unclassified", "keywords": []},
        ],
        "curated_nodes": nodes,
        "links": [],
        "documents": ["research", "reflect_05_day"],
        "journal": "research/bandit_tree/journal.jsonl",
    }
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def _args(root: Path, command: str) -> list[str]:
    return ["--root", str(root), command]


def _tree(root: Path) -> dict[str, object]:
    path = root / "research" / "bandit_tree" / "generated" / "tree.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_duplicate_events_and_legacy_state_are_preserved(tmp_path: Path, capsys: object) -> None:
    config_path = _fixture(tmp_path)
    assert research_tree.main(_args(tmp_path, "build")) == 0
    tree = _tree(tmp_path)
    nodes = {node["id"]: node for node in tree["nodes"]}
    h1 = nodes["hypothesis:H-1"]
    e1 = nodes["experiment:E-1"]
    assert len(h1["raw_rows"]) == 2
    assert len(e1["raw_rows"]) == 2
    assert h1["latest_recorded_status"] == "supported"
    assert h1["raw_rows"][0]["source"]["line"] == 1
    raw_line = (
        (tmp_path / "research" / "hypotheses.jsonl").read_bytes().splitlines(keepends=True)[0]
    )
    assert h1["raw_rows"][0]["source"]["sha256"] == hashlib.sha256(raw_line).hexdigest()
    assert h1["raw_rows"][0]["raw_record"] == raw_line.decode("utf-8")
    assert [item["kind"] for item in h1["evidence_refs"]] == ["experiment", "document", "note"]
    journal = tree["metadata"]["journal_records"]
    journal_line = (
        (tmp_path / "research" / "bandit_tree" / "journal.jsonl")
        .read_bytes()
        .splitlines(keepends=True)[0]
    )
    assert len(journal) == 2
    assert journal[0]["raw_record"].encode("utf-8") == journal_line
    assert journal[0]["source"]["sha256"] == hashlib.sha256(journal_line).hexdigest()
    legacy = nodes["legacy:old"]
    assert legacy["allocation"]["eligible"] is False
    assert legacy["raw_arm"]["mu"] == 0.4
    assert (
        tree["metadata"]["legacy_archive"]["raw"]["continuation_evidence"]["round"]["status"]
        == "preserved"
    )
    edges = tree["related_edges"]
    assert any(edge["target_id"] == "hypothesis:H-2" and edge["resolved"] for edge in edges)
    e2 = nodes["experiment:E-2"]
    assert e2["parent_id"] == "family:unclassified"
    assert sum(edge["source_id"] == "experiment:E-2" for edge in edges) == 2
    current = nodes["arm:current"]
    assert [ref["status"] for ref in current["artifact_refs"]] == ["ambiguous", "missing"]
    index = (tmp_path / "research" / "bandit_tree" / "generated" / "INDEX.md").read_text(
        encoding="utf-8"
    )
    assert "hypotheses.jsonl#L1" in index
    assert "../../a/REPORT.md" in index
    assert "ambiguous" in index and "not been independently verified" in index
    catalog = json.loads(
        (tmp_path / "research" / "bandit_tree" / "generated" / "catalog.json").read_text(
            encoding="utf-8"
        )
    )
    assert all("generated/" not in entry["path"] for entry in catalog["files"])
    assert capsys.readouterr().out.startswith("Built ")
    assert config_path.is_file()


def test_parent_cycle_is_rejected(tmp_path: Path, capsys: object) -> None:
    _fixture(tmp_path, cyclic=True)
    assert research_tree.main(_args(tmp_path, "build")) == 1
    assert "Cycle in parent tree" in capsys.readouterr().err


def test_invalid_source_reports_path_and_line(tmp_path: Path, capsys: object) -> None:
    _fixture(tmp_path)
    (tmp_path / "research" / "hypotheses.jsonl").write_text("{not json}\n", encoding="utf-8")
    assert research_tree.main(_args(tmp_path, "build")) == 1
    assert "research/hypotheses.jsonl:1" in capsys.readouterr().err


def test_legacy_prior_cannot_rank_until_current_override(tmp_path: Path, capsys: object) -> None:
    _fixture(tmp_path)
    config_path = tmp_path / "research" / "bandit_tree" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["curated_nodes"].append(
        {
            **config["curated_nodes"][1],
            "id": "arm:old-current",
            "title": "Explicitly refreshed old recipe",
            "current_override": True,
            "allocation": {
                "eligible": True,
                "priority": 1,
                "reason": "reviewed",
                "launch_authorized": False,
            },
            "belief": {
                "mu": None,
                "sd": None,
                "noise_sd": None,
                "unit": "absolute_global_wp_delta",
                "reviewed": False,
            },
            "cost_h": None,
        }
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    assert research_tree.main(_args(tmp_path, "build")) == 0
    assert research_tree.main(_args(tmp_path, "rank")) == 0
    output = capsys.readouterr().out
    assert "arm:old-current" in output
    assert "expected_public_gain=unknown" in output
    assert "arm:old-stale" not in output
    assert "legacy:old" not in output


def test_check_detects_stale_generation_and_missing_inputs(tmp_path: Path, capsys: object) -> None:
    _fixture(tmp_path)
    args = _args(tmp_path, "build")
    assert research_tree.main(args) == 0
    assert research_tree.main(_args(tmp_path, "check")) == 0
    capsys.readouterr()
    source = tmp_path / "research" / "a" / "REPORT.md"
    source.unlink()
    assert research_tree.main(_args(tmp_path, "check")) == 1
    assert "stale generated files" in capsys.readouterr().err
    (tmp_path / "research" / "experiments.jsonl").unlink()
    assert research_tree.main(_args(tmp_path, "check")) == 1
    assert "No such file" in capsys.readouterr().err
