"""Adversarial tiny fixtures for tree provenance, linkage and inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import research_tree


def _repository(
    root: Path,
    *,
    hypotheses: bytes | None = None,
    experiments: bytes | None = None,
    legacy_arms: dict[str, object] | None = None,
) -> Path:
    research = root / "research"
    config_dir = research / "bandit_tree"
    config_dir.mkdir(parents=True)
    (research / "bandit").mkdir()
    reflect = root / "reflect_05_day"
    reflect.mkdir()

    default_hypotheses = b'{"id":"H-1","status":"open","statement":"alpha"}\n'
    default_experiments = b'{"id":"E-1","hypotheses":["H-1"],"status":"registered"}\n'
    (research / "hypotheses.jsonl").write_bytes(hypotheses or default_hypotheses)
    (research / "experiments.jsonl").write_bytes(experiments or default_experiments)
    arms = legacy_arms or {"old": {"mu": 4, "sd": 2, "pulls": []}}
    (research / "bandit" / "arms.json").write_text(
        json.dumps({"_doc": "historical", "arms": arms, "future": {"kept": True}}),
        encoding="utf-8",
    )
    (config_dir / "journal.jsonl").write_bytes(
        b'{ "id" : "TREE-1", "decision" : "preserve" }\r\n'
    )
    (research / "a").mkdir()
    (research / "b").mkdir()
    (research / "a" / "REPORT.md").write_text("first", encoding="utf-8")
    (research / "b" / "REPORT.md").write_text("second", encoding="utf-8")

    config = {
        "version": 1,
        "root": {
            "id": "root",
            "title": "Fixture research",
            "objective_mode": "competition",
            "metric_id": "global_wp_clip2_v1",
            "public_best": 0.666,
            "public_milestone": 0.7,
            "public_milestone_status": "unmet",
            "evaluation_exposure": "exposed fixture data",
        },
        "families": [
            {"id": "family:alpha", "title": "Alpha", "keywords": ["alpha"]},
            {"id": "family:unclassified", "title": "Unclassified", "keywords": []},
        ],
        "curated_nodes": [
            {
                "id": "arm:current",
                "parent_id": "family:alpha",
                "title": "Current fixture arm",
                "kind": "arm",
                "hypothesis_ids": ["H-1"],
                "experiment_ids": ["E-1"],
                "artifacts": ["REPORT.md", "research/absent.md"],
                "related_nodes": [],
                "status": "not_tested",
                "next_test": "fixture only",
                "evidence_states": [
                    {"scope": "attempt A", "ledger": "admitted", "receipt": "interrupted"},
                    {"scope": "attempt B", "ledger": "prepared", "receipt": "missing"},
                ],
                "planned_experiment_ids": ["E-404"],
                "future_extension": {"retain": [1, 2, 3]},
                "allocation": {
                    "eligible": True,
                    "priority": 3,
                    "reason": "manual fixture priority",
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
                "id": "arm:interaction",
                "parent_id": "family:alpha",
                "title": "Explicit cross-family interaction fixture",
                "kind": "arm",
                "hypothesis_ids": [],
                "experiment_ids": [],
                "related_nodes": ["arm:current"],
                "status": "not_tested",
                "allocation": {"eligible": False, "priority": 0, "launch_authorized": False},
            },
        ],
        "links": [],
        "documents": ["research", "reflect_05_day"],
        "journal": "research/bandit_tree/journal.jsonl",
    }
    path = config_dir / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _args(root: Path, command: str) -> list[str]:
    return ["--root", str(root), command]


def _generated(root: Path, name: str) -> Path:
    return root / "research" / "bandit_tree" / "generated" / name


def test_exact_source_bytes_duplicate_events_and_unresolved_links(tmp_path: Path, capsys: object) -> None:
    hypothesis_lines = [
        b'{ "id" : "H-1", "status" : "recorded", "statement" : "alpha", "future_extension" : {"k":1} }\r\n',
        b'{"id":"H-1","status":"updated","statement":"alpha updated","parent_ids":["H-2","H-404"],"evidence":["E-1","E-404"]} \n',
        b'{"id":"H-2","status":"open","statement":"alpha parent"}\n',
    ]
    experiment_lines = [
        b'{"id":"E-1","hypotheses":["H-1"],"status":"registered"}\r\n',
        b'{ "id" : "E-1", "status" : "latest row omits prior links", "future_experiment_field" : 5 }\n',
        b'{"id":"E-2","hypotheses":["H-1","H-2"],"status":"recorded"}\n',
        b'{"id":"E-3","hypotheses":["H-1"],"status":"early link"}\n',
        b'{"id":"E-3","status":"latest row omits prior links"}\n',
    ]
    arms = {f"old-{index:02}": {"mu": index - 8, "sd": 2, "pulls": []} for index in range(18)}
    _repository(
        tmp_path,
        hypotheses=b"".join(hypothesis_lines),
        experiments=b"".join(experiment_lines),
        legacy_arms=arms,
    )

    assert research_tree.main(_args(tmp_path, "build")) == 0
    tree = json.loads(_generated(tmp_path, "tree.json").read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in tree["nodes"]}

    h1 = nodes["hypothesis:H-1"]
    assert h1["latest_recorded_status"] == "updated"
    assert len(h1["raw_rows"]) == 2
    for row, raw_line, line_number in zip(h1["raw_rows"], hypothesis_lines[:2], (1, 2), strict=True):
        assert row["raw_record"].encode("utf-8") == raw_line
        assert row["source"] == {
            "path": "research/hypotheses.jsonl",
            "line": line_number,
            "sha256": hashlib.sha256(raw_line).hexdigest(),
        }
    assert h1["raw_rows"][0]["record"]["future_extension"] == {"k": 1}

    e1 = nodes["experiment:E-1"]
    assert len(e1["raw_rows"]) == 2
    assert e1["raw_rows"][0]["raw_record"].encode("utf-8") == experiment_lines[0]
    assert e1["raw_rows"][1]["raw_record"].encode("utf-8") == experiment_lines[1]
    assert e1["parent_id"] == "arm:current", "curated membership has explicit precedence"
    assert e1["raw_rows"][1]["record"]["future_experiment_field"] == 5
    assert nodes["experiment:E-2"]["parent_id"] == "family:unclassified"
    e3 = nodes["experiment:E-3"]
    assert len(e3["raw_rows"]) == 2
    assert e3["parent_id"] == "hypothesis:H-1", "association in an earlier row must not vanish"

    edge_by_target = {edge["target_id"]: edge for edge in tree["related_edges"] if edge["source_id"] == "hypothesis:H-1"}
    assert edge_by_target["hypothesis:H-2"]["resolved"] is True
    assert edge_by_target["H-404"]["resolved"] is False
    assert edge_by_target["experiment:E-1"]["resolved"] is True
    assert edge_by_target["experiment:E-404"]["resolved"] is False
    assert any("H-404" in warning for warning in tree["warnings"])
    assert any("E-404" in warning for warning in tree["warnings"])
    e2_edges = [edge for edge in tree["related_edges"] if edge["source_id"] == "experiment:E-2"]
    assert {edge["target_id"] for edge in e2_edges} == {"hypothesis:H-1", "hypothesis:H-2"}
    assert all(edge["resolved"] for edge in e2_edges)

    current = nodes["arm:current"]
    assert current["evidence_states"] == [
        {"scope": "attempt A", "ledger": "admitted", "receipt": "interrupted"},
        {"scope": "attempt B", "ledger": "prepared", "receipt": "missing"},
    ]
    assert current["planned_experiment_ids"] == ["E-404"]
    assert current["future_extension"] == {"retain": [1, 2, 3]}
    assert [item["status"] for item in current["artifact_refs"]] == ["ambiguous", "missing"]
    interaction_edge = next(
        edge
        for edge in tree["related_edges"]
        if edge["source_id"] == "arm:interaction" and edge["relation"] == "curated_related_node"
    )
    assert interaction_edge["target_id"] == "arm:current" and interaction_edge["resolved"] is True

    legacy_nodes = [node for node in tree["nodes"] if node["type"] == "legacy_arm"]
    assert len(legacy_nodes) == 18
    assert all(node["allocation"]["eligible"] is False for node in legacy_nodes)
    assert {node["raw_arm"]["mu"] for node in legacy_nodes} == set(range(-8, 10))
    assert research_tree.main(_args(tmp_path, "rank")) == 0
    ranking = capsys.readouterr().out
    assert "expected_public_gain=unknown" in ranking
    assert "KG/cost=" not in ranking
    assert "legacy:old-" not in ranking


def test_filesystem_catalog_includes_ignored_hidden_provider_and_symlink_entries(
    tmp_path: Path, capsys: object
) -> None:
    _repository(tmp_path)
    research = tmp_path / "research"
    (research / ".gitignore").write_text("ignored.data\n", encoding="utf-8")
    (research / "ignored.data").write_text("still index this", encoding="utf-8")
    (research / ".hidden_snapshot").mkdir()
    (research / ".hidden_snapshot" / "hypotheses.snapshot.jsonl").write_text("[]", encoding="utf-8")
    (research / "provider_output").mkdir()
    (research / "provider_output" / "RECEIPT.json").write_text('{"state":"ERROR"}', encoding="utf-8")
    (research / "opaque.bin").write_bytes(b"\x00\xff\x10")
    (tmp_path / "outside_document_root").mkdir()
    (tmp_path / "outside_document_root" / "must_not_be_walked.json").write_text("secret", encoding="utf-8")
    (research / "reflect.snapshot.md").write_text("snapshot", encoding="utf-8")
    (tmp_path / "reflect_05_day" / "provider_export.json").write_text("{}", encoding="utf-8")

    symlink_supported = True
    try:
        (research / "provider_link").symlink_to("../outside_document_root", target_is_directory=True)
        (research / "self_loop").symlink_to("self_loop")
    except OSError:
        symlink_supported = False

    assert research_tree.main(_args(tmp_path, "build")) == 0
    catalog = json.loads(_generated(tmp_path, "catalog.json").read_text(encoding="utf-8"))
    by_path = {entry["path"]: entry for entry in catalog["files"]}
    required = {
        "research/.gitignore",
        "research/ignored.data",
        "research/.hidden_snapshot/hypotheses.snapshot.jsonl",
        "research/provider_output/RECEIPT.json",
        "research/opaque.bin",
        "research/reflect.snapshot.md",
        "reflect_05_day/provider_export.json",
    }
    assert required <= by_path.keys()
    assert "outside_document_root/must_not_be_walked.json" not in by_path
    assert "research/generated/tree.json" not in by_path
    assert by_path["research/opaque.bin"]["sha256"] is None
    assert by_path["research/opaque.bin"]["hash_reason"] == "not_small_text"
    if symlink_supported:
        assert by_path["research/provider_link"]["type"] == "symlink"
        assert by_path["research/self_loop"]["type"] == "symlink"
        assert "research/provider_link/must_not_be_walked.json" not in by_path
    assert catalog["file_count"] == len(catalog["files"])
    capsys.readouterr()


def test_config_snapshots_are_content_addressed_and_immutable(tmp_path: Path, capsys: object) -> None:
    config_path = _repository(tmp_path)
    assert research_tree.main(_args(tmp_path, "build")) == 0
    integrity = json.loads(_generated(tmp_path, "integrity.json").read_text(encoding="utf-8"))
    original_snapshot = tmp_path / "research" / "bandit_tree" / "generated" / integrity["config_snapshot"].split("generated/", 1)[-1]
    original_bytes = original_snapshot.read_bytes()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["root"]["revision"] = "second"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    assert research_tree.main(_args(tmp_path, "build")) == 0

    snapshots = sorted((_generated(tmp_path, "tree.json").parent / "snapshots").glob("config-*.json"))
    assert len(snapshots) == 2
    assert original_snapshot.read_bytes() == original_bytes
    assert any(path.read_bytes() == config_path.read_bytes() for path in snapshots)
    assert research_tree.main(_args(tmp_path, "check")) == 0
    capsys.readouterr()


def test_malformed_ledger_line_reports_exact_source_location(tmp_path: Path, capsys: object) -> None:
    _repository(tmp_path, hypotheses=b'{"id":"H-1","status":"open"}\n{oops}\n')
    assert research_tree.main(_args(tmp_path, "build")) == 1
    assert "research/hypotheses.jsonl:2" in capsys.readouterr().err
    assert not _generated(tmp_path, "tree.json").exists()


def test_config_cannot_grant_launch_authority(tmp_path: Path, capsys: object) -> None:
    config_path = _repository(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["curated_nodes"][0]["allocation"]["launch_authorized"] = True
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    assert research_tree.main(_args(tmp_path, "build")) == 1
    assert "cannot authorize a launch" in capsys.readouterr().err
    assert not _generated(tmp_path, "tree.json").exists()
