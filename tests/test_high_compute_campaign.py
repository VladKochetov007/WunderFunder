from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/experiment_drafts/b300-transformer-distillation-runtime/high_compute_r4"


def _packet_modules() -> dict[str, Any]:
    """Import metadata-only campaign modules from the integrated packet."""
    if str(PACKET) not in sys.path:
        sys.path.insert(0, str(PACKET))
    names = (
        "campaign_io",
        "source_bindings",
        "catalog",
        "failure_receipts",
        "endpoints",
        "materialize",
        "stage_inner",
        "stage_outer",
    )
    return {name: importlib.import_module(name) for name in names}


@pytest.fixture
def packet() -> dict[str, Any]:
    return _packet_modules()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_split_pair(packet: dict[str, Any], directory: Path, fold: int) -> tuple[Path, Path]:
    """Make a small valid source-qualified outer/inner pair with component isolation."""
    campaign_io = packet["campaign_io"]
    train_groups, valid_groups, outer_eval = [1, 2, 3], [10, 11, 12], [20]
    sequences = {
        "train": [{"row_group": group, "component": 100 + group} for group in train_groups],
        "valid": [
            {"row_group": group, "component": 200 + group} for group in valid_groups + outer_eval
        ],
    }
    sources = [
        {"source": "train", "sha256": "a" * 64},
        {"source": "valid", "sha256": "b" * 64},
    ]
    outer_path = directory / f"outer_f{fold}.json"
    outer = {
        "schema": "wnn.sequence-campaign-split.v1",
        "version": 1,
        "fold": fold,
        "n_folds": 3,
        "sources": sources,
        "sequences": sequences,
        "fit": {"train": train_groups, "valid": valid_groups},
        "evaluation": {"valid": outer_eval},
    }
    _write_json(outer_path, outer)
    outer_hash = campaign_io.sha256_file(outer_path)

    seed = 773
    components = [200 + group for group in valid_groups]
    ordered = sorted(
        components,
        key=lambda component: (
            hashlib.sha256(f"wnn.sequence-inner.v1:{seed}:{fold}:{component}".encode()).hexdigest(),
            component,
        ),
    )
    chosen = {ordered[0]}
    evaluation_valid = [group for group in valid_groups if 200 + group in chosen]
    fit_valid = [group for group in valid_groups if group not in evaluation_valid]
    inner_path = directory / f"inner_f{fold}.json"
    inner = {
        "schema": "wnn.sequence-inner-split.v1",
        "version": 1,
        "fold": fold,
        "n_folds": 3,
        "parent": {"path": str(outer_path.resolve()), "sha256": outer_hash},
        "sources": sources,
        "sequences": sequences,
        "selection": {
            "algorithm": "sha256-component-order-v1",
            "seed": seed,
            "evaluation_fraction": 0.1,
        },
        "fit": {"train": train_groups, "valid": fit_valid},
        "evaluation": {"valid": evaluation_valid},
    }
    _write_json(inner_path, inner)
    return outer_path, inner_path


@pytest.fixture
def campaign_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packet: dict[str, Any]
) -> dict[str, Any]:
    """Create an isolated campaign using small source inventories and split files."""
    source_bindings = packet["source_bindings"]
    inventories = {
        "runtime": {"fixture/runtime.py": "1" * 64},
        "packet": {"fixture/packet.json": "2" * 64},
        "r3": {"fixture/r3.json": "3" * 64},
    }
    monkeypatch.setattr(source_bindings, "runtime_source_files", lambda: inventories["runtime"])
    monkeypatch.setattr(source_bindings, "packet_source_files", lambda: inventories["packet"])
    monkeypatch.setattr(source_bindings, "r3_source_files", lambda: inventories["r3"])

    data_root = tmp_path / "data"
    data_root.mkdir()
    split_root = tmp_path / "splits"
    splits = {fold: _write_split_pair(packet, split_root, fold) for fold in (0, 2)}
    root = tmp_path / "campaign"
    campaign = packet["campaign_io"].create_campaign(
        root,
        data_root,
        {fold: pair[0] for fold, pair in splits.items()},
        {fold: pair[1] for fold, pair in splits.items()},
    )
    (root / "cells").mkdir()
    (root / "stages").mkdir()
    return {"root": root, "campaign": campaign, "splits": splits}


def test_b_materialization_profiles_hashes_and_ineligible_failure(
    campaign_case: dict[str, Any], packet: dict[str, Any]
) -> None:
    """B writes six paired configs and rejects drift or a frozen failed cell."""
    root = campaign_case["root"]
    materialize = packet["materialize"]
    campaign_io = packet["campaign_io"]
    failure_receipts = packet["failure_receipts"]
    prepared = materialize.prepare_stage(root, "B")
    loaded = materialize.load_stage(root, "B")

    assert prepared["accounting"] == {
        "new_teacher_fits": 6,
        "reused_teacher_fits": 0,
        "new_student_fits": 0,
    }
    assert len(loaded["cells"]) == 6
    assert {(cell["fold"], cell["seed"]) for cell in loaded["cells"]} == {
        (0, 201001),
        (2, 201002),
    }
    assert {cell["variant"] for cell in loaded["cells"]} == {
        "lr_1e-4",
        "lr_3e-4",
        "lr_1e-3",
    }
    for cell in loaded["cells"]:
        config = campaign_io.read_object(root / cell["config"])
        one = campaign_io.read_object(root / cell["profile_one"])
        peak = campaign_io.read_object(root / cell["profile_peak"])
        assert one == config | {"required_world_size": 1}
        assert peak == config | {"warmup_fraction": 0.0}
        assert config["epochs"] == 30 and config["optimizer"] == "muon"
        assert config["required_world_size"] == 8
        assert config["batch_size"] == 2
        assert config["precision_mode"] == "bf16_bf16x9"
        assert "precision_mode" not in config["model"]
        assert config["model"]["attention_backend"] == "efficient"
        assert config["model"]["compile_blocks"] is True
        assert config["model"]["checkpoint_blocks"] is False
        assert (
            packet["catalog"].normalized_runtime_config(config)["model"]["precision_mode"]
            == "bf16_bf16x9"
        )
        assert campaign_io.sha256_file(root / cell["profile_one"]) == cell["profile_one_sha256"]
        assert campaign_io.sha256_file(root / cell["profile_peak"]) == cell["profile_peak_sha256"]

    cell = loaded["cells"][0]
    run = root / cell["run_dir"]
    status = run / "fit" / "terminal_status.json"
    status.parent.mkdir(parents=True)
    _write_json(status, {"status": "completed"})
    evidence = root / "evaluation-failure.log"
    evidence.write_text("evaluation export failed\n", encoding="utf-8")
    campaign = campaign_io.load_campaign(root)
    receipt = failure_receipts.record_failure(
        root, campaign, cell, "evaluation export failed", [evidence], phase="evaluate"
    )
    assert receipt["selection_eligible"] is False
    assert failure_receipts.validate_failure_receipt(root, campaign, cell)["state"] == "failed"
    with pytest.raises(ValueError, match="terminal failure receipt"):
        materialize.get_cell(root, "B", cell["id"])

    score = run / "fit" / "score.json"
    _write_json(score, {"score": "must not coexist with evaluation failure"})
    with pytest.raises(ValueError, match="completed score"):
        failure_receipts.validate_failure_receipt(root, campaign, cell)

    profile = root / cell["profile_one"]
    profile.write_bytes(profile.read_bytes() + b" ")
    with pytest.raises(ValueError, match="profile config changed"):
        materialize.load_stage(root, "B")


def test_endpoint_accepts_loader_normalized_precision_identity(
    campaign_case: dict[str, Any], packet: dict[str, Any]
) -> None:
    """Endpoint provenance binds the loader's injected field while raw config stays unchanged."""
    root, campaign = campaign_case["root"], campaign_case["campaign"]
    cell = packet["materialize"].prepare_stage(root, "B")["cells"][0]
    config = packet["campaign_io"].read_object(root / cell["config"])
    runtime_config = packet["catalog"].normalized_runtime_config(config)
    split_path = Path(cell["split_manifest"])
    split = packet["campaign_io"].read_object(split_path)
    split_hash = cell["split_sha256"]
    fit_members = [["train", int(group)] for group in split["fit"]["train"]] + [
        ["valid", int(group)] for group in split["fit"]["valid"]
    ]
    run = root / cell["run_dir"]
    model_path = run / "fit" / "model.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"fixture checkpoint")
    weight_hash = packet["campaign_io"].sha256_file(model_path)
    identity = {
        "config_sha256": packet["campaign_io"].canonical_sha256(runtime_config),
        "source_sha256": campaign["runtime_source_sha256"],
        "world_size": config["required_world_size"],
        "data": {
            "fold": cell["fold"],
            "split_sha256": split_hash,
            "fit_members": fit_members,
        },
    }
    evaluation_path = run / "evaluation" / "manifest.json"
    evaluation_entries = [
        {"split": "valid", "row_group": int(group)} for group in split["evaluation"]["valid"]
    ]
    producer = {
        "checkpoint_sha256": weight_hash,
        "config": runtime_config,
        "provenance": {"identity": identity},
        "split_sha256": split_hash,
        "fit_members": fit_members,
    }
    _write_json(
        evaluation_path,
        {
            "schema": "sequence-predictions-v1",
            "complete": True,
            "membership": "evaluation",
            "split_sha256": split_hash,
            "producer": producer,
            "sequences": evaluation_entries,
        },
    )
    components = {
        str(int(item["component"]))
        for item in split["sequences"]["valid"]
        if int(item["row_group"]) in set(split["evaluation"]["valid"])
    }
    _write_json(
        run / "fit" / "terminal_status.json",
        {
            "status": "completed",
            "completed_epochs": config["epochs"],
            "identity": identity,
            "model_sha256": weight_hash,
        },
    )
    _write_json(
        run / "fit" / "score.json",
        {
            "score": {"metric_id": "global_wp_clip2_v1", "wp": 0.25},
            "producer": producer,
            "split_sha256": split_hash,
            "sequences": len(evaluation_entries),
            "predictions_manifest_sha256": packet["campaign_io"].sha256_file(evaluation_path),
            "component_centered_moments": {
                component: [[0.0] * 6, [0.0] * 6] for component in components
            },
        },
    )

    endpoint = packet["endpoints"].validate_completed_endpoint(root, campaign, cell)
    assert endpoint["state"] == "completed"
    assert endpoint["config_sha256"] == cell["config_sha256"]


def test_c_reuses_selected_endpoints_without_rewriting_artifact_bindings(
    campaign_case: dict[str, Any], packet: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """C's selected 5% arm reuses both exact B endpoint paths and identities."""
    root, campaign = campaign_case["root"], campaign_case["campaign"]
    b_stage = packet["materialize"].prepare_stage(root, "B")
    selected = {
        fold: next(
            cell
            for cell in b_stage["cells"]
            if cell["fold"] == fold and cell["variant"] == "lr_3e-4"
        )
        for fold in (0, 2)
    }
    dependencies = {"parent_stage": "B"}
    monkeypatch.setattr(
        packet["stage_inner"],
        "_load_selected_parent",
        lambda _root, stage: ({"variant": "lr_3e-4"}, selected, dependencies),
    )

    records, accounting, observed_dependencies = packet["stage_inner"].prepare_inner_stage(
        root, campaign, "C"
    )
    assert accounting["new_teacher_fits"] == 4
    assert accounting["reused_teacher_fits"] == 2
    assert observed_dependencies == dependencies
    reused = [record for record in records if record["reused"]]
    assert len(reused) == 2
    for record in reused:
        source = selected[record["fold"]]
        assert record["reuse_of"] == source["id"]
        for key in (
            "config",
            "profile_one",
            "profile_peak",
            "run_dir",
            "teacher_cache",
            "config_sha256",
            "config_file_sha256",
            "split_manifest",
            "split_manifest_sha256",
        ):
            assert record[key] == source[key]


def test_g_aliases_identical_reference_and_h_binds_ordered_teacher_caches(
    campaign_case: dict[str, Any], packet: dict[str, Any]
) -> None:
    """G collapses an identical candidate pair and H preserves three-seed ancestry order."""
    root, campaign = campaign_case["root"], campaign_case["campaign"]
    catalog, campaign_io = packet["catalog"], packet["campaign_io"]
    selected: dict[int, dict[str, Any]] = {}
    for fold, seed in ((0, 201001), (2, 201002)):
        config_path = root / "selected" / f"F{fold}.json"
        config = catalog.teacher_base(fold, seed)
        campaign_io.write_json_new(config_path, config)
        selected[fold] = {
            "id": f"selected_f{fold}",
            "variant": "selected",
            "fold": fold,
            "seed": seed,
            "config": config_path.relative_to(root).as_posix(),
        }

    g_records, g_accounting = packet["stage_outer"]._prepare_g(root, campaign, selected)
    assert g_accounting["candidate_aliases_reference"] is True
    assert g_accounting["new_teacher_fits"] == 6
    assert g_accounting["reused_teacher_fits"] == 6
    candidates = [record for record in g_records if record["variant"] == "candidate"]
    assert len(candidates) == 6
    assert all(record["reused"] for record in candidates)
    assert all(record["run_dir"].startswith("runs/G_reference_") for record in candidates)

    h_records, h_accounting = packet["stage_outer"]._prepare_h(root, campaign, {"cells": g_records})
    assert h_accounting["new_student_fits"] == 6
    assert len(h_records) == 6
    for fold in (0, 2):
        mean = next(
            record
            for record in h_records
            if record["fold"] == fold and record["variant"] == "mean3_kd"
        )
        single = next(
            record
            for record in h_records
            if record["fold"] == fold and record["variant"] == "single_kd"
        )
        labels = next(
            record
            for record in h_records
            if record["fold"] == fold and record["variant"] == "labels"
        )
        student_config = campaign_io.read_object(root / labels["config"])
        assert student_config["precision_mode"] == "bf16_bf16x9"
        assert "precision_mode" not in student_config["model"]
        assert student_config["model"]["architecture"] == "gru"
        assert student_config["model"]["attention_backend"] == "math"
        assert student_config["model"]["compile_blocks"] is False
        assert (
            catalog.normalized_runtime_config(student_config)["model"]["precision_mode"]
            == "bf16_bf16x9"
        )
        assert [item["seed"] for item in mean["cache_producers"]] == [301001, 301002, 301003]
        assert mean["cache_inputs"] == [item["path"] for item in mean["cache_producers"]]
        assert mean["parent_run_dirs"] == [
            item["producer_run_dir"] for item in mean["cache_producers"]
        ]
        assert mean["teacher_cache"] == f"caches/g_candidate_f{fold}_mean"
        assert single["cache_inputs"] == [mean["cache_inputs"][0]]
        assert single["parent_run_dirs"] == [mean["parent_run_dirs"][0]]
        assert single["teacher_cache"] == single["cache_inputs"][0]
        assert labels["teacher_cache"] is None
        assert labels["cache_inputs"] == []
        assert {record["seed"] for record in (mean, single, labels)} == {
            401001 if fold == 0 else 401002
        }
