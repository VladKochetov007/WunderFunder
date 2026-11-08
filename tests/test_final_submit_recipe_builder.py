from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import wnn.final_submit_manifest as final_manifest
from scripts.final_submit_recipe_builder import (
    FINAL_FIT_SCHEMA,
    FINAL_RUN_PREFIX,
    TOTAL_SEQUENCES,
    build_final_fit_membership,
    materialize_plan,
    validate_final_fit_membership,
)
from scripts.tl3_chronological_recipes import FINAL_WEIGHTS, W18_COEFFICIENTS, build_nodes
from wnn.data import FEATURE_COLUMNS, TARGET_COLUMNS
from wnn.training.cli import build_parser

TRAIN_SHA256 = "a" * 64
VALID_SHA256 = "b" * 64
VALIDATION_SHA256 = "c" * 64


def _materialize_plan(tmp_path: Path) -> tuple[dict[str, object], Path, str]:
    """Build a plan against a local receipt fixture and return its pinned path/hash."""
    packet = tmp_path / "packet"
    packet.mkdir()
    receipt_path = packet / "SOURCE_VALIDATION_R1.json"
    receipt_path.write_bytes(b"source-validation receipt fixture")
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    plan = materialize_plan(
        packet / "final-fit-membership.json",
        packet / "runs",
        TRAIN_SHA256,
        VALID_SHA256,
        receipt_sha256,
    )
    return plan, receipt_path, receipt_sha256


def test_final_fit_membership_covers_all_source_row_groups_and_forbids_scoring() -> None:
    manifest = build_final_fit_membership(TRAIN_SHA256, VALID_SHA256, VALIDATION_SHA256)

    assert manifest["schema"] == FINAL_FIT_SCHEMA
    assert manifest["fit_sequences"] == TOTAL_SEQUENCES == 12480
    assert manifest["fit_members"] == {
        "train": list(range(10607)),
        "valid": list(range(1873)),
    }
    assert manifest["independent_evaluation_members"] == []
    assert manifest["independent_score_claim"] == "forbidden"
    assert manifest["post_fit_primary_valid_scoring"] == "forbidden"
    assert manifest["source_validation"]["checked_member"] == ["train", 528]
    assert (
        validate_final_fit_membership(manifest, TRAIN_SHA256, VALID_SHA256, VALIDATION_SHA256)
        == manifest
    )


def test_final_fit_schema_rejects_train_only_or_campaign_manifest_identity() -> None:
    manifest = build_final_fit_membership(TRAIN_SHA256, VALID_SHA256, VALIDATION_SHA256)
    altered = {**manifest, "fit_sequences": 10607}
    with pytest.raises(ValueError, match="frozen all-labeled-data"):
        validate_final_fit_membership(altered, TRAIN_SHA256, VALID_SHA256, VALIDATION_SHA256)
    altered = {**manifest, "chronological_primary_split_sha256": "0" * 64}
    with pytest.raises(ValueError, match="frozen all-labeled-data"):
        validate_final_fit_membership(altered, TRAIN_SHA256, VALID_SHA256, VALIDATION_SHA256)
    with pytest.raises(ValueError, match="source identities"):
        validate_final_fit_membership(manifest, "d" * 64, VALID_SHA256, VALIDATION_SHA256)
    altered = {**manifest, "source_validation": {**manifest["source_validation"], "rows": 19999}}
    with pytest.raises(ValueError, match="frozen all-labeled-data"):
        validate_final_fit_membership(altered, TRAIN_SHA256, VALID_SHA256, VALIDATION_SHA256)


def test_final_plan_preserves_all_seventeen_frozen_nodes_and_fixed_ancestry(tmp_path: Path) -> None:
    plan, _, _ = _materialize_plan(tmp_path)
    run_root = tmp_path / "packet" / "runs"
    historical = build_nodes(Path("historical-primary-manifest.json"), run_root)

    assert plan["schema"] == "wnn.final-submit-recipes.v1"
    assert plan["fit_sequences"] == 12480
    assert plan["configured_epochs"] == 54
    assert plan["expected_total_updates"] == 312000
    assert len(plan["nodes"]) == len(historical) == 17
    assert list(plan["final_members"].values()) == list(FINAL_WEIGHTS.values())
    assert [node["id"] for node in plan["nodes"]] == [node["id"] for node in historical]
    assert [node["parents"] for node in plan["nodes"]] == [node["parents"] for node in historical]
    assert plan["nodes"][-1]["parents"] == list(W18_COEFFICIENTS)
    assert len({node["run"] for node in plan["nodes"]}) == 17
    assert all(str(node["run"]).startswith(FINAL_RUN_PREFIX) for node in plan["nodes"])
    assert not set(plan["final_members"]) & {node["run"] for node in historical}


def test_final_plan_uses_the_new_membership_policy_without_scoring_or_old_cache_reuse(
    tmp_path: Path,
) -> None:
    plan, receipt_path, receipt_sha256 = _materialize_plan(tmp_path)
    run_root = tmp_path / "packet" / "runs"
    w18 = plan["nodes"][-1]
    parser = build_parser()

    assert plan["independent_evaluation_sequences"] == 0
    assert plan["independent_score_claim"] == "forbidden"
    assert plan["status"] == "trainer_bridge_ready"
    assert plan["fit_started"] is False
    assert plan["source_validation_receipt"] == {
        "path": str(receipt_path),
        "sha256": receipt_sha256,
    }
    assert isinstance(plan["plan_sha256"], str)
    for node in plan["nodes"]:
        args = list(node["args"])
        assert "--final-fit-manifest" in args
        assert "--final-fit-source-validation-sha256" in args
        assert "--final-fit-source-validation-receipt" in args
        assert args[args.index("--final-fit-source-validation-receipt") + 1] == str(receipt_path)
        for option in (
            "--final-fit-manifest",
            "--final-fit-source-validation-sha256",
            "--final-fit-source-validation-receipt",
        ):
            extension_index = args.index(option)
            del args[extension_index : extension_index + 2]
        parsed = parser.parse_args(args)
        assert parsed.train_source == "train+valid"
        assert parsed.train_seqs == 10607
        assert parsed.include_heldout_valid
        assert parsed.no_eval_during_fit
        assert parsed.checkpoint_policy == "fixed"
        assert parsed.checkpoint_epoch == node["endpoint"]
        assert parsed.propensity == ""
        assert not parsed.prop_input and not parsed.mask_weight and not parsed.prop_cache
        assert "--split-manifest" not in args
        assert "--eval-source" not in args
        assert "--full-valid" not in args
        assert "--train-eval-seqs" not in args
        assert all("tl3c_" not in token for token in args)
    assert w18["distillation_cache_root"] == str(run_root / "final-fit-w18-mixed-targets")
    assert w18["args"][w18["args"].index("--distill-cache-root") + 1] == str(
        run_root / "final-fit-w18-mixed-targets"
    )
    receipt_path.write_bytes(b"changed receipt fixture")
    with pytest.raises(ValueError, match="receipt bytes"):
        materialize_plan(
            receipt_path.parent / "final-fit-membership.json",
            run_root,
            TRAIN_SHA256,
            VALID_SHA256,
            receipt_sha256,
        )


def test_source4_candidate_waits_for_explicit_compatibility_receipt(tmp_path: Path) -> None:
    plan, _, _ = _materialize_plan(tmp_path)

    assert plan["optional_source4_lstm"]["included"] is False
    assert "compatibility receipt" in plan["optional_source4_lstm"]["reason"]


def test_manifest_loader_binds_live_parquets_and_complete_whole_sequence_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    (data_root / "datasets").mkdir(parents=True)
    train_path = data_root / "datasets" / "train.parquet"
    valid_path = data_root / "datasets" / "valid.parquet"
    train_path.write_bytes(b"train parquet identity")
    valid_path.write_bytes(b"valid parquet identity")
    train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
    valid_sha = hashlib.sha256(valid_path.read_bytes()).hexdigest()
    manifest = build_final_fit_membership(train_sha, valid_sha, VALIDATION_SHA256)
    manifest_path = tmp_path / "membership.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class FakeMetadata:
        def __init__(self, count: int) -> None:
            self.num_row_groups = count

        def row_group(self, group: int) -> SimpleNamespace:
            assert 0 <= group < self.num_row_groups
            return SimpleNamespace(num_rows=20_000)

    class FakeParquetFile:
        def __init__(self, path: Path) -> None:
            count = 10_607 if path.name == "train.parquet" else 1_873
            self.metadata = FakeMetadata(count)
            self.schema_arrow = SimpleNamespace(names=[*FEATURE_COLUMNS, *TARGET_COLUMNS])

    monkeypatch.setattr(final_manifest.pq, "ParquetFile", FakeParquetFile)
    loaded = final_manifest.load_final_fit_membership(manifest_path, data_root, VALIDATION_SHA256)

    assert len(loaded.train_groups) == 10_607
    assert loaded.train_groups[528] == 528
    assert len(loaded.valid_groups) == 1_873
    assert loaded.train_sha256 == train_sha
    assert loaded.valid_sha256 == valid_sha
    assert not hasattr(loaded, "evaluation_groups")
    with pytest.raises(ValueError, match="source-validation receipt"):
        final_manifest.load_final_fit_membership(manifest_path, data_root, "d" * 64)
    valid_path.write_bytes(b"changed valid source")
    with pytest.raises(ValueError, match="source SHA256 changed"):
        final_manifest.load_final_fit_membership(manifest_path, data_root, VALIDATION_SHA256)
