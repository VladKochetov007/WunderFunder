"""CPU-only tests for the frozen validation exporter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from adapter_common import adapter_bundle_identity
from export_cell import _DEFAULT_GRADER, _check_fit_binding, predictor_argv


def _builder() -> dict:
    return json.loads((Path(__file__).parent / "command_builder.json").read_text())


def test_bundled_source_manifest_matches_command_builder_pin() -> None:
    code_dir = Path(__file__).parent
    recipe = _builder()
    manifest_path = code_dir / recipe["source_manifest"]
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert hashlib.sha256(manifest_bytes).hexdigest() == recipe["source_manifest_sha256"]
    assert manifest["source_identity_sha256"] == recipe["expected_source_identity_sha256"]
    assert hashlib.sha256(_DEFAULT_GRADER.read_bytes()).hexdigest() == recipe["r4_grader_sha256"]
    tree_path = code_dir / recipe["source_tree_manifest"]
    tree_bytes = tree_path.read_bytes()
    tree_manifest = json.loads(tree_bytes)
    assert hashlib.sha256(tree_bytes).hexdigest() == recipe["source_tree_manifest_sha256"]
    assert tree_manifest["archive_sha256"] == recipe["source_tree_archive_sha256"]
    assert (
        tree_manifest["deployed_tree_identity_sha256"]
        == recipe["expected_source_tree_identity_sha256"]
    )
    assert tree_manifest["file_count"] == 533


def test_adapter_bundle_hashes_all_producer_helper_files() -> None:
    identity = adapter_bundle_identity()
    assert identity["schema"] == "tabdata-l4-adapter-bundle.v1"
    assert set(identity["files"]) == set(_builder()["adapter_bundle_files"])
    assert identity["sha256"]
    assert (
        identity["source_tree_identity_sha256"]
        == _builder()["expected_source_tree_identity_sha256"]
    )


def test_export_invokes_frozen_predictor_at_epoch5_eval_only_fp32() -> None:
    script = Path("/frozen/scripts/predict_run.py")
    split = Path("/splits/fold0.json")
    argv = predictor_argv(script, "tabdata_wide_f0_s1002", split)
    assert argv == [
        str(script),
        "tabdata_wide_f0_s1002",
        "valid",
        "--epoch",
        "5",
        "--campaign-split-manifest",
        str(split),
        "--campaign-evaluation-only",
        "--device",
        "cuda",
        "--inference-precision",
        "fp32",
    ]


def test_export_binding_rejects_changed_fit_or_evaluation_membership(tmp_path) -> None:
    run_dir = tmp_path / "attempt"
    run_dir.mkdir()
    run_name = "tabdata_wide_f0_s1002"
    split = tmp_path / "fold0.json"
    groups = [4, 8]
    split.write_text(
        json.dumps(
            {
                "fold": 0,
                "n_folds": 4,
                "evaluation": {"valid": groups},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = run_dir / "artifacts" / "runs" / run_name / "model_ep5.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixed-epoch-5")
    split_sha = hashlib.sha256(split.read_bytes()).hexdigest()
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ancestry = {"split_manifest_sha256": split_sha, "fold": 0, "evaluation_groups": groups}
    (run_dir / "producer_receipt.json").write_text(
        json.dumps(
            {
                "schema": "tabdata-l4-fit-producer.v1",
                "status": "completed_fixed_epoch5",
                "run_name": run_name,
                "fold": 0,
                "seed": 1002,
                "selected_epoch": 5,
                "input_ancestry": ancestry,
                "outputs": {f"artifacts/runs/{run_name}/model_ep5.pt": {"sha256": checkpoint_sha}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "command.json").write_text(
        json.dumps({"mode": "fit", "run_name": run_name, "fold": 0, "seed": 1002}),
        encoding="utf-8",
    )
    (run_dir / "adapter_preflight.json").write_text(
        json.dumps({"split_manifest_sha256": split_sha, "fold": 0, "evaluation_groups": groups}),
        encoding="utf-8",
    )

    receipt, _, selected_groups = _check_fit_binding(run_dir, split, run_name, 0, 1002)
    assert receipt["selected_epoch"] == 5
    assert selected_groups == groups
    split.write_text(
        json.dumps({"fold": 0, "n_folds": 4, "evaluation": {"valid": list(reversed(groups))}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split manifest"):
        _check_fit_binding(run_dir, split, run_name, 0, 1002)


def test_r4_grader_manifest_contract_accepts_ordered_feature_only_export() -> None:
    import ast
    from collections.abc import Mapping, Sequence

    import numpy as np

    module = ast.parse(_DEFAULT_GRADER.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"membership_sha256", "validate_export_manifest"}
    }
    namespace = {"hashlib": hashlib, "np": np, "Mapping": Mapping, "Sequence": Sequence}
    exec(
        compile(
            ast.Module(
                body=[functions["membership_sha256"], functions["validate_export_manifest"]],
                type_ignores=[],
            ),
            str(_DEFAULT_GRADER),
            "exec",
        ),
        namespace,
    )
    validate_export_manifest = namespace["validate_export_manifest"]
    groups = [4, 8, 9]
    membership = hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
    manifest = {
        "run": "tabdata_wide_f0_s1002",
        "split": "valid",
        "epoch": 5,
        "split_manifest_sha256": "a" * 64,
        "campaign_split_manifest_sha256": "a" * 64,
        "campaign_export_scope": "evaluation",
        "feature_only": True,
        "inference_precision": "fp32",
        "row_groups": groups,
        "evaluation_membership_sha256": membership,
    }
    validate_export_manifest(
        manifest,
        run="tabdata_wide_f0_s1002",
        epoch=5,
        split_sha256="a" * 64,
        row_groups=groups,
    )
    with pytest.raises(ValueError, match="row_groups"):
        validate_export_manifest(
            manifest,
            run="tabdata_wide_f0_s1002",
            epoch=5,
            split_sha256="a" * 64,
            row_groups=list(reversed(groups)),
        )
