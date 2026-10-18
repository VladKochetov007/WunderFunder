"""CPU-only adapter checks; these never import the trainer or CUDA runtime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from adapter_common import (
    THREAD_ENVIRONMENT,
    apply_thread_environment,
    verify_deployed_tree_manifest,
    verify_source_manifest,
)
from fit_receipt import extract_fit_receipt
from fit_recipe import build_training_argv
from fit_workspace import prepare_export_workspace, prepare_workspace


def _builder() -> dict:
    return json.loads((Path(__file__).parent / "command_builder.json").read_text())


def test_command_builder_freezes_both_shapes_and_fit_policy(tmp_path) -> None:
    split = tmp_path / "fold0.json"
    split.write_text("{}", encoding="utf-8")
    recipe = _builder()

    wide = build_training_argv(
        recipe,
        arm="wide",
        fold=0,
        seed=1002,
        run_name="tabdata_wide_f0_s1002",
        split_manifest=split,
        mode="fit",
    )
    deep = build_training_argv(
        recipe,
        arm="deep",
        fold=3,
        seed=1002,
        run_name="tabdata_deep_f3_s1002",
        split_manifest=split,
        mode="fit",
    )
    assert wide[wide.index("--hidden") + 1] == "512"
    assert wide[wide.index("--layers") + 1] == "1"
    assert deep[deep.index("--hidden") + 1] == "384"
    assert deep[deep.index("--layers") + 1] == "2"
    for argv in (wide, deep):
        assert argv[argv.index("--batch") + 1] == "8"
        assert argv[argv.index("--chunk") + 1] == "2000"
        assert argv[argv.index("--epochs") + 1] == "6"
        assert argv[argv.index("--checkpoint-epoch") + 1] == "5"
        assert "--no-eval-during-fit" in argv
        assert "--precision" in argv and argv[argv.index("--precision") + 1] == "fp32"
    with pytest.raises(ValueError, match="seed 1001 is excluded"):
        build_training_argv(
            recipe,
            arm="wide",
            fold=0,
            seed=1001,
            run_name="owned-cell",
            split_manifest=split,
            mode="fit",
        )


def test_profile_and_fit_modes_add_only_expected_flags(tmp_path) -> None:
    split = tmp_path / "fold1.json"
    split.write_text("{}", encoding="utf-8")
    recipe = _builder()
    common = {
        "recipe": recipe,
        "arm": "wide",
        "fold": 1,
        "seed": 1002,
        "run_name": "profile",
        "split_manifest": split,
    }
    profile = build_training_argv(**common, mode="profile")
    fit = build_training_argv(**common, mode="fit")
    assert profile[-4:] == ["--profile-steps", "3", "--profile-warmup", "1"]
    assert "--save-training-state" not in profile
    assert fit[-1] == "--save-training-state"
    assert "--profile-steps" not in fit


def test_workspace_keeps_outputs_local_and_links_inputs_without_copying(tmp_path) -> None:
    data = tmp_path / "shared-data"
    (data / "datasets").mkdir(parents=True)
    (data / "cache").mkdir()
    (data / "datasets" / "train.parquet").write_bytes(b"train-input")
    (data / "datasets" / "valid.parquet").write_bytes(b"valid-input")
    for name in ("chains.npz", "chain_folds.npz", "train_targets.npz", "valid_targets.npz"):
        (data / "cache" / name).write_bytes(name.encode())
    split = tmp_path / "fold0.json"
    split.write_text(
        json.dumps(
            {
                "fold": 0,
                "n_folds": 4,
                "fit": {"train": [1], "valid": [2]},
                "evaluation": {"valid": [3]},
                "sources": {},
            }
        ),
        encoding="utf-8",
    )
    run = tmp_path / "owned-run"
    run.mkdir()

    try:
        receipt = prepare_workspace(run, data, split, 0)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("directory symlinks require Windows Developer Mode or privilege")
        raise

    assert (run / "datasets").is_symlink()
    assert not (run / "cache").is_symlink()
    assert (run / "cache" / "chains.npz").is_symlink()
    assert receipt["workspace_links"]["cache_is_local_overlay"] is True
    assert not (data / "cache" / "pred_test_valid.npy").exists()


def test_export_retry_workspace_links_immutable_fit_without_replacing_it(tmp_path) -> None:
    fit = tmp_path / "fit-attempt"
    (fit / "artifacts").mkdir(parents=True)
    (fit / "datasets").mkdir()
    (fit / "cache").mkdir()
    checkpoint = fit / "artifacts" / "model_ep5.pt"
    checkpoint.write_bytes(b"frozen-epoch-5")
    dataset = fit / "datasets" / "valid.parquet"
    dataset.write_bytes(b"feature-data")
    for name in ("chains.npz", "chain_folds.npz", "train_targets.npz", "valid_targets.npz"):
        (fit / "cache" / name).write_bytes(name.encode())
    export = tmp_path / "export-retry-1"

    try:
        receipt = prepare_export_workspace(export, fit)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("directory symlinks require Windows Developer Mode or privilege")
        raise

    assert receipt["cache_is_local_overlay"] is True
    assert (export / "artifacts").is_symlink()
    assert (export / "datasets").is_symlink()
    assert not (export / "cache").is_symlink()
    assert (export / "artifacts" / "model_ep5.pt").read_bytes() == b"frozen-epoch-5"
    assert (fit / "artifacts" / "model_ep5.pt").read_bytes() == b"frozen-epoch-5"
    with pytest.raises(FileExistsError):
        prepare_export_workspace(export, fit)


def test_source_manifest_is_content_addressed_and_does_not_need_git(tmp_path) -> None:
    source = tmp_path / "archive"
    (source / "scripts").mkdir(parents=True)
    (source / "src" / "wnn" / "training").mkdir(parents=True)
    files = {
        "scripts/train_rnn.py": b"cli",
        "pyproject.toml": b"project",
        "uv.lock": b"lock",
        "src/wnn/training/runner.py": b"runner",
    }
    for relative, data in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    import hashlib

    hashes = {path: hashlib.sha256(data).hexdigest() for path, data in files.items()}
    aggregate = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema": "tabdata-l4-source-manifest.v1",
        "source_revision": "a266",
        "files": hashes,
        "source_identity_sha256": aggregate,
    }

    receipt = verify_source_manifest(source, manifest)
    assert receipt["source_identity_sha256"] == aggregate
    assert receipt["file_count"] == 4
    (source / "src" / "wnn" / "training" / "runner.py").write_bytes(b"edited")
    with pytest.raises(ValueError, match="changed=.*runner.py"):
        verify_source_manifest(source, manifest)


def test_deployed_tree_manifest_rejects_changed_or_added_dependencies(tmp_path) -> None:
    import hashlib

    source = tmp_path / "archive"
    payloads = {
        "src/wnn/nn/model.py": b"model-v1",
        "src/wnn/features/sequence.py": b"features-v1",
        "src/wnn/data.py": b"data-v1",
        "src/wnn/optim.py": b"optim-v1",
        "src/wnn/splits.py": b"splits-v1",
        "src/wnn/resources.py": b"resources-v1",
        "scripts/train_rnn.py": b"trainer-v1",
    }
    for relative, payload in payloads.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    aggregate = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema": "tabdata-l4-deployed-tree-manifest.v1",
        "roots": ["src", "scripts", "configs"],
        "source_revision": "a266",
        "archive_sha256": "a" * 64,
        "file_count": len(hashes),
        "files": hashes,
        "deployed_tree_identity_sha256": aggregate,
    }

    receipt = verify_deployed_tree_manifest(source, manifest, manifest_sha256="b" * 64)
    assert receipt["deployed_tree_identity_sha256"] == aggregate
    assert receipt["file_count"] == len(payloads)

    (source / "src/wnn/data.py").write_bytes(b"changed dependency")
    with pytest.raises(ValueError, match="changed=.*src/wnn/data.py"):
        verify_deployed_tree_manifest(source, manifest, manifest_sha256="b" * 64)

    (source / "src/wnn/data.py").write_bytes(payloads["src/wnn/data.py"])
    (source / "src/wnn/nn/new_dependency.py").write_bytes(b"unmanifested")
    with pytest.raises(ValueError, match="extra=.*src/wnn/nn/new_dependency.py"):
        verify_deployed_tree_manifest(source, manifest, manifest_sha256="b" * 64)


def test_thread_environment_sets_explicit_two_thread_caps(monkeypatch) -> None:
    for name in THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "192")
    assert apply_thread_environment() == THREAD_ENVIRONMENT
    assert {name: __import__("os").environ[name] for name in THREAD_ENVIRONMENT} == {
        name: "2" for name in THREAD_ENVIRONMENT
    }


def test_fit_receipt_requires_all_fixed_checkpoints_and_records_ancestry(tmp_path) -> None:
    run = tmp_path / "run"
    output = run / "artifacts" / "runs" / "tabdata_wide_f0_s1002"
    output.mkdir(parents=True)
    args = {
        "name": "tabdata_wide_f0_s1002",
        "valid_fold": 0,
        "seed": 1002,
        "epochs": 6,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 5,
        "no_eval_during_fit": True,
        "precision": "fp32",
        "hidden": 512,
        "layers": 1,
        "batch": 8,
        "chunk": 2000,
        "loss": "wmse",
        "loss_norm": "chunk",
        "batch_order": "seeded",
        "sched": "onecycle",
        "lr": 0.0001,
        "beta2": 0.999,
        "wd": 0.01,
        "head_lr_mult": 1.0,
        "clip": 1.0,
        "sync_free": True,
        "fused_adam": True,
        "deterministic": True,
        "init": "",
        "propensity": "",
        "prop_input": "",
        "distill_cache_root": None,
        "distill": None,
    }
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "args": args,
                "split_manifest_sha256": "a" * 64,
                "selected_epoch": 5,
                "fit_row_groups": {"train": [1, 2], "valid": [3]},
                "trainer_implementation": {
                    "sha256": "b" * 64,
                    "files": {"scripts/train_rnn.py": "c" * 64},
                },
            }
        ),
        encoding="utf-8",
    )
    (output / "history.json").write_text(
        json.dumps(
            [
                {
                    "evaluation": "withheld",
                    "update_diagnostics": {"epoch": {"optimizer_completed_updates": 12}},
                }
                for _ in range(6)
            ]
        ),
        encoding="utf-8",
    )
    output_names = [f"model_ep{epoch}.pt" for epoch in range(6)] + [
        "model.pt",
        "training_state.pt",
    ]
    for name in output_names:
        (output / name).write_bytes(name.encode())

    receipt = extract_fit_receipt(
        run,
        output,
        run_name="tabdata_wide_f0_s1002",
        arm="wide",
        fold=0,
        seed=1002,
        input_receipt={
            "split_manifest_sha256": "a" * 64,
            "fit_train_groups": [1, 2],
            "fit_valid_groups": [3],
            "trainer_source_identity": {
                "source_identity_sha256": "b" * 64,
                "files": {"scripts/train_rnn.py": "c" * 64},
            },
        },
        adapter_bundle_identity={"sha256": "d" * 64, "files": {}},
    )

    assert receipt["checkpoint_count"] == 7
    assert len(receipt["outputs"]) == 10
    assert receipt["fit_group_counts"] == {"train": 2, "valid": 1}
    assert receipt["selected_epoch"] == 5
