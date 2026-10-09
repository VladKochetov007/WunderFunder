from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

PACKET_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PACKET_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "scripts"))
sys.path.insert(0, str(CODE_ROOT / "src"))

import loss_pair_gpu_worker as worker  # noqa: E402
import train_rnn as trainer  # noqa: E402
from wnn.fold0_assets import verify_and_link_fold0_assets  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402
from wnn.splits import VerifiedParsedCacheSources, load_campaign_split  # noqa: E402
from wnn.training_sampling import EpochOrderSampler  # noqa: E402

SOURCE_STAGE_FIXTURE = Path(__file__).parent / "fixtures" / "parsed_cache_source_stage"
SPLIT_RELATIVE = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
REGISTRY_RELATIVE = Path("research/campaign_2026-10-04/portable_inputs.json")
ASSET_MANIFEST_RELATIVE = Path(
    "research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json"
)


def _cache_fixture() -> SimpleNamespace:
    fit_rows = [
        {
            "source": "datasets/train.parquet",
            "row_group": index,
            "sequence_id": f"train:{index:04d}",
        }
        for index in range(350)
    ]
    eval_rows = [
        {
            "source": "datasets/valid.parquet",
            "row_group": index,
            "sequence_id": f"valid:{index:04d}",
        }
        for index in range(468)
    ]
    return SimpleNamespace(
        membership={"memberships": {"fit_train": fit_rows, "evaluation_valid": eval_rows}},
        manifest={"membership_artifact": {"sha256": "a" * 64}},
    )


def test_actual_trainer_producers_write_tpu_contract_fixture(tmp_path, monkeypatch) -> None:
    arm_dir = tmp_path / "wmse"
    arm_dir.mkdir()
    monkeypatch.setenv("WNN_LOSS_PAIR_ARM_DIR", str(arm_dir))
    monkeypatch.setenv("WNN_LOSS_PAIR_RUN_NAME", "fixture_wmse")
    loader = DataLoader(
        TensorDataset(torch.zeros(350, 1)),
        batch_size=32,
        sampler=EpochOrderSampler(350, seed=20261009, batch_size=32, drop_last=False),
        drop_last=False,
    )
    model = RNNModel(211, hidden=8, layers=1, cell="gru", enc="linear")
    trainer.export_loss_pair_prefit_artifacts(model, loader, np.arange(350), _cache_fixture())
    prefit = arm_dir / "prefit"
    assert {path.name for path in prefit.iterdir()} == {
        "initial_state.npz",
        "normalization.npz",
        "normalization.json",
        "sample_orders.json",
        "prefit_receipt.json",
    }

    row_mask = torch.arange(2000)[None, :] >= 99
    trainer.export_loss_pair_profile_inputs(
        torch.zeros(32, 2000, 211),
        torch.zeros(32, 2000, 2),
        row_mask.expand(32, -1),
        arm_dir,
    )
    with np.load(prefit / "profile_inputs.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"inputs", "targets", "row_mask"}
        assert arrays["inputs"].shape == (32, 2000, 211)
        assert arrays["inputs"].dtype == np.dtype("<f4")
        assert arrays["targets"].shape == (32, 2000, 2)
        assert arrays["targets"].dtype == np.dtype("<f4")
        assert arrays["row_mask"].shape == (32, 2000)
        assert arrays["row_mask"].dtype == np.dtype("|b1")
        assert np.array_equal(
            arrays["row_mask"], np.broadcast_to(np.arange(2000) >= 99, (32, 2000))
        )
    orders = json.loads((prefit / "sample_orders.json").read_text(encoding="utf-8"))
    assert orders["schema"] == "wnn.loss-pair.sample-orders.v1"
    assert set(orders["epochs"]) == {"-1", "0", "1"}
    assert all(len(order) == 350 for order in orders["epochs"].values())
    assert len(orders["evaluation_ids"]) == 468


def test_gpu_arms_differ_only_by_loss_and_use_matched_fp32(tmp_path) -> None:
    validation = tmp_path / "validation.json"
    args = worker._common_train_argv(
        "pair", "wmse", tmp_path / "manifest.json", validation, "b" * 64
    )
    huber = worker._common_train_argv(
        "pair", "huber", tmp_path / "manifest.json", validation, "b" * 64
    )
    loss_index = args.index("--loss")
    assert args[loss_index + 1] == "wmse"
    assert huber[loss_index + 1] == "huber"
    args[loss_index + 1] = huber[loss_index + 1]
    assert args == huber
    assert args[args.index("--precision") + 1] == "fp32"
    assert "--save-training-state" in args


def test_actual_trainer_parser_accepts_disposable_profile_argv(
    tmp_path, monkeypatch
) -> None:
    command = worker._device_shape_profile_argv(
        "profile_parser_fixture",
        tmp_path / "manifest.json",
        tmp_path / "validation.json",
        "b" * 64,
    )
    assert "--save-training-state" not in command
    assert command[-4:] == ["--profile-steps", "2", "--profile-warmup", "0"]

    class ReachedPinnedCacheOpen(Exception):
        pass

    def stop_before_cache_io(*_args, **_kwargs):
        raise ReachedPinnedCacheOpen

    monkeypatch.setattr(trainer.ParsedSequenceCache, "open", stop_before_cache_io)
    monkeypatch.setattr(sys, "argv", [command[1], *command[2:]])
    with pytest.raises(ReachedPinnedCacheOpen):
        trainer.main()


def test_actual_split_loader_consumes_verified_mounted_auxiliary_assets(
    tmp_path, monkeypatch
) -> None:
    worktree_root = PACKET_ROOT.parents[5]
    main_root = worktree_root.parents[1] / "wnn_connectome_starterpack"
    manifest_path = SOURCE_STAGE_FIXTURE / "manifest.json"
    validation_path = SOURCE_STAGE_FIXTURE / "reader_validation.json"
    source_receipt_path = SOURCE_STAGE_FIXTURE / "source_stage_receipt.json"
    source_verification_path = SOURCE_STAGE_FIXTURE / "source_verification.json"
    manifest_raw = manifest_path.read_bytes()
    validation_raw = validation_path.read_bytes()
    source_verification_raw = source_verification_path.read_bytes()
    source_stage_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads(manifest_raw)
    validation_receipt = json.loads(validation_raw)
    source_verification = json.loads(source_verification_raw)

    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    validation_sha = hashlib.sha256(validation_raw).hexdigest()
    source_verification_sha = hashlib.sha256(source_verification_raw).hexdigest()
    assert source_stage_receipt["state"] == "complete"
    assert source_stage_receipt["cache_manifest_sha256"] == manifest_sha
    assert source_stage_receipt["reader_validation_sha256"] == validation_sha
    assert source_stage_receipt["raw_source_verification_sha256"] == source_verification_sha
    assert cache_manifest["source_verification_artifact"]["sha256"] == source_verification_sha
    assert validation_receipt["manifest_sha256"] == manifest_sha
    assert source_verification["split_manifest_sha256"] == cache_manifest["split_manifest_sha256"]

    parsed_sources = VerifiedParsedCacheSources(
        split_manifest_sha256=cache_manifest["split_manifest_sha256"],
        cache_manifest_sha256=manifest_sha,
        validation_receipt_sha256=validation_sha,
        source_verification_sha256=source_verification_sha,
        sources={
            source["logical_name"]: {
                "bytes": int(source["bytes"]),
                "sha256": source["sha256"],
            }
            for source in cache_manifest["sources"].values()
        },
    )

    dataset_mount = tmp_path / "input" / "datasets" / "owner" / "fold0-assets"
    dataset_mount.mkdir(parents=True)
    real_manifest = main_root / ASSET_MANIFEST_RELATIVE
    (dataset_mount / "assets-manifest.json").symlink_to(real_manifest)
    asset_manifest = json.loads(real_manifest.read_text(encoding="utf-8"))
    required_assets = {
        "cache/chain_folds.npz",
        "cache/chains.npz",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
    }
    for row in asset_manifest["files"]:
        if row["path"] in required_assets:
            (dataset_mount / row["filename"]).symlink_to(main_root / row["path"])

    source_root = tmp_path / "source"
    cache_root = source_root / "cache"
    registry_path = CODE_ROOT / REGISTRY_RELATIVE
    split_path = CODE_ROOT / SPLIT_RELATIVE
    receipt_path = tmp_path / "fold0_assets_validation.json"
    receipt = verify_and_link_fold0_assets(
        dataset_mount / "assets-manifest.json",
        registry_path,
        split_path,
        cache_root,
        receipt_path,
    )
    assert receipt["telemetry"]["total_bytes_hashed"] == 2_247_722_202
    assert receipt["telemetry"]["fadvise_dontneed_advised_bytes"] <= 2_247_722_202
    assert set(receipt["files"]) == required_assets
    assert all((cache_root / Path(name).name).is_symlink() for name in required_assets)

    monkeypatch.setenv("WNN_PORTABLE_DATA_MANIFEST", str(registry_path))
    monkeypatch.setenv("WNN_FOLD0_ASSET_VALIDATION_RECEIPT", str(receipt_path))
    monkeypatch.setenv(
        "WNN_FOLD0_ASSET_VALIDATION_SHA256", hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    )
    from wnn import portability

    def reject_rehash(*_args, **_kwargs):
        raise AssertionError("verified auxiliary assets must not be rehashed per fit process")

    monkeypatch.setattr(portability, "verify_portable_source", reject_rehash)
    split = load_campaign_split(split_path, source_root, preverified_sources=parsed_sources)
    assert split.fold == 0
    assert len(split.evaluation_groups) == 468
    assert split.sha256 == cache_manifest["split_manifest_sha256"]
