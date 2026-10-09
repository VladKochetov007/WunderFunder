"""Focused offline validation for the admitted GPU R2 terminal collector target."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from collect_terminal_outputs import (
    GPU_D02_R3_KERNEL,
    GPU_R2_KERNEL,
    GPU_R4_KERNEL,
    GPU_R5_KERNEL,
    PINNED_TARGETS,
    TPU_V7_KERNEL,
    _allowed_target_sets,
    validate_config,
)

# JSON receipt fixtures are open-schema records, matching the collector's provider inputs.

PACKET_BINDINGS = {
    "packet_sha256": "6aeda2f8a0839dee7c93d782bc765092c859c8336a762f7e42afb94e7ea6c7d9",
    "bootstrap_sha256": "229b93f9e61de82dd1dac5755303ec97a7a0077d4e805b66bc147d7a47871ae6",
    "metadata_sha256": "89580cc03ec96dfb2901dac1775110e8bed62fe985f32e51b3ddad0ecac7e6ec",
    "root_sealed_review_sha256": "9c60105be2d1421b61c01f5fe89c6387d2e8274f6a42fe60af1704f94dfe1907",
    "readiness_quota_observation_sha256": (
        "62b454dafa093d02d74f8d4590fd77e0ac2e0b3196a45de222993c43ef12e178"
    ),
}
SOURCE_DATASET = "vladkochetov07/wnn-loss-pair-code-20261009-r2"
SOURCE_MANIFEST_SHA256 = "d6ae4e62f36d1ddbc3ac298a511d31b52fbba9494d7127e8c0c38552b86009be"
ROOT_ADMISSION_SHA256 = "df7e5265f8deabb407253427d4924f6f0e638b126efbd8f197b9087a66a4d14d"
PUSH_RECEIPT_SHA256 = "d3c9ce15e2de612ed604e750008bc699d94df1fcd530c2337289cf7304fa59ef"

R4_PACKET_BINDINGS = {
    "packet_sha256": "dd1f57dc288b8b75df349f150170c6fc60cdff474f2f6f14897baa60b9fcc16a",
    "bootstrap_sha256": "86346f9c1addefbd38d3dcec52b18f514698a9514e7917aeb6081cc32f7eb039",
    "metadata_sha256": "ced255b82699ff6d354b12e7b7687d122814a3b75f80827213dadbc7830e567d",
    "root_sealed_review_sha256": "69bddcad812dcbe5b6abcee8853b53cd43330a411a215aea2ae2d5652979565c",
}
R4_SOURCE_DATASET = "vladkochetov07/wnn-loss-pair-code-20261009-r4"
R4_SOURCE_MANIFEST_SHA256 = "8ef360536ca117232198a4ee6dd8fabf072fb7bfad9812a83972d8bfb750f584"
R4_ROOT_ADMISSION_SHA256 = "890e429a496a701542825f13208dc27377cf8ca97966e6c17465a957f9901fda"
R4_ASSET_ADMISSION_SHA256 = "8260e87bf4c0b46c6878d81cd9e22660284a91ac037812180d102cac1334685a"
R4_PUSH_RECEIPT_SHA256 = "6112ed6fbfcb10f541cd17f9f3a1b900742e504d2279fedca40733cf25beeceb"
R4_READY_RECEIPT_SHA256 = "1d0f3af9a3ec3f1d2d4f8ca8c1a776f514c22c01ac42c1f091cd32a775aa7184"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one small private-operator fixture file for validation tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    """Return a file digest without loading its contents as a large object."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Create an exact-schema local fixture with bytes pinned by the config."""
    root_admission = {
        "state": "one_private_kernel_push_conditionally_authorized",
        "kernel": GPU_R2_KERNEL,
        **PACKET_BINDINGS,
        "provisional_gpu_seconds": 7200,
        "provider_timeout_seconds": 3600,
        "worker_wall_seconds": 3300,
    }
    root_path = tmp_path / "root-admission.json"
    _write_json(root_path, root_admission)
    root_sha = _sha256(root_path)

    push_receipt = {
        "canonical_provider_slug": GPU_R2_KERNEL,
        "root_admission_sha256": root_sha,
        "push_count": 1,
        "kernel_version": 1,
        "source_dataset": SOURCE_DATASET,
        "source_dataset_version": 1,
        "source_dataset_ready_at_push": True,
        "source_manifest_sha256_verified": SOURCE_MANIFEST_SHA256,
        "packet": PACKET_BINDINGS,
    }
    push_path = tmp_path / "push-receipt.json"
    _write_json(push_path, push_receipt)
    push_sha = _sha256(push_path)

    pinned: dict[str, dict[str, Any]] = {
        GPU_R2_KERNEL: {
            "resource": "gpu",
            "metadata_id": GPU_R2_KERNEL,
            "root_admission": "root-admission.json",
            "root_admission_sha256": root_sha,
            "artifact_dir": "gpu_results",
            "terminal_receipt": "gpu_terminal.json",
            "push_receipt": "push-receipt.json",
            "push_receipt_sha256": push_sha,
            "admission_bindings": PACKET_BINDINGS,
            "source_dataset": SOURCE_DATASET,
            "source_dataset_version": 1,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "provisional_reservation_seconds": 7200,
            "provider_timeout_seconds": 3600,
            "worker_max_seconds": 3300,
        }
    }
    target = {"kernel": GPU_R2_KERNEL, **pinned[GPU_R2_KERNEL]}
    config_path = tmp_path / "collector.json"
    cli_path = tmp_path / ".venv" / "bin" / "kaggle"
    cli_path.parent.mkdir(parents=True)
    cli_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli_path.chmod(0o700)
    _write_json(
        config_path,
        {
            "schema": "wnn.kaggle-terminal-collector-targets.v1",
            "kaggle_cli": str(cli_path),
            "expected_cli_version": "Kaggle CLI 2.2.4",
            "minimum_free_bytes": 4 * 1024**3,
            "maximum_download_bytes": 8 * 1024**3,
            "max_runtime_seconds": 800,
            "max_attempts": 3,
            "poll_log": "poll.jsonl",
            "targets": [target],
        },
    )
    return config_path, pinned


def _r4_fixture(tmp_path: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Create pinned R4 kernel, source admission, and READY receipt fixtures."""
    root_admission = {
        "state": "one_private_kernel_push_conditionally_authorized",
        "kernel": GPU_R4_KERNEL,
        **R4_PACKET_BINDINGS,
        "source_dataset": R4_SOURCE_DATASET,
        "source_asset_manifest_sha256": R4_SOURCE_MANIFEST_SHA256,
        "provisional_gpu_seconds": 7200,
        "provider_timeout_seconds": 3600,
        "worker_wall_seconds": 3300,
    }
    root_path = tmp_path / "root-admission.json"
    _write_json(root_path, root_admission)
    root_sha = _sha256(root_path)

    asset_admission = {
        "state": "one_exact_private_asset_upload_conditionally_authorized",
        "kernel": GPU_R4_KERNEL,
        "dataset": R4_SOURCE_DATASET,
        "source_asset_manifest_sha256": R4_SOURCE_MANIFEST_SHA256,
    }
    asset_admission_path = tmp_path / "asset-upload-admission.json"
    _write_json(asset_admission_path, asset_admission)
    asset_admission_sha = _sha256(asset_admission_path)

    ready_receipt = {
        "dataset": R4_SOURCE_DATASET,
        "private": True,
        "provider_version": 1,
        "root_admission": {"sha256": asset_admission_sha},
        "kernel_admission": {"sha256": root_sha},
        "readiness": {
            "status": "ready",
            "remote_manifest_sha256": R4_SOURCE_MANIFEST_SHA256,
            "remote_manifest_matches_staged_bytes": True,
        },
    }
    ready_path = tmp_path / "asset-ready.json"
    _write_json(ready_path, ready_receipt)
    ready_sha = _sha256(ready_path)

    push_receipt = {
        "canonical_provider_slug": GPU_R4_KERNEL,
        "root_admission": {"sha256": root_sha},
        "asset_upload_admission": {"sha256": asset_admission_sha},
        "push_count": 1,
        "kernel_version": 1,
        "source_dataset": R4_SOURCE_DATASET,
        "source_dataset_version": 1,
        "source_dataset_ready_at_push": True,
        "source_manifest_sha256_verified": R4_SOURCE_MANIFEST_SHA256,
        "packet": R4_PACKET_BINDINGS,
    }
    push_path = tmp_path / "push-receipt.json"
    _write_json(push_path, push_receipt)
    push_sha = _sha256(push_path)

    pinned: dict[str, dict[str, Any]] = {
        GPU_R4_KERNEL: {
            "resource": "gpu",
            "metadata_id": GPU_R4_KERNEL,
            "root_admission": "root-admission.json",
            "root_admission_sha256": root_sha,
            "artifact_dir": "gpu_results",
            "terminal_receipt": "gpu_terminal.json",
            "push_receipt": "push-receipt.json",
            "push_receipt_sha256": push_sha,
            "asset_ready_receipt": "asset-ready.json",
            "asset_ready_receipt_sha256": ready_sha,
            "asset_upload_admission": "asset-upload-admission.json",
            "asset_upload_admission_sha256": asset_admission_sha,
            "admission_bindings": R4_PACKET_BINDINGS,
            "source_dataset": R4_SOURCE_DATASET,
            "source_dataset_version": 1,
            "source_manifest_sha256": R4_SOURCE_MANIFEST_SHA256,
            "provisional_reservation_seconds": 7200,
            "provider_timeout_seconds": 3600,
            "worker_max_seconds": 3300,
        }
    }
    target = {"kernel": GPU_R4_KERNEL, **pinned[GPU_R4_KERNEL]}
    config_path = tmp_path / "collector.json"
    cli_path = tmp_path / ".venv" / "bin" / "kaggle"
    cli_path.parent.mkdir(parents=True)
    cli_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli_path.chmod(0o700)
    _write_json(
        config_path,
        {
            "schema": "wnn.kaggle-terminal-collector-targets.v1",
            "kaggle_cli": str(cli_path),
            "expected_cli_version": "Kaggle CLI 2.2.4",
            "minimum_free_bytes": 4 * 1024**3,
            "maximum_download_bytes": 8 * 1024**3,
            "max_runtime_seconds": 800,
            "max_attempts": 3,
            "poll_log": "poll.jsonl",
            "targets": [target],
        },
    )
    return config_path, pinned


def test_only_exact_original_pair_and_admitted_gpu_extensions_are_supported() -> None:
    """Allow old fixed sets and exact admitted GPU extensions only."""
    allowed = _allowed_target_sets(PINNED_TARGETS)
    legacy = frozenset(PINNED_TARGETS) - {
        GPU_R2_KERNEL,
        GPU_R4_KERNEL,
        GPU_R5_KERNEL,
        GPU_D02_R3_KERNEL,
    }
    assert allowed == {
        legacy,
        legacy | {GPU_R2_KERNEL},
        legacy | {GPU_R4_KERNEL},
        legacy | {GPU_R2_KERNEL, GPU_R4_KERNEL},
        frozenset({TPU_V7_KERNEL, GPU_R5_KERNEL}),
        frozenset({GPU_D02_R3_KERNEL}),
    }


def test_r2_config_validates_root_admission_and_actual_receipt_schema(
    tmp_path: Path,
) -> None:
    """The target resolves only when its receipt fields match the root binding."""
    config_path, pinned = _fixture(tmp_path)
    config, targets = validate_config(config_path, tmp_path, pinned_targets=pinned)
    assert config["maximum_download_bytes"] == 8 * 1024**3
    assert config["minimum_free_bytes"] == 4 * 1024**3
    assert targets[0]["kernel"] == GPU_R2_KERNEL


def test_r2_config_rejects_changed_push_receipt_bytes(tmp_path: Path) -> None:
    """A changed provider receipt cannot remain a collected target binding."""
    config_path, pinned = _fixture(tmp_path)
    (tmp_path / "push-receipt.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact provider push receipt hash mismatch"):
        validate_config(config_path, tmp_path, pinned_targets=pinned)


def test_r2_config_rejects_unpinned_source_manifest(tmp_path: Path) -> None:
    """A lookalike source version cannot replace the admitted code manifest."""
    config_path, pinned = _fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["targets"][0]["source_manifest_sha256"] = "0" * 64
    _write_json(config_path, config)
    with pytest.raises(ValueError, match="source_manifest_sha256 does not match"):
        validate_config(config_path, tmp_path, pinned_targets=pinned)


def test_r4_config_validates_root_push_and_remote_ready_bindings(tmp_path: Path) -> None:
    """The R4 target is accepted only with its exact source and push receipts."""
    config_path, pinned = _r4_fixture(tmp_path)
    config, targets = validate_config(config_path, tmp_path, pinned_targets=pinned)
    assert config["maximum_download_bytes"] == 8 * 1024**3
    assert targets[0]["kernel"] == GPU_R4_KERNEL


def test_r4_config_rejects_changed_push_receipt_bytes(tmp_path: Path) -> None:
    """A rewritten accepted-push receipt cannot pass the pinned collector."""
    config_path, pinned = _r4_fixture(tmp_path)
    (tmp_path / "push-receipt.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact provider push receipt hash mismatch"):
        validate_config(config_path, tmp_path, pinned_targets=pinned)


def test_r4_config_rejects_changed_readiness_receipt_bytes(tmp_path: Path) -> None:
    """A changed READY receipt cannot pass just because the dataset still exists."""
    config_path, pinned = _r4_fixture(tmp_path)
    (tmp_path / "asset-ready.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact source readiness receipt hash mismatch"):
        validate_config(config_path, tmp_path, pinned_targets=pinned)
