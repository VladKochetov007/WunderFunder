"""Validate the exact admitted D02 R5 collector target and receipts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from collect_terminal_outputs import GPU_D02_R5_KERNEL, validate_config

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_RELATIVE_PATH = Path(
    "research/campaign_2026-10-06/cloud_preflight/"
    "terminal_collection_targets_20261009_d02_r5_r1.json"
)
TARGET_PATH_FIELDS = (
    "root_admission",
    "source_admission",
    "push_receipt",
    "asset_ready_receipt",
)


def _main_repo_root(worktree_root: Path) -> Path:
    """Find the shared main checkout containing the root-owned final admission."""
    git_file = worktree_root / ".git"
    if not git_file.is_file():
        return worktree_root
    gitdir_line = git_file.read_text(encoding="utf-8").strip()
    prefix = "gitdir: "
    if not gitdir_line.startswith(prefix):
        raise ValueError(f"unexpected worktree .git file: {git_file}")
    gitdir = Path(gitdir_line.removeprefix(prefix)).resolve()
    for candidate in gitdir.parents:
        if (candidate / "specs/README.md").is_file():
            return candidate
    raise ValueError("could not locate the main project checkout")


def _copy_receipt(
    source_root: Path,
    target_root: Path,
    relative_path: str,
) -> None:
    """Copy one exact small receipt into a temporary validation repository."""
    relative_to_config = Path(CONFIG_RELATIVE_PATH.parent, relative_path)
    source = (source_root / relative_to_config).resolve()
    target = (target_root / relative_to_config).resolve()
    source.relative_to(source_root.resolve())
    target.relative_to(target_root.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _actual_config_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Build a temporary target from the admitted root and actual push receipts."""
    main_root = _main_repo_root(WORKTREE_ROOT)
    config = json.loads((WORKTREE_ROOT / CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))
    repo_root = tmp_path / "repo"
    config_path = repo_root / CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)

    target = config["targets"][0]
    for field in TARGET_PATH_FIELDS:
        _copy_receipt(main_root, repo_root, target[field])

    cli_path = repo_root / ".venv/bin/kaggle"
    cli_path.parent.mkdir(parents=True)
    cli_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli_path.chmod(0o700)
    config["kaggle_cli"] = str(cli_path)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path, repo_root, config


def test_d02_r5_collector_accepts_exact_admission_and_provider_receipts(
    tmp_path: Path,
) -> None:
    """Pin only the exact admitted private D02 R5 version for terminal collection."""
    config_path, repo_root, config = _actual_config_fixture(tmp_path)

    validated_config, targets = validate_config(config_path, repo_root)

    assert [target["kernel"] for target in targets] == [GPU_D02_R5_KERNEL]
    assert validated_config["maximum_download_bytes"] == 8 * 1024**3
    assert targets[0]["worker_max_seconds"] == 3300
    assert targets[0]["source_dataset_version"] == 1
    assert targets[0]["other_open_gpu_holds"] == {"native_gpu_fp16_r1": 7200}


def test_d02_r5_collector_rejects_provider_receipt_byte_drift(tmp_path: Path) -> None:
    """Reject changed provider evidence even when parsed fields remain plausible."""
    config_path, repo_root, config = _actual_config_fixture(tmp_path)
    push_path = (config_path.parent / config["targets"][0]["push_receipt"]).resolve()
    receipt = json.loads(push_path.read_text(encoding="utf-8"))
    receipt["provider_status_observed_after_push"] = "KernelWorkerStatus.ERROR"
    push_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact provider push receipt hash mismatch"):
        validate_config(config_path, repo_root)


def test_d02_r5_collector_rejects_unpinned_target_slug(tmp_path: Path) -> None:
    """Do not accept a config that redirects the finite collector target."""
    config_path, repo_root, config = _actual_config_fixture(tmp_path)
    config["targets"][0]["kernel"] = "vladkochetov07/unreviewed-kernel"
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact pinned target set"):
        validate_config(config_path, repo_root)
