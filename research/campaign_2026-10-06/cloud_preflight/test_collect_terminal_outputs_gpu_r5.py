"""Exercise the finite R5 collector config against the actual admitted receipts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from collect_terminal_outputs import GPU_R5_KERNEL, TPU_V7_KERNEL, validate_config

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_RELATIVE_PATH = Path(
    "research/campaign_2026-10-06/cloud_preflight/"
    "terminal_collection_targets_20261009_gpu_r5_r1.json"
)
TARGET_PATH_FIELDS = (
    "root_admission",
    "push_receipt",
    "asset_ready_receipt",
    "asset_upload_admission",
    "recovery_asset_manifest",
)


def _main_repo_root(worktree_root: Path) -> Path:
    """Find the shared main checkout that holds root-owned admission receipts."""
    git_file = worktree_root / ".git"
    if not git_file.is_file():
        return worktree_root
    gitdir_line = git_file.read_text(encoding="utf-8").strip()
    gitdir_prefix = "gitdir: "
    if not gitdir_line.startswith(gitdir_prefix):
        raise ValueError(f"unexpected worktree .git file: {git_file}")
    gitdir = Path(gitdir_line.removeprefix(gitdir_prefix)).resolve()
    for candidate in gitdir.parents:
        if (candidate / "specs/README.md").is_file():
            return candidate
    raise ValueError("could not locate the main project checkout from the worktree")


def _copy_receipt(source_root: Path, target_root: Path, relative_path: str) -> None:
    """Copy one small exact receipt into the isolated validation repository."""
    source = (source_root / CONFIG_RELATIVE_PATH.parent / relative_path).resolve()
    target = (target_root / CONFIG_RELATIVE_PATH.parent / relative_path).resolve()
    source.relative_to(source_root.resolve())
    target.relative_to(target_root.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _actual_config_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Copy the candidate config and its real pinned receipts into a temp root."""
    main_root = _main_repo_root(WORKTREE_ROOT)
    source_config_path = WORKTREE_ROOT / CONFIG_RELATIVE_PATH
    config = json.loads(source_config_path.read_text(encoding="utf-8"))
    repo_root = tmp_path / "repo"
    config_path = repo_root / CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True)

    for target in config["targets"]:
        kernel = target["kernel"]
        for field in TARGET_PATH_FIELDS:
            relative_path = target.get(field)
            if relative_path is not None:
                source_root = (
                    main_root
                    if kernel != GPU_R5_KERNEL
                    or field in {"root_admission", "asset_upload_admission"}
                    else WORKTREE_ROOT
                )
                _copy_receipt(source_root, repo_root, relative_path)

    cli_path = repo_root / ".venv/bin/kaggle"
    cli_path.parent.mkdir(parents=True)
    cli_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli_path.chmod(0o700)
    config["kaggle_cli"] = str(cli_path)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path, repo_root, config


def test_r5_collector_config_accepts_exact_real_admission_and_asset_receipts(
    tmp_path: Path,
) -> None:
    """Check the two finite live targets against their actual pinned bytes."""
    config_path, repo_root, config = _actual_config_fixture(tmp_path)

    validated_config, targets = validate_config(config_path, repo_root)

    assert validated_config["maximum_download_bytes"] == 8 * 1024**3
    assert [target["kernel"] for target in targets] == [TPU_V7_KERNEL, GPU_R5_KERNEL]
    r5_target = next(target for target in targets if target["kernel"] == GPU_R5_KERNEL)
    assert (
        r5_target["recovery_manifest_sha256"] == (config["targets"][1]["recovery_manifest_sha256"])
    )
    assert r5_target["worker_max_seconds"] == 3300


def test_r5_collector_rejects_recovery_manifest_pin_drift(tmp_path: Path) -> None:
    """Reject an edited manifest identity before any provider download begins."""
    config_path, repo_root, config = _actual_config_fixture(tmp_path)
    config["targets"][1]["recovery_manifest_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recovery_manifest_sha256 does not match"):
        validate_config(config_path, repo_root)
