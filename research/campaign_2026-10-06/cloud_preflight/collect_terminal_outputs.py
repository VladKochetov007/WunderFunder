"""Collect terminal Kaggle outputs for the finite root-pinned campaign kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from poll_provider import _receipt_files_match, append_jsonl
    from poll_provider import collect as collect_provider_snapshot
except ImportError:  # pragma: no cover - package import for focused tests
    from .poll_provider import _receipt_files_match, append_jsonl
    from .poll_provider import collect as collect_provider_snapshot

GIB = 1024**3
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("terminal_collection_targets_20261009_gpu_r5_r1.json")
TERMINAL_STATES = {"COMPLETE", "COMPLETED", "ERROR", "FAILED", "CANCELED", "CANCELLED"}
MAX_ATTEMPTS = 3
GPU_R2_KERNEL = "vladkochetov07/wnn-loss-pair-gpu-20261009-v2"
GPU_R4_KERNEL = "vladkochetov07/wnn-loss-pair-gpu-20261009-v4"
GPU_R5_KERNEL = "vladkochetov07/wnn-loss-pair-export-recovery-20261009-r1"
GPU_D02_R3_KERNEL = "vladkochetov07/wnn-d02-muon-loss-gpu-20261009-v3"
GPU_D02_R4_KERNEL = "vladkochetov07/wnn-d02-muon-loss-gpu-20261009-v4"
TPU_V7_KERNEL = "vladkochetov07/e463-tpu-real-data-reload-v7-20261008"

# Only the original fixed pair and its exact root-pinned GPU extensions are accepted.
PINNED_TARGETS: dict[str, dict[str, Any]] = {
    "vladkochetov07/wnn-warm-cohort-smoke-20261008-v4": {
        "resource": "gpu",
        "metadata_id": "vladkochetov07/wnn-warm-cohort-smoke-20261008-v4",
        "root_admission": (
            "../bandit_continuation/controller_claude_20261008/KAGGLE_SMOKE4_ROOT_ADMISSION_R1.json"
        ),
        "root_admission_sha256": (
            "79adf2e9b11dc737459bb770124e4401586920a36c68a3384050fc1c99886a23"
        ),
        "artifact_dir": (
            "../bandit_continuation/kaggle_warm_cohort/smoke4_20261008/terminal_collect_r1"
        ),
        "terminal_receipt": (
            "../bandit_continuation/kaggle_warm_cohort/"
            "smoke4_20261008/KAGGLE_SMOKE4_TERMINAL_OPERATOR_RECEIPT_R1.json"
        ),
        "push_receipt": (
            "../bandit_continuation/kaggle_warm_cohort/"
            "smoke4_20261008/KAGGLE_SMOKE4_PROVIDER_PUSH_R1.json"
        ),
        "admission_bindings": {
            "kernel_script_sha256": (
                "5e75bdeade37964848e8e4f1023566e979ce29545edbd75ef64a59bb4204c538"
            ),
            "metadata_sha256": ("246c8415fd195827ef2432fabf6ac664858f27a6e3ee7281596abf42f043fae6"),
            "packet_sha256": ("f7472dfd64f505de7e65a080691ae24917a4504c35e033f74084e88505e26179"),
            "independent_review_sha256": (
                "62cf342c59b19af9c16bf5a00021ea7d6fb780dec384d32c8420eecb964d2469"
            ),
        },
    },
    "vladkochetov07/e463-tpu-real-data-reload-v7-20261008": {
        "resource": "tpu",
        "metadata_id": "vladkochetov07/e463-tpu-realdata-reload-v7-20261008",
        "root_admission": (
            "../bandit_continuation/controller_claude_20261008/KAGGLE_TPU_V7_ROOT_ADMISSION_R1.json"
        ),
        "root_admission_sha256": (
            "dbb32828d38d3b3f0f294a4f08b3d7fd189740e973493a212ba29a36dcdc4bd4"
        ),
        "artifact_dir": "tpu_v7_terminal_20261008_r1",
        "terminal_receipt": "E463_TPU_V7_TERMINAL_OPERATOR_RECEIPT_20261008_R1.json",
        "push_receipt": "KAGGLE_TPU_V7_PROVIDER_PUSH_R1.json",
        "admission_bindings": {
            "code_sha256": ("d22947623bbe638280183ab019d4159ea37038b73840f3058e0137379bb4dad7"),
            "metadata_sha256": ("23e67bce8659cbd98bcc9c985c73637e8e3b161e17f56f5cdb185e030714c4ef"),
            "bundle_sha256": ("053638cf52a82f00b2da8fce82fcb63f21f7bfaccaf1b766b64e0ffd8ab1c528"),
            "independent_review_sha256": (
                "26f4119a0a469d28588652bf5faca08a210ac472312af769606f64d41f999d2a"
            ),
        },
    },
    GPU_R2_KERNEL: {
        "resource": "gpu",
        "metadata_id": GPU_R2_KERNEL,
        "root_admission": (
            "../bandit_continuation/diversity_stack_20261009/ROOT_GPU_R2_KERNEL_ADMISSION_R1.json"
        ),
        "root_admission_sha256": (
            "df7e5265f8deabb407253427d4924f6f0e638b126efbd8f197b9087a66a4d14d"
        ),
        "artifact_dir": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "gpu_loss_pair_r2_terminal"
        ),
        "terminal_receipt": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "GPU_R2_TERMINAL_OPERATOR_RECEIPT_R1.json"
        ),
        "push_receipt": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "GPU_R2_PROVIDER_PUSH_RECEIPT_R1.json"
        ),
        "push_receipt_sha256": ("d3c9ce15e2de612ed604e750008bc699d94df1fcd530c2337289cf7304fa59ef"),
        "admission_bindings": {
            "packet_sha256": ("6aeda2f8a0839dee7c93d782bc765092c859c8336a762f7e42afb94e7ea6c7d9"),
            "bootstrap_sha256": (
                "229b93f9e61de82dd1dac5755303ec97a7a0077d4e805b66bc147d7a47871ae6"
            ),
            "metadata_sha256": ("89580cc03ec96dfb2901dac1775110e8bed62fe985f32e51b3ddad0ecac7e6ec"),
            "root_sealed_review_sha256": (
                "9c60105be2d1421b61c01f5fe89c6387d2e8274f6a42fe60af1704f94dfe1907"
            ),
            "readiness_quota_observation_sha256": (
                "62b454dafa093d02d74f8d4590fd77e0ac2e0b3196a45de222993c43ef12e178"
            ),
        },
        "source_dataset": "vladkochetov07/wnn-loss-pair-code-20261009-r2",
        "source_dataset_version": 1,
        "source_manifest_sha256": (
            "d6ae4e62f36d1ddbc3ac298a511d31b52fbba9494d7127e8c0c38552b86009be"
        ),
        "provisional_reservation_seconds": 7200,
        "provider_timeout_seconds": 3600,
        "worker_max_seconds": 3300,
    },
    GPU_R4_KERNEL: {
        "resource": "gpu",
        "metadata_id": GPU_R4_KERNEL,
        "root_admission": (
            "../bandit_continuation/diversity_stack_20261009/ROOT_GPU_R4_KERNEL_ADMISSION_R1.json"
        ),
        "root_admission_sha256": (
            "890e429a496a701542825f13208dc27377cf8ca97966e6c17465a957f9901fda"
        ),
        "artifact_dir": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "gpu_loss_pair_r4/terminal_collect_r1"
        ),
        "terminal_receipt": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "gpu_loss_pair_r4/GPU_R4_TERMINAL_OPERATOR_RECEIPT_R1.json"
        ),
        "push_receipt": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "gpu_loss_pair_r4/GPU_R4_PROVIDER_PUSH_RECEIPT_R1.json"
        ),
        "push_receipt_sha256": ("6112ed6fbfcb10f541cd17f9f3a1b900742e504d2279fedca40733cf25beeceb"),
        "asset_ready_receipt": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "gpu_loss_pair_r4/GPU_R4_ASSET_READY_RECEIPT_R1.json"
        ),
        "asset_ready_receipt_sha256": (
            "1d0f3af9a3ec3f1d2d4f8ca8c1a776f514c22c01ac42c1f091cd32a775aa7184"
        ),
        "asset_upload_admission": (
            "../bandit_continuation/diversity_stack_20261009/ROOT_GPU_R4_ASSET_UPLOAD_ADMISSION_R1.json"
        ),
        "asset_upload_admission_sha256": (
            "8260e87bf4c0b46c6878d81cd9e22660284a91ac037812180d102cac1334685a"
        ),
        "admission_bindings": {
            "packet_sha256": ("dd1f57dc288b8b75df349f150170c6fc60cdff474f2f6f14897baa60b9fcc16a"),
            "bootstrap_sha256": (
                "86346f9c1addefbd38d3dcec52b18f514698a9514e7917aeb6081cc32f7eb039"
            ),
            "metadata_sha256": ("ced255b82699ff6d354b12e7b7687d122814a3b75f80827213dadbc7830e567d"),
            "root_sealed_review_sha256": (
                "69bddcad812dcbe5b6abcee8853b53cd43330a411a215aea2ae2d5652979565c"
            ),
        },
        "source_dataset": "vladkochetov07/wnn-loss-pair-code-20261009-r4",
        "source_dataset_version": 1,
        "source_manifest_sha256": (
            "8ef360536ca117232198a4ee6dd8fabf072fb7bfad9812a83972d8bfb750f584"
        ),
        "provisional_reservation_seconds": 7200,
        "provider_timeout_seconds": 3600,
        "worker_max_seconds": 3300,
    },
    GPU_R5_KERNEL: {
        "resource": "gpu",
        "metadata_id": GPU_R5_KERNEL,
        "root_admission": (
            "../bandit_continuation/diversity_stack_20261009/ROOT_GPU_R5_KERNEL_ADMISSION_R1.json"
        ),
        "root_admission_sha256": (
            "fc6872d0ef27ad7d7da843ce9b1939c340882dfcb99e8211a8f31df8317d4e7c"
        ),
        "artifact_dir": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "gpu_loss_pair_r5_export/terminal_collect_r1"
        ),
        "terminal_receipt": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "gpu_loss_pair_r5_export/GPU_R5_TERMINAL_OPERATOR_RECEIPT_R1.json"
        ),
        "push_receipt": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "gpu_loss_pair_r5_export/GPU_R5_PROVIDER_PUSH_RECEIPT_R1.json"
        ),
        "push_receipt_sha256": ("33dd6c5c8718976ee6f1466ef8f67bc5707f4c3cc772a3c064b27ab8edb2cb61"),
        "asset_ready_receipt": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "gpu_loss_pair_r5_export/asset_upload_receipt_R1.json"
        ),
        "asset_ready_receipt_sha256": (
            "b6e0d5acced34780a734c8770bc3b1969661ca983010dd21a1f3d9a84485e619"
        ),
        "asset_upload_admission": (
            "../bandit_continuation/diversity_stack_20261009/"
            "ROOT_GPU_R5_ASSET_UPLOAD_ADMISSION_R2.json"
        ),
        "asset_upload_admission_sha256": (
            "19767481c504d198419e0f4ad2143e98ecd195751945bfeab5f5101689f72f1f"
        ),
        "recovery_asset_manifest": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "gpu_loss_pair_r5_export/source_asset_stage/assets-manifest.json"
        ),
        "admission_bindings": {
            "packet_sha256": ("ce58293e8124ef7e51d1e32c773c286ada5ca180c3e81a308bc31a95c3c108cc"),
            "bootstrap_sha256": (
                "ddb6401a6bf3b59a8242e387d11da9c7dbc1713236e085bb3cade27de775fc19"
            ),
            "metadata_sha256": ("e57d5b86eca15818b18c0fd11d01190f5f4cd00ec868187f7ef89a1527dc9494"),
            "root_source_review_sha256": (
                "9aaab0bb6514c652e30bce15c37964a790ef44401cdd6de811a8a76b05aefd7a"
            ),
            "recovery_asset_admission_sha256": (
                "19767481c504d198419e0f4ad2143e98ecd195751945bfeab5f5101689f72f1f"
            ),
        },
        "source_dataset": "vladkochetov07/wnn-loss-pair-code-20261009-r4",
        "source_dataset_version": 1,
        "source_manifest_sha256": (
            "8ef360536ca117232198a4ee6dd8fabf072fb7bfad9812a83972d8bfb750f584"
        ),
        "cache_source_kernel": "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1",
        "cache_manifest_sha256": (
            "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b"
        ),
        "recovery_dataset": "vladkochetov07/wnn-loss-pair-r4-recovery-20261009-r1",
        "recovery_dataset_version": 1,
        "recovery_manifest_sha256": (
            "0d9d5b6d53017b8ef837c369d7aaa47ce2a676c37a1165cda71d6735b446bcfd"
        ),
        "rehearsal_commit": "fdc3c694",
        "provisional_reservation_seconds": 7200,
        "provider_timeout_seconds": 3600,
        "worker_max_seconds": 3300,
    },
    GPU_D02_R3_KERNEL: {
        "resource": "gpu",
        "metadata_id": GPU_D02_R3_KERNEL,
        "root_admission": (
            "../bandit_continuation/diversity_stack_20261009/ROOT_D02_R3_KERNEL_ADMISSION_R1.json"
        ),
        "root_admission_sha256": (
            "3fb53641b78245c780433031988089577ebaf061f89aa7070d96be7595f96fba"
        ),
        "source_admission": (
            "../bandit_continuation/diversity_stack_20261009/ROOT_D02_R3_SOURCE_ADMISSION_R1.json"
        ),
        "source_admission_sha256": (
            "81ca4132e166793b91614d4bbb5f604bc9fec155aa3f118939cfecffd15bfd1e"
        ),
        "artifact_dir": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "d02_r3/terminal_collect_r1"
        ),
        "terminal_receipt": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "d02_r3/GPU_D02_R3_TERMINAL_OPERATOR_RECEIPT_R1.json"
        ),
        "push_receipt": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "d02_r3/GPU_D02_R3_PROVIDER_PUSH_RECEIPT_R1.json"
        ),
        "push_receipt_sha256": ("592289a4fbeb9fc8c20c16ca0857d28ae1d265dafec8b727ecd6a485a7d34b76"),
        "asset_ready_receipt": (
            "../bandit_continuation/diversity_stack_20261009/cloud_delivery/"
            "d02_r3/GPU_D02_R3_ASSET_READY_RECEIPT_R1.json"
        ),
        "asset_ready_receipt_sha256": (
            "3fd9041c1c2f11165f5bf6fb9c61cd37ae748020f47199004d24bf3c99dd6b01"
        ),
        "source_dataset": "vladkochetov07/wnn-d02-muon-loss-code-20261009-r3",
        "source_dataset_version": 1,
        "source_manifest_sha256": (
            "c3807b6d6cf4a2456d4689934be9e5afee9d9123a69014152384d731afae8d9d"
        ),
        "admission_bindings": {
            "packet_sha256": ("6d4404bb5aca9e8e616b24a46211c524b4c023631138f1b913e5faef30f1536f"),
            "bootstrap_sha256": (
                "9bc8a4aa069c00207ac4abccecfee999d99151bac1564523692b576d36861146"
            ),
            "metadata_sha256": ("254a63961a23194929f21c8f0ea231c32a1e038b661f593b9089cc59093d0e04"),
            "embedded_admission_sha256": (
                "d7b5880ef51e4d3ae9e4c06c253ad2d4d27074ccbeb3f6456a99bd464d460c34"
            ),
        },
        "provisional_reservation_seconds": 7200,
        "provider_timeout_seconds": 3600,
        "worker_max_seconds": 3300,
    },
    GPU_D02_R4_KERNEL: {
        "resource": "gpu",
        "metadata_id": GPU_D02_R4_KERNEL,
        "root_admission": (
            "../bandit_continuation/diversity_stack_20261009/ROOT_D02_R4_KERNEL_ADMISSION_R1.json"
        ),
        "root_admission_sha256": (
            "6feb7099aaf65fa6b1fa190bcfa1e6a1d12b2711b7f07cca0d15e1fbc8321805"
        ),
        "source_admission": (
            "../bandit_continuation/diversity_stack_20261009/ROOT_D02_R4_SOURCE_ADMISSION_R1.json"
        ),
        "source_admission_sha256": (
            "c678becf01175b7227dc5667c810150a631bd7acdfd78b45bcdf146986d49792"
        ),
        "artifact_dir": (
            "../bandit_continuation/diversity_stack_20261009/"
            "d02_muon_loss_successor_r4_admitted_r2/terminal_collect_r1"
        ),
        "terminal_receipt": (
            "../bandit_continuation/diversity_stack_20261009/"
            "d02_muon_loss_successor_r4_admitted_r2/GPU_D02_R4_TERMINAL_OPERATOR_RECEIPT_R1.json"
        ),
        "push_receipt": (
            "../bandit_continuation/diversity_stack_20261009/"
            "d02_muon_loss_successor_r4_admitted_r2/GPU_D02_R4_PROVIDER_PUSH_RECEIPT_R1.json"
        ),
        "push_receipt_sha256": (
            "226f7bee47f5bd348e33e47312e1d4a820096b680f125e7f637fb36d5a2ff20a"
        ),
        "asset_ready_receipt": (
            "../bandit_continuation/diversity_stack_20261009/"
            "d02_muon_loss_successor_r4_source_r2/SOURCE_ASSET_READY_R1.json"
        ),
        "asset_ready_receipt_sha256": (
            "fa4c7951a6b8c060a41ec290c59ef53767db545dd8535d77bf5246891c1e8689"
        ),
        "source_dataset": "vladkochetov07/wnn-d02-muon-loss-code-20261009-r4",
        "source_dataset_version": 1,
        "source_manifest_sha256": (
            "c7bf7880a73a87b33d4c9c7d800384a5b245321721014ac214df3bd6b864f28f"
        ),
        "admission_bindings": {
            "packet_sha256": (
                "e3c80582b15b1e1f71d172022c895402ef8ed1d4b4c787c7498db9287ef44650"
            ),
            "bootstrap_sha256": (
                "0a49f42afff170eb48d03d054ab0bbc66c091ecdda63cecffc17b24d0224f310"
            ),
            "metadata_sha256": (
                "8bcea8b07aecd567e62bbe0f730eff6d3d5e1e824afc745d7960b5633d89d623"
            ),
            "embedded_admission_sha256": (
                "08911f92985f0de59aa78f86cce516ce8582e241403a74f6ce26c9568d7d4c9f"
            ),
        },
        "provisional_reservation_seconds": 7200,
        "provider_timeout_seconds": 3600,
        "worker_max_seconds": 3300,
    },
}


def utc_now() -> str:
    """Return a stable UTC timestamp for a durable receipt."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large model outputs are never loaded at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace a JSON receipt without exposing a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def resolve_repo_path(repo_root: Path, config_dir: Path, relative_path: str) -> Path:
    """Resolve a config path while rejecting absolute paths and repo escapes."""
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"expected a repository-relative path: {relative_path}")
    resolved = (config_dir / candidate).resolve()
    if not resolved.is_relative_to(repo_root.resolve()):
        raise ValueError(f"path escapes repository root: {relative_path}")
    return resolved


def _allowed_target_sets(pinned_targets: dict[str, dict[str, Any]]) -> set[frozenset[str]]:
    """Keep the original fixed pair valid with either or both exact GPU extensions."""
    all_targets = frozenset(pinned_targets)
    legacy_targets = all_targets - {
        GPU_R2_KERNEL,
        GPU_R4_KERNEL,
        GPU_R5_KERNEL,
        GPU_D02_R3_KERNEL,
        GPU_D02_R4_KERNEL,
    }
    allowed = {legacy_targets}
    if GPU_R2_KERNEL in all_targets:
        allowed.add(legacy_targets | {GPU_R2_KERNEL})
    if GPU_R4_KERNEL in all_targets:
        allowed.add(legacy_targets | {GPU_R4_KERNEL})
    if GPU_R2_KERNEL in all_targets and GPU_R4_KERNEL in all_targets:
        allowed.add(legacy_targets | {GPU_R2_KERNEL, GPU_R4_KERNEL})
    if GPU_R5_KERNEL in all_targets and TPU_V7_KERNEL in all_targets:
        allowed.add(frozenset({TPU_V7_KERNEL, GPU_R5_KERNEL}))
    if GPU_D02_R3_KERNEL in all_targets:
        allowed.add(frozenset({GPU_D02_R3_KERNEL}))
    if GPU_D02_R4_KERNEL in all_targets:
        allowed.add(frozenset({GPU_D02_R4_KERNEL}))
    return allowed


def validate_config(
    config_path: Path,
    repo_root: Path,
    *,
    pinned_targets: dict[str, dict[str, Any]] = PINNED_TARGETS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the finite target list against root admissions and pinned hashes."""
    config_path = config_path.resolve()
    repo_root = repo_root.resolve()
    if not config_path.is_relative_to(repo_root):
        raise ValueError("collector config must live inside the repository")
    config = json.loads(config_path.read_text())
    if config.get("schema") != "wnn.kaggle-terminal-collector-targets.v1":
        raise ValueError("unsupported terminal collector config schema")
    targets = config.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("collector targets must be a non-empty list")
    kernels = [target.get("kernel") for target in targets]
    kernel_set = frozenset(kernels)
    allowed_target_sets = _allowed_target_sets(pinned_targets)
    if len(kernels) != len(set(kernels)) or kernel_set not in allowed_target_sets:
        raise ValueError("collector targets must match one exact pinned target set")

    config_dir = config_path.parent
    for target in targets:
        kernel = target["kernel"]
        pinned = pinned_targets[kernel]
        for key, expected in pinned.items():
            if target.get(key) != expected:
                raise ValueError(f"{kernel}: {key} does not match the pinned value")

        admission_path = resolve_repo_path(repo_root, config_dir, target["root_admission"])
        if sha256_file(admission_path) != target["root_admission_sha256"]:
            raise ValueError(f"{kernel}: root admission hash mismatch")
        admission = json.loads(admission_path.read_text())
        if kernel in {GPU_D02_R3_KERNEL, GPU_D02_R4_KERNEL}:
            if admission.get("state") != "one_exact_private_kernel_push_authorized":
                raise ValueError(f"{kernel}: root admission state mismatch")
            if admission.get("kernel") != target["metadata_id"]:
                raise ValueError(f"{kernel}: admission kernel ID mismatch")
            for key, expected in target["admission_bindings"].items():
                if admission.get(key) != expected:
                    raise ValueError(f"{kernel}: admission binding mismatch for {key}")
            if (
                admission.get("provisional_hold_seconds")
                != target["provisional_reservation_seconds"]
            ):
                raise ValueError(f"{kernel}: admission reservation mismatch")
            if admission.get("provider_timeout_seconds") != target["provider_timeout_seconds"]:
                raise ValueError(f"{kernel}: admission timeout mismatch")
            if admission.get("worker_wall_seconds") != target["worker_max_seconds"]:
                raise ValueError(f"{kernel}: admission worker bound mismatch")
            if admission.get("accelerator") != "NvidiaTeslaT4":
                raise ValueError(f"{kernel}: admission accelerator mismatch")
            if (
                admission.get("source_readiness_receipt_sha256")
                != target["asset_ready_receipt_sha256"]
            ):
                raise ValueError(f"{kernel}: admission source readiness receipt mismatch")
            if admission.get("root_source_admission_sha256") != target["source_admission_sha256"]:
                raise ValueError(f"{kernel}: admission source admission mismatch")
            checks = admission.get("checks", {})
            expected_offline_check = "pass" if kernel == GPU_D02_R3_KERNEL else "root_pass"
            if (
                checks.get("exact_final_bootstrap_offline") != expected_offline_check
                or checks.get("embedded_payload_matches_root") is not True
            ):
                raise ValueError(f"{kernel}: final sealed-packet checks are incomplete")
        elif kernel in {GPU_R2_KERNEL, GPU_R4_KERNEL, GPU_R5_KERNEL}:
            if admission.get("state") != "one_private_kernel_push_conditionally_authorized":
                raise ValueError(f"{kernel}: root admission state mismatch")
            if admission.get("kernel") != target["metadata_id"]:
                raise ValueError(f"{kernel}: admission kernel ID mismatch")
            for key, expected in target["admission_bindings"].items():
                if admission.get(key) != expected:
                    raise ValueError(f"{kernel}: admission binding mismatch for {key}")
            if (
                admission.get("provisional_gpu_seconds")
                != target["provisional_reservation_seconds"]
            ):
                raise ValueError(f"{kernel}: admission reservation mismatch")
            if admission.get("provider_timeout_seconds") != target["provider_timeout_seconds"]:
                raise ValueError(f"{kernel}: admission timeout mismatch")
            if admission.get("worker_wall_seconds") != target["worker_max_seconds"]:
                raise ValueError(f"{kernel}: admission worker bound mismatch")
            if kernel == GPU_R4_KERNEL:
                if admission.get("source_dataset") != target["source_dataset"]:
                    raise ValueError(f"{kernel}: source dataset mismatch in root admission")
                if (
                    admission.get("source_asset_manifest_sha256")
                    != target["source_manifest_sha256"]
                ):
                    raise ValueError(f"{kernel}: source manifest mismatch in root admission")
            if kernel == GPU_R5_KERNEL:
                if admission.get("source_dataset") != target["recovery_dataset"]:
                    raise ValueError(f"{kernel}: recovery dataset mismatch in root admission")
                if (
                    admission.get("source_asset_manifest_sha256")
                    != target["recovery_manifest_sha256"]
                ):
                    raise ValueError(f"{kernel}: recovery manifest mismatch in root admission")
        else:
            if admission.get("authorized") is not True:
                raise ValueError(f"{kernel}: root admission is not authorized")
            if admission.get("kernel_id") != target["metadata_id"]:
                raise ValueError(f"{kernel}: admission metadata ID mismatch")
            for key, expected in target["admission_bindings"].items():
                if admission.get(key) != expected:
                    raise ValueError(f"{kernel}: admission binding mismatch for {key}")

        push_receipt_path = resolve_repo_path(repo_root, config_dir, target["push_receipt"])
        push_receipt = json.loads(push_receipt_path.read_text())
        if kernel in {GPU_D02_R3_KERNEL, GPU_D02_R4_KERNEL}:
            if sha256_file(push_receipt_path) != target["push_receipt_sha256"]:
                raise ValueError(f"{kernel}: exact provider push receipt hash mismatch")
            if (
                push_receipt.get("schema") != "wnn.kaggle-d02-gpu-push-receipt.v1"
                or push_receipt.get("kernel") != kernel
                or push_receipt.get("canonical_provider_slug") != kernel
            ):
                raise ValueError(f"{kernel}: provider push receipt slug/schema mismatch")
            if (
                push_receipt.get("private") is not True
                or push_receipt.get("title_derived_slug_match") is not True
                or push_receipt.get("provider_version") != 1
                or push_receipt.get("kernel_version") != 1
                or push_receipt.get("push_count") != 1
                or push_receipt.get("push_exit_code") != 0
            ):
                raise ValueError(f"{kernel}: provider receipt does not prove one accepted push")
            if (
                push_receipt.get("root_kernel_admission", {}).get("sha256")
                != target["root_admission_sha256"]
            ):
                raise ValueError(f"{kernel}: provider receipt root admission mismatch")
            if (
                push_receipt.get("root_source_admission", {}).get("sha256")
                != target["source_admission_sha256"]
            ):
                raise ValueError(f"{kernel}: provider receipt source admission mismatch")
            if (
                push_receipt.get("source_dataset") != target["source_dataset"]
                or push_receipt.get("source_dataset_version") != target["source_dataset_version"]
                or push_receipt.get("source_dataset_ready_at_push") is not True
                or push_receipt.get("source_manifest_sha256_verified")
                != target["source_manifest_sha256"]
                or push_receipt.get("source_file_inventory_sha256_bound")
                != target["source_manifest_sha256"]
                or push_receipt.get("source_readiness_receipt_sha256")
                != target["asset_ready_receipt_sha256"]
            ):
                raise ValueError(f"{kernel}: provider receipt source binding mismatch")
            for key, expected in target["admission_bindings"].items():
                if push_receipt.get("packet", {}).get(key) != expected:
                    raise ValueError(
                        f"{kernel}: provider receipt packet binding mismatch for {key}"
                    )
            quota = push_receipt.get("quota_snapshot", {})
            if (
                quota.get("provisional_gpu_hold_seconds")
                != target["provisional_reservation_seconds"]
                or quota.get("other_open_gpu_holds") != 0
                or float(quota.get("campaign_gpu_headroom_after_d02_hold_seconds", 0)) < 1000
            ):
                raise ValueError(f"{kernel}: provider receipt quota does not meet admission")
            bounds = push_receipt.get("worker_bounds", {})
            if (
                bounds.get("provider_timeout_seconds") != target["provider_timeout_seconds"]
                or bounds.get("worker_wall_seconds") != target["worker_max_seconds"]
                or bounds.get("device_count") != 2
                or bounds.get("device_type") != "T4"
                or bounds.get("training_enabled") is not True
                or bounds.get("precision") != "FP32"
                or bounds.get("tf32_enabled") is not False
            ):
                raise ValueError(f"{kernel}: provider receipt worker bounds mismatch")

            source_admission_path = resolve_repo_path(
                repo_root, config_dir, target["source_admission"]
            )
            if sha256_file(source_admission_path) != target["source_admission_sha256"]:
                raise ValueError(f"{kernel}: exact root source admission hash mismatch")
            source_admission = json.loads(source_admission_path.read_text())
            if (
                source_admission.get("state")
                != "private_source_upload_and_admitted_packet_preparation_authorized"
                or source_admission.get("kernel_id") != kernel
                or source_admission.get("source_dataset") != target["source_dataset"]
                or source_admission.get("source_asset_manifest_sha256")
                != target["source_manifest_sha256"]
                or source_admission.get("root_admission_payload", {}).get("push_allowed")
                is not True
                or source_admission.get("root_admission_payload", {}).get("conditional_fit_allowed")
                is not True
            ):
                raise ValueError(f"{kernel}: root source admission binding mismatch")

            asset_ready_path = resolve_repo_path(
                repo_root, config_dir, target["asset_ready_receipt"]
            )
            if sha256_file(asset_ready_path) != target["asset_ready_receipt_sha256"]:
                raise ValueError(f"{kernel}: exact source readiness receipt hash mismatch")
            asset_ready = json.loads(asset_ready_path.read_text())
            readiness = asset_ready.get("readiness", {})
            if (
                asset_ready.get("schema") != "wnn.kaggle-d02-private-source-ready-receipt.v1"
                or asset_ready.get("dataset") != target["source_dataset"]
                or asset_ready.get("private") is not True
                or asset_ready.get("provider_version") != target["source_dataset_version"]
                or asset_ready.get("root_source_admission", {}).get("sha256")
                != target["source_admission_sha256"]
                or readiness.get("status") != "ready"
                or readiness.get("remote_manifest_sha256") != target["source_manifest_sha256"]
                or readiness.get("remote_manifest_matches_staged_bytes") is not True
                or readiness.get("remote_inventory_exactly_matches_manifest_names_and_sizes")
                is not True
                or readiness.get("remote_data_file_count") != 6
                or readiness.get("remote_payload_file_count") != 5
            ):
                raise ValueError(f"{kernel}: source readiness evidence mismatch")
            if push_receipt.get("source_readiness_receipt_sha256") != sha256_file(asset_ready_path):
                raise ValueError(f"{kernel}: push receipt does not bind exact readiness bytes")
        elif kernel == GPU_R2_KERNEL:
            if sha256_file(push_receipt_path) != target["push_receipt_sha256"]:
                raise ValueError(f"{kernel}: exact provider push receipt hash mismatch")
            if push_receipt.get("canonical_provider_slug") != kernel:
                raise ValueError(f"{kernel}: provider push receipt slug mismatch")
            if push_receipt.get("root_admission_sha256") != target["root_admission_sha256"]:
                raise ValueError(f"{kernel}: provider push receipt admission mismatch")
            if push_receipt.get("push_count") != 1 or push_receipt.get("kernel_version") != 1:
                raise ValueError(f"{kernel}: expected exactly one accepted version")
            if push_receipt.get("source_dataset") != target["source_dataset"]:
                raise ValueError(f"{kernel}: source dataset mismatch")
            if push_receipt.get("source_dataset_version") != target["source_dataset_version"]:
                raise ValueError(f"{kernel}: source dataset version mismatch")
            if push_receipt.get("source_dataset_ready_at_push") is not True:
                raise ValueError(f"{kernel}: source dataset was not READY at push")
            if (
                push_receipt.get("source_manifest_sha256_verified")
                != target["source_manifest_sha256"]
            ):
                raise ValueError(f"{kernel}: source manifest mismatch")
            packet = push_receipt.get("packet", {})
            for key, expected in target["admission_bindings"].items():
                if packet.get(key) != expected:
                    raise ValueError(f"{kernel}: push receipt packet binding mismatch for {key}")
        elif kernel == GPU_R4_KERNEL:
            if sha256_file(push_receipt_path) != target["push_receipt_sha256"]:
                raise ValueError(f"{kernel}: exact provider push receipt hash mismatch")
            if push_receipt.get("canonical_provider_slug") != kernel:
                raise ValueError(f"{kernel}: provider push receipt slug mismatch")
            receipt_admission = push_receipt.get("root_admission", {})
            if receipt_admission.get("sha256") != target["root_admission_sha256"]:
                raise ValueError(f"{kernel}: provider push receipt admission mismatch")
            receipt_asset_admission = push_receipt.get("asset_upload_admission", {})
            if receipt_asset_admission.get("sha256") != target["asset_upload_admission_sha256"]:
                raise ValueError(f"{kernel}: provider push receipt asset admission mismatch")
            if push_receipt.get("push_count") != 1 or push_receipt.get("kernel_version") != 1:
                raise ValueError(f"{kernel}: expected exactly one accepted version")
            if push_receipt.get("source_dataset") != target["source_dataset"]:
                raise ValueError(f"{kernel}: source dataset mismatch")
            if push_receipt.get("source_dataset_version") != target["source_dataset_version"]:
                raise ValueError(f"{kernel}: source dataset version mismatch")
            if push_receipt.get("source_dataset_ready_at_push") is not True:
                raise ValueError(f"{kernel}: source dataset was not READY at push")
            if (
                push_receipt.get("source_manifest_sha256_verified")
                != target["source_manifest_sha256"]
            ):
                raise ValueError(f"{kernel}: source manifest mismatch")
            packet = push_receipt.get("packet", {})
            for key, expected in target["admission_bindings"].items():
                if packet.get(key) != expected:
                    raise ValueError(f"{kernel}: push receipt packet binding mismatch for {key}")

            asset_admission_path = resolve_repo_path(
                repo_root, config_dir, target["asset_upload_admission"]
            )
            if sha256_file(asset_admission_path) != target["asset_upload_admission_sha256"]:
                raise ValueError(f"{kernel}: exact asset upload admission hash mismatch")
            asset_admission = json.loads(asset_admission_path.read_text())
            if asset_admission.get("dataset") != target["source_dataset"]:
                raise ValueError(f"{kernel}: asset upload admission dataset mismatch")
            if asset_admission.get("kernel") != kernel:
                raise ValueError(f"{kernel}: asset upload admission kernel mismatch")
            if (
                asset_admission.get("source_asset_manifest_sha256")
                != target["source_manifest_sha256"]
            ):
                raise ValueError(f"{kernel}: asset upload admission source manifest mismatch")

            asset_ready_path = resolve_repo_path(
                repo_root, config_dir, target["asset_ready_receipt"]
            )
            if sha256_file(asset_ready_path) != target["asset_ready_receipt_sha256"]:
                raise ValueError(f"{kernel}: exact source readiness receipt hash mismatch")
            asset_ready = json.loads(asset_ready_path.read_text())
            if asset_ready.get("dataset") != target["source_dataset"]:
                raise ValueError(f"{kernel}: source readiness dataset mismatch")
            if asset_ready.get("private") is not True or asset_ready.get("provider_version") != 1:
                raise ValueError(f"{kernel}: source readiness privacy/version mismatch")
            if (
                asset_ready.get("root_admission", {}).get("sha256")
                != target["asset_upload_admission_sha256"]
            ):
                raise ValueError(f"{kernel}: asset upload admission mismatch")
            if (
                asset_ready.get("kernel_admission", {}).get("sha256")
                != target["root_admission_sha256"]
            ):
                raise ValueError(f"{kernel}: kernel admission mismatch in source readiness receipt")
            readiness = asset_ready.get("readiness", {})
            if readiness.get("status") != "ready":
                raise ValueError(f"{kernel}: private source dataset is not READY")
            if readiness.get("remote_manifest_sha256") != target["source_manifest_sha256"]:
                raise ValueError(f"{kernel}: remote source manifest mismatch")
            if readiness.get("remote_manifest_matches_staged_bytes") is not True:
                raise ValueError(f"{kernel}: remote source manifest did not match staged bytes")
        elif kernel == GPU_R5_KERNEL:
            if sha256_file(push_receipt_path) != target["push_receipt_sha256"]:
                raise ValueError(f"{kernel}: exact provider push receipt hash mismatch")
            if push_receipt.get("canonical_provider_slug") != kernel:
                raise ValueError(f"{kernel}: provider push receipt slug mismatch")
            if push_receipt.get("kernel_version") != 1 or push_receipt.get("push_count") != 1:
                raise ValueError(f"{kernel}: expected exactly one accepted version")
            if push_receipt.get("title_derived_slug_match") is not True:
                raise ValueError(f"{kernel}: pushed title does not bind the canonical slug")
            if push_receipt.get("private") is not True:
                raise ValueError(f"{kernel}: push receipt does not establish a private kernel")
            if (
                push_receipt.get("root_admission", {}).get("sha256")
                != target["root_admission_sha256"]
            ):
                raise ValueError(f"{kernel}: provider push receipt admission mismatch")
            source = push_receipt.get("source", {})
            for key in ("packet_sha256", "bootstrap_sha256", "metadata_sha256"):
                if source.get(key) != target["admission_bindings"][key]:
                    raise ValueError(
                        f"{kernel}: provider receipt source binding mismatch for {key}"
                    )
            for key, expected in (
                ("code_dataset", target["source_dataset"]),
                ("code_dataset_version", target["source_dataset_version"]),
                ("code_asset_manifest_sha256", target["source_manifest_sha256"]),
                ("parsed_cache_kernel", target["cache_source_kernel"]),
                ("parsed_cache_manifest_sha256", target["cache_manifest_sha256"]),
                ("recovery_dataset", target["recovery_dataset"]),
                ("recovery_dataset_version", target["recovery_dataset_version"]),
                ("recovery_asset_manifest_sha256", target["recovery_manifest_sha256"]),
                ("recovery_asset_upload_receipt_sha256", target["asset_ready_receipt_sha256"]),
            ):
                if source.get(key) != expected:
                    raise ValueError(f"{kernel}: provider receipt source mismatch for {key}")
            if push_receipt.get("rehearsal", {}).get("commit") != target["rehearsal_commit"]:
                raise ValueError(f"{kernel}: shared-artifact rehearsal commit mismatch")
            quota = push_receipt.get("quota_snapshot", {})
            if (
                float(quota.get("gpu_campaign_headroom_before_hold_seconds", 0)) < 8200
                or quota.get("other_open_gpu_holds_seconds") != 0
                or quota.get("provisional_gpu_hold_seconds")
                != target["provisional_reservation_seconds"]
            ):
                raise ValueError(f"{kernel}: push receipt quota does not meet admission")
            bounds = push_receipt.get("worker_bounds", {})
            if (
                bounds.get("provider_timeout_seconds") != target["provider_timeout_seconds"]
                or bounds.get("worker_wall_seconds") != target["worker_max_seconds"]
                or bounds.get("device_count") != 2
                or bounds.get("device_type") != "T4"
                or bounds.get("training_enabled") is not False
                or bounds.get("prediction_precision") != "fp32"
            ):
                raise ValueError(f"{kernel}: export worker bounds differ from the admission")

            asset_admission_path = resolve_repo_path(
                repo_root, config_dir, target["asset_upload_admission"]
            )
            if sha256_file(asset_admission_path) != target["asset_upload_admission_sha256"]:
                raise ValueError(f"{kernel}: exact recovery asset admission hash mismatch")
            asset_admission = json.loads(asset_admission_path.read_text())
            if (
                asset_admission.get("state") != "one_exact_private_dataset_create_authorized"
                or asset_admission.get("dataset") != target["recovery_dataset"]
                or asset_admission.get("version") != target["recovery_dataset_version"]
                or asset_admission.get("asset_manifest_sha256")
                != target["recovery_manifest_sha256"]
                or asset_admission.get("payload_file_count") != 39
                or asset_admission.get("payload_bytes") != 74811058
                or asset_admission.get("private") is not True
            ):
                raise ValueError(f"{kernel}: recovery asset admission does not match the packet")

            asset_ready_path = resolve_repo_path(
                repo_root, config_dir, target["asset_ready_receipt"]
            )
            if sha256_file(asset_ready_path) != target["asset_ready_receipt_sha256"]:
                raise ValueError(f"{kernel}: exact recovery readiness receipt hash mismatch")
            asset_ready = json.loads(asset_ready_path.read_text())
            if (
                asset_ready.get("dataset") != target["recovery_dataset"]
                or asset_ready.get("version") != target["recovery_dataset_version"]
                or asset_ready.get("private") is not True
                or asset_ready.get("provider_status", {}).get("status") != "ready"
                or asset_ready.get("provider_status", {}).get("current_version_number") != 1
            ):
                raise ValueError(f"{kernel}: recovery dataset is not private READY version1")
            if (
                asset_ready.get("admission", {}).get("sha256")
                != target["asset_upload_admission_sha256"]
            ):
                raise ValueError(f"{kernel}: recovery asset receipt admission mismatch")
            readiness = asset_ready.get("local_stage", {})
            if (
                readiness.get("physical_file_count") != 41
                or readiness.get("unique_payload_file_count") != 39
                or readiness.get("unique_payload_bytes") != 74811058
                or readiness.get("manifest_logical_row_count") != 45
                or readiness.get("manifest_sha256") != target["recovery_manifest_sha256"]
                or readiness.get("stage_receipt_uploaded") is not False
            ):
                raise ValueError(f"{kernel}: recovery asset stage inventory mismatch")
            remote_inventory = asset_ready.get("remote_inventory", {})
            manifest_path = resolve_repo_path(
                repo_root, config_dir, target["recovery_asset_manifest"]
            )
            if sha256_file(manifest_path) != target["recovery_manifest_sha256"]:
                raise ValueError(f"{kernel}: pinned recovery manifest bytes changed")
            manifest = json.loads(manifest_path.read_text())
            unique_payloads = {row["filename"]: int(row["size"]) for row in manifest["files"]}
            expected_remote = {
                **unique_payloads,
                "assets-manifest.json": manifest_path.stat().st_size,
            }
            remote_files = remote_inventory.get("files", [])
            observed_remote = {row["name"]: int(row["size"]) for row in remote_files}
            if (
                len(manifest["files"]) != 45
                or len(unique_payloads) != 39
                or remote_inventory.get("file_count") != 40
                or remote_inventory.get("payload_file_count") != 39
                or len(remote_files) != 40
                or observed_remote != expected_remote
                or remote_inventory.get("dataset_metadata_json_listed_as_data_file") is not False
                or asset_ready.get("remote_manifest_sha256") != target["recovery_manifest_sha256"]
            ):
                raise ValueError(f"{kernel}: remote recovery dataset inventory mismatch")
        else:
            if push_receipt.get("canonical_provider_slug") != kernel:
                raise ValueError(f"{kernel}: provider push receipt slug mismatch")
            if push_receipt.get("root_admission_sha256") != target["root_admission_sha256"]:
                raise ValueError(f"{kernel}: provider push receipt admission mismatch")
            if push_receipt.get("push_count") != 1 or push_receipt.get("kernel_version") != 1:
                raise ValueError(f"{kernel}: expected exactly one accepted version")
            packet = push_receipt.get("packet", {})
            for key, expected in target["admission_bindings"].items():
                receipt_value = packet.get(key, push_receipt.get(key, admission.get(key)))
                if receipt_value != expected:
                    raise ValueError(f"{kernel}: provider receipt binding mismatch for {key}")

        for path_field in (
            "artifact_dir",
            "terminal_receipt",
            "push_receipt",
            "asset_ready_receipt",
            "asset_upload_admission",
            "recovery_asset_manifest",
            "source_admission",
        ):
            if path_field in target:
                resolve_repo_path(repo_root, config_dir, target[path_field])

    cli_path = Path(config.get("kaggle_cli", ""))
    if not cli_path.is_absolute() or not cli_path.is_file() or not os.access(cli_path, os.X_OK):
        raise ValueError("configured Kaggle CLI must be an executable absolute path")
    if cli_path.resolve() != (repo_root / ".venv/bin/kaggle").resolve():
        raise ValueError("collector must use the repository-pinned Kaggle CLI executable")
    if config.get("expected_cli_version") != "Kaggle CLI 2.2.4":
        raise ValueError("collector must pin the reviewed Kaggle CLI version")
    if int(config.get("minimum_free_bytes", 0)) < 2 * GIB:
        raise ValueError("minimum disk reserve must be at least 2 GiB")
    if int(config.get("maximum_download_bytes", 0)) < int(config["minimum_free_bytes"]):
        raise ValueError("maximum download size must be at least the disk reserve")
    if int(config["maximum_download_bytes"]) > 8 * GIB:
        raise ValueError("maximum_download_bytes may not exceed the reviewed 8 GiB cap")
    if not 1 <= int(config.get("max_attempts", 0)) <= MAX_ATTEMPTS:
        raise ValueError(f"max_attempts must be in [1, {MAX_ATTEMPTS}]")
    if not 60 <= int(config.get("max_runtime_seconds", 0)) <= 850:
        raise ValueError("max_runtime_seconds must be in [60, 850]")
    resolve_repo_path(repo_root, config_dir, config["poll_log"])
    return config, targets


def terminal_state(status: str) -> bool:
    """Return whether a Kaggle status string represents a terminal state."""
    return status.rsplit(".", maxsplit=1)[-1].upper() in TERMINAL_STATES


def file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    """Return a relative-path, byte-count, and SHA-256 inventory."""
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"refusing to inventory a symlink: {path}")
        if path.is_file():
            inventory[path.relative_to(root).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return inventory


def directory_bytes(root: Path) -> int:
    """Count regular-file bytes beneath a directory without following symlinks."""
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            total += path.stat().st_size
    return total


def collection_is_complete(receipt_path: Path, artifact_dir: Path) -> bool:
    """Check that terminal files still match their receipt sizes and hashes."""
    try:
        receipt = json.loads(receipt_path.read_text())
        collection = receipt["terminal_collection"]
        files = collection["files"]
        root = artifact_dir.resolve()
        return (
            terminal_state(receipt["terminal_status"])
            and collection["output_exit_code"] == 0
            and collection["log_exit_code"] == 0
            and _receipt_files_match(files, root)
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def next_attempt_number(artifact_dir: Path) -> int:
    """Choose a new attempt number while preserving every earlier partial result."""
    numbers = []
    for path in artifact_dir.glob("attempt_*"):
        if path.is_dir() and path.name.removeprefix("attempt_").isdigit():
            numbers.append(int(path.name.removeprefix("attempt_")))
    return max(numbers, default=0) + 1


def run_bounded_command(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    monitor_dir: Path,
    min_free_bytes: int,
    max_bytes: int,
    deadline: float,
) -> dict[str, Any]:
    """Run one Kaggle read-only download, enforcing disk, byte, and wall bounds."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic()
    reason: str | None = None
    process: subprocess.Popen[bytes] | None = None
    with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_stream,
                stderr=stderr_stream,
                shell=False,
            )
            while process.poll() is None:
                if shutil.disk_usage(monitor_dir).free < min_free_bytes:
                    reason = "minimum_free_disk_reserve_reached"
                elif directory_bytes(monitor_dir) > max_bytes:
                    reason = "download_byte_limit_reached"
                elif time.monotonic() >= deadline:
                    reason = "collector_runtime_limit_reached"
                if reason is not None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    break
                time.sleep(1)
            return_code = process.wait()
        except OSError as error:
            return {
                "exit_code": None,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "stop_reason": "process_start_error",
                "error_type": type(error).__name__,
            }
    final_size = directory_bytes(monitor_dir)
    final_free = shutil.disk_usage(monitor_dir).free
    if reason is None and final_size > max_bytes:
        reason = "download_byte_limit_reached_after_process_exit"
    if reason is None and final_free < min_free_bytes:
        reason = "minimum_free_disk_reserve_reached_after_process_exit"
    if reason is not None and return_code == 0:
        return_code = 125
    return {
        "exit_code": return_code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stop_reason": reason,
        "bytes_in_monitor_dir": final_size,
        "free_bytes_after": final_free,
    }


def kaggle_cli_version(cli_path: str) -> dict[str, Any]:
    """Read the exact Kaggle CLI version without contacting the provider."""
    try:
        result = subprocess.run(
            [cli_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"exit_code": None, "version": "", "error_type": type(error).__name__}
    output = (result.stdout or result.stderr).strip()
    return {"exit_code": result.returncode, "version": output}


def _target_status(snapshot: dict[str, Any], kernel: str) -> dict[str, Any] | None:
    return next(
        (row for row in snapshot.get("status", []) if row.get("kernel") == kernel),
        None,
    )


def _terminal_receipt(
    *,
    target: dict[str, Any],
    status_row: dict[str, Any],
    attempt_dir: Path,
    artifact_dir: Path,
    result: dict[str, Any],
    cli_info: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create an auditable receipt, including partial attempts and all file hashes."""
    files = file_inventory(artifact_dir)
    output_count = sum(1 for name in files if name.startswith(f"{attempt_dir.name}/outputs/"))
    output_code = result.get("output", {}).get("exit_code")
    log_code = result.get("logs", {}).get("exit_code")
    complete = output_code == 0 and log_code == 0
    return {
        "schema": "wnn.kaggle-terminal-collector-receipt.v1",
        "recorded_utc": utc_now(),
        "kernel": target["kernel"],
        "resource": target["resource"],
        "root_admission_sha256": target["root_admission_sha256"],
        "push_receipt_sha256": sha256_file(
            resolve_repo_path(
                Path(config["repo_root"]),
                Path(config["config_dir"]),
                target["push_receipt"],
            )
        ),
        "terminal_status": status_row["status"],
        "provider_failure_message": status_row.get("failure_message", ""),
        "terminal_collection": {
            "collection_complete": complete,
            "output_exit_code": output_code,
            "log_exit_code": log_code,
            "attempt_directory": attempt_dir.relative_to(artifact_dir).as_posix(),
            "provider_output_file_count": output_count,
            "empty_provider_outputs_note": (
                "A successful output command returned no provider artifacts; this alone "
                "does not show whether the kernel script executed."
                if complete and output_count == 0
                else None
            ),
            "minimum_free_bytes": int(config["minimum_free_bytes"]),
            "maximum_download_bytes": int(config["maximum_download_bytes"]),
            "cli_executable": config["kaggle_cli"],
            "cli_version": cli_info,
            "files": files,
        },
        "attempt_result": result,
    }


def collect_terminal_target(
    *,
    target: dict[str, Any],
    status_row: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
    config_dir: Path,
) -> dict[str, Any]:
    """Download one terminal kernel's logs and outputs into a new private attempt."""
    artifact_dir = resolve_repo_path(repo_root, config_dir, target["artifact_dir"])
    receipt_path = resolve_repo_path(repo_root, config_dir, target["terminal_receipt"])
    artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(artifact_dir, 0o700)
    if collection_is_complete(receipt_path, artifact_dir):
        return {"kernel": target["kernel"], "state": "already_collected"}

    existing_attempts = next_attempt_number(artifact_dir) - 1
    if existing_attempts >= int(config["max_attempts"]):
        return {
            "kernel": target["kernel"],
            "state": "retry_limit_reached",
            "attempts": existing_attempts,
        }

    attempt_dir = artifact_dir / f"attempt_{existing_attempts + 1:03d}"
    attempt_dir.mkdir(mode=0o700)
    os.chmod(attempt_dir, 0o700)
    snapshot_status_path = attempt_dir / "provider_status.json"
    atomic_write_json(
        snapshot_status_path,
        {
            "observed_utc": utc_now(),
            "kernel": target["kernel"],
            "status": status_row.get("status"),
            "failure_message": status_row.get("failure_message", ""),
        },
    )
    result: dict[str, Any] = {"logs": None, "output": None}
    cli_info = kaggle_cli_version(config["kaggle_cli"])
    cli_matches = (
        cli_info.get("exit_code") == 0 and cli_info.get("version") == config["expected_cli_version"]
    )
    result["cli_version_match"] = cli_matches
    if not cli_matches:
        result["stop_reason"] = "pinned_kaggle_cli_version_unavailable"
    elif shutil.disk_usage(artifact_dir).free < int(config["minimum_free_bytes"]):
        result["stop_reason"] = "minimum_free_disk_reserve_not_met_before_download"
    else:
        deadline = time.monotonic() + int(config["max_runtime_seconds"])
        cli_path = config["kaggle_cli"]
        slug = target["kernel"]
        result["logs"] = run_bounded_command(
            [cli_path, "kernels", "logs", slug],
            stdout_path=attempt_dir / "provider_logs.txt",
            stderr_path=attempt_dir / "provider_logs.stderr.txt",
            monitor_dir=attempt_dir,
            min_free_bytes=int(config["minimum_free_bytes"]),
            max_bytes=int(config["maximum_download_bytes"]),
            deadline=deadline,
        )
        if result["logs"]["exit_code"] == 0 and time.monotonic() < deadline:
            if shutil.disk_usage(artifact_dir).free < int(config["minimum_free_bytes"]):
                result["output"] = {
                    "exit_code": None,
                    "stop_reason": "minimum_free_disk_reserve_not_met_before_output_download",
                }
            else:
                output_dir = attempt_dir / "outputs"
                output_dir.mkdir(mode=0o700)
                result["output"] = run_bounded_command(
                    [cli_path, "kernels", "output", "-p", str(output_dir), "-o", "-q", slug],
                    stdout_path=attempt_dir / "output_command.stdout.txt",
                    stderr_path=attempt_dir / "output_command.stderr.txt",
                    monitor_dir=attempt_dir,
                    min_free_bytes=int(config["minimum_free_bytes"]),
                    max_bytes=int(config["maximum_download_bytes"]),
                    deadline=deadline,
                )
        elif result["logs"]["exit_code"] != 0:
            result["output"] = {
                "exit_code": None,
                "stop_reason": "logs_download_failed; outputs preserved for retry",
            }
        else:
            result["output"] = {
                "exit_code": None,
                "stop_reason": "collector_runtime_limit_reached_before_output_download",
            }

    atomic_write_json(attempt_dir / "attempt_result.json", result)
    for path in artifact_dir.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            os.chmod(path, 0o700)
        elif path.is_file() and not path.is_symlink():
            os.chmod(path, 0o600)
    receipt = _terminal_receipt(
        target=target,
        status_row=status_row,
        attempt_dir=attempt_dir,
        artifact_dir=artifact_dir,
        result=result,
        cli_info=cli_info,
        config=config,
    )
    atomic_write_json(receipt_path, receipt)
    return {
        "kernel": target["kernel"],
        "state": (
            "terminal_outputs_and_logs_collected"
            if receipt["terminal_collection"]["collection_complete"]
            else "terminal_collection_partial"
        ),
        "attempt": attempt_dir.name,
        "provider_output_file_count": receipt["terminal_collection"]["provider_output_file_count"],
        "receipt": str(receipt_path),
        "file_count": len(receipt["terminal_collection"]["files"]),
    }


def run_once(config_path: Path, repo_root: Path) -> dict[str, Any]:
    """Poll the exact configured admitted kernels and collect terminal results."""
    config, targets = validate_config(config_path, repo_root)
    config["repo_root"] = str(repo_root.resolve())
    config["config_dir"] = str(config_path.resolve().parent)
    log_path = resolve_repo_path(
        repo_root.resolve(), config_path.resolve().parent, config["poll_log"]
    )
    try:
        snapshot = collect_provider_snapshot(config_path)
    except Exception as error:
        failure = {
            "observed_at_utc": utc_now(),
            "source": "terminal collector provider status/quota query failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        append_jsonl(log_path, failure)
        append_jsonl(
            log_path,
            {
                "collector_result": {
                    "observed_utc": failure["observed_at_utc"],
                    "state": "provider_poll_error",
                    "error_type": failure["error_type"],
                }
            },
        )
        return {"state": "provider_poll_error", "error_type": failure["error_type"]}
    append_jsonl(log_path, snapshot)
    outcomes = []
    for target in targets:
        row = _target_status(snapshot, target["kernel"])
        if row is None:
            outcomes.append({"kernel": target["kernel"], "state": "status_missing"})
        elif row.get("monitoring_state") == "stopped_after_terminal_evidence_downloaded":
            outcomes.append({"kernel": target["kernel"], "state": "already_collected"})
        elif not terminal_state(str(row.get("status", ""))):
            outcomes.append(
                {
                    "kernel": target["kernel"],
                    "state": "provider_not_terminal",
                    "status": row.get("status"),
                }
            )
        else:
            outcomes.append(
                collect_terminal_target(
                    target=target,
                    status_row=row,
                    config=config,
                    repo_root=repo_root.resolve(),
                    config_dir=config_path.resolve().parent,
                )
            )
    summary = {
        "observed_utc": snapshot.get("observed_at_utc"),
        "quota": snapshot.get("quota"),
        "targets": outcomes,
    }
    append_jsonl(log_path, {"collector_result": summary})
    return summary


def parse_args() -> argparse.Namespace:
    """Parse one-shot collector inputs; scheduling belongs to systemd."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    """Execute one finite poll/download pass and exit without waiting for the timer."""
    arguments = parse_args()
    try:
        result = run_once(arguments.config, arguments.repo_root)
    except Exception as error:  # keep a bounded durable failure instead of retrying a push
        print(
            json.dumps(
                {
                    "state": "collector_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 1 if result.get("state") == "provider_poll_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
