"""R4 binds the reviewed R5 control and requires explicit fit headroom policy."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tpu_packet_runtime_r4 as runtime  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _producer(tmp_path: Path) -> tuple[Path, dict]:
    """Build a tiny producer-shaped fixture with every contracted role."""
    artifacts = {}
    for name in sorted(runtime.REQUIRED_SHARED_ROLES):
        role_path = tmp_path / "shared" / f"{name}.bin"
        role_path.parent.mkdir(parents=True, exist_ok=True)
        role_path.write_bytes(f"fixture:{name}".encode())
        artifacts[name] = {
            "path": f"shared/{name}.bin",
            "size_bytes": role_path.stat().st_size,
            "sha256": _sha(role_path),
        }

    profile = {
        "state": "passed",
        "profile": {
            "batch_size": 32,
            "chunk_rows": 2000,
            "input_width": 211,
            "hidden_size": 256,
            "layers": 2,
            "precision": "float32",
            "tf32": False,
            "attempted_updates": 20,
        },
    }
    _write_json(tmp_path / "shared/device_profile_gate.bin", profile)
    artifacts["device_profile_gate"].update(
        {
            "size_bytes": (tmp_path / "shared/device_profile_gate.bin").stat().st_size,
            "sha256": _sha(tmp_path / "shared/device_profile_gate.bin"),
        }
    )
    receipt = {
        "profile_inputs_sha256": artifacts["profile_inputs"]["sha256"],
        "initial_state_sha256": artifacts["initial_state"]["sha256"],
        "cache_validation_receipt_sha256": artifacts["cache_validation_receipt"]["sha256"],
    }
    _write_json(tmp_path / "shared/profile_inputs_receipt.bin", receipt)
    artifacts["profile_inputs_receipt"].update(
        {
            "size_bytes": (tmp_path / "shared/profile_inputs_receipt.bin").stat().st_size,
            "sha256": _sha(tmp_path / "shared/profile_inputs_receipt.bin"),
        }
    )

    cache = {
        "manifest_sha256": runtime.CACHE_MANIFEST_SHA,
        "cache_key": runtime.CACHE_KEY,
        "membership_sha256": runtime.MEMBERSHIP_SHA,
        "split_manifest_sha256": runtime.SPLIT_SHA,
        "portable_registry_sha256": runtime.REGISTRY_SHA,
        "fold": 0,
        "validation_receipt_sha256": artifacts["cache_validation_receipt"]["sha256"],
        "reader_profile_sha256": artifacts["cache_reader_profile"]["sha256"],
        "source_verification_sha256": artifacts["source_verification"]["sha256"],
        "fit_train_count": 350,
        "fit_valid_count_not_used_for_fit": 28,
        "evaluation_valid_count": 468,
    }
    recovery = {
        "mode": "export_only",
        "predecessor_kernel": runtime.R4_FIT_KERNEL,
        "predecessor_terminal_receipt_sha256": runtime.R4_TERMINAL_RECEIPT_SHA,
        "predecessor_root_fit_review_sha256": runtime.R4_FIT_REVIEW_SHA,
        "recovery_asset_manifest_sha256": runtime.R5_RECOVERY_ASSET_SHA,
        "checkpoint_cache_receipt_sha256": runtime.R4_ORIGINAL_CACHE_RECEIPT_SHA,
        "predictor_original_sha256": runtime.R5_PREDICTOR_ORIGINAL_SHA,
        "fresh_runtime_cache_receipt_sha256": artifacts["cache_validation_receipt"]["sha256"],
        "fits_retrained": False,
        "checkpoint_files_rewritten": False,
    }
    contract = {
        "schema": "wnn.loss-pair.training-contract.v1",
        "format_version": 1,
        "state": "complete",
        "recovery": recovery,
        "evaluation": {
            "state": "pending_independent_grade",
            "metric_id": "global_wp_clip2_v1",
            "fold": 0,
            "fixed_evaluation_count": 468,
        },
        "cache": cache,
        "source": {
            "dataset": runtime.R4_CODE_DATASET,
            "source_stage_kernel": runtime.CPU_KERNEL,
        },
        "model": {
            "family": "two-layer GRU",
            "input_width": 211,
            "hidden_size": 256,
            "layers": 2,
            "activation": "GELU",
            "input_layout": {"raw_features": 112, "causal_event_features": 99},
            "outputs": 2,
            "precision": "float32",
            "tf32": False,
        },
        "fit": {
            "batch_size": 32,
            "chunk_rows": 2000,
            "epochs": 2,
            "updates_per_arm": 220,
            "optimizer": "AdamW",
            "scheduler": "OneCycleLR",
            "global_gradient_clip": 1.0,
            "learning_rate_max": 0.001,
            "checkpoint_epoch_zero_indexed": 1,
        },
        "arms": {
            "wmse": {"loss": "wmse", "huber_delta": None, "updates_attempted": 220},
            "huber_d1": {"loss": "huber", "huber_delta": 1.0, "updates_attempted": 220},
        },
        "artifacts": artifacts,
    }
    contract_path = tmp_path / "training_contract.json"
    _write_json(contract_path, contract)
    admission = {"gpu_training_contract_sha256": _sha(contract_path)}
    return contract_path, admission


def test_actual_contract_shape_and_all_fourteen_roles_are_required():
    assert len(runtime.REQUIRED_SHARED_ROLES) == 14
    assert runtime.GPU_KERNEL == "vladkochetov07/wnn-loss-pair-export-recovery-20261009-r1"
    assert runtime.R4_FIT_KERNEL != runtime.GPU_KERNEL


def test_producer_shaped_contract_and_role_hashes_pass(tmp_path):
    path, admission = _producer(tmp_path)
    contract, resolved = runtime.verify_shared_contract(path, admission)
    assert contract["recovery"]["mode"] == "export_only"
    assert set(resolved) == runtime.REQUIRED_SHARED_ROLES


def test_changed_role_bytes_fail_even_when_contract_itself_is_pinned(tmp_path):
    path, admission = _producer(tmp_path)
    (tmp_path / "shared/initial_state.bin").write_bytes(b"changed producer state")
    with pytest.raises(ValueError, match="shared artifact changed"):
        runtime.verify_shared_contract(path, admission)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda contract: contract["artifacts"].pop("sample_orders"), "omits shared roles"),
        (lambda contract: contract["recovery"].update(fits_retrained=True), "export-only recovery"),
        (
            lambda contract: contract["recovery"].update(predecessor_kernel="wrong"),
            "export-only recovery",
        ),
        (lambda contract: contract["fit"].update(batch_size=16), "completed matched FP32 control"),
    ],
)
def test_wrong_producer_identity_or_recipe_fails(tmp_path, mutate, message):
    path, admission = _producer(tmp_path)
    contract = json.loads(path.read_text())
    mutate(contract)
    _write_json(path, contract)
    admission["gpu_training_contract_sha256"] = _sha(path)
    with pytest.raises(ValueError, match=message):
        runtime.verify_shared_contract(path, admission)


def test_pending_or_mismatched_root_admission_cannot_launch():
    manifest_sha = "a" * 64
    contract_sha = "b" * 64
    manifest = {
        "gpu_export_recovery_producer_r5": {
            "training_contract_sha256": contract_sha,
            "root_terminal_review_sha256": runtime.R5_ROOT_REVIEW_SHA,
        }
    }
    admission = {
        "schema": "wnn.loss-pair-tpu.root-admission.v1",
        "source_manifest_sha256": manifest_sha,
        "cache_manifest_sha256": runtime.CACHE_MANIFEST_SHA,
        "cache_key": runtime.CACHE_KEY,
        "gpu_kernel": runtime.GPU_KERNEL,
        "cpu_kernel": runtime.CPU_KERNEL,
        "gpu_training_contract_sha256": contract_sha,
        "gpu_root_terminal_review_sha256": runtime.R5_ROOT_REVIEW_SHA,
        "push_allowed": False,
        "conditional_fit_allowed": False,
        "runtime_seconds": 3300,
        "tpu_compute_precision": "fp32",
        "native_bf16_fit_allowed": False,
    }
    with pytest.raises(ValueError, match="pending root/provider"):
        runtime.check_admission(admission, manifest_sha, manifest)
    runtime.check_admission(admission, manifest_sha, manifest, require_push=False)
    with pytest.raises(ValueError, match="identity differs"):
        runtime.check_admission(admission, "d" * 64, manifest, require_push=False)
    admission["push_allowed"] = True
    runtime.check_admission(admission, manifest_sha, manifest)
    admission["gpu_training_contract_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="identity differs"):
        runtime.check_admission(admission, manifest_sha, manifest)


def test_conditional_fit_requires_root_specified_headroom_policy():
    manifest_sha = "a" * 64
    contract_sha = "b" * 64
    manifest = {
        "gpu_export_recovery_producer_r5": {
            "training_contract_sha256": contract_sha,
            "root_terminal_review_sha256": runtime.R5_ROOT_REVIEW_SHA,
        }
    }
    admission = {
        "schema": "wnn.loss-pair-tpu.root-admission.v1",
        "source_manifest_sha256": manifest_sha,
        "cache_manifest_sha256": runtime.CACHE_MANIFEST_SHA,
        "cache_key": runtime.CACHE_KEY,
        "gpu_kernel": runtime.GPU_KERNEL,
        "cpu_kernel": runtime.CPU_KERNEL,
        "gpu_training_contract_sha256": contract_sha,
        "gpu_root_terminal_review_sha256": runtime.R5_ROOT_REVIEW_SHA,
        "push_allowed": True,
        "conditional_fit_allowed": True,
        "runtime_seconds": 3300,
        "tpu_compute_precision": "fp32",
    }
    with pytest.raises(ValueError, match="approved headroom reserves"):
        runtime.check_admission(admission, manifest_sha, manifest)
    admission["fit_headroom_policy"] = {
        "schema": "wnn.loss-pair-tpu.fit-headroom-policy.v1",
        "minimum_device_free_bytes": 1,
        "minimum_host_available_bytes": 1,
        "minimum_cgroup_headroom_bytes": 1,
        "minimum_disk_free_bytes": 1,
    }
    runtime.check_admission(admission, manifest_sha, manifest)
