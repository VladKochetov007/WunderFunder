from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[5]
R4_OUTPUTS = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    / "cloud_delivery/gpu_loss_pair_r4/terminal_collect_r1/attempt_001/outputs"
)
KERNEL = HERE / "kernel/wnn-loss-pair-export-recovery-20261009-r1.py"
PACKET = HERE / "gpu_loss_pair_export_recovery.packet.json"


def load_kernel():
    spec = importlib.util.spec_from_file_location("gpu_loss_pair_r5_export", KERNEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_packet_binds_exact_kernel_metadata_and_recovery_manifest() -> None:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    metadata_path = HERE / "kernel/kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    code_path = HERE / "kernel" / metadata["code_file"]
    asset_dir = HERE / "source_asset_stage"
    asset_manifest = json.loads((asset_dir / "assets-manifest.json").read_text(encoding="utf-8"))
    stage_receipt = json.loads((asset_dir / "stage_receipt.json").read_text(encoding="utf-8"))

    assert packet["packet_id"] == metadata["id"]
    assert metadata["title"].lower().replace(" ", "-") == packet["packet_id"].split("/", 1)[1]
    assert packet["kernel"]["code_file_sha256"] == digest(code_path)
    assert packet["kernel"]["code_file_size_bytes"] == code_path.stat().st_size
    assert packet["kernel"]["metadata_sha256"] == digest(metadata_path)
    assert packet["recovery_asset"]["manifest_sha256"] == digest(asset_dir / "assets-manifest.json")
    assert packet["recovery_asset"]["metadata_sha256"] == digest(asset_dir / "dataset-metadata.json")
    assert stage_receipt["manifest_sha256"] == packet["recovery_asset"]["manifest_sha256"]
    assert stage_receipt["files"] == asset_manifest["files"]
    assert stage_receipt["file_count"] == len(asset_manifest["files"]) == 45
    assert packet["offline_checks"]["test_sha256"] == digest(Path(__file__))


def test_actual_r4_checkpoints_are_fixed_epoch_one_and_unchanged() -> None:
    module = load_kernel()
    for arm, run_name in module.ARM_RUN.items():
        receipt = module.verify_checkpoint(arm, run_name, R4_OUTPUTS / "out/arms" / arm)
        assert receipt["checkpoint_sha256"] == module.MODEL_SHA[arm]
        assert receipt["epoch1_sha256"] == module.EPOCH1_SHA[arm]
        assert receipt["checkpoint_epoch"] == 1
        assert receipt["provenance_sha256"] == module.PROVENANCE_SHA[arm]


def test_fold0_discovery_hashes_manifest_without_opening_unused_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_kernel()
    actual_manifest = ROOT / "research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json"
    source = json.loads(actual_manifest.read_text(encoding="utf-8"))
    required = {
        "cache/chain_folds.npz",
        "cache/chains.npz",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
    }
    unused_large = [
        row
        for row in source["files"]
        if row["path"] not in required and int(row["size"]) > 1_000_000_000
    ]
    assert unused_large, "fixture should include large mounted prediction artifacts"

    mounted = tmp_path / "datasets/owner/fold0-assets"
    mounted.mkdir(parents=True)
    fixture_manifest = mounted / "assets-manifest.json"
    shutil.copyfile(actual_manifest, fixture_manifest)
    opened_payloads: list[Path] = []
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.suffix == ".wnn":
            opened_payloads.append(path)
            raise AssertionError(f"unused payload was opened: {path.name}")
        return original_open(path, *args, **kwargs)

    digest_calls: list[Path] = []
    original_sha = module.sha

    def guarded_sha(path: Path) -> str:
        digest_calls.append(path)
        assert path == fixture_manifest, f"payload was unexpectedly hashed: {path}"
        return original_sha(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(module, "sha", guarded_sha)
    found, observed = module.discover_fold0_asset_manifest(tmp_path / "datasets")
    assert found == fixture_manifest
    assert observed == source
    assert digest_calls == [fixture_manifest]
    assert opened_payloads == []


def test_prediction_command_is_cuda_fp32_without_unsupported_device_flag() -> None:
    module = load_kernel()
    argv = module.prediction_argv(
        Path("/repo"),
        "loss_pair_r3_wmse",
        Path("/cache/manifest.json"),
        Path("/cache/validation.json"),
        "a" * 64,
    )
    assert "--device" not in argv
    assert argv[argv.index("--inference-precision") + 1] == "fp32"
    assert argv[argv.index("--expected-checkpoint-cache-validation-sha256") + 1] == (
        module.R4_OLD_CACHE_RECEIPT_SHA
    )


def test_patch_targets_actual_predictor_and_preserves_checkpoint_receipt_pin(
    tmp_path: Path,
) -> None:
    module = load_kernel()
    source = R4_OUTPUTS / "wnn/scripts/predict_run.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == module.PREDICTOR_ORIGINAL_SHA
    module.REPO = tmp_path / "repo"
    module.OUT = tmp_path / "out"
    predictor = module.REPO / "scripts/predict_run.py"
    predictor.parent.mkdir(parents=True)
    shutil.copyfile(source, predictor)

    receipt = module.patch_predictor()
    patched = predictor.read_text(encoding="utf-8")
    assert receipt["original_sha256"] == module.PREDICTOR_ORIGINAL_SHA
    assert receipt["patched_sha256"] == hashlib.sha256(predictor.read_bytes()).hexdigest()
    assert patched.count("--expected-checkpoint-cache-validation-sha256") == 1
    assert "!= args.expected_checkpoint_cache_validation_sha256" in patched
    assert "!= args.expected_parsed_cache_validation_sha256" not in patched


def test_prediction_identity_requires_exact_checkpoint_and_fresh_cache() -> None:
    import numpy as np

    module = load_kernel()
    groups = list(range(468))
    group_sha = hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()
    cache_manifest = {
        "membership_artifact": {"sha256": module.MEMBERSHIP_SHA},
    }
    validation_sha = "b" * 64
    sidecar = {
        "run": module.ARM_RUN["wmse"],
        "split": "valid",
        "epoch": 1,
        "checkpoint_sha256": module.MODEL_SHA["wmse"],
        "prediction_sha256": "c" * 64,
        "inference_precision": "fp32",
        "split_manifest_sha256": module.SPLIT_SHA,
        "row_groups": groups,
        "evaluation_membership_sha256": group_sha,
        "parsed_sequence_cache": {
            "cache_key": module.CACHE_KEY,
            "manifest_sha256": module.CACHE_MANIFEST_SHA,
            "validation_receipt_sha256": validation_sha,
            "membership_sha256": module.MEMBERSHIP_SHA,
        },
    }
    module.validate_prediction_metadata(
        "wmse", module.ARM_RUN["wmse"], sidecar, "c" * 64, groups, validation_sha, cache_manifest
    )

    for field, value in (
        ("checkpoint_sha256", "d" * 64),
        ("epoch", 0),
        ("inference_precision", "fp16"),
    ):
        changed = {**sidecar, field: value}
        with pytest.raises(module.Rejected):
            module.validate_prediction_metadata(
                "wmse", module.ARM_RUN["wmse"], changed, "c" * 64, groups, validation_sha, cache_manifest
            )

    stale = json.loads(json.dumps(sidecar))
    stale["parsed_sequence_cache"]["validation_receipt_sha256"] = module.R4_OLD_CACHE_RECEIPT_SHA
    with pytest.raises(module.Rejected):
        module.validate_prediction_metadata(
            "wmse", module.ARM_RUN["wmse"], stale, "c" * 64, groups, validation_sha, cache_manifest
        )
