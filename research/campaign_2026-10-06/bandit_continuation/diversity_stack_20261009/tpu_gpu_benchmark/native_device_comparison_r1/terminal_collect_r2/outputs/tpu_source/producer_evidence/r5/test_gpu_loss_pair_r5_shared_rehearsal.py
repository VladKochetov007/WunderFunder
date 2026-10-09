from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[5]
R4_OUTPUTS = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    / "cloud_delivery/gpu_loss_pair_r4/terminal_collect_r1/attempt_001/outputs"
)
KERNEL = HERE / "kernel/wnn-loss-pair-export-recovery-20261009-r1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_actual_collect_shared_rehearsal_with_synthetic_prediction_records(
    tmp_path: Path,
) -> None:
    producer = _load(KERNEL, "gpu_loss_pair_r5_export_rehearsal")
    worker = _load(
        R4_OUTPUTS / "wnn/scripts/loss_pair_gpu_worker.py",
        "r4_loss_pair_gpu_worker_rehearsal",
    )

    out = tmp_path / "rehearsal/out"
    out.mkdir(parents=True)
    (out / "recovery").mkdir()
    producer.OUT = out
    recovery_root = HERE / "source_asset_stage"
    recovery_manifest_path = recovery_root / "assets-manifest.json"
    recovery_manifest = json.loads(recovery_manifest_path.read_text(encoding="utf-8"))
    assert _digest(recovery_manifest_path) == producer.RECOVERY_MANIFEST_SHA
    assert producer.recover_outputs(recovery_root, recovery_manifest) == 45

    # Production R5 writes a fresh receipt. The rehearsal uses the preserved R4 receipt.
    shutil.copyfile(
        out / "recovery/r4_fold0_assets_validation.json",
        out / "shared/fold0_assets_validation.json",
    )
    arms = {arm: out / "arms" / arm for arm in producer.ARM_RUN}
    cpu_output = (
        ROOT
        / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
        / "cloud_delivery/cpu_source_stage_artifact_observation_r1/provider_output"
    )
    manifest_path = cpu_output / "out/parsed_cache/manifest.json"
    runtime_validation = R4_OUTPUTS / "out/shared/cache_validation_runtime.json"
    runtime_profile = R4_OUTPUTS / "out/shared/cache_reader_profile_runtime.json"
    source_receipt = json.loads(
        (cpu_output / "out/source_stage_receipt.json").read_text(encoding="utf-8")
    )
    bootstrap_receipt = json.loads(
        (R4_OUTPUTS / "out/bootstrap_receipt.json").read_text(encoding="utf-8")
    )

    worker.ROOT = R4_OUTPUTS / "wnn"
    worker.OUT = out
    worker.WORK = out.parent
    worker.SOURCE_DATASET = producer.R4_CODE
    worker._STARTED = time.monotonic()
    worker._SAMPLES = []
    worker._COMMANDS = []
    fit_rows = [
        {"label": "fit_wmse", "wall_seconds": None},
        {"label": "fit_huber_d1", "wall_seconds": None},
    ]
    prediction_rows = {
        arm: {
            "label": f"predict_{arm}",
            "wall_seconds": None,
            "record_kind": "synthetic_rehearsal_no_prediction_process_was_run",
        }
        for arm in producer.ARM_RUN
    }
    contract, profile_receipt = worker._collect_shared(
        arms,
        producer.ARM_RUN,
        runtime_validation,
        runtime_profile,
        source_receipt,
        bootstrap_receipt,
        manifest_path,
        fit_rows,
        prediction_rows,
    )

    required_roles = {
        "cache_manifest",
        "cache_validation_receipt",
        "cache_reader_profile",
        "source_stage_receipt",
        "source_verification",
        "initial_state",
        "normalization",
        "normalization_metadata",
        "sample_orders",
        "onecycle_trace",
        "profile_inputs",
        "profile_inputs_receipt",
        "device_profile_gate",
        "fold0_asset_validation_receipt",
    }
    assert required_roles.issubset(contract["artifacts"])
    for artifact in contract["artifacts"].values():
        path = out / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["size_bytes"]
        assert _digest(path) == artifact["sha256"]
    assert profile_receipt["schema"] == "wnn.loss-pair.profile-inputs-receipt.v1"
    for arm in ("wmse", "huber_d1"):
        timing = contract["arms"][arm]["timing"]
        assert timing["prediction_export_wall_seconds"] is None
        assert not (out / f"arms/{arm}/predictions_valid.npy").exists()
    assert len(contract["artifacts"]) == 14
