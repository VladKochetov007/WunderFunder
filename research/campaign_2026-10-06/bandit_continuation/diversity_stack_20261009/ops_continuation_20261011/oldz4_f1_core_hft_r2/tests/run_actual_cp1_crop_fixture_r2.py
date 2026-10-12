from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

PACKET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OPS_ROOT = PROJECT_ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/"
    "diversity_stack_20261009/ops_continuation_20261011"
)
R5_PACKET = OPS_ROOT / "oldz4_student_core_hft_f0_r5"
SOURCE_SCRIPT = R5_PACKET / "source_snapshots/core_crop_fe8/scripts/make_core_checkpoint.py"
VERIFIER_SCRIPT = R5_PACKET / "prepare_core_crop_r5.py"
SOURCE_RUN = "oldz4_h230_student_f1_s3_r4"
DESTINATION_RUN = "fixture_oldz4_h230_core_f1_s3_r2"
RECEIPT_PATH = PACKET_ROOT / "ACTUAL_F1_CP1_CROP_FIXTURE_R2.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_from_path(path: Path, module_name: str):
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("set PYTHONDONTWRITEBYTECODE=1 before importing frozen producers")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen producer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"preserve existing fixture receipt: {RECEIPT_PATH}")
    binding_path = PACKET_ROOT / "SOURCE_INPUT_BINDINGS_R2.json"
    bindings = json.loads(binding_path.read_text(encoding="utf-8"))
    pins = bindings["f1_student_cp1"]
    source_checkpoint = Path(pins["paths"]["checkpoint"])
    source_provenance = Path(pins["paths"]["provenance"])
    for key, path in (("checkpoint", source_checkpoint), ("provenance", source_provenance)):
        if sha256_file(path) != pins["sha256"][key]:
            raise ValueError(f"actual F1 parent differs from its binding: {key}")

    r5 = bindings["core_crop"]
    if sha256_file(SOURCE_SCRIPT) != r5["r5_crop_source_sha256"]:
        raise ValueError("actual F0 crop producer differs from the R2 binding")
    if sha256_file(VERIFIER_SCRIPT) != r5["r5_actual_output_verifier_sha256"]:
        raise ValueError("actual F0 crop verifier differs from the R2 binding")

    temp_root = Path("/tmp/wnn_f1_r2_actual_cp1_crop_fixture")
    if temp_root.exists():
        raise FileExistsError(f"preserve existing fixture directory: {temp_root}")
    source_dir = temp_root / "artifacts/runs" / SOURCE_RUN
    source_dir.mkdir(parents=True)
    (source_dir / "model_ep1.pt").symlink_to(source_checkpoint)
    (source_dir / "provenance.json").symlink_to(source_provenance)

    producer = import_from_path(SOURCE_SCRIPT, "f1_r2_actual_crop_producer")
    verifier = import_from_path(VERIFIER_SCRIPT, "f1_r2_actual_crop_verifier")
    producer_result = producer.derive_core(temp_root, SOURCE_RUN, 1, DESTINATION_RUN)
    destination = temp_root / "artifacts/runs" / DESTINATION_RUN
    output_checkpoint = destination / "model_ep1.pt"
    output_provenance = destination / "provenance.json"
    verification = verifier.verify_crop(
        source_checkpoint,
        source_provenance,
        output_checkpoint,
        output_provenance,
        source_run=SOURCE_RUN,
        source_epoch=1,
    )
    if producer_result.get("checkpoint_sha256") != verification["output_checkpoint_sha256"]:
        raise ValueError("actual producer result differs from the frozen downstream verifier")

    receipt = {
        "schema": "wnn.f1-actual-cp1-crop-consumer-fixture.r2",
        "status": "PASS_ACTUAL_SOURCE_AND_DOWNSTREAM_CONSUMER",
        "fixture_only": True,
        "root_admission_used": False,
        "main_artifacts_written": False,
        "device_started": False,
        "created_utc": datetime.now(UTC).isoformat(),
        "temporary_project_root": str(temp_root),
        "source_run": SOURCE_RUN,
        "source_checkpoint_sha256": pins["sha256"]["checkpoint"],
        "source_provenance_sha256": pins["sha256"]["provenance"],
        "source_training_state_sha256": pins["sha256"]["training_state"],
        "producer_source_sha256": r5["r5_crop_source_sha256"],
        "consumer_source_sha256": r5["r5_actual_output_verifier_sha256"],
        "producer_result": producer_result,
        "consumer_verification": verification,
        "output_checkpoint_path": str(output_checkpoint),
        "output_provenance_path": str(output_provenance),
    }
    with RECEIPT_PATH.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
