"""Build an isolated, admission-bound D02 R4 push directory after root approval."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

CANDIDATE_ROOT = Path(__file__).resolve().parent
KERNEL_ID = "vladkochetov07/wnn-d02-muon-loss-gpu-20261009-v4"
CONTROL_KERNEL = "vladkochetov07/wnn-loss-pair-export-recovery-20261009-r1"
CONTROL_CONTRACT_SHA256 = "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7"
SOURCE_DATASET = "vladkochetov07/wnn-d02-muon-loss-code-20261009-r4"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_r4_sealer() -> ModuleType:
    path = CANDIDATE_ROOT / "seal_d02_candidate_r4.py"
    spec = importlib.util.spec_from_file_location("d02_r4_sealer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load frozen R4 sealer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_admission(
    admission: dict[str, Any], *, expected_manifest_sha256: str
) -> None:
    """Require the embedded execution admission to bind this exact D02 source."""
    if admission.get("schema") != "wnn.d02.root-admission.v1":
        raise ValueError("unsupported D02 root admission schema")
    if admission.get("kernel_id") != KERNEL_ID or admission.get("push_allowed") is not True:
        raise ValueError("root admission does not authorize this exact D02 kernel push")
    if admission.get("source_code_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("root admission binds a different source-code manifest")
    if admission.get("gpu_control_contract_sha256") != CONTROL_CONTRACT_SHA256:
        raise ValueError("root admission binds a different completed GPU control")
    if admission.get("control_gpu_kernel") != CONTROL_KERNEL:
        raise ValueError("root admission names a different GPU control kernel")
    runtime_seconds = admission.get("runtime_seconds")
    if type(runtime_seconds) is not int or not 1 <= runtime_seconds <= 3300:
        raise ValueError("root admission runtime must be an integer in [1, 3300]")
    if type(admission.get("conditional_fit_allowed")) is not bool:
        raise ValueError("root admission must explicitly bind the conditional-fit gate")


def _load_admitted_payload(
    admission_path: Path,
    readiness_path: Path,
    *,
    frozen_packet: dict[str, Any],
    sealer: ModuleType,
) -> tuple[dict[str, Any], bytes, str, str]:
    """Validate root's source-admission receipt and the exact READY asset evidence."""
    source_admission_raw = admission_path.read_bytes()
    source_admission_sha256 = hashlib.sha256(source_admission_raw).hexdigest()
    source_admission = json.loads(source_admission_raw)
    if (
        source_admission.get("schema") != "wnn.d02.r4.root-source-admission.v1"
        or source_admission.get("state")
        != "private_source_upload_and_admitted_packet_preparation_authorized"
    ):
        raise ValueError("D02 source-admission receipt is not the admitted R1 record")
    for key, expected in (
        ("pending_packet_sha256", _sha256_file(CANDIDATE_ROOT / "gpu_loss_pair.packet.json")),
        ("source_code_manifest_sha256", frozen_packet["source"]["code_manifest_sha256"]),
        ("source_archive_sha256", frozen_packet["source"]["archive_sha256"]),
        ("source_asset_manifest_sha256", frozen_packet["source"]["asset_manifest_sha256"]),
        ("source_dataset", SOURCE_DATASET),
        ("kernel_id", KERNEL_ID),
        ("control_contract_sha256", CONTROL_CONTRACT_SHA256),
    ):
        if source_admission.get(key) != expected:
            raise ValueError(f"D02 source-admission binding mismatch for {key}")

    readiness_raw = readiness_path.read_bytes()
    readiness_sha256 = hashlib.sha256(readiness_raw).hexdigest()
    readiness = json.loads(readiness_raw)
    if (
        readiness.get("schema") != "wnn.kaggle-d02-private-source-ready-receipt.v1"
        or readiness.get("root_source_admission", {}).get("sha256") != source_admission_sha256
        or readiness.get("dataset") != SOURCE_DATASET
        or readiness.get("private") is not True
        or readiness.get("provider_version") != 1
        or readiness.get("readiness", {}).get("status") != "ready"
        or readiness.get("readiness", {}).get("remote_manifest_sha256")
        != frozen_packet["source"]["asset_manifest_sha256"]
        or readiness.get("readiness", {}).get("remote_manifest_matches_staged_bytes") is not True
    ):
        raise ValueError("source readiness receipt does not prove the exact private version 1")

    payload = source_admission.get("root_admission_payload")
    if not isinstance(payload, dict):
        raise ValueError("root source-admission receipt lacks its embedded execution admission")
    payload_raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    return payload, payload_raw, source_admission_sha256, readiness_sha256


def finalize_candidate(
    admission_path: Path, readiness_path: Path, output_path: Path
) -> dict[str, Any]:
    """Create a fresh review directory; never write into the frozen candidate."""
    candidate = CANDIDATE_ROOT
    output = output_path.resolve()
    if output == candidate or output.is_relative_to(candidate):
        raise ValueError("finalization output must be outside the frozen candidate")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite a finalization directory: {output}")

    frozen_packet_path = candidate / "gpu_loss_pair.packet.json"
    frozen_packet = json.loads(frozen_packet_path.read_text(encoding="utf-8"))
    source_commit = frozen_packet.get("source", {}).get("source_commit")
    if not isinstance(source_commit, str):
        raise ValueError("frozen D02 packet lacks its source commit")

    sealer = _load_r4_sealer()
    base = sealer._load_base_sealer()
    source_manifest_sha256 = sealer._source_manifest_sha256(base, source_commit)
    if source_manifest_sha256 != frozen_packet["source"]["code_manifest_sha256"]:
        raise ValueError("candidate code no longer matches the reviewed manifest")
    admission, admission_raw, source_admission_sha256, readiness_sha256 = _load_admitted_payload(
        admission_path,
        readiness_path,
        frozen_packet=frozen_packet,
        sealer=sealer,
    )
    _validate_admission(admission, expected_manifest_sha256=source_manifest_sha256)

    output.mkdir(parents=True)
    try:
        shutil.copytree(candidate / "code", output / "code", symlinks=False)
        (output / "kernel").mkdir()
        (output / "root_admission.json").write_bytes(admission_raw)
        packet = base.seal_candidate(output, source_commit, output / "root_admission.json")
        packet["schema"] = "wnn.d02.kaggle-muon-loss-packet.v4"
        packet["r5_control_parent"] = sealer._r5_parent_binding()
        packet["control_gpu_kernel"] = CONTROL_KERNEL
        packet["recovery_parent"] = frozen_packet["recovery_parent"]
        packet["root_admission_required_before_dataset_creation_or_kernel_push"] = True
        packet["authorized_for_execution"] = False
        packet["status"] = "root_execution_admission_embedded_waiting_exact_kernel_push_admission"
        packet["experiment"]["control_basis"] = {
            "training_contract_sha256": CONTROL_CONTRACT_SHA256,
            "kernel": CONTROL_KERNEL,
            "retrain": False,
            "science_recipe_change": False,
        }
        packet["experiment"]["batch_tail_semantics"] = (
            "final batch has 30 real sequences plus 2 zero-weight padded sequences"
        )
        packet_path = output / "gpu_loss_pair.packet.json"
        base.write_json(packet_path, packet)

        for field in ("code_manifest_sha256", "archive_sha256", "asset_manifest_sha256"):
            if packet["source"][field] != frozen_packet["source"][field]:
                raise ValueError(f"finalization changed frozen source binding: {field}")
        if packet["kernel"]["metadata_sha256"] != frozen_packet["kernel"]["metadata_sha256"]:
            raise ValueError("finalization changed the reviewed Kaggle metadata")
        if packet["kernel"]["code_file_sha256"] == frozen_packet["kernel"]["code_file_sha256"]:
            raise ValueError("admission was not embedded into a new bootstrap")
        if _sha256_file(output / "source_asset" / "assets-manifest.json") != packet[
            "source"
        ]["asset_manifest_sha256"]:
            raise ValueError("generated source dataset inventory failed its manifest check")
        original_assets = {
            path.name: (path.stat().st_size, _sha256_file(path))
            for path in (candidate / "source_asset").iterdir()
            if path.is_file()
        }
        final_assets = {
            path.name: (path.stat().st_size, _sha256_file(path))
            for path in (output / "source_asset").iterdir()
            if path.is_file()
        }
        if final_assets != original_assets:
            raise ValueError("admission finalization changed the uploaded source dataset bytes")

        receipt = {
            "schema": "wnn.d02.r4.admitted-finalization.v1",
            "state": "finalized_for_root_review_only",
            "provider_mutations": False,
            "source_candidate_packet_sha256": _sha256_file(frozen_packet_path),
            "source_admission_receipt_sha256": source_admission_sha256,
            "source_readiness_receipt_sha256": readiness_sha256,
            "embedded_root_admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
            "source_code_manifest_sha256": source_manifest_sha256,
            "final_packet_sha256": _sha256_file(packet_path),
            "final_bootstrap_sha256": packet["kernel"]["code_file_sha256"],
            "metadata_sha256": packet["kernel"]["metadata_sha256"],
            "asset_manifest_sha256": packet["source"]["asset_manifest_sha256"],
            "kernel_push_admission_required": True,
        }
        (output / "finalization_receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return receipt
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-admission", type=Path, required=True)
    parser.add_argument("--source-readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_candidate(args.source_admission, args.source_readiness, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
