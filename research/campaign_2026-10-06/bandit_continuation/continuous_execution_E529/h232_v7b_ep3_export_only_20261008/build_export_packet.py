"""Build and locally verify the frozen H232 fixed-epoch export-only packet."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import re
import textwrap
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

PACKET_ID = "E529_H232_V7B_EP3_FP32_EXPORT_ONLY_20261008_R2"
KERNEL_ID = "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r3"
KERNEL_TITLE = "H232 GRU384 V7B EP3 Export 20261008 R3"
CHECKPOINT_SHA256 = "aaa731da1692ee5083c80296fd665b0a60815c9dc18430457f702cf779c2b829"
SOURCE_ARCHIVE_SHA256 = "f1333c63eca38e69149e14dbc94c6a701ba97e113d6679b8939d7c860f1a95e4"
SOURCE_MANIFEST_SHA256 = "6ee80717ab4cec6da8b9ef3f124439a7598c2dfde2fa25de9fb35bf47a0adc42"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PARENT_RUN = "bd26_h232_full_s3_f0_amp4096_v7_proactive_20261008"
PARENT_FIT_ATTEMPT = "H232_V7B_SHORTSLUG"
WALL_CAP_SECONDS = 4_500
EXPORT_CAP_SECONDS = 1_800

BASIS_NAMES = {
    "source_archive.tar": "source_archive.tar",
    "source_archive_manifest.json": "source_archive_manifest.json",
    "parent_packet_meta.json": "parent_packet_meta.json",
    "parent_profile_plan.json": "H232_PROFILE_PLAN.json",
    "parent_recipe.json": "H232_AMP4096_SUCCESSOR_RECIPE.json",
    "parent_external_review.json": "external_review.json",
    "parent_profile_worker.py": "profile_worker.py",
    "parent_profile_kernel.py": "parent_profile_kernel.py",
    "cache_pressure_recovery.py": "cache_pressure_recovery.py",
    "gpu_memory_domains.py": "gpu_memory_domains.py",
    "verified_digest_handoff.py": "verified_digest_handoff.py",
    "prior_input_identity_v2.json": "prior_input_identity_v2.json",
    "fit/model_ep3.pt": "model_ep3.pt",
    "fit/provenance.json": "provenance.json",
    "fit/checkpoint_source_review.json": "checkpoint_source_review.json",
    "fit/terminal_receipt.json": "terminal_receipt.json",
}
REVIEWED_PARENT_NAMES = {
    "H232_PROFILE_PLAN.json",
    "H232_AMP4096_SUCCESSOR_RECIPE.json",
    "cache_pressure_recovery.py",
    "gpu_memory_domains.py",
    "prior_input_identity_v2.json",
    "profile_worker.py",
    "profile_kernel.py",
    "source_archive.tar",
    "source_archive_manifest.json",
    "verified_digest_handoff.py",
}
PAYLOAD_NAMES = (
    "H232_AMP4096_SUCCESSOR_RECIPE.json",
    "H232_PROFILE_PLAN.json",
    "cache_pressure_recovery.py",
    "checkpoint_source_review.json",
    "export_worker.py",
    "external_review.json",
    "gpu_memory_domains.py",
    "membership_audit.py",
    "model_ep3.pt",
    "packet_meta.json",
    "prior_input_identity_v2.json",
    "profile_worker.py",
    "provenance.json",
    "source_archive.tar",
    "source_archive_manifest.json",
    "terminal_receipt.json",
    "verified_digest_handoff.py",
)


def sha256_bytes(data: bytes) -> str:
    """Return a SHA-256 hex digest for the supplied bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file using bounded sequential reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_kernel_identity(kernel_id: str, title: str) -> None:
    """Reject provider metadata whose title-derived slug differs from its ID."""
    owner, separator, slug = kernel_id.partition("/")
    title_slug = "-".join(re.findall(r"[a-z0-9]+", title.casefold()))
    if not separator or not owner or not slug or title_slug != slug:
        raise ValueError(f"kernel title slug {title_slug!r} does not match kernel ID {kernel_id!r}")


def _json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object documents."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def patch_worker_identity(source: bytes) -> bytes:
    """Change only the packet identity and wall cap in the prior guard module."""
    text = source.decode("utf-8")
    original_identity = (
        'PACKET_ID = "E529_H232_GRU384_B8_S3_AMP4096_PROACTIVE_CACHE_RECOVERY_20261008_V7A"'
    )
    replacement_identity = f'PACKET_ID = "{PACKET_ID}"'
    original_wall = "WORKER_WALL_CAP_S = 16_000"
    replacement_wall = f"WORKER_WALL_CAP_S = {WALL_CAP_SECONDS:,}".replace(",", "_")
    if text.count(original_identity) != 1 or text.count(original_wall) != 1:
        raise ValueError("parent profile worker execution literals are not unique")
    updated = text.replace(original_identity, replacement_identity).replace(
        original_wall, replacement_wall
    )
    reverted = updated.replace(replacement_identity, original_identity).replace(
        replacement_wall, original_wall
    )
    if reverted != text:
        raise ValueError("profile worker diff extends beyond identity and wall cap")
    return updated.encode("utf-8")


def _copy_basis(basis: Path, packet: Path) -> dict[str, str]:
    """Copy the admitted V7B checkpoint/source evidence into a fresh packet."""
    copied: dict[str, str] = {}
    for basis_relative, packet_name in BASIS_NAMES.items():
        source = basis / basis_relative
        if not source.is_file():
            raise FileNotFoundError(f"required frozen V7B artifact is missing: {source}")
        data = source.read_bytes()
        if packet_name == "parent_profile_kernel.py":
            # Its hash is checked against V7B's reviewed map, but this obsolete
            # training wrapper is deliberately not shipped in the export packet.
            continue
        destination = packet / packet_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if packet_name == "profile_worker.py":
            data = patch_worker_identity(data)
        destination.write_bytes(data)
        copied[packet_name] = sha256_bytes(data)
    for source_name in ("export_worker.py", "membership_audit.py"):
        source = packet.parent / source_name
        if not source.is_file():
            raise FileNotFoundError(f"packet builder source is missing: {source}")
        data = source.read_bytes()
        (packet / source_name).write_bytes(data)
        copied[source_name] = sha256_bytes(data)
    return copied


def _verify_basis(basis: Path, copied: dict[str, str]) -> dict[str, Any]:
    """Bind the chosen model bytes to the accepted V7B source and terminal receipts."""
    parent_meta = _json(basis / "parent_packet_meta.json")
    current = parent_meta.get("current_packet_hashes")
    reviewed = parent_meta.get("reviewed_hashes")
    if not isinstance(current, dict) or not isinstance(reviewed, dict):
        raise ValueError("V7B parent packet does not carry its reviewed hash map")
    if current != reviewed or set(current) != REVIEWED_PARENT_NAMES:
        raise ValueError("V7B source/recipe hashes differ from the admitted reviewed packet")
    for name in REVIEWED_PARENT_NAMES:
        basis_name = (
            "parent_profile_kernel.py"
            if name == "profile_kernel.py"
            else next(key for key, value in BASIS_NAMES.items() if value == name)
        )
        actual = sha256_file(basis / basis_name)
        if actual != current[name]:
            raise ValueError(f"V7B reviewed file digest mismatch: {name}")
    if sha256_file(basis / "source_archive.tar") != SOURCE_ARCHIVE_SHA256:
        raise ValueError("frozen V7B source archive hash changed")
    if sha256_file(basis / "source_archive_manifest.json") != SOURCE_MANIFEST_SHA256:
        raise ValueError("frozen V7B source archive manifest hash changed")
    if sha256_file(basis / "fit/model_ep3.pt") != CHECKPOINT_SHA256:
        raise ValueError("selected fixed-epoch checkpoint bytes changed")
    external_review_sha256 = sha256_file(basis / "parent_external_review.json")
    if external_review_sha256 != parent_meta.get("external_review_sha256"):
        raise ValueError("parent external-review receipt hash differs from V7B packet metadata")
    provenance = _json(basis / "fit/provenance.json")
    if (
        provenance.get("selected_epoch") != 3
        or provenance.get("split_manifest_sha256") != SPLIT_SHA256
        or provenance.get("args", {}).get("name") != PARENT_RUN
        or provenance.get("checkpoint_sha256", {}).get("model_ep3.pt") != CHECKPOINT_SHA256
    ):
        raise ValueError("checkpoint provenance does not bind the fixed V7B epoch-3 attempt")
    review = _json(basis / "fit/checkpoint_source_review.json")
    reviewed_ep3 = (
        review.get("checkpoint_review", {}).get("checkpoint_files", {}).get("model_ep3.pt", {})
    )
    if (
        reviewed_ep3.get("sha256") != CHECKPOINT_SHA256
        or reviewed_ep3.get("checkpoint_epoch_arg") != 3
        or reviewed_ep3.get("all_floating_tensors_finite") is not True
        or reviewed_ep3.get("matches_trainer_provenance") is not True
    ):
        raise ValueError("independent fixed-epoch checkpoint tensor review is not admitted")
    receipt = _json(basis / "fit/terminal_receipt.json")
    if (
        receipt.get("schema") != "h232_v7b_shortslug_terminal_delivery_v1"
        or receipt.get("fit_run_name") != PARENT_RUN
        or receipt.get("phase") != "failed"
        or receipt.get("terminal_failure", {}).get("prediction_artifact_found") is not False
        or receipt.get("source_identity", {}).get("archive_sha256_downloaded")
        != SOURCE_ARCHIVE_SHA256
        or receipt.get("source_identity", {}).get("external_review_sha256")
        != external_review_sha256
    ):
        raise ValueError("V7B terminal receipt does not match the audited failed export attempt")
    terminal_ep3 = receipt.get("fit", {}).get("checkpoint_files", {}).get("model_ep3.pt", {})
    if terminal_ep3.get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("V7B terminal receipt does not bind the fixed epoch-3 checkpoint")
    return {
        "parent_packet_id": parent_meta.get("packet_id"),
        "parent_source_recipe_sha256": parent_meta.get("source_recipe_sha256"),
        "parent_external_review_sha256": external_review_sha256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_archive_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "provenance_sha256": copied["provenance.json"],
        "checkpoint_source_review_sha256": copied["checkpoint_source_review.json"],
        "terminal_receipt_sha256": copied["terminal_receipt.json"],
        "split_manifest_sha256": SPLIT_SHA256,
        "selected_epoch": 3,
        "run_name": PARENT_RUN,
    }


def _zip_payload(packet: Path) -> tuple[bytes, dict[str, str]]:
    """Create a deterministic self-contained ZIP containing exactly the payload set."""
    contents: dict[str, bytes] = {}
    for name in PAYLOAD_NAMES:
        path = packet / name
        if not path.is_file():
            raise FileNotFoundError(f"packet payload is missing: {name}")
        contents[name] = path.read_bytes()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(contents.items()):
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe payload member: {name}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)
    return output.getvalue(), {name: sha256_bytes(data) for name, data in sorted(contents.items())}


def _wrapper(archive_bytes: bytes, payload_hashes: dict[str, str]) -> str:
    """Generate the Kaggle code file with a hash-pinned embedded packet."""
    encoded = base64.b64encode(archive_bytes).decode("ascii")
    encoded_lines = "\n".join(f"    {line!r}" for line in textwrap.wrap(encoded, 100))
    return f'''"""Self-contained H232 fixed-epoch FP32 export-only kernel wrapper."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WNN_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

PACKET_ID = {PACKET_ID!r}
PAYLOAD_ZIP_SHA256 = {sha256_bytes(archive_bytes)!r}
PAYLOAD_FILE_SHA256 = {payload_hashes!r}
PAYLOAD_B64 = (
{encoded_lines}
)


def main() -> None:
    """Verify and unpack the exact frozen packet, then execute its export worker."""
    data = base64.b64decode(PAYLOAD_B64, validate=True)
    if hashlib.sha256(data).hexdigest() != PAYLOAD_ZIP_SHA256:
        raise RuntimeError("H232 export packet ZIP digest mismatch")
    packet_root = Path("/kaggle/working/h232_v7b_ep3_export_packet")
    packet_root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
        names = set()
        for item in archive.infolist():
            relative = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                raise RuntimeError(f"unsafe embedded path: {{item.filename}}")
            if item.filename in names:
                raise RuntimeError(f"duplicate embedded path: {{item.filename}}")
            names.add(item.filename)
            destination = packet_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(item))
    if names != set(PAYLOAD_FILE_SHA256):
        raise RuntimeError("H232 export packet member set differs")
    for name, expected in PAYLOAD_FILE_SHA256.items():
        observed = hashlib.sha256((packet_root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"H232 export payload digest mismatch: {{name}}")
    sys.path.insert(0, str(packet_root))
    from export_worker import main as run_export

    run_export(packet_root)


if __name__ == "__main__":
    main()
'''


def build(basis: Path, packet: Path) -> dict[str, Any]:
    """Assemble packet and build receipts without provider access or execution."""
    if packet.exists() and any(packet.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty packet directory: {packet}")
    packet.mkdir(parents=True, exist_ok=True)
    copied_hashes = _copy_basis(basis, packet)
    frozen_identity = _verify_basis(basis, copied_hashes)
    meta = {
        "schema": "h232-export-only-packet-v1",
        "packet_id": PACKET_ID,
        "parent_fit_attempt": PARENT_FIT_ATTEMPT,
        "parent_run_name": PARENT_RUN,
        "scientific_numeric_successor": False,
        "training_enabled": False,
        "profile_enabled": False,
        "scoring_enabled": False,
        "provider_push_authorized": False,
        "fit_enabled": False,
        "prediction_export_enabled": True,
        "prediction_split": "valid",
        "selected_epoch_zero_based": 3,
        "inference_precision": "fp32",
        "expected_prediction_shape": [1_873, 20_000, 2],
        "expected_evaluation_group_count": 468,
        "worker_wall_cap_seconds": WALL_CAP_SECONDS,
        "prediction_export_cap_seconds": EXPORT_CAP_SECONDS,
        "worker_output_directory": "/kaggle/working/h232_v7b_ep3_export_only_r2_20261008",
        "prediction_is_all_valid_rows": True,
        "scoring_claim_scope": "none; root-owned grader may later score only the exposed 468 evaluation groups",
        "frozen_identity": frozen_identity,
        "payload_file_sha256": copied_hashes,
    }
    (packet / "packet_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    payload_zip, payload_hashes = _zip_payload(packet)
    code = _wrapper(payload_zip, payload_hashes)
    code_path = packet / "h232_v7b_ep3_export_kernel_r2.py"
    code_path.write_text(code, encoding="utf-8")
    kernel_meta = {
        "code_file": code_path.name,
        "competition_sources": [],
        "dataset_sources": [
            "vladkochetov07/wunder-funder",
            "vladkochetov07/wnn-c20261004-fold0-assets",
        ],
        "enable_gpu": True,
        "enable_internet": False,
        "id": KERNEL_ID,
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "script",
        "language": "python",
        "title": KERNEL_TITLE,
    }
    metadata_path = packet / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(kernel_meta, indent=2, sort_keys=True) + "\n")
    verify_packet(packet, payload_zip, payload_hashes)
    receipt = {
        "schema": "h232-export-only-build-receipt-v1",
        "packet_id": PACKET_ID,
        "kernel_id": KERNEL_ID,
        "kernel_title": KERNEL_TITLE,
        "packet_path": str(packet),
        "provider_push_authorized": False,
        "payload_zip_sha256": sha256_bytes(payload_zip),
        "payload_zip_bytes": len(payload_zip),
        "code_file_sha256": sha256_file(code_path),
        "code_file_bytes": code_path.stat().st_size,
        "source_archive_sha256": frozen_identity["source_archive_sha256"],
        "source_archive_manifest_sha256": frozen_identity["source_archive_manifest_sha256"],
        "checkpoint_sha256": frozen_identity["checkpoint_sha256"],
        "provenance_sha256": frozen_identity["provenance_sha256"],
        "external_review_sha256": sha256_file(packet / "external_review.json"),
        "terminal_receipt_sha256": frozen_identity["terminal_receipt_sha256"],
        "checkpoint_source_review_sha256": frozen_identity["checkpoint_source_review_sha256"],
        "split_manifest_sha256": SPLIT_SHA256,
        "payload_member_sha256": payload_hashes,
        "packet_file_sha256": {
            path.name: sha256_file(path) for path in sorted(packet.iterdir()) if path.is_file()
        },
        "parent_reviewed_hashes": _json(basis / "parent_packet_meta.json")["reviewed_hashes"],
        "profile_worker_allowed_changes": ["PACKET_ID literal", "WORKER_WALL_CAP_S:16000→4500"],
        "no_local_provider_or_torch_or_parquet_or_npz_execution": True,
    }
    receipt_path = packet.parent / "H232_V7B_EP3_EXPORT_R3_BUILD_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_packet(packet: Path, payload_zip: bytes, payload_hashes: dict[str, str]) -> None:
    """Check deterministic payload/member hashes, metadata scope and no-fit guard."""
    metadata = _json(packet / "kernel-metadata.json")
    if metadata.get("id") != KERNEL_ID or metadata.get("title") != KERNEL_TITLE:
        raise ValueError("kernel metadata differs from the new export-only identity")
    validate_kernel_identity(metadata["id"], metadata["title"])
    if metadata.get("enable_internet") is not False or metadata.get("enable_gpu") is not True:
        raise ValueError("kernel provider flags differ from the admitted offline GPU lane")
    meta = _json(packet / "packet_meta.json")
    for name in (
        "training_enabled",
        "profile_enabled",
        "scoring_enabled",
        "provider_push_authorized",
    ):
        if meta.get(name) is not False:
            raise ValueError(f"export packet must explicitly disable {name}")
    if (
        meta.get("selected_epoch_zero_based") != 3
        or meta.get("inference_precision") != "fp32"
        or meta.get("prediction_split") != "valid"
        or meta.get("worker_wall_cap_seconds") != WALL_CAP_SECONDS
        or meta.get("prediction_export_cap_seconds") != EXPORT_CAP_SECONDS
    ):
        raise ValueError("export-only execution contract is incomplete")
    worker_text = (packet / "export_worker.py").read_text()
    if "profile_worker.run_profile" in worker_text or "train_rnn.main" in worker_text:
        raise ValueError("export worker contains a training/profile invocation")
    if "--epoch" not in worker_text or '"3"' not in worker_text or '"fp32"' not in worker_text:
        raise ValueError("worker no longer selects fixed epoch 3 FP32 export")
    if len(payload_hashes) != len(PAYLOAD_NAMES) or sha256_bytes(payload_zip) == "":
        raise ValueError("payload archive/member manifest is incomplete")
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as archive:
        if set(archive.namelist()) != set(PAYLOAD_NAMES):
            raise ValueError("payload ZIP contains missing or unexpected members")
        for name, expected in payload_hashes.items():
            if sha256_bytes(archive.read(name)) != expected:
                raise ValueError(f"payload ZIP member hash mismatch: {name}")
    code = (packet / "h232_v7b_ep3_export_kernel_r2.py").read_text()
    if re.search(r"kaggle kernels (push|pull)|subprocess\.run\([^\n]*kaggle", code):
        raise ValueError("kernel code file includes provider control or push behavior")
    assignments: dict[str, Any] = {}
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "PAYLOAD_ZIP_SHA256",
                    "PAYLOAD_FILE_SHA256",
                    "PAYLOAD_B64",
                }:
                    assignments[target.id] = ast.literal_eval(node.value)
    embedded = base64.b64decode("".join(assignments["PAYLOAD_B64"]), validate=True)
    if (
        sha256_bytes(embedded) != assignments.get("PAYLOAD_ZIP_SHA256")
        or embedded != payload_zip
        or assignments.get("PAYLOAD_FILE_SHA256") != payload_hashes
    ):
        raise ValueError("Kaggle code file does not embed the exact reviewed payload")


def main() -> None:
    """Build a unique packet using the precollected private terminal evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", type=Path, default=Path(__file__).parent / "basis")
    parser.add_argument(
        "--packet", type=Path, default=Path(__file__).parent / "packet_r4_metadata_slugfix"
    )
    args = parser.parse_args()
    receipt = build(args.basis.resolve(), args.packet.resolve())
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
