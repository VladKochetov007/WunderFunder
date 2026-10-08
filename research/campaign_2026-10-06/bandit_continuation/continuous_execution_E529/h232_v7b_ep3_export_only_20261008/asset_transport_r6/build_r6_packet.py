"""Build an identified private-asset successor with the omitted parent receipt."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

# Receipts mirror heterogeneous provider JSON fields, so their decoded values
# cannot be expressed as one narrower static type without a larger schema layer.
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(__file__).resolve().parent
R5_ROOT = PACKAGE_ROOT / "asset_transport_r5"
R5_BUILD_RECEIPT = R5_ROOT / "H232_V7B_EP3_ASSET_TRANSPORT_R5_BUILD_RECEIPT_R1.json"
R5_VERIFICATION_RECEIPT = R5_ROOT / "H232_V7B_EP3_ASSET_TRANSPORT_R5_VERIFICATION_RECEIPT.json"
R5_TEMPLATE = R5_ROOT / "h232_export_asset_bootstrap.py"
EXPECTED_R5_TEMPLATE_SHA256 = "5abbb4f4540b2db939083d94660f0da532790eca876d27cf6e902ae5da880525"
EXPECTED_PARENT_METADATA_BYTES = 4_204
EXPECTED_PARENT_METADATA_SHA256 = "97e01d0c09a4e7dd67c13da6757e6f5bc72e2906c4232ec52415af8fdbc95e20"
KERNEL_TITLE = "H232 GRU384 V7B EP3 Export 20261008 R6"
KERNEL_ID = "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r6"
DATASET_SOURCES = (
    "vladkochetov07/wunder-funder",
    "vladkochetov07/wnn-c20261004-fold0-assets",
)


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file through bounded sequential reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject other document shapes."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def kernel_slug(title: str) -> str:
    """Return the lowercase alphanumeric Kaggle slug for a title."""
    return "-".join(re.findall(r"[a-z0-9]+", title.casefold()))


def validate_kernel_identity(kernel_id: str, title: str) -> None:
    """Require exact title-to-ID slug normalization."""
    owner, separator, slug = kernel_id.partition("/")
    if not owner or not separator or not slug or kernel_slug(title) != slug:
        raise ValueError("R6 kernel title does not normalize to its canonical ID")


def repackage_payload(
    r5_payload_path: Path,
    parent_metadata_path: Path,
    expected_r5_bytes: int,
    expected_r5_sha256: str,
    expected_r5_members: dict[str, str],
) -> tuple[bytes, dict[str, str]]:
    """Add the exact omitted parent metadata while retaining every R5 member."""
    if (
        r5_payload_path.stat().st_size != expected_r5_bytes
        or sha256_file(r5_payload_path) != expected_r5_sha256
    ):
        raise ValueError("R5 private payload differs from its immutable build receipt")
    if (
        parent_metadata_path.stat().st_size != EXPECTED_PARENT_METADATA_BYTES
        or sha256_file(parent_metadata_path) != EXPECTED_PARENT_METADATA_SHA256
    ):
        raise ValueError("frozen parent packet metadata differs from the required identity")

    with zipfile.ZipFile(r5_payload_path, "r") as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or set(names) != set(expected_r5_members):
            raise ValueError("R5 archive members differ from its immutable build receipt")
        members = {name: archive.read(name) for name in names}
    for name, expected in expected_r5_members.items():
        if sha256_bytes(members[name]) != expected:
            raise ValueError(f"R5 archive member changed: {name}")

    packet_metadata = json.loads(members["packet_meta.json"])
    required_payload_members = packet_metadata.get("payload_file_sha256")
    if not isinstance(required_payload_members, dict) or any(
        not isinstance(name, str) or not isinstance(digest, str)
        for name, digest in required_payload_members.items()
    ):
        raise ValueError("embedded packet_meta payload hash map is malformed")
    if "parent_packet_meta.json" in members:
        raise ValueError("R5 payload unexpectedly contains the missing parent member")
    if set(required_payload_members) - {"parent_packet_meta.json"} != set(members) - {
        "packet_meta.json"
    }:
        raise ValueError(
            "R5 archive and inner worker-required member sets drift beyond parent metadata"
        )

    parent_bytes = parent_metadata_path.read_bytes()
    if len(parent_bytes) != EXPECTED_PARENT_METADATA_BYTES:
        raise ValueError("frozen parent packet metadata has an unexpected byte count")
    parent_sha256 = sha256_bytes(parent_bytes)
    if required_payload_members.get("parent_packet_meta.json") != parent_sha256:
        raise ValueError("parent metadata does not match the embedded worker-required digest")
    for name, expected in required_payload_members.items():
        if name != "parent_packet_meta.json" and sha256_bytes(members[name]) != expected:
            raise ValueError(f"inner worker-required member differs from packet_meta: {name}")

    members["parent_packet_meta.json"] = parent_bytes
    if set(members) - {"packet_meta.json"} != set(required_payload_members):
        raise ValueError("R6 archive does not close the exact inner payload member set")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name])
    payload = output.getvalue()
    member_hashes = {name: sha256_bytes(data) for name, data in sorted(members.items())}
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        if set(archive.namelist()) != set(member_hashes):
            raise ValueError("R6 archive writer changed the exact member set")
        for name, expected in member_hashes.items():
            if sha256_bytes(archive.read(name)) != expected:
                raise ValueError(f"R6 archive writer changed member bytes: {name}")
    return payload, member_hashes


def render_bootstrap(
    template: str,
    *,
    dataset_id: str,
    asset_filename: str,
    asset_bytes: int,
    asset_sha256: str,
    payload_bytes: int,
    payload_sha256: str,
    payload_members: dict[str, str],
) -> str:
    """Bind the R6 private asset and exact 18-member payload into the bootstrap."""
    replacements = {
        'ASSET_DATASET_ID = "__ASSET_DATASET_ID__"': f"ASSET_DATASET_ID = {dataset_id!r}",
        "ASSET_DATASET_VERSION = 0": "ASSET_DATASET_VERSION = 1",
        'ASSET_FILENAME = "__ASSET_FILENAME__"': f"ASSET_FILENAME = {asset_filename!r}",
        "ASSET_BYTES = 0": f"ASSET_BYTES = {asset_bytes}",
        'ASSET_SHA256 = "__ASSET_SHA256__"': f"ASSET_SHA256 = {asset_sha256!r}",
        "PAYLOAD_ZIP_BYTES = 0": f"PAYLOAD_ZIP_BYTES = {payload_bytes}",
        'PAYLOAD_ZIP_SHA256 = "__PAYLOAD_ZIP_SHA256__"': (
            f"PAYLOAD_ZIP_SHA256 = {payload_sha256!r}"
        ),
        "PAYLOAD_MEMBER_SHA256: dict[str, str] = {}": (
            f"PAYLOAD_MEMBER_SHA256: dict[str, str] = {payload_members!r}"
        ),
        'KERNEL_ID = "__KERNEL_ID__"': f"KERNEL_ID = {KERNEL_ID!r}",
        'KERNEL_TITLE = "__KERNEL_TITLE__"': f"KERNEL_TITLE = {KERNEL_TITLE!r}",
        "h232_export_asset_transport_receipt_r5.json": "h232_export_asset_transport_receipt_r6.json",
        "h232_v7b_ep3_verified_payload_r5": "h232_v7b_ep3_verified_payload_r6",
    }
    rendered = template
    for old, new in replacements.items():
        if rendered.count(old) != 1:
            raise ValueError(f"R5 bootstrap binding is not unique: {old}")
        rendered = rendered.replace(old, new)
    ast.parse(rendered)
    import_line = "from export_worker import main as run_export"
    if rendered.index('os.environ["CUDA_VISIBLE_DEVICES"]') > rendered.index(import_line):
        raise ValueError("R6 bootstrap CUDA visibility is set after worker import")
    for name in ("WNN_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if rendered.index(f'os.environ["{name}"]') > rendered.index(import_line):
            raise ValueError(f"R6 bootstrap {name} is set after worker import")
    return rendered


def build(
    *,
    asset_source: Path,
    asset_upload: Path,
    kernel_packet: Path,
) -> dict[str, Any]:
    """Stage the corrected asset and kernel without provider calls."""
    r5 = read_json(R5_BUILD_RECEIPT)
    if r5.get("kernel_id") != "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r5-asset":
        raise ValueError("R5 build receipt has an unexpected canonical identity")
    verification = read_json(R5_VERIFICATION_RECEIPT)
    if (
        verification.get("payload_identity", {}).get(
            "scientific_worker_profile_checkpoint_and_payload_changed"
        )
        is not False
    ):
        raise ValueError(
            "R5 runtime repair receipt does not attest unchanged science and checkpoint"
        )
    if any(path.exists() for path in (asset_source, asset_upload, kernel_packet)):
        raise FileExistsError("refusing to overwrite R6 asset or kernel packet paths")
    validate_kernel_identity(KERNEL_ID, KERNEL_TITLE)

    r5_payload_path = R5_ROOT / "asset_upload_v1" / str(r5["asset_file"])
    parent_metadata_path = PACKAGE_ROOT / "packet/parent_packet_meta.json"
    payload, member_hashes = repackage_payload(
        r5_payload_path,
        parent_metadata_path,
        int(r5["payload_zip_bytes"]),
        str(r5["payload_zip_sha256"]),
        r5["payload_member_sha256"],
    )
    if len(member_hashes) != 18 or "parent_packet_meta.json" not in member_hashes:
        raise ValueError("R6 transport archive does not contain all 18 expected members")
    basis_dir = ASSET_ROOT / "basis"
    basis_dir.mkdir(exist_ok=False)
    (basis_dir / "parent_packet_meta.json").write_bytes(parent_metadata_path.read_bytes())
    payload_sha256 = sha256_bytes(payload)
    asset_filename = f"{payload_sha256}.wnn"
    asset_dataset = f"vladkochetov07/h232-v7b-ep3-payload-r6-{payload_sha256[:12]}"
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        packet_metadata = json.loads(archive.read("packet_meta.json"))
    inner_required_members = packet_metadata["payload_file_sha256"]
    inner_required_digest = sha256_bytes(
        json.dumps(inner_required_members, sort_keys=True, separators=(",", ":")).encode()
    )
    asset_source.mkdir(parents=True, exist_ok=False)
    (asset_source / "payload.wnn").write_bytes(payload)

    sys.path.insert(0, str(PACKAGE_ROOT.parents[4] / "scripts"))
    from kaggle_assets import prepare

    prepare(asset_source, ["payload.wnn"], asset_upload, asset_dataset)
    asset_manifest = read_json(asset_upload / "assets-manifest.json")
    dataset_metadata = read_json(asset_upload / "dataset-metadata.json")
    staged_asset = asset_upload / asset_filename
    if (
        asset_manifest.get("dataset") != asset_dataset
        or asset_manifest.get("private") is not True
        or asset_manifest.get("files")
        != [
            {
                "path": "payload.wnn",
                "size": len(payload),
                "sha256": payload_sha256,
                "filename": asset_filename,
            }
        ]
        or dataset_metadata.get("id") != asset_dataset
        or dataset_metadata.get("isPrivate") is not True
        or staged_asset.stat().st_size != len(payload)
        or sha256_file(staged_asset) != payload_sha256
    ):
        raise ValueError("R6 private asset helper staged an unexpected payload identity")

    if sha256_file(R5_TEMPLATE) != EXPECTED_R5_TEMPLATE_SHA256:
        raise ValueError("R5 reviewed bootstrap template changed before R6 rendering")
    code = render_bootstrap(
        R5_TEMPLATE.read_text(encoding="utf-8"),
        dataset_id=asset_dataset,
        asset_filename=asset_filename,
        asset_bytes=len(payload),
        asset_sha256=payload_sha256,
        payload_bytes=len(payload),
        payload_sha256=payload_sha256,
        payload_members=member_hashes,
    )
    kernel_packet.mkdir(parents=True, exist_ok=False)
    code_path = kernel_packet / "kernel.py"
    code_path.write_text(code, encoding="utf-8")
    subprocess.run(["ruff", "format", str(code_path)], check=True, capture_output=True, text=True)
    metadata = {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": [*DATASET_SOURCES, f"{asset_dataset}/1"],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": KERNEL_ID,
        "is_private": True,
        "kernel_type": "script",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "title": KERNEL_TITLE,
        "kernel_sources": [],
    }
    (kernel_packet / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if code_path.stat().st_size >= 1_000_000:
        raise ValueError("R6 bootstrap exceeds Kaggle's 1 MB source limit")
    receipt = {
        "schema": "h232-v7b-ep3-r6-asset-transport-build-v1",
        "record_id": "H232_V7B_EP3_ASSET_TRANSPORT_R6_BUILD_RECEIPT_R1",
        "status": "prepared_for_independent_root_review",
        "kernel_id": KERNEL_ID,
        "kernel_title": KERNEL_TITLE,
        "kernel_packet_path": str(kernel_packet.relative_to(PACKAGE_ROOT)),
        "kernel_source_bytes": code_path.stat().st_size,
        "kernel_source_sha256": sha256_file(code_path),
        "kernel_metadata_sha256": sha256_file(kernel_packet / "kernel-metadata.json"),
        "kernel_metadata": metadata,
        "asset_dataset_id": asset_dataset,
        "asset_dataset_version": 1,
        "asset_dataset_version_pin": f"{asset_dataset}/1",
        "asset_upload_path": str(asset_upload.relative_to(PACKAGE_ROOT)),
        "asset_filename": asset_filename,
        "asset_bytes": len(payload),
        "asset_sha256": payload_sha256,
        "asset_manifest_sha256": sha256_file(asset_upload / "assets-manifest.json"),
        "payload_member_count": len(member_hashes),
        "payload_member_sha256": member_hashes,
        "parent_packet_meta_sha256": member_hashes["parent_packet_meta.json"],
        "r5_payload_sha256": r5["payload_zip_sha256"],
        "r5_payload_bytes": r5["payload_zip_bytes"],
        "r5_payload_member_sha256": r5["payload_member_sha256"],
        "r5_template_sha256": EXPECTED_R5_TEMPLATE_SHA256,
        "export_worker_sha256": member_hashes["export_worker.py"],
        "checkpoint_sha256": member_hashes["model_ep3.pt"],
        "source_archive_sha256": r5["source_archive_sha256"],
        "source_archive_manifest_sha256": r5["source_archive_manifest_sha256"],
        "inner_required_member_count": len(inner_required_members),
        "inner_required_member_map_sha256": inner_required_digest,
        "inner_required_member_sha256": inner_required_members,
        "root_kernel_push_authorized": False,
        "provider_actions": 0,
        "fit_enabled": False,
        "profile_enabled": False,
        "scoring_enabled": False,
        "scientific_source_worker_checkpoint_unchanged": True,
        "correction": "Add exact parent_packet_meta.json required by unchanged embedded packet_meta.json; no R5 members removed or modified.",
    }
    payload_path = ASSET_ROOT / "r6_payload.wnn"
    if payload_path.exists():
        raise FileExistsError("refusing to overwrite the R6 payload archive")
    payload_path.write_bytes(payload)
    receipt["payload_path"] = str(payload_path.relative_to(PACKAGE_ROOT))
    receipt["payload_zip_bytes"] = len(payload)
    receipt["payload_zip_sha256"] = payload_sha256
    receipt_path = ASSET_ROOT / "H232_V7B_EP3_ASSET_TRANSPORT_R6_BUILD_RECEIPT_R1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    """Build the corrected R6 private asset and small bootstrap packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-source", type=Path, default=ASSET_ROOT / "asset_source_r1")
    parser.add_argument("--asset-upload", type=Path, default=ASSET_ROOT / "asset_upload_v1")
    parser.add_argument("--kernel-packet", type=Path, default=ASSET_ROOT / "kernel_packet_r6_v1")
    args = parser.parse_args()
    receipt = build(
        asset_source=args.asset_source.resolve(),
        asset_upload=args.asset_upload.resolve(),
        kernel_packet=args.kernel_packet.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
