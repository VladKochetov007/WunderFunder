"""Build an identified export-only successor with a scoped payload manifest."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(__file__).resolve().parent
R5_ROOT = PACKAGE_ROOT / "asset_transport_r5"
R6_ROOT = PACKAGE_ROOT / "asset_transport_r6"
R6_BUILD_RECEIPT = R6_ROOT / "H232_V7B_EP3_ASSET_TRANSPORT_R6_BUILD_RECEIPT_R1.json"
R6_TERMINAL_RECEIPT = ASSET_ROOT / "reference/H232_R6_TERMINAL_RECEIPT_20261008_R1.json"
R6_INPUT_INVENTORY = ASSET_ROOT / "reference/H232_R6_PROVIDER_INPUT_INVENTORY_20261008_R1.json"
R6_TERMINAL_LOG = ASSET_ROOT / "reference/H232_R6_LIVE_LOGS_FOLLOW_R1.txt"
R5_TEMPLATE = R5_ROOT / "h232_export_asset_bootstrap.py"
EXPECTED_R5_TEMPLATE_SHA256 = "5abbb4f4540b2db939083d94660f0da532790eca876d27cf6e902ae5da880525"
EXPECTED_R6_PAYLOAD_SHA256 = "dda776b3f9b77f143eae3642ac36c7f7b58d751557de8e440c8775b01df4857b"
EXPECTED_R6_CHECKPOINT_SHA256 = "aaa731da1692ee5083c80296fd665b0a60815c9dc18430457f702cf779c2b829"
EXPECTED_R6_SOURCE_ARCHIVE_SHA256 = (
    "f1333c63eca38e69149e14dbc94c6a701ba97e113d6679b8939d7c860f1a95e4"
)
KERNEL_TITLE = "H232 GRU384 V7B EP3 Export 20261008 R7"
KERNEL_ID = "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r7"
ASSET_DATASET_ID = f"vladkochetov07/h232-v7b-ep3-payload-r7-{EXPECTED_R6_PAYLOAD_SHA256[:12]}"
ASSET_MANIFEST_FILENAME = "h232-v7b-ep3-payload-manifest-r7.json"
DATASET_SOURCES = (
    "vladkochetov07/wunder-funder",
    "vladkochetov07/wnn-c20261004-fold0-assets",
)


def sha256_file(path: Path) -> str:
    """Hash a file with bounded sequential reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject other document shapes."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def kernel_slug(title: str) -> str:
    """Normalize a Kaggle title using lowercase alphanumeric tokens."""
    return "-".join(re.findall(r"[a-z0-9]+", title.casefold()))


def validate_kernel_identity(kernel_id: str, title: str) -> None:
    """Require the title to normalize to the provider kernel slug."""
    owner, separator, slug = kernel_id.partition("/")
    if not owner or not separator or not slug or kernel_slug(title) != slug:
        raise ValueError("R7 kernel title does not normalize to its canonical ID")


def validate_r6_basis() -> tuple[dict[str, Any], Path, dict[str, str]]:
    """Bind the R7 transport to the exact failed R6 archive and terminal evidence."""
    receipt = read_json(R6_BUILD_RECEIPT)
    terminal = read_json(R6_TERMINAL_RECEIPT)
    inventory = read_json(R6_INPUT_INVENTORY)
    terminal_log = R6_TERMINAL_LOG.read_text(encoding="utf-8")
    payload_path = PACKAGE_ROOT / str(receipt["payload_path"])
    members = receipt.get("payload_member_sha256")
    if not isinstance(members, dict) or not all(
        isinstance(name, str) and isinstance(digest, str) for name, digest in members.items()
    ):
        raise ValueError("R6 payload member identity map is malformed")
    if (
        receipt.get("payload_zip_sha256") != EXPECTED_R6_PAYLOAD_SHA256
        or sha256_file(payload_path) != EXPECTED_R6_PAYLOAD_SHA256
    ):
        raise ValueError("R6 immutable payload differs from the reviewed identity")
    if (
        receipt.get("checkpoint_sha256") != EXPECTED_R6_CHECKPOINT_SHA256
        or receipt.get("source_archive_sha256") != EXPECTED_R6_SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("R6 checkpoint or source archive identity changed")
    if (
        terminal.get("status") != "terminal_failure_before_export"
        or terminal.get("execution", {}).get("phase") != "input_identity"
        or "found 2" not in terminal_log
    ):
        raise ValueError("R6 terminal receipt does not document the manifest collision")
    if (
        inventory.get("status") != "observed_provider_file_inventories"
        or inventory.get("basename_collisions_across_all_three_inputs") != ["assets-manifest.json"]
        or inventory.get("r7_remote_inventory_created") is not False
    ):
        raise ValueError("R6 provider inventory does not isolate the single manifest collision")
    with zipfile.ZipFile(payload_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(members):
            raise ValueError("R6 payload member set changed before R7 packaging")
        for name, expected in members.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"R6 payload member changed before R7 packaging: {name}")
    return receipt, payload_path, members


def render_bootstrap(
    template: str,
    *,
    dataset_id: str,
    manifest_filename: str,
    asset_filename: str,
    asset_bytes: int,
    asset_sha256: str,
    payload_bytes: int,
    payload_sha256: str,
    payload_members: dict[str, str],
) -> str:
    """Bind the R7 payload and private manifest without changing worker code."""
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
        'manifest_path = asset.parent / "assets-manifest.json"': (
            f"manifest_path = asset.parent / {manifest_filename!r}"
        ),
        "h232_export_asset_transport_receipt_r5.json": "h232_export_asset_transport_receipt_r7.json",
        "h232_v7b_ep3_verified_payload_r5": "h232_v7b_ep3_verified_payload_r7",
    }
    rendered = template
    for old, new in replacements.items():
        if rendered.count(old) != 1:
            raise ValueError(f"R5 bootstrap binding is not unique: {old}")
        rendered = rendered.replace(old, new)
    ast.parse(rendered)
    if rendered.count(f"manifest_path = asset.parent / {manifest_filename!r}") != 1:
        raise ValueError("R7 bootstrap does not select the unique private manifest by sibling path")
    import_line = "from export_worker import main as run_export"
    if rendered.index('os.environ["CUDA_VISIBLE_DEVICES"]') > rendered.index(import_line):
        raise ValueError("R7 bootstrap CUDA visibility is set after worker import")
    for name in ("WNN_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if rendered.index(f'os.environ["{name}"]') > rendered.index(import_line):
            raise ValueError(f"R7 bootstrap {name} is set after worker import")
    return rendered


def build(*, asset_source: Path, asset_upload: Path, kernel_packet: Path) -> dict[str, Any]:
    """Stage a local R7 packet without making provider calls or changing the payload."""
    if any(path.exists() for path in (asset_source, asset_upload, kernel_packet)):
        raise FileExistsError("refusing to overwrite R7 staging paths")
    validate_kernel_identity(KERNEL_ID, KERNEL_TITLE)
    r6, payload_path, member_hashes = validate_r6_basis()
    r5_template = R5_ROOT / "h232_export_asset_bootstrap.py"
    if sha256_file(r5_template) != EXPECTED_R5_TEMPLATE_SHA256:
        raise ValueError("reviewed R5 bootstrap template changed before R7 rendering")

    payload_sha256 = sha256_file(payload_path)
    payload_bytes = payload_path.stat().st_size
    asset_filename = f"{payload_sha256}.wnn"
    asset_source.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(payload_path, asset_source / "payload.wnn")

    sys.path.insert(0, str(PACKAGE_ROOT.parents[4] / "scripts"))
    from kaggle_assets import prepare

    prepare(asset_source, ["payload.wnn"], asset_upload, ASSET_DATASET_ID)
    generated_manifest = asset_upload / "assets-manifest.json"
    manifest_bytes = generated_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    metadata = read_json(asset_upload / "dataset-metadata.json")
    staged_asset = asset_upload / asset_filename
    if (
        manifest.get("dataset") != ASSET_DATASET_ID
        or manifest.get("private") is not True
        or manifest.get("files")
        != [
            {
                "path": "payload.wnn",
                "size": payload_bytes,
                "sha256": payload_sha256,
                "filename": asset_filename,
            }
        ]
        or metadata.get("id") != ASSET_DATASET_ID
        or metadata.get("isPrivate") is not True
        or staged_asset.stat().st_size != payload_bytes
        or sha256_file(staged_asset) != payload_sha256
    ):
        raise ValueError("R7 private asset helper staged an unexpected payload identity")
    custom_manifest_path = asset_upload / ASSET_MANIFEST_FILENAME
    generated_manifest.rename(custom_manifest_path)
    if generated_manifest.exists() or custom_manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("R7 private payload manifest was not renamed byte-for-byte")
    if any(path.name == "assets-manifest.json" for path in asset_upload.iterdir()):
        raise ValueError("R7 upload still exposes the globally ambiguous manifest basename")

    template = r5_template.read_text(encoding="utf-8")
    code = render_bootstrap(
        template,
        dataset_id=ASSET_DATASET_ID,
        manifest_filename=ASSET_MANIFEST_FILENAME,
        asset_filename=asset_filename,
        asset_bytes=payload_bytes,
        asset_sha256=payload_sha256,
        payload_bytes=payload_bytes,
        payload_sha256=payload_sha256,
        payload_members=member_hashes,
    )
    kernel_packet.mkdir(parents=True, exist_ok=False)
    code_path = kernel_packet / "kernel.py"
    code_path.write_text(code, encoding="utf-8")
    subprocess.run(["ruff", "format", str(code_path)], check=True, capture_output=True, text=True)
    kernel_metadata = {
        "code_file": "kernel.py",
        "competition_sources": [],
        "dataset_sources": [*DATASET_SOURCES, f"{ASSET_DATASET_ID}/1"],
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
    metadata_path = kernel_packet / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(kernel_metadata, indent=2, sort_keys=True) + "\n")
    if code_path.stat().st_size >= 1_000_000:
        raise ValueError("R7 bootstrap exceeds Kaggle's 1 MB source limit")

    root_review_path = R6_ROOT / "H232_R6_ROOT_INDEPENDENT_REVIEW_20261008_R1.json"
    receipt = {
        "schema": "h232-v7b-ep3-r7-manifest-transport-build-v1",
        "record_id": "H232_V7B_EP3_ASSET_TRANSPORT_R7_BUILD_RECEIPT_R1",
        "status": "prepared_for_independent_root_review",
        "kernel_id": KERNEL_ID,
        "kernel_title": KERNEL_TITLE,
        "kernel_packet_path": str(kernel_packet.relative_to(PACKAGE_ROOT)),
        "kernel_source_bytes": code_path.stat().st_size,
        "kernel_source_sha256": sha256_file(code_path),
        "kernel_metadata_sha256": sha256_file(metadata_path),
        "kernel_metadata": kernel_metadata,
        "asset_dataset_id": ASSET_DATASET_ID,
        "asset_dataset_version": 1,
        "asset_dataset_version_pin": f"{ASSET_DATASET_ID}/1",
        "asset_upload_path": str(asset_upload.relative_to(PACKAGE_ROOT)),
        "asset_filename": asset_filename,
        "asset_bytes": payload_bytes,
        "asset_sha256": payload_sha256,
        "asset_manifest_filename": ASSET_MANIFEST_FILENAME,
        "asset_manifest_bytes": custom_manifest_path.stat().st_size,
        "asset_manifest_sha256": sha256_file(custom_manifest_path),
        "r6_payload_path": str(payload_path.relative_to(PACKAGE_ROOT)),
        "r6_payload_sha256": r6["payload_zip_sha256"],
        "r6_payload_bytes": r6["payload_zip_bytes"],
        "payload_member_count": len(member_hashes),
        "payload_member_sha256": member_hashes,
        "checkpoint_sha256": r6["checkpoint_sha256"],
        "source_archive_sha256": r6["source_archive_sha256"],
        "r6_terminal_receipt_sha256": sha256_file(R6_TERMINAL_RECEIPT),
        "r6_input_inventory_receipt_sha256": sha256_file(R6_INPUT_INVENTORY),
        "r6_root_review_path": str(root_review_path.relative_to(PACKAGE_ROOT)),
        "r6_root_review_sha256": sha256_file(root_review_path),
        "input_consumer_basename_collision_repaired": "assets-manifest.json",
        "scientific_source_worker_checkpoint_payload_unchanged": True,
        "root_kernel_push_authorized": False,
        "provider_actions": 0,
        "fit_enabled": False,
        "profile_enabled": False,
        "scoring_enabled": False,
        "correction": "R6's payload manifest collided with the fold0 assets manifest in the unchanged worker's global basename lookup. R7 renames only the new private payload manifest and has the bootstrap open that exact sibling filename.",
    }
    receipt_path = ASSET_ROOT / "H232_V7B_EP3_ASSET_TRANSPORT_R7_BUILD_RECEIPT_R1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    """Build the local R7 packet after exact R6 evidence validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-source", type=Path, default=ASSET_ROOT / "asset_source_r1")
    parser.add_argument("--asset-upload", type=Path, default=ASSET_ROOT / "asset_upload_v1")
    parser.add_argument("--kernel-packet", type=Path, default=ASSET_ROOT / "kernel_packet_r7_v1")
    args = parser.parse_args()
    receipt = build(
        asset_source=args.asset_source.resolve(),
        asset_upload=args.asset_upload.resolve(),
        kernel_packet=args.kernel_packet.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
