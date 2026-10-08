from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

# JSON documents and provider receipts are validated at an external-file boundary.
CONTINUATION_ROOT = Path(__file__).resolve().parents[2]
R8_ROOT = Path(__file__).resolve().parent
R7_ROOT = R8_ROOT.parent / "asset_transport_r7"
R7_ID = "vladkochetov07/h232-gru384-v7b-ep3-export-20261008-r7"
R8_ID = "vladkochetov07/h232-ep3-export-r8-20261008"
R8_TITLE = "H232 EP3 Export R8 20261008"
ASSET_ID = "vladkochetov07/h232-v7b-ep3-payload-r7-dda776b3f9b7"
ASSET_VERSION = 1
PAYLOAD_SHA256 = "dda776b3f9b77f143eae3642ac36c7f7b58d751557de8e440c8775b01df4857b"
SOURCE_ARCHIVE_SHA256 = "f1333c63eca38e69149e14dbc94c6a701ba97e113d6679b8939d7c860f1a95e4"
SOURCE_ROOT_NAME = "h232_v7b_ep3_export_source_r2"
CHECKPOINT_SHA256 = "aaa731da1692ee5083c80296fd665b0a60815c9dc18430457f702cf779c2b829"
WORKER_SHA256 = "e5568c871f6c918330c424dafc03ece7018e2059d019c6f87ac72352fdf82201"
EXTERNAL_REVIEW_SHA256 = "3d00f9cf9796c1511625af368fb59a3c4574a505ca9bdeaa88e4fe14a32e5949"
R7_TERMINAL_RECEIPT = R8_ROOT / "reference/H232_R7_TERMINAL_POLL_RECEIPT_20261008_R1.json"
R7_STDERR = R8_ROOT / "reference/H232_R7_PREDICTION_STDERR_R1.txt"
R7_ASSET_VERIFICATION = (
    R8_ROOT / "reference/H232_R7_PRIVATE_ASSET_UPLOAD_VERIFICATION_20261008_R1.json"
)
R7_ROOT_REVIEW = (
    R8_ROOT / "reference/H232_R7_ROOT_REVIEW_CONDITIONAL_EXPORT_AUTHORIZATION_20261008_R1.json"
)
R7_WORKER = R8_ROOT / "reference/R7_export_worker.py"
R8_SOURCE_ARCHIVE = R8_ROOT / "reference/R7_source_archive.tar"
SMOKE_RECEIPT = R8_ROOT / "H232_R8_ACTUAL_PREDICTOR_IMPORT_SMOKE_20261008_R1.json"
KERNEL_PACKET = R8_ROOT / "kernel_packet_r8_v5"
BUILD_RECEIPT = R8_ROOT / "H232_V7B_EP3_IMPORTPATH_R8_BUILD_RECEIPT_R5.json"


def sha256_file(path: Path) -> str:
    """Hash a file with bounded sequential reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load and validate one JSON object from a packet or receipt."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def kernel_slug(title: str) -> str:
    """Normalize a provider title into its lowercase alphanumeric slug."""
    return "-".join(re.findall(r"[a-z0-9]+", title.casefold()))


def validate_r7_basis() -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind R8 to R7's exact payload and diagnosed terminal import failure."""
    build_receipt = read_json(R7_ROOT / "H232_V7B_EP3_ASSET_TRANSPORT_R7_BUILD_RECEIPT_R1.json")
    terminal = read_json(R7_TERMINAL_RECEIPT)
    asset_verification = read_json(R7_ASSET_VERIFICATION)
    root_review = read_json(R7_ROOT_REVIEW)
    smoke = read_json(SMOKE_RECEIPT)
    members = build_receipt.get("payload_member_sha256")
    download_entries = asset_verification.get("download_archive", {}).get("entry_hashes", {})
    payload_entry = download_entries.get(f"{PAYLOAD_SHA256}.wnn", {})
    manifest_entry = download_entries.get("h232-v7b-ep3-payload-manifest-r7.json", {})
    if not isinstance(members, dict):
        raise ValueError("R7 payload member receipt is malformed")
    if (
        build_receipt.get("kernel_id") != R7_ID
        or build_receipt.get("asset_dataset_version_pin") != f"{ASSET_ID}/{ASSET_VERSION}"
        or build_receipt.get("asset_sha256") != PAYLOAD_SHA256
        or build_receipt.get("asset_bytes") != payload_entry.get("bytes")
        or payload_entry.get("sha256") != PAYLOAD_SHA256
        or build_receipt.get("asset_manifest_sha256") != manifest_entry.get("sha256")
        or sha256_file(R7_ROOT / "asset_upload_v1/h232-v7b-ep3-payload-manifest-r7.json")
        != manifest_entry.get("sha256")
        or build_receipt.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or build_receipt.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or build_receipt.get("scientific_source_worker_checkpoint_payload_unchanged") is not True
        or asset_verification.get("record_id")
        != "H232_R7_PRIVATE_ASSET_UPLOAD_VERIFICATION_20261008_R1"
        or asset_verification.get("dataset_id") != ASSET_ID
        or asset_verification.get("version_pin") != f"{ASSET_ID}/{ASSET_VERSION}"
        or asset_verification.get("expected_remote_identity_verified") is not True
        or asset_verification.get("source_payload_unchanged") is not True
        or asset_verification.get("dataset_status", {}).get("status") != "ready"
        or root_review.get("canonical_kernel") != R7_ID
        or root_review.get("science_payload_unchanged") is not True
        or root_review.get("conditional_one_push_authorized") is not True
        or root_review.get("artifacts", {}).get("kernel_packet_r7_v1/kernel.py")
        != build_receipt.get("kernel_source_sha256")
    ):
        raise ValueError("R7 frozen payload or build identity changed")
    if (
        len(members) != 18
        or members.get("source_archive.tar") != SOURCE_ARCHIVE_SHA256
        or members.get("model_ep3.pt") != CHECKPOINT_SHA256
        or members.get("export_worker.py") != WORKER_SHA256
        or members.get("external_review.json") != EXTERNAL_REVIEW_SHA256
        or sha256_file(R7_WORKER) != WORKER_SHA256
        or sha256_file(R8_SOURCE_ARCHIVE) != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("R7 member receipts or source fixtures changed")
    worker_text = R7_WORKER.read_text(encoding="utf-8")
    if (
        f'SOURCE_ROOT_NAME = "{SOURCE_ROOT_NAME}"' not in worker_text
        or "environment = os.environ.copy()" not in worker_text
    ):
        raise ValueError("R8 import-path binding differs from the unchanged child worker")
    if (
        terminal.get("kernel") != R7_ID
        or terminal.get("kernel_version") != 1
        or terminal.get("kernel_statuses", {}).get(R7_ID, {}).get("status")
        != "KernelWorkerStatus.ERROR"
        or terminal.get("diagnosis", {}).get("exception")
        != "ModuleNotFoundError: No module named 'wnn'"
        or terminal.get("terminal_collection", {}).get("file_count") != 98
        or "ModuleNotFoundError: No module named 'wnn'" not in R7_STDERR.read_text()
    ):
        raise ValueError("R7 terminal record does not establish the import-path failure")
    if (
        smoke.get("status") != "passed"
        or smoke.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or smoke.get("return_code") != 0
        or "usage:" not in str(smoke.get("stdout", "")).lower()
    ):
        raise ValueError("actual predictor import smoke receipt is missing or failed")
    return build_receipt, terminal


def render_kernel(template: str) -> str:
    """Render a new kernel identity and propagate extracted src to child Python."""
    replacements = {
        f'KERNEL_ID = "{R7_ID}"': f'KERNEL_ID = "{R8_ID}"',
        'KERNEL_TITLE = "H232 GRU384 V7B EP3 Export 20261008 R7"': (f'KERNEL_TITLE = "{R8_TITLE}"'),
        "h232_export_asset_transport_receipt_r7.json": (
            "h232_export_asset_transport_receipt_r8.json"
        ),
        "h232_v7b_ep3_verified_payload_r7": "h232_v7b_ep3_verified_payload_r8",
    }
    rendered = template
    for old, new in replacements.items():
        if rendered.count(old) != 1:
            raise ValueError(f"R7 template binding is not unique: {old}")
        rendered = rendered.replace(old, new)
    insertion = "    sys.path.insert(0, str(destination))\n"
    environment = (
        '    working_root = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))\n'
        f'    source_root = working_root / "{SOURCE_ROOT_NAME}"\n'
        '    source_package = source_root / "src"\n'
        '    inherited_pythonpath = os.environ.get("PYTHONPATH", "")\n'
        '    os.environ["PYTHONPATH"] = os.pathsep.join(\n'
        "        [str(source_package), *([inherited_pythonpath] if inherited_pythonpath else [])]\n"
        "    )\n"
    )
    if rendered.count(insertion) != 1:
        raise ValueError("R7 template import entry point changed")
    rendered = rendered.replace(insertion, environment + insertion)
    ast.parse(rendered)
    if rendered.index('os.environ["PYTHONPATH"]') > rendered.index(
        "from export_worker import main as run_export"
    ):
        raise ValueError("child PYTHONPATH is set after the export worker import")
    if kernel_slug(R8_TITLE) != R8_ID.split("/", maxsplit=1)[1]:
        raise ValueError("R8 title does not normalize to its provider kernel ID")
    return rendered


def build() -> dict[str, Any]:
    """Build the R8 kernel packet without changing or reuploading its payload."""
    if KERNEL_PACKET.exists() or BUILD_RECEIPT.exists():
        raise FileExistsError("refusing to overwrite an existing R8 packet")
    basis, terminal = validate_r7_basis()
    template_path = R7_ROOT / "kernel_packet_r7_v1" / "kernel.py"
    metadata_path = R7_ROOT / "kernel_packet_r7_v1" / "kernel-metadata.json"
    kernel_code = render_kernel(template_path.read_text(encoding="utf-8"))
    metadata = read_json(metadata_path)
    if (
        metadata.get("id") != R7_ID
        or metadata.get("title") != "H232 GRU384 V7B EP3 Export 20261008 R7"
        or metadata.get("dataset_sources", [])[-1] != f"{ASSET_ID}/{ASSET_VERSION}"
    ):
        raise ValueError("R7 metadata source identity differs from the frozen payload")
    metadata["id"] = R8_ID
    metadata["title"] = R8_TITLE
    KERNEL_PACKET.mkdir(parents=True, exist_ok=False)
    kernel_path = KERNEL_PACKET / "kernel.py"
    metadata_out = KERNEL_PACKET / "kernel-metadata.json"
    kernel_path.write_text(kernel_code, encoding="utf-8")
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    receipt: dict[str, Any] = {
        "record_id": "H232_V7B_EP3_IMPORTPATH_R8_BUILD_RECEIPT_R5",
        "schema": "h232-v7b-ep3-importpath-r8-source-only-v1",
        "status": "prepared_for_independent_root_review",
        "created_utc": "2026-10-08",
        "kernel_id": R8_ID,
        "kernel_title": R8_TITLE,
        "kernel_slug_matches_title": kernel_slug(R8_TITLE) == R8_ID.split("/", maxsplit=1)[1],
        "kernel_packet_path": str(KERNEL_PACKET.relative_to(CONTINUATION_ROOT)),
        "kernel_source_bytes": kernel_path.stat().st_size,
        "kernel_source_sha256": sha256_file(kernel_path),
        "kernel_metadata_sha256": sha256_file(metadata_out),
        "reused_private_asset_dataset_version": f"{ASSET_ID}/{ASSET_VERSION}",
        "payload_sha256": PAYLOAD_SHA256,
        "payload_bytes": basis["asset_bytes"],
        "payload_member_count": len(basis["payload_member_sha256"]),
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "export_worker_sha256": WORKER_SHA256,
        "external_review_sha256": EXTERNAL_REVIEW_SHA256,
        "r7_terminal_receipt_sha256": sha256_file(R7_TERMINAL_RECEIPT),
        "r7_prediction_stderr_sha256": sha256_file(R7_STDERR),
        "r7_remote_asset_verification_sha256": sha256_file(R7_ASSET_VERIFICATION),
        "r7_root_review_sha256": sha256_file(R7_ROOT_REVIEW),
        "actual_child_import_smoke_receipt_sha256": sha256_file(SMOKE_RECEIPT),
        "source_change": (
            "bootstrap prepends the unchanged worker's extracted source-root src to inherited "
            "PYTHONPATH before worker import"
        ),
        "science_or_model_changes": False,
        "payload_or_worker_changes": False,
        "training_profile_scoring_or_grading_enabled": False,
        "provider_actions": 0,
        "fresh_quota_admission_performed": False,
        "root_push_authorized": False,
        "r7_parent_kernel_id": R7_ID,
        "r7_terminal_status": terminal.get("kernel_statuses", {}).get(R7_ID, {}).get("status"),
        "r7_failure": terminal.get("diagnosis", {}).get("exception"),
    }
    BUILD_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    build()
