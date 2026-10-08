from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "packet"
PARENT = ROOT.parent / "h232_conditional_v7b_reviewed_20261008"
PARENT_PACKET = PARENT / "packet"
PARENT_BUILD_RECEIPT = PARENT / "V7B_BUILD_RECEIPT.json"
PARENT_TEST_RECEIPT = PARENT / "V7B_TEST_RECEIPT.json"
PUSH_DIAGNOSIS = PARENT / "V7B_PUSH_FAILURE_DIAGNOSIS_20261008_R1.json"
PRIOR_ROOT_ADMISSION = ROOT / "basis/H232_V7B_ROOT_PUSH_ADMISSION_20261008_R1.json"
PUSH_FAILURE_RECEIPT = ROOT / "basis/H232_V7B_PROVIDER_PUSH_FAILURE_20261008_R1.json"
KERNEL_ID = "vladkochetov07/h232-gru384-b8-amp4096-v7b-shortslug"
KERNEL_TITLE = "H232 GRU384 B8 AMP4096 V7B Shortslug"
KERNEL_SLUG = KERNEL_ID.split("/", maxsplit=1)[1]
EXPECTED_PACKET_FILE_COUNT = 14


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_basis() -> dict[str, str]:
    basis = ROOT / "basis"
    basis.mkdir(parents=True, exist_ok=True)
    for source in (PARENT / "basis").iterdir():
        if not source.is_file():
            continue
        destination = basis / source.name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Successor basis file differs from its V7B source: {source.name}")
        if not destination.exists():
            shutil.copyfile(source, destination)
    hashes = {path.name: sha256_file(path) for path in sorted(basis.iterdir()) if path.is_file()}
    required = {
        "V7B_BUILD_RECEIPT.json",
        "V7B_TEST_RECEIPT.json",
        "V7B_ROOT_DELIVERY_AUTHORIZATION_20261008_R1.json",
        "V7B_PUSH_FAILURE_DIAGNOSIS_20261008_R1.json",
        "H232_V7B_ROOT_PUSH_ADMISSION_20261008_R1.json",
        "H232_V7B_PROVIDER_PUSH_FAILURE_20261008_R1.json",
        "H232_V7_ROOT_PUSH_ADMISSION_20261008_R1.json",
        "H232_V7_PROVIDER_PUSH_20261008_R1.json",
    }
    missing = required - set(hashes)
    if missing:
        raise ValueError(f"Successor provenance basis is incomplete: {sorted(missing)}")
    return hashes


def prepare_packet(parent_hashes: dict[str, str]) -> dict[str, str]:
    if len(parent_hashes) != EXPECTED_PACKET_FILE_COUNT:
        raise ValueError(f"Expected {EXPECTED_PACKET_FILE_COUNT} parent packet hashes")
    if not PARENT_PACKET.is_dir():
        raise FileNotFoundError(f"Parent packet is missing: {PARENT_PACKET}")

    PACKET.mkdir(parents=True, exist_ok=True)
    expected_names = set(parent_hashes)
    existing_names = {path.name for path in PACKET.iterdir() if path.is_file()}
    if existing_names - expected_names:
        raise ValueError(f"Unexpected successor packet files: {sorted(existing_names - expected_names)}")

    for name in sorted(expected_names):
        source = PARENT_PACKET / name
        destination = PACKET / name
        if not source.is_file() or sha256_file(source) != parent_hashes[name]:
            raise ValueError(f"Parent V7B file no longer matches its build receipt: {name}")
        if name == "kernel-metadata.json":
            continue
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Successor changed protected V7B packet file: {name}")
        if not destination.exists():
            shutil.copyfile(source, destination)

    parent_metadata = read_json(PARENT_PACKET / "kernel-metadata.json")
    metadata = dict(parent_metadata)
    metadata["id"] = KERNEL_ID
    metadata["title"] = KERNEL_TITLE
    rendered = json.dumps(metadata, indent=2) + "\n"
    metadata_path = PACKET / "kernel-metadata.json"
    if metadata_path.exists():
        current = read_json(metadata_path)
        if current != metadata:
            allowed_before_rebuild = dict(current)
            allowed_before_rebuild["id"] = parent_metadata["id"]
            allowed_before_rebuild["title"] = parent_metadata["title"]
            if allowed_before_rebuild != parent_metadata:
                raise ValueError("Existing successor kernel metadata has unexpected changes")
    metadata_path.write_text(rendered, encoding="utf-8")

    observed_names = {path.name for path in PACKET.iterdir() if path.is_file()}
    if observed_names != expected_names:
        raise ValueError(f"Successor packet file set differs: {sorted(observed_names ^ expected_names)}")
    hashes = {name: sha256_file(PACKET / name) for name in sorted(expected_names)}
    for name in sorted(expected_names - {"kernel-metadata.json"}):
        if hashes[name] != parent_hashes[name]:
            raise ValueError(f"Metadata-only successor changed reviewed packet member: {name}")
    if hashes["kernel-metadata.json"] == parent_hashes["kernel-metadata.json"]:
        raise ValueError("Successor metadata unexpectedly matches the rejected 51-character slug")
    return hashes


def prepare_source_tree() -> dict[str, str]:
    manifest = read_json(PARENT_PACKET / "source_archive_manifest.json")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != manifest.get("member_count"):
        raise ValueError("V7B source archive manifest is malformed")
    parent_source = PARENT / "source"
    successor_source = ROOT / "source"
    successor_source.mkdir(parents=True, exist_ok=True)

    expected: dict[str, str] = {}
    for member in members:
        if not isinstance(member, dict):
            raise ValueError("V7B source archive manifest contains a malformed member")
        relative = PurePosixPath(str(member["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe V7B source path: {relative}")
        name = relative.as_posix()
        source = parent_source.joinpath(*relative.parts)
        destination = successor_source.joinpath(*relative.parts)
        if (
            not source.is_file()
            or source.stat().st_size != member["bytes"]
            or sha256_file(source) != member["sha256"]
        ):
            raise ValueError(f"Parent V7B extracted source differs from its archive manifest: {name}")
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Successor source changed protected V7B source member: {name}")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        expected[name] = member["sha256"]

    observed_names = {
        path.relative_to(successor_source).as_posix()
        for path in successor_source.rglob("*")
        if path.is_file()
    }
    if observed_names != set(expected):
        raise ValueError(f"Successor source tree differs: {sorted(observed_names ^ set(expected))}")
    return expected


def build() -> dict[str, Any]:
    parent_receipt = read_json(PARENT_BUILD_RECEIPT)
    parent_hashes = parent_receipt["packet_files_14"]
    if not isinstance(parent_hashes, dict):
        raise ValueError("V7B parent receipt has no packet hash map")
    diagnosis = read_json(PUSH_DIAGNOSIS)
    if diagnosis.get("sdk_error_body", {}).get("status") != "not_available_to_diagnosing_agent":
        raise ValueError("Expected the documented, uncaptured V7B SaveKernel error body")
    prior_admission = read_json(PRIOR_ROOT_ADMISSION)
    push_failure = read_json(PUSH_FAILURE_RECEIPT)
    if (
        prior_admission.get("authorized") is not True
        or prior_admission.get("canonical_expected")
        != "vladkochetov07/h232-gru384-b8-amp4096-v7b-proactive-cache-recovery"
        or prior_admission.get("push_count_authorized") != 1
        or prior_admission.get("reviewed_packet_hashes") != parent_hashes
    ):
        raise ValueError("Prior V7B push admission is missing or does not bind the reviewed parent payload")
    if (
        push_failure.get("push_count") != 1
        or push_failure.get("http_status") != 400
        or push_failure.get("operation") != "SaveKernel"
        or push_failure.get("response_body_captured") is not False
        or push_failure.get("accepted_execution_proven") is not False
        or push_failure.get("canonical_expected") != prior_admission.get("canonical_expected")
    ):
        raise ValueError("Root V7B push-failure receipt does not bind the single failed SaveKernel attempt")

    basis_hashes = prepare_basis()
    successor_hashes = prepare_packet(parent_hashes)
    source_tree_hashes = prepare_source_tree()
    packet_meta = read_json(PACKET / "packet_meta.json")
    external_review = read_json(PACKET / "external_review.json")
    kernel_meta = read_json(PACKET / "kernel-metadata.json")
    if kernel_meta.get("id") != KERNEL_ID or kernel_meta.get("title") != KERNEL_TITLE:
        raise ValueError("Successor Kaggle metadata does not match the declared canonical title and slug")
    title_slug = "-".join(KERNEL_TITLE.lower().split())
    if title_slug != KERNEL_SLUG:
        raise ValueError("Successor title does not normalize to its canonical kernel slug")
    if len(KERNEL_SLUG) >= 50:
        raise ValueError("Successor slug must remain below the suspected provider limit")
    if packet_meta.get("provider_push_authorized") is not False:
        raise ValueError("Packet metadata must keep provider push authorization disabled")
    if external_review.get("provider_push_authorized") is not False:
        raise ValueError("External review must not authorize provider push")

    report = """# H232 V7B short-slug metadata successor

The single V7B `SaveKernel` push returned HTTP 400. Its public response body was
not captured, so the cause remains unconfirmed. The rejected V7B slug has 51
characters; accepted V7A's slug has 50. This successor uses the distinct,
title-aligned 36-character slug `h232-gru384-b8-amp4096-v7b-shortslug` as a
metadata-only correction candidate.

The successor packet retains the exact approved V7B external-review receipt,
packet metadata, embedded execution wrapper, scientific source archive, source
manifest, recipe, split, profile, and resource guards. Only
`kernel-metadata.json` differs from V7B. The independent review still does not
authorize a provider push; a fresh root admission is required. This builder
does not contact Kaggle and makes no fit or score claim.
"""
    report_path = ROOT / "H232_V7B_SHORTSLUG_REPORT_V1.md"
    report_path.write_text(report, encoding="utf-8")

    receipt = {
        "schema": "h232-v7b-shortslug-metadata-successor-v1",
        "variant": "V7B_SHORTSLUG",
        "experiment_id": parent_receipt["experiment_id"],
        "status": "metadata_successor_built_pending_fresh_root_admission",
        "provider_push_authorized": False,
        "provider_status": "not_queried",
        "kernel_id": KERNEL_ID,
        "kernel_title": KERNEL_TITLE,
        "slug_characters": len(KERNEL_SLUG),
        "builder_sha256": sha256_file(Path(__file__)),
        "focused_preflight_test_sha256": sha256_file(
            ROOT / "tests/test_shortslug_packet_preflight.py"
        ),
        "parent_v7b_build_receipt_sha256": sha256_file(PARENT_BUILD_RECEIPT),
        "parent_v7b_test_receipt_sha256": sha256_file(PARENT_TEST_RECEIPT),
        "v7b_push_diagnosis_sha256": sha256_file(PUSH_DIAGNOSIS),
        "prior_v7b_root_push_admission_sha256": sha256_file(PRIOR_ROOT_ADMISSION),
        "v7b_push_failure_receipt_sha256": sha256_file(PUSH_FAILURE_RECEIPT),
        "basis_files_sha256": basis_hashes,
        "v7b_external_review_sha256": sha256_file(PARENT_PACKET / "external_review.json"),
        "external_review_sha256": sha256_file(PACKET / "external_review.json"),
        "external_review_exactly_preserved": (
            (PACKET / "external_review.json").read_bytes()
            == (PARENT_PACKET / "external_review.json").read_bytes()
        ),
        "packet_meta_exactly_preserved": (
            (PACKET / "packet_meta.json").read_bytes()
            == (PARENT_PACKET / "packet_meta.json").read_bytes()
        ),
        "embedded_wrapper_exactly_preserved": (
            (PACKET / "h232_profile_kernel_bundle.py").read_bytes()
            == (PARENT_PACKET / "h232_profile_kernel_bundle.py").read_bytes()
        ),
        "packet_files_14": successor_hashes,
        "source_tree_members_sha256": source_tree_hashes,
        "unchanged_parent_packet_files": sorted(set(parent_hashes) - {"kernel-metadata.json"}),
        "changed_parent_packet_files": ["kernel-metadata.json"],
        "source_archive_sha256": successor_hashes["source_archive.tar"],
        "source_archive_manifest_sha256": successor_hashes["source_archive_manifest.json"],
        "external_review_fit_authorization": external_review.get("conditional_fit_authorized"),
        "external_review_provider_push_authorization": external_review.get("provider_push_authorized"),
        "packet_provider_push_authorization": packet_meta.get("provider_push_authorized"),
        "root_push_authorization_required": True,
        "push_failure_diagnosis": "51-character V7B slug is a hypothesis only; original HTTP 400 body was unavailable",
        "build_report_sha256": sha256_file(report_path),
        "fit_or_score_claim": None,
    }
    if not all(
        receipt[key]
        for key in (
            "external_review_exactly_preserved",
            "packet_meta_exactly_preserved",
            "embedded_wrapper_exactly_preserved",
        )
    ):
        raise ValueError("V7B approved review/package payload was not preserved exactly")
    write_json(ROOT / "V7B_SHORTSLUG_BUILD_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
