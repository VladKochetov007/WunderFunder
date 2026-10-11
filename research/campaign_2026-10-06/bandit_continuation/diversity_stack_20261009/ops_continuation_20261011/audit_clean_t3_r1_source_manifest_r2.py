from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

OPS = Path(__file__).resolve().parent
ROOT = OPS.parents[4]
PACKET = OPS / "clean_t3_fold0_h390_r1"
SNAPSHOT = PACKET / "source_snapshot"
BUNDLE = PACKET / "bundle_teacher_profile_r1"
RECEIPT = OPS / "CLEAN_T3_F0_SOURCE_MANIFEST_RECONCILIATION_R2.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def main() -> None:
    baseline_path = SNAPSHOT / "SOURCE_MANIFEST_H390.json"
    bindings_path = PACKET / "SOURCE_BINDINGS_CLEAN_T3_F0_R1.json"
    bundle_manifest_path = BUNDLE / "bundle-manifest.json"
    archive_path = BUNDLE / "bundle.wnn"
    static_receipt_path = PACKET / "PROFILE_BUNDLE_STATIC_VALIDATION_R1.json"

    baseline_manifest = read_json(baseline_path)
    bindings = read_json(bindings_path)
    bundle_manifest = read_json(bundle_manifest_path)
    static_receipt = read_json(static_receipt_path)
    baseline = baseline_manifest.get("sources")
    current_bound = bindings.get("source_snapshot_sha256_by_path")
    deltas = bindings.get("allowed_deltas_from_h390")
    if (
        not isinstance(baseline, dict)
        or not isinstance(current_bound, dict)
        or not isinstance(deltas, list)
    ):
        raise ValueError("source lineage manifests lack the required typed fields")

    actual: dict[str, str] = {}
    for source in sorted(SNAPSHOT.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        actual[source.relative_to(SNAPSHOT).as_posix()] = sha256_file(source)
    if actual != current_bound:
        missing = sorted(set(actual) - set(current_bound))
        stale = sorted(
            path for path in set(actual) & set(current_bound) if actual[path] != current_bound[path]
        )
        extra = sorted(set(current_bound) - set(actual))
        raise ValueError(
            f"current source snapshot does not match outer bindings: missing={missing}; stale={stale}; extra={extra}"
        )

    changed = {
        path: {"parent_h390_sha256": baseline.get(path), "current_sha256": actual.get(path)}
        for path in sorted(set(baseline) | set(actual))
        if baseline.get(path) != actual.get(path)
    }
    declared = {item.get("path"): item for item in deltas if isinstance(item, dict)}
    if len(declared) != len(deltas) or set(declared) != set(changed):
        raise ValueError(
            f"allowed-delta list must equal exactly the baseline mismatches/new sources; "
            f"missing={sorted(set(changed) - set(declared))}; spurious={sorted(set(declared) - set(changed))}"
        )
    for path, item in declared.items():
        if item.get("prior_h390_sha256") != baseline.get(path):
            raise ValueError(f"allowed-delta baseline pin differs for {path}")
        if item.get("current_sha256") != actual.get(path):
            raise ValueError(f"allowed-delta current pin differs for {path}")
        if not item.get("reason"):
            raise ValueError(f"allowed-delta reason missing for {path}")

    if baseline_manifest.get("schema") != "wnn.h390-source-snapshot.v3":
        raise ValueError("inner source manifest is not the frozen H390 parent baseline")
    for ancestry_key in (
        "parent_archive_sha256",
        "parent_manifest_sha256",
        "cache_reuse_lineage_sha256",
    ):
        if (
            not isinstance(baseline_manifest.get(ancestry_key), str)
            or len(baseline_manifest[ancestry_key]) != 64
        ):
            raise ValueError(f"H390 parent baseline lacks pinned ancestry field: {ancestry_key}")

    manifest_digest = sha256_file(baseline_path)
    if bindings.get("source_manifest_sha256") != manifest_digest:
        raise ValueError("outer source bindings do not pin the exact H390 parent-baseline file")
    if bindings.get("source_manifest_path") != "source_snapshot/SOURCE_MANIFEST_H390.json":
        raise ValueError("outer source manifest path differs from the preserved baseline")
    code_paths = set(bindings.get("code_package", {}).get("paths", []))
    packet_rel = PACKET.relative_to(ROOT).as_posix()
    packaged_sources = {f"{packet_rel}/source_snapshot/{path}" for path in actual}
    manifest_rel = f"{packet_rel}/source_snapshot/SOURCE_MANIFEST_H390.json"
    if not packaged_sources.issubset(code_paths) or manifest_rel not in code_paths:
        raise ValueError(
            "outer frozen code package omits source snapshot files or its parent baseline"
        )
    if (
        bindings.get("code_package", {}).get("worker_sources") != len(code_paths)
        or len(code_paths) != len(actual) + 4
    ):
        raise ValueError(
            "outer worker-source count differs from the Python snapshot, parent manifest and three runtime tools"
        )

    archive_digest = sha256_file(archive_path)
    if archive_digest != bundle_manifest.get("archive_sha256") or archive_digest != bindings.get(
        "teacher_profile_bundle_archive_sha256"
    ):
        raise ValueError(
            "profile archive digest differs across the archive, bundle manifest and source binding"
        )
    if archive_digest != static_receipt.get("archive_sha256"):
        raise ValueError("static archive-consumer receipt does not name the same profile archive")
    if bundle_manifest.get("spec_paths") != static_receipt.get("spec_paths"):
        raise ValueError("static archive receipt and bundle disagree on frozen profile specs")

    archive_file_pins = {item["path"]: item for item in bundle_manifest.get("files", [])}
    archive_snapshot_checks: dict[str, str] = {}
    with tarfile.open(archive_path, "r:*") as archive:
        for relative, expected_digest in actual.items():
            member_path = f"{packet_rel}/source_snapshot/{relative}"
            descriptor = archive_file_pins.get(member_path)
            member = archive.extractfile(member_path)
            if descriptor is None or member is None:
                raise ValueError(f"archive lacks a manifest-bound current source file: {relative}")
            payload = member.read()
            observed_digest = sha256_bytes(payload)
            if observed_digest != expected_digest or observed_digest != descriptor.get("sha256"):
                raise ValueError(
                    f"archived source bytes differ from current/outer bindings: {relative}"
                )
            archive_snapshot_checks[relative] = observed_digest
        member_path = f"{packet_rel}/source_snapshot/SOURCE_MANIFEST_H390.json"
        descriptor = archive_file_pins.get(member_path)
        member = archive.extractfile(member_path)
        if descriptor is None or member is None or sha256_bytes(member.read()) != manifest_digest:
            raise ValueError(
                "archive's embedded parent-baseline manifest differs from the reviewed file"
            )

    receipt = {
        "schema": "wnn.clean_t3_f0_source_manifest_reconciliation.r2",
        "status": "pass_parent_baseline_and_current_archive_bindings",
        "packet": packet_rel,
        "interpretation": (
            "SOURCE_MANIFEST_H390.json is the immutable parent H390 baseline. Its `sources` values are the parent hashes; "
            "the outer SOURCE_BINDINGS file records current child hashes and the exact allowed deltas."
        ),
        "parent_baseline": {
            "path": "source_snapshot/SOURCE_MANIFEST_H390.json",
            "sha256": manifest_digest,
            "schema": baseline_manifest["schema"],
            "source_count": len(baseline),
            "parent_archive_sha256": baseline_manifest["parent_archive_sha256"],
            "parent_manifest_sha256": baseline_manifest["parent_manifest_sha256"],
            "cache_reuse_lineage_sha256": baseline_manifest["cache_reuse_lineage_sha256"],
        },
        "current_snapshot": {
            "source_count": len(actual),
            "outer_binding_sha256": sha256_file(bindings_path),
            "outer_binding_matches_every_current_source": True,
            "archive_contains_matching_bytes_for_every_current_source": True,
        },
        "baseline_comparison": {
            "unchanged_source_count": len(actual) - len(changed),
            "declared_allowed_delta_count": len(declared),
            "allowed_delta_set_equals_all_and_only_missing_or_changed_sources": True,
            "allowed_deltas": [
                {
                    "path": path,
                    "parent_h390_sha256": changed[path]["parent_h390_sha256"],
                    "current_sha256": changed[path]["current_sha256"],
                    "reason": declared[path]["reason"],
                }
                for path in sorted(changed)
            ],
        },
        "profile_archive": {
            "path": "bundle_teacher_profile_r1/bundle.wnn",
            "sha256": archive_digest,
            "static_consumer_receipt_sha256": sha256_file(static_receipt_path),
            "spec_paths": bundle_manifest["spec_paths"],
            "full_teacher_fits_present": False,
        },
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "scope": "Read-only lineage reconciliation. No source, archive, profile, fit, provider, upload, queue or quota mutation.",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
