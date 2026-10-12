from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any  # Frozen manifests have schema-defined heterogeneous values.

R10_ID = "vladkochetov07/wnn-clean-t3-f0-h384-exp-20261012-r10"
R10_TITLE = "wnn-clean-t3-f0-h384-exp-20261012-r10"
R9_PACKET_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    "/ops_continuation_20261011/clean_t3_f0_h390_r9_export_from_owned_checkpoints_20261012_r2"
)
R7_PACKET_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    "/ops_continuation_20261011/clean_t3_f0_h390_r7_export_recovery_20261012_r1/kernel"
)
R5_PRODUCER_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    "/ops_continuation_20261011/clean_t3_fold0_h390_r5_teacher_cohort/package_r9"
    "/bundle_teacher_cohort_r5"
)
R9_KERNEL_SHA256 = "8b0e7ee4fd02615bd6503fb534fd977d77828bbdf5afa97336d2b0656a988f95"
R9_METADATA_SHA256 = "d048e4ed4cfa8a1afddaceaf7bd38bd369fb0090d3f324e519bf27249d409dbd"
R9_TEMPLATE_SHA256 = "96d98bbc8fa21aa5211b10f1264634b51c8acb90fde87bbcacb3ce3789ae8bc5"
R9_BUILD_RECEIPT_SHA256 = "19a691922305358596d6bf13d8dabc535a297fd02c1d70ee182bbcc03e32d88c"
R9_CHECKPOINT_MANIFEST_SHA256 = "30c4b2edfe6d39f13713a6dd82d1355e1975143a64ec8b8f422d8c316566053d"
R5_ARCHIVE_SHA256 = "88641b3c9fc629a7a2ce8b11cfd98ecf8b53fe943ffb2d7f632e1dfb997fa549"
R5_MANIFEST_SHA256 = "5fc7423ebb8f8ef3c5c5bb0ec8be35352f3470dd75d99d47cad4180ac663c689"
R9_BODY_SHA256 = "4b1c96cd30d4346a034a1162b2d661d3d85b992e195a98040a6353fb9c4edb36"
R9_BINDINGS_SHA256 = "5875f0eba62ec40c837c00a1b469a8dcb683b56c1f8b389ff6ae5939d68a5f31"
R9_RELEASE_SHA256 = "f8a6830af4ced9f330c6b052b8f3bd2ad7df7a940ff8ee4318dc5a6a0579cc91"
R10_RECEIPT_NAME = "R10_BUILD_RECEIPT_R1.json"

EXTERNAL_PATH_HELPER = b'''
def authorized_external_declared_paths(
    manifest: dict[str, Any], bindings: dict[str, Any]
) -> set[str]:
    """Authorize only the exact externally mounted paths sealed in R7 bindings."""
    files = manifest.get("files")
    spec_paths = manifest.get("spec_paths")
    external = manifest.get("external_mounts")
    expected_external = bindings.get("r5_external_mounts")
    if (
        not isinstance(files, list)
        or not isinstance(spec_paths, list)
        or not isinstance(external, dict)
        or not isinstance(expected_external, dict)
        or external != expected_external
    ):
        raise ValueError("R10 R5 external mount map differs from its R7 binding")
    file_paths = [row.get("path") if isinstance(row, dict) else None for row in files]
    if any(not isinstance(path, str) or not path for path in file_paths):
        raise ValueError("R10 R5 archive file inventory is malformed")
    if len(file_paths) != len(set(file_paths)):
        raise ValueError("R10 R5 archive file inventory repeats a path")
    if any(not isinstance(path, str) or not path for path in spec_paths):
        raise ValueError("R10 R5 spec path inventory is malformed")
    if len(spec_paths) != len(set(spec_paths)):
        raise ValueError("R10 R5 spec path inventory repeats a path")
    if any(not isinstance(path, str) or not path for path in external):
        raise ValueError("R10 R5 external mount key is malformed")
    declared_paths = set(file_paths) | set(spec_paths)
    external_paths = set(external)
    overlap = declared_paths & external_paths
    if overlap:
        raise ValueError(
            f"R10 external mount keys overlap archive/spec members: {sorted(overlap)[:3]}"
        )
    return declared_paths | external_paths
'''


def sha256(payload: bytes) -> str:
    """Return the SHA-256 digest for a byte string."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading its contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    """Serialize deterministic JSON for frozen local receipts."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_module(name: str, path: Path):
    """Load a reviewed local wrapper module without adding dependencies."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import reviewed R9 wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(payload: bytes, old: bytes, new: bytes, label: str) -> bytes:
    """Apply one source-pinned replacement, failing on drift or ambiguity."""
    if payload.count(old) != 1:
        raise ValueError(f"expected one R9 {label} insertion point")
    return payload.replace(old, new)


def patch_worker(body: bytes) -> bytes:
    """Fix only the R5 external-key declaration at the R9 worker callsite."""
    old_call = (
        b'        declared = {row["path"] for row in manifest["files"]} '
        b'| set(manifest["spec_paths"])\n'
        b"        mounts.extend(\n"
        b'            mount_verified_inputs(input_root, root, manifest["external_mounts"], declared)\n'
        b"        )"
    )
    new_call = (
        b"        declared = authorized_external_declared_paths(manifest, bindings)\n"
        b"        mounts.extend(\n"
        b'            mount_verified_inputs(input_root, root, manifest["external_mounts"], declared)\n'
        b"        )"
    )
    patched = replace_once(body, old_call, new_call, "external mount callsite")
    return replace_once(
        patched,
        b"\ndef verify_export_specs(\n",
        EXTERNAL_PATH_HELPER + b"\n\ndef verify_export_specs(\n",
        "external mount validator",
    )


def make_wrapper(template: str, body: bytes, bindings: bytes, release: bytes) -> bytes:
    """Create the one-file R10 transport wrapper from the reviewed R8 template."""
    source = template.replace("R8_KERNEL_ID", "R10_KERNEL_ID").replace("R8_TITLE", "R10_TITLE")
    source = source.replace("wnn-clean-t3-f0-h384-exp-20261012-r8", R10_TITLE).replace(
        ".wnn-r8-export-control", ".wnn-r10-export-control"
    )
    source = (
        source.replace("R8 refuses", "R10 refuses")
        .replace("R8 control", "R10 control")
        .replace("this R8 transport target", "this R10 transport target")
    )
    substitutions = {
        "@BODY_B64@": base64_lines(body),
        "@BINDINGS_B64@": base64_lines(bindings),
        "@RELEASE_B64@": base64_lines(release),
        "@BODY_SHA@": sha256(body),
        "@BINDINGS_SHA@": sha256(bindings),
        "@RELEASE_SHA@": sha256(release),
    }
    for marker, value in substitutions.items():
        source = source.replace(marker, value)
    if any(marker in source for marker in substitutions):
        raise ValueError("R10 wrapper contains an unexpanded template marker")
    if f'R10_KERNEL_ID = "{R10_ID}"' not in source:
        raise ValueError("R10 wrapper target ID did not bind")
    return source.encode("utf-8")


def base64_lines(payload: bytes) -> str:
    """Format base64 as source-safe lines for the single-file kernel."""
    encoded = base64.b64encode(payload).decode("ascii")
    return "\n".join(f'    "{encoded[index : index + 76]}"' for index in range(0, len(encoded), 76))


def build(main_root: Path, packet_root: Path) -> dict[str, Any]:
    """Build an identified R10 packet from immutable, SHA-bound R9/R5 inputs."""
    r9_packet = main_root / R9_PACKET_RELATIVE
    r9_kernel = r9_packet / "kernel/kernel.py"
    r9_metadata = r9_packet / "kernel/kernel-metadata.json"
    r9_template = r9_packet / "r8_wrapper_template.py"
    r9_receipt_path = r9_packet / "R9_BUILD_RECEIPT_R2.json"
    r9_manifest_path = r9_packet / "R9_CHECKPOINT_INPUT_MANIFEST_R2.json"
    r5_source = main_root / R5_PRODUCER_RELATIVE
    fixture_dir = packet_root / "fixtures/r5_bundle_package_r9"
    fixture_manifest = fixture_dir / "bundle-manifest.json"
    fixture_archive = fixture_dir / "bundle.wnn"
    if sha256_file(r9_kernel) != R9_KERNEL_SHA256:
        raise ValueError("R9 kernel source bytes differ from the reviewed packet")
    if sha256_file(r9_metadata) != R9_METADATA_SHA256:
        raise ValueError("R9 metadata bytes differ from the reviewed packet")
    if sha256_file(r9_template) != R9_TEMPLATE_SHA256:
        raise ValueError("R8 wrapper template bytes differ from the reviewed packet")
    if sha256_file(r9_receipt_path) != R9_BUILD_RECEIPT_SHA256:
        raise ValueError("R9 build receipt bytes differ from the reviewed packet")
    if sha256_file(r9_manifest_path) != R9_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("R9 checkpoint input manifest bytes changed")
    for source, fixture in (
        (r5_source / "bundle-manifest.json", fixture_manifest),
        (r5_source / "bundle.wnn", fixture_archive),
    ):
        if not source.is_file() or not fixture.is_file():
            raise FileNotFoundError(f"R5 actual producer input is missing: {source}")
        if source.stat().st_size != fixture.stat().st_size:
            raise ValueError(f"R5 producer fixture size differs from package_r9: {source.name}")
        if sha256_file(source) != sha256_file(fixture):
            raise ValueError(f"R5 producer fixture differs from package_r9: {source.name}")
    if (
        sha256_file(fixture_manifest) != R5_MANIFEST_SHA256
        or sha256_file(fixture_archive) != R5_ARCHIVE_SHA256
    ):
        raise ValueError("R5 producer fixture differs from the R7 bound source")

    r9_wrapper = load_module("r9_wrapper_for_r10_build", r9_kernel)
    body, bindings_bytes, r9_release_bytes = r9_wrapper._decode_embedded()
    if (
        sha256(body) != R9_BODY_SHA256
        or sha256(bindings_bytes) != R9_BINDINGS_SHA256
        or sha256(r9_release_bytes) != R9_RELEASE_SHA256
    ):
        raise ValueError("R9 embedded worker, binding or release changed")
    bindings = json.loads(bindings_bytes)
    manifest = json.loads(fixture_manifest.read_bytes())
    if (
        manifest.get("external_mounts") != bindings.get("r5_external_mounts")
        or manifest.get("archive_sha256") != R5_ARCHIVE_SHA256
        or manifest.get("archive_bytes") != fixture_archive.stat().st_size
    ):
        raise ValueError("actual R5 manifest does not match the frozen R7 external/archive pins")

    patched_body = patch_worker(body)
    release = json.loads(r9_release_bytes)
    if (
        release.get("status") != "pending"
        or release.get("authorized") is not False
        or release.get("binding") != bindings.get("root_release_binding")
        or release.get("logical_recipe_target_kernel_id")
        != bindings["root_release_binding"].get("target_kernel_id")
        or release.get("transport_target_kernel_id") != r9_wrapper.R9_KERNEL_ID
    ):
        raise ValueError("R9 root release is not the expected pending transport authority")
    release["transport_target_kernel_id"] = R10_ID
    release["note"] = "Root R10 dispatch admission is pending; no export worker is authorized."
    release_bytes = canonical_json(release)
    template_module = load_module("r8_wrapper_template_for_r10", r9_template)
    template = template_module.WRAPPER_TEMPLATE
    wrapper_bytes = make_wrapper(template, patched_body, bindings_bytes, release_bytes)
    if len(wrapper_bytes) >= 1_000_000:
        raise ValueError("R10 single-file kernel exceeds the 1 MB source envelope")

    metadata = json.loads(r9_metadata.read_text(encoding="utf-8"))
    if metadata.get(
        "kernel_sources"
    ) != [] or "vladkochetov07/wnn-clean-t3-f0-h384-cp2-20261012-r9" not in metadata.get(
        "dataset_sources", []
    ):
        raise ValueError("R9 mount inventory differs from its reviewed source closure")
    metadata["id"] = R10_ID
    metadata["title"] = R10_TITLE
    metadata_bytes = canonical_json(metadata)

    kernel_dir = packet_root / "kernel"
    receipt_path = packet_root / R10_RECEIPT_NAME
    if kernel_dir.exists() or kernel_dir.is_symlink() or receipt_path.exists():
        raise FileExistsError("R10 output paths are create-only")
    kernel_dir.mkdir(parents=True, exist_ok=False)
    (kernel_dir / "kernel.py").write_bytes(wrapper_bytes)
    (kernel_dir / "kernel-metadata.json").write_bytes(metadata_bytes)
    receipt = {
        "schema": "wnn.kaggle.r10_external_mount_repair_build.r1",
        "status": "pending_root_release_and_independent_review",
        "transport_target_kernel_id": R10_ID,
        "transport_title": R10_TITLE,
        "logical_recipe_target_kernel_id": bindings["root_release_binding"]["target_kernel_id"],
        "r9_source": {
            "build_receipt_sha256": R9_BUILD_RECEIPT_SHA256,
            "checkpoint_manifest_sha256": R9_CHECKPOINT_MANIFEST_SHA256,
            "embedded_bindings_sha256": sha256(bindings_bytes),
            "embedded_release_sha256": sha256(r9_release_bytes),
            "embedded_worker_sha256": sha256(body),
            "kernel_sha256": R9_KERNEL_SHA256,
            "metadata_sha256": R9_METADATA_SHA256,
            "wrapper_template_sha256": R9_TEMPLATE_SHA256,
        },
        "r5_actual_producer": {
            "package_path": str(R5_PRODUCER_RELATIVE),
            "bundle_archive_bytes": fixture_archive.stat().st_size,
            "bundle_archive_sha256": sha256_file(fixture_archive),
            "bundle_manifest_bytes": fixture_manifest.stat().st_size,
            "bundle_manifest_sha256": sha256_file(fixture_manifest),
            "manifest_external_keys": len(manifest["external_mounts"]),
            "manifest_equals_r7_hash_bound_external_mounts": True,
            "producer_package_version": "package_r9",
        },
        "repair": {
            "changed_worker_logic": [
                "require the manifest external_mounts object to equal the R7 hash-bound map",
                "reject overlap between external mount keys and archive/spec paths",
                "add exactly the verified external keys to mount_verified_inputs declared paths",
            ],
            "science_recipe_changed": False,
            "checkpoint_selection_changed": False,
            "export_specs_changed": False,
            "worker_limits_changed": False,
        },
        "selected_exports": bindings["export_names"],
        "fit_workers": 0,
        "provider_actions": 0,
        "root_release_required": True,
        "root_release_embedded_status": release["status"],
        "upload_files": ["kernel.py", "kernel-metadata.json"],
        "files": {
            "kernel/kernel.py": {
                "bytes": len(wrapper_bytes),
                "sha256": sha256(wrapper_bytes),
            },
            "kernel/kernel-metadata.json": {
                "bytes": len(metadata_bytes),
                "sha256": sha256(metadata_bytes),
            },
        },
    }
    receipt_path.write_bytes(canonical_json(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--packet-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = build(args.main_root.resolve(), args.packet_root.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
