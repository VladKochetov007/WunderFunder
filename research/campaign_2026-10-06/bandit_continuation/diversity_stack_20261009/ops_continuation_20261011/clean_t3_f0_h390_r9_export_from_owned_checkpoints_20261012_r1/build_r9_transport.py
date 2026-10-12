from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any  # Frozen JSON bindings have schema-dependent fields.

R9_ID = "vladkochetov07/wnn-clean-t3-f0-h384-exp-20261012-r9"
R9_TITLE = "wnn-clean-t3-f0-h384-exp-20261012-r9"
R9_DATASET_ID = "vladkochetov07/wnn-clean-t3-f0-h384-cp2-20261012-r9"
R9_DATASET_SLUG = R9_DATASET_ID.split("/", maxsplit=1)[1]
R9_MANIFEST_NAME = "R9_CHECKPOINT_INPUT_MANIFEST_R1.json"
R7_PACKET = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    "/ops_continuation_20261011/clean_t3_f0_h390_r7_export_recovery_20261012_r1/kernel"
)
R8_BUILDER_SHA256 = "96d98bbc8fa21aa5211b10f1264634b51c8acb90fde87bbcacb3ce3789ae8bc5"
R7_BODY_SHA256 = "f38ac7bd72c762b265e5951096e57b799cf632f43f7f96819a45d38772277cb6"
R7_BINDINGS_SHA256 = "39127244077e0937934fce1fedc131f7fa440ff18f7bc7fc3508f6f69e318d0f"
R7_PENDING_RELEASE_SHA256 = "7b76769bf34bf1f43b3ff23c9f5c59616cd80188c232b46df667083793b70940"
OMITTED_LOCAL_EXPORT = "export_clean_t3_f0_muonall_h384_e3_s3_r5_fit_train"

WORKER_MANIFEST_FUNCTION = '''

def validate_checkpoint_input_dataset(source_root: Path, bindings: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact source-dataset closure and every mounted checkpoint byte."""
    manifest_path = source_root / CHECKPOINT_MANIFEST_FILENAME
    expected_manifest_sha = bindings.get("checkpoint_source_manifest_sha256")
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or sha256_file(manifest_path) != expected_manifest_sha
    ):
        raise ValueError("R9 checkpoint dataset manifest is absent or changed")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != "wnn.kaggle.r9_checkpoint_input_manifest.r1"
        or manifest.get("dataset_id") != bindings.get("checkpoint_source_dataset_id")
        or manifest.get("dataset_slug") != R6_KERNEL_SLUG
    ):
        raise ValueError("R9 checkpoint dataset identity differs from its binding")
    expected_files: dict[str, tuple[int, str]] = {}
    for expected in bindings["checkpoints"].values():
        expected_files[(Path("experiment") / expected["model_ep2_path"]).as_posix()] = (
            expected["model_ep2_bytes"], expected["model_ep2_sha256"]
        )
        expected_files[(Path("experiment") / expected["provenance_path"]).as_posix()] = (
            expected["provenance_bytes"], expected["provenance_sha256"]
        )
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("R9 checkpoint dataset file inventory is malformed")
    declared = {row.get("path"): (row.get("bytes"), row.get("sha256")) for row in rows}
    if len(declared) != len(rows) or declared != expected_files:
        raise ValueError("R9 checkpoint dataset manifest does not bind the exact six files")
    observed: set[str] = set()
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("R9 checkpoint dataset must not contain symlinks")
        if path.is_file():
            observed.add(path.relative_to(source_root).as_posix())
    if observed != set(expected_files) | {CHECKPOINT_MANIFEST_FILENAME}:
        raise ValueError("R9 checkpoint dataset has missing or undeclared files")
    for relative, (expected_bytes, expected_sha) in expected_files.items():
        path = source_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha
        ):
            raise ValueError(f"R9 source checkpoint/provenance changed: {relative}")
    return manifest
'''

R9_RELEASE_PENDING = {
    "authorized": False,
    "binding": {},
    "logical_recipe_target_kernel_id": "vladkochetov07/wnn-clean-t3-f0-h384-exp-20261012-r7",
    "note": "Root admission is pending; this template authorizes no workers.",
    "schema": "wnn.root.clean_t3_f0_export_only_release.r7.r1",
    "status": "pending",
    "transport_target_kernel_id": R9_ID,
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def b64_lines(payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return "\n".join(f'    "{encoded[i : i + 76]}"' for i in range(0, len(encoded), 76))


def load_r8_builder(packet_root: Path):
    path = packet_root / "r8_wrapper_template.py"
    if sha256_file(path) != R8_BUILDER_SHA256:
        raise PermissionError("copied reviewed R8 wrapper template changed")
    spec = importlib.util.spec_from_file_location("r8_transport_template", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the reviewed R8 single-file wrapper template")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_checkpoint_stage(
    main_repo: Path, staging_parent: Path, bindings: dict[str, Any]
) -> tuple[Path, dict[str, Any], str]:
    stage = staging_parent / R9_DATASET_SLUG
    if staging_parent.exists() or stage.exists():
        raise FileExistsError(f"R9 dataset stage is create-only: {staging_parent}")
    stage.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for run_name, expected in bindings["checkpoints"].items():
        for field, sha_field, size_field in (
            ("model_ep2_path", "model_ep2_sha256", "model_ep2_bytes"),
            ("provenance_path", "provenance_sha256", "provenance_bytes"),
        ):
            relative = Path("experiment") / expected[field]
            source = main_repo / "artifacts" / "runs" / run_name / Path(expected[field]).name
            if (
                not source.is_file()
                or source.is_symlink()
                or source.stat().st_size != expected[size_field]
                or sha256_file(source) != expected[sha_field]
            ):
                raise ValueError(f"owned checkpoint input differs from R7 binding: {source}")
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as input_file, destination.open("xb") as output_file:
                shutil.copyfileobj(input_file, output_file, length=1 << 20)
            if (
                destination.stat().st_size != expected[size_field]
                or sha256_file(destination) != expected[sha_field]
            ):
                raise OSError(f"staged checkpoint copy failed verification: {relative}")
            rows.append(
                {
                    "bytes": expected[size_field],
                    "path": relative.as_posix(),
                    "run_name": run_name,
                    "sha256": expected[sha_field],
                    "type": "checkpoint" if field == "model_ep2_path" else "provenance",
                }
            )
    manifest = {
        "dataset_id": R9_DATASET_ID,
        "dataset_slug": R9_DATASET_SLUG,
        "files": sorted(rows, key=lambda row: row["path"]),
        "origin": {
            "r6_collector_receipt_sha256": bindings["root_release_binding"][
                "r6_complete_collector_receipt_sha256"
            ],
            "r6_kernel_id": bindings["r6_kernel_id"],
            "r6_kernel_version": bindings["r6_kernel_version"],
            "r6_operator_receipt_sha256": bindings["root_release_binding"][
                "r6_operator_receipt_sha256"
            ],
            "r6_root_stage_receipt_sha256": bindings["root_release_binding"][
                "r6_root_stage_receipt_sha256"
            ],
        },
        "schema": "wnn.kaggle.r9_checkpoint_input_manifest.r1",
    }
    manifest_bytes = canonical_json(manifest)
    manifest_path = stage / R9_MANIFEST_NAME
    with manifest_path.open("xb") as output_file:
        output_file.write(manifest_bytes)
    return stage, manifest, sha256(manifest_bytes)


def derive_bindings(
    original: dict[str, Any], manifest_sha256: str
) -> tuple[dict[str, Any], list[str]]:
    bindings = json.loads(json.dumps(original))
    names = [name for name in bindings["export_names"] if name != OMITTED_LOCAL_EXPORT]
    if len(names) != 5 or OMITTED_LOCAL_EXPORT in names:
        raise ValueError("R9 must select exactly the five remaining remote exports")
    if names != [
        "export_clean_t3_f0_muonall_h384_e3_s3_r5_fit_valid",
        "export_clean_t3_f0_muonall_h384_e3_s4_r5_fit_train",
        "export_clean_t3_f0_muonall_h384_e3_s4_r5_fit_valid",
        "export_clean_t3_f0_muonall_h384_e3_s5_r5_fit_train",
        "export_clean_t3_f0_muonall_h384_e3_s5_r5_fit_valid",
    ]:
        raise ValueError("R9 export selection differs from the root-authorized five-spec plan")
    specs = {name: bindings["export_specs"][name] for name in names}
    expected_bytes = sum(
        row["expected_row_groups_count"]
        * bindings["array_schema"]["sequence_length"]
        * bindings["array_schema"]["columns"]
        * 4
        for row in specs.values()
    )
    if expected_bytes != 4_045_600_000:
        raise ValueError("R9 five-export NPY byte total differs from the exact producer shapes")
    bindings["export_names"] = names
    bindings["export_specs"] = specs
    bindings["expected_total_npy_bytes"] = expected_bytes
    bindings["disk_and_collection"]["expected_prediction_npy_bytes"] = expected_bytes
    bindings["checkpoint_source_dataset_id"] = R9_DATASET_ID
    bindings["checkpoint_source_dataset_slug"] = R9_DATASET_SLUG
    bindings["checkpoint_source_manifest_filename"] = R9_MANIFEST_NAME
    bindings["checkpoint_source_manifest_sha256"] = manifest_sha256
    root_binding = bindings["root_release_binding"]
    root_binding["export_count"] = len(names)
    root_binding["export_specs_sha256"] = {name: specs[name]["spec_sha256"] for name in names}
    root_binding["disk_and_collection"] = dict(bindings["disk_and_collection"])
    root_binding["selected_export_names"] = names
    root_binding["checkpoint_source_dataset_id"] = R9_DATASET_ID
    root_binding["checkpoint_source_manifest_sha256"] = manifest_sha256
    root_binding["transport_target_kernel_id"] = R9_ID
    bindings["root_release_binding"] = root_binding
    return bindings, names


def patch_worker_body(body: bytes, bindings_sha256: str) -> bytes:
    old_slug = b'R6_KERNEL_SLUG = "wnn-clean-t3-f0-teachers-20261011-r5-runner"'
    new_slug = f'R6_KERNEL_SLUG = "{R9_DATASET_SLUG}"'.encode()
    old_binding = b'EXPECTED_BINDINGS_SHA256 = "39127244077e0937934fce1fedc131f7fa440ff18f7bc7fc3508f6f69e318d0f"'
    new_binding = f'EXPECTED_BINDINGS_SHA256 = "{bindings_sha256}"'.encode()
    if body.count(old_slug) != 1 or body.count(old_binding) != 1:
        raise ValueError("R7 worker constants differ from the reviewed input")
    patched = body.replace(old_slug, new_slug).replace(old_binding, new_binding)
    anchor = b"\ndef import_source_module(name: str, path: Path):"
    if patched.count(anchor) != 1:
        raise ValueError("R7 worker manifest-validation insertion point changed")
    patched = patched.replace(anchor, WORKER_MANIFEST_FUNCTION.encode() + anchor)
    main_call = b"source_root = find_source_mount(input_root)\n        mounts.extend("
    replacement = (
        b"source_root = find_source_mount(input_root)\n"
        b"        validate_checkpoint_input_dataset(source_root, bindings)\n"
        b"        mounts.extend("
    )
    if patched.count(main_call) != 1:
        raise ValueError("R7 checkpoint-mount callsite changed")
    patched = patched.replace(main_call, replacement)
    if b'CHECKPOINT_MANIFEST_FILENAME = "R9_CHECKPOINT_INPUT_MANIFEST_R1.json"' not in patched:
        patched = patched.replace(
            new_slug + b"\n",
            new_slug + b'\nCHECKPOINT_MANIFEST_FILENAME = "R9_CHECKPOINT_INPUT_MANIFEST_R1.json"\n',
            1,
        )
    patched = patched.replace(b"all six exports", b"all five selected exports")
    return patched


def make_wrapper(template: str, body: bytes, binding: bytes, release: bytes) -> bytes:
    source = template.replace("R8_KERNEL_ID", "R9_KERNEL_ID").replace("R8_TITLE", "R9_TITLE")
    source = source.replace("wnn-clean-t3-f0-h384-exp-20261012-r8", R9_TITLE)
    source = source.replace(".wnn-r8-export-control", ".wnn-r9-export-control")
    source = source.replace("this R8 transport target", "this R9 transport target")
    source = source.replace("R8 refuses", "R9 refuses").replace("R8 control", "R9 control")
    if f'R9_KERNEL_ID = "{R9_ID}"' not in source:
        raise ValueError("R8 wrapper template did not bind the R9 kernel id")
    substitutions = {
        "@BODY_B64@": b64_lines(body),
        "@BINDINGS_B64@": b64_lines(binding),
        "@RELEASE_B64@": b64_lines(release),
        "@BODY_SHA@": sha256(body),
        "@BINDINGS_SHA@": sha256(binding),
        "@RELEASE_SHA@": sha256(release),
    }
    for marker, value in substitutions.items():
        source = source.replace(marker, value)
    if "@" in source:
        raise ValueError("R9 wrapper contains an unexpanded template marker")
    return source.encode()


def build(
    main_repo: Path,
    output: Path,
    staging_parent: Path,
    receipt_path: Path,
    root_release_path: Path | None = None,
) -> dict[str, Any]:
    builder_root = Path(__file__).resolve().parent
    packet_root = output.parent.resolve()
    manifest_output_path = packet_root / R9_MANIFEST_NAME
    if (
        output.exists()
        or output.is_symlink()
        or receipt_path.exists()
        or manifest_output_path.exists()
    ):
        raise FileExistsError("R9 output and receipt are create-only")
    r7_kernel = main_repo / R7_PACKET
    original_body = (r7_kernel / "kernel.py").read_bytes()
    original_binding_bytes = (r7_kernel / "R7_EXPORT_BINDINGS_R1.json").read_bytes()
    original_release_bytes = (r7_kernel / "ROOT_EXPORT_RELEASE_R7.json").read_bytes()
    r8_builder_hash = sha256_file(builder_root / "r8_wrapper_template.py")
    r8_builder = load_r8_builder(builder_root)
    if (
        sha256(original_body) != R7_BODY_SHA256
        or sha256(original_binding_bytes) != R7_BINDINGS_SHA256
        or sha256(original_release_bytes) != R7_PENDING_RELEASE_SHA256
    ):
        raise ValueError("R7 input bytes differ from the preserved R7 packet")
    original = json.loads(original_binding_bytes)
    stage, source_manifest, source_manifest_sha = create_checkpoint_stage(
        main_repo, staging_parent, original
    )
    bindings, names = derive_bindings(original, source_manifest_sha)
    bindings_bytes = canonical_json(bindings)
    body = patch_worker_body(original_body, sha256(bindings_bytes))
    pending_release = dict(R9_RELEASE_PENDING)
    pending_release["binding"] = bindings["root_release_binding"]
    pending_release["logical_recipe_target_kernel_id"] = bindings["root_release_binding"][
        "target_kernel_id"
    ]
    release_bytes = canonical_json(pending_release)
    if root_release_path is not None:
        release_bytes = root_release_path.read_bytes()
        r8_builder._validate_release_bytes(bindings_bytes, release_bytes)
        admitted = json.loads(release_bytes)
        if admitted.get("transport_target_kernel_id") != R9_ID:
            raise PermissionError("ROOT release does not authorize the R9 transport")
    wrapper_template = r8_builder.WRAPPER_TEMPLATE
    wrapper = make_wrapper(wrapper_template, body, bindings_bytes, release_bytes)
    if len(wrapper) >= 1_000_000:
        raise ValueError("R9 single-file kernel exceeds the 1 MB code envelope")
    output.mkdir(parents=True, exist_ok=False)
    kernel_path = output / "kernel.py"
    with kernel_path.open("xb") as target:
        target.write(wrapper)
    r7_metadata = json.loads((r7_kernel / "kernel-metadata.json").read_text())
    if r7_metadata.get("kernel_sources") != [
        "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5-runner"
    ]:
        raise ValueError("R7 historical metadata source inventory differs")
    metadata = dict(r7_metadata)
    metadata["id"] = R9_ID
    metadata["title"] = R9_TITLE
    metadata["kernel_sources"] = []
    metadata["dataset_sources"] = list(r7_metadata["dataset_sources"]) + [R9_DATASET_ID]
    metadata_path = output / "kernel-metadata.json"
    with metadata_path.open("x", encoding="utf-8") as target:
        json.dump(metadata, target, indent=2, sort_keys=True)
        target.write("\n")
    stage_files = [row["path"] for row in source_manifest["files"]] + [R9_MANIFEST_NAME]
    packet_manifest_path = manifest_output_path
    packet_manifest_path.write_bytes(canonical_json(source_manifest))
    if sha256_file(packet_manifest_path) != source_manifest_sha:
        raise OSError("packet and physical R9 source manifests differ")
    file_inventory = []
    for relative in sorted(stage_files):
        path = stage / relative
        file_inventory.append(
            {"bytes": path.stat().st_size, "path": relative, "sha256": sha256_file(path)}
        )
    result = {
        "schema": "wnn.kaggle.r9_owned_checkpoint_export_transport_build.r1",
        "status": "pending_root_release" if root_release_path is None else "root_release_embedded",
        "transport_target_kernel_id": R9_ID,
        "transport_title": R9_TITLE,
        "logical_recipe_target_kernel_id": bindings["root_release_binding"]["target_kernel_id"],
        "checkpoint_dataset_id": R9_DATASET_ID,
        "checkpoint_dataset_slug": R9_DATASET_SLUG,
        "checkpoint_stage_directory": str(stage),
        "checkpoint_manifest_filename": R9_MANIFEST_NAME,
        "checkpoint_manifest_sha256": source_manifest_sha,
        "checkpoint_files": file_inventory,
        "checkpoint_total_bytes_excluding_manifest": sum(
            row["bytes"] for row in file_inventory if row["path"] != R9_MANIFEST_NAME
        ),
        "r7_source_kernel_sha256": sha256(original_body),
        "r7_source_bindings_sha256": sha256(original_binding_bytes),
        "r7_pending_release_sha256": sha256(original_release_bytes),
        "r9_embedded_worker_sha256": sha256(body),
        "r9_embedded_bindings_sha256": sha256(bindings_bytes),
        "r9_pending_or_admitted_release_sha256": sha256(release_bytes),
        "kernel_py_sha256": sha256(wrapper),
        "kernel_py_bytes": len(wrapper),
        "metadata_sha256": sha256(metadata_path.read_bytes()),
        "upload_files": ["kernel.py", "kernel-metadata.json"],
        "selected_exports": names,
        "omitted_export": {
            "name": OMITTED_LOCAL_EXPORT,
            "reason": "root admitted the same s3 fit-train checkpoint for a local export; no duplicate R9 worker",
        },
        "expected_prediction_npy_bytes": bindings["expected_total_npy_bytes"],
        "collector_cap_bytes": bindings["disk_and_collection"]["collector_cap_bytes"],
        "minimum_free_disk_bytes": bindings["disk_and_collection"]["minimum_free_disk_bytes"],
        "worker_limits": bindings["root_release_binding"]["worker_limits"],
        "kernel_wall_ceiling_seconds": bindings["root_release_binding"]["runtime_limits"][
            "kernel_wall_ceiling_seconds"
        ],
        "fit_workers": 0,
        "provider_actions": 0,
        "root_release_required": True,
        "r8_wrapper_template_sha256": r8_builder_hash,
        "receipt_path": str(receipt_path),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging-parent", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root-release", type=Path)
    args = parser.parse_args()
    result = build(
        args.main_repo.resolve(),
        args.output.resolve(),
        args.staging_parent.resolve(),
        args.receipt.resolve(),
        args.root_release,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
