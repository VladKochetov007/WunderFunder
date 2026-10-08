from __future__ import annotations

import base64
import hashlib
import json
import sys
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"
PACKET = ROOT / "packet"
BASE_PROBE_SHA256 = "258590e4a1213a3c4a844a62b928efd82bbb32bef46109d01f67984867204d8f"
PARENT_PROVENANCE_SHA256 = "9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f"
SOURCE_ROOT = (
    Path(__file__).resolve().parents[4] / "campaign_2026-10-04" / "staging" / "jax_tpu_e465"
)
PACKET_ID = "vladkochetov07/e463-tpu-realdata-reload-v6-20261008"
PACKET_TITLE = "E463 TPU Real Data Reload V6 20261008"

SUPPORT_FILES = {
    "distributed_combo.py": SOURCE_ROOT / "src/wnn_jax/distributed_combo.py",
    "e465_host_contract.py": SOURCE_ROOT / "src/wnn_jax/e465_host_contract.py",
    "onecycle.py": SOURCE_ROOT / "src/wnn_jax/onecycle.py",
}
EXPECTED_SUPPORT_SHA256 = {
    "distributed_combo.py": "24e4ea77a3c2524d3c62b2beb5e83e48a9789c7dc5bad24faabd7fd52501cb68",
    "e465_host_contract.py": "a40807275271d5ccbb1a38867bf8c9328756e9e663b89c278bf6439126213370",
    "onecycle.py": "f857fe857a361249c2742e3486cd32b846e0f47413a5c4db2ae1fb6c65509ded",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _literal(value: str) -> str:
    chunks = textwrap.wrap(value, width=100)
    return "(\n" + "\n".join(f"    {chunk!r}" for chunk in chunks) + "\n)"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one {label} marker, found {count}")
    return text.replace(old, new, 1)


def _write_embedded_support() -> str:
    lines = [
        "def prepare_embedded_support() -> Path:",
        '    """Materialize hash-pinned helpers before the isolated TPU child imports them."""',
        '    modules = {"combo_loss_reference.py": (COMBO_LOSS_REFERENCE_B64, COMBO_LOSS_REFERENCE_SHA256)}',
        "    modules.update(",
        "        {",
        "            name: (E465_SUPPORT_B64[name], E465_SUPPORT_SHA256[name])",
        "            for name in E465_SUPPORT_B64",
        "        }",
        "    )",
        '    support_root = OUTPUT / "support"',
        "    support_root.mkdir(parents=True, exist_ok=True)",
        "    for name, (encoded, expected_hash) in modules.items():",
        "        module_bytes = base64.b64decode(encoded, validate=True)",
        "        if hashlib.sha256(module_bytes).hexdigest() != expected_hash:",
        '            raise RuntimeError(f"embedded helper digest mismatch: {name}")',
        "        module_path = support_root / name",
        "        if module_path.exists():",
        "            if module_path.is_symlink() or module_path.read_bytes() != module_bytes:",
        '                raise RuntimeError(f"materialized helper differs from pinned bytes: {name}")',
        "        else:",
        '            temporary = module_path.with_suffix(module_path.suffix + ".tmp")',
        "            temporary.write_bytes(module_bytes)",
        "            temporary.replace(module_path)",
        "    support_path = str(support_root)",
        "    if support_path not in sys.path:",
        "        sys.path.insert(0, support_path)",
        "    return support_root",
    ]
    return "\n".join(lines) + "\n\n"


def build_probe() -> tuple[bytes, dict[str, object]]:
    base_path = REFERENCE / "v5_probe.py"
    base_bytes = base_path.read_bytes()
    if sha256_bytes(base_bytes) != BASE_PROBE_SHA256:
        raise ValueError("frozen V5 probe bytes changed")

    provenance_bytes = (REFERENCE / "parent_provenance.json").read_bytes()
    provenance_sha = sha256_bytes(provenance_bytes)
    if provenance_sha != PARENT_PROVENANCE_SHA256:
        raise ValueError("own epoch-3 provenance bytes changed")
    support_bytes: dict[str, bytes] = {}
    for name, source_path in SUPPORT_FILES.items():
        copied = (REFERENCE / name).read_bytes()
        original = source_path.read_bytes()
        if copied != original:
            raise ValueError(f"reviewed E465 helper differs from its owned source: {name}")
        if sha256_bytes(copied) != EXPECTED_SUPPORT_SHA256[name]:
            raise ValueError(f"reviewed E465 helper digest changed: {name}")
        support_bytes[name] = copied

    text = base_bytes.decode("utf-8")
    text = replace_once(
        text,
        'PACKET_ID = "wnn-e463-combo-reduction-tpu-v5e8-20261007-v5"',
        f'PACKET_ID = "{PACKET_ID.rsplit("/", 1)[1]}"',
        "packet id",
    )
    text = replace_once(
        text,
        'OUTPUT = Path("/kaggle/working/cloud_preflight_tpu_v5")',
        'OUTPUT = Path("/kaggle/working/cloud_preflight_tpu_v6")',
        "output directory",
    )
    text = replace_once(
        text,
        "from typing import Any\n",
        "from collections.abc import Sequence\nfrom typing import Any\n",
        "typing import",
    )
    text = replace_once(
        text,
        '"""Run the TPU payload under an internal bound and always preserve terminal status."""\n    global CHILD, PHASE, TERMINAL',
        '"""Run the TPU payload under an internal bound and always preserve terminal status."""\n    global CHILD, PHASE, TERMINAL, SOURCE_MOUNT_ROOT',
        "parent globals",
    )

    support_writer = _write_embedded_support()
    prepare_start = text.index("def prepare_embedded_support()")
    prepare_end = text.index("def record_phase(", prepare_start)
    text = text[:prepare_start] + support_writer + text[prepare_end:]

    runtime = (REFERENCE / "v6_runtime_additions.py").read_text(encoding="utf-8")
    runtime = runtime[runtime.index("PACKET_ID = ") :]

    embedded_support = "E465_SUPPORT_B64: dict[str, str] = {}"
    support_literal = (
        "E465_SUPPORT_B64: dict[str, str] = {\n"
        + ",\n".join(
            f"    {name!r}: {_literal(base64.b64encode(content).decode('ascii'))}"
            for name, content in sorted(support_bytes.items())
        )
        + "\n}"
    )
    runtime = replace_once(runtime, embedded_support, support_literal, "E465 base64 placeholder")
    hashes_literal = "E465_SUPPORT_SHA256: dict[str, str] = " + repr(
        {name: sha256_bytes(content) for name, content in sorted(support_bytes.items())}
    )
    runtime = replace_once(
        runtime,
        "E465_SUPPORT_SHA256: dict[str, str] = {}",
        hashes_literal,
        "E465 hash placeholder",
    )
    runtime = replace_once(
        runtime,
        'PARENT_PROVENANCE_B64 = "__PARENT_PROVENANCE_B64__"',
        "PARENT_PROVENANCE_B64 = " + _literal(base64.b64encode(provenance_bytes).decode("ascii")),
        "parent provenance placeholder",
    )

    realdata_stage = """        realdata_input_identity = materialize_realdata_project(source_root, source_materialization)
        write_json(OUTPUT / "realdata_input_identity.json", realdata_input_identity)
"""
    text = replace_once(
        text,
        '        write_json(OUTPUT / "e463_checkpoint_identity.json", input_identity)\n',
        '        write_json(OUTPUT / "e463_checkpoint_identity.json", input_identity)\n'
        + realdata_stage,
        "real-data parent staging",
    )
    text = replace_once(
        text,
        '        write_json(OUTPUT / "input_mounts.json", input_mount_inventory())\n',
        "        SOURCE_MOUNT_ROOT = discover_source_mount()\n"
        '        write_json(OUTPUT / "input_mounts.json", input_mount_inventory())\n',
        "dynamic source mount call",
    )
    text = replace_once(
        text,
        '                "WNN_E463_SOURCE_ROOT": str(source_root),\n',
        '                "WNN_E463_SOURCE_ROOT": str(source_root),\n'
        '                "WNN_E463_DATA_ROOT": str(realdata_input_identity["project_root"]),\n',
        "child project root environment",
    )

    realdata_call = """    realdata_update = run_realdata_one_update(jax, jnp, np, params, local_devices, source_root)
    write_json(OUTPUT / "realdata_update_receipt.json", realdata_update)
"""
    text = replace_once(
        text,
        '    write_json(OUTPUT / "tpu_fullshape_profile.json", profile_records)\n',
        '    write_json(OUTPUT / "tpu_fullshape_profile.json", profile_records)\n' + realdata_call,
        "real-data update call",
    )
    text = replace_once(
        text,
        '        "classification": "actual TPUv5e eight-device unequal-mass combo-objective gradient diagnostic and full-shape synthetic profile",\n',
        '        "classification": "TPUv5e eight-device combo-objective diagnostic plus one real-data E463 update, checkpoint roundtrip, and causal inference check",\n',
        "result classification",
    )
    text = replace_once(
        text,
        '        "no_training_data_used": True,\n',
        '        "no_training_data_used": False,\n',
        "training-data result flag",
    )
    text = replace_once(
        text,
        '        "no_checkpoint_saved_or_exported": True,\n',
        '        "no_checkpoint_saved_or_exported": False,\n',
        "checkpoint result flag",
    )
    text = replace_once(
        text,
        '        "fullshape_profile": profile_records,\n',
        '        "fullshape_profile": profile_records,\n'
        '        "realdata_update": realdata_update,\n',
        "real-data result receipt",
    )

    text = replace_once(
        text, "def worker_main()", runtime + "\ndef worker_main()", "runtime addition insertion"
    )

    # Parent staging must resolve the dynamically selected source mount before
    # the inventory and source verifier consume SOURCE_MOUNT_ROOT.
    probe_bytes = text.encode("utf-8")
    manifest = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "packet_title": PACKET_TITLE,
        "packet_code_file": "probe.py",
        "code_bytes": len(probe_bytes),
        "code_sha256": sha256_bytes(probe_bytes),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "test_sha256": sha256_file(ROOT / "tests/test_v6_packet_contract.py"),
        "recipe_audit_sha256": sha256_file(ROOT / "V6_RECIPE_AUDIT.md"),
        "readme_sha256": sha256_file(ROOT / "README.md"),
        "base_v5_probe_sha256": BASE_PROBE_SHA256,
        "base_v5_packet": "vladkochetov07/wnn-e463-combo-reduction-tpu-v5e8-20261007-v5/1",
        "runtime_additions_sha256": sha256_file(REFERENCE / "v6_runtime_additions.py"),
        "source_manifest_sha256": "851898475dfa701f2089ec76cfe9592538aa2e7f9b70ebcb68c0a614e332d6fb",
        "source_archive_sha256": "a29bdf21a910a6d7b403d9ed8c52cf0761618f5ef6576b5f06d70d0a8fd5ec9e",
        "checkpoint_sha256": "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e",
        "checkpoint_npz_sha256": "677f604fd86d993c3d4a66a10638755253bfe52f1f578009dbd781c73fe6d04a",
        "checkpoint_sidecar_sha256": "3e65d2a88352d22bc90d88b829f05beee3faa7a66b9de41c7507f521aab12668",
        "model_source_sha256": "294eff11d906fe434dba2c4d0ed4741ad2191f756c4a35204474cb229ff939dc",
        "parent_provenance_sha256": provenance_sha,
        "fold_manifest_sha256": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
        "assets_manifest_sha256": "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7",
        "portable_inputs_sha256": "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c",
        "raw_parquet": {
            "train": {
                "bytes": 29240219011,
                "sha256": "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
            },
            "valid": {
                "bytes": 5615509128,
                "sha256": "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
            },
        },
        "helper_modules": {
            name: {"bytes": len(content), "sha256": sha256_bytes(content)}
            for name, content in sorted(support_bytes.items())
        },
        "support_path": "research/campaign_2026-10-04/staging/jax_tpu_e465/src/wnn_jax",
        "requested_inputs": [
            "vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1",
            "vladkochetov07/wnn-c20261004-fold0-assets",
            "vladkochetov07/wunder-funder",
        ],
        "execution": {
            "machine_shape": "TpuV5E8",
            "gpu": False,
            "internet": False,
            "parent_timeout_seconds": 1650,
            "worker_timeout_seconds": 1500,
            "global_batch": 16,
            "local_batch_per_tpu": 2,
            "chunk_rows": 2000,
            "sequence_rows": 20000,
            "model_dimensions": [211, 256, 2, 23],
            "real_optimizer_updates": 1,
            "profile_updates_are_synthetic": 4,
            "training_groups": 10535,
            "fit_valid_groups": 1405,
            "evaluation_groups": 468,
            "selected_probe_groups": 16,
            "selected_probe_split": "valid-fit only; exposed",
            "heldout_claim": None,
            "scoring": False,
        },
        "provider_push_authorized": False,
        "source_only": True,
    }
    return probe_bytes, manifest


def _write_archive() -> str:
    archive_path = ROOT / "packet.bundle.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for source in sorted(PACKET.iterdir(), key=lambda path: path.name):
            if not source.is_file():
                continue
            info = zipfile.ZipInfo(source.name, date_time=(2026, 10, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    return sha256_file(archive_path)


def main() -> int:
    if PACKET.exists() and any(PACKET.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty packet: {PACKET}")
    PACKET.mkdir(parents=True, exist_ok=True)
    probe, manifest = build_probe()
    metadata = {
        "id": PACKET_ID,
        "title": PACKET_TITLE,
        "code_file": "probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": False,
        "machine_shape": "TpuV5E8",
        "dataset_sources": [
            "vladkochetov07/wnn-jax-realdata-e463-tpu-20261004-v1",
            "vladkochetov07/wnn-c20261004-fold0-assets",
            "vladkochetov07/wunder-funder",
        ],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (PACKET / "probe.py").write_bytes(probe)
    (PACKET / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest["metadata_sha256"] = sha256_file(PACKET / "kernel-metadata.json")
    manifest["bundle_sha256"] = _write_archive()
    (ROOT / "packet_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    sys.stdout.write(json.dumps({"packet": str(PACKET), **manifest}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
