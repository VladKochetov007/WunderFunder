"""Stage an offline native-GPU FP16 asset dataset and sealed Kaggle packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

PREFIX = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "tpu_gpu_benchmark/native_precision_gpu_r1"
)
SOURCE_DATASET = "vladkochetov07/wnn-native-gpu-fp16-assets-20261009-r1"
KERNEL_ID = "vladkochetov07/wnn-native-gpu-fp16-20261009-r1"
KERNEL_TITLE = "WNN Native GPU FP16 20261009 R1"
R4_DATASET = "vladkochetov07/wnn-loss-pair-code-20261009-r4"
FOLD0_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
CACHE_KERNEL = "vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1"
R4_KERNEL_SHA256 = "86346f9c1addefbd38d3dcec52b18f514698a9514e7917aeb6081cc32f7eb039"
R4_ASSET_MANIFEST_SHA256 = "8ef360536ca117232198a4ee6dd8fabf072fb7bfad9812a83972d8bfb750f584"
R4_CODE_MANIFEST_SHA256 = "0a6fa1567dde9ac5fc41d27eed8d9edb6f90a93afb891db5a4080770b65e4351"
R4_TRAINER_SHA256 = "971c0336a71d374b507b0a5b83c312afc9ae8dbae8262cb492f0611e6d5bcc0b"
R4_PREDICTOR_SHA256 = "74d1eac17add3f9461e1205bac0eb5b6a78facd567898d679b15bdfa5bed4f5c"
R5_CONTRACT_SHA256 = "5b63ed39aa6046e9f02878e7dcb4b48982240bffa45ec09b7466dfb9eeac63f7"
R5_REVIEW_SHA256 = "60e9770de432a6b39d3ae41c818d60b2d47b5298f630b33ee3483a0e7fecc7f9"
R5_SHARED = {
    "cache_manifest": (
        "shared/cache_manifest.json",
        "7400b75504f592011577dd3b126e9813749c21ce9126e5c4fa04c8e6b61e8f3b",
    ),
    "cache_reader_profile": (
        "shared/cache_reader_profile_runtime.json",
        "fbac76445ff91dbe8d8336bc3ab2bc3e2ddddd3c4be90d25550c478ea774091e",
    ),
    "cache_validation_receipt": (
        "shared/cache_validation_runtime.json",
        "bf43ade9f08fb0447fc1df18dbb91f1d4723c828eaa3c14a7f5bffd73049f6ed",
    ),
    "device_profile_gate": (
        "shared/device_profile_gate.json",
        "70b8b40e994a74387a0b7e9055a532f21fbbb2ea516ead3ef3201d1ad3c575ae",
    ),
    "fold0_asset_validation_receipt": (
        "shared/fold0_assets_validation.json",
        "93ecabd180a608d72f111b9886e8804cb084d4ace694de57d188b745c33dbd43",
    ),
    "initial_state": (
        "shared/initial_state.npz",
        "10e4b56002b52faaaf55dbfcbd28b37f479d2dbd69e727740250dbb90d873af1",
    ),
    "normalization": (
        "shared/normalization.npz",
        "d096115cb59a7462608d5d11df29ba77142aac03527f26fbb2a8534077e57849",
    ),
    "normalization_metadata": (
        "shared/normalization.json",
        "0e4762ebb51f95589b93e79545c769376b42afaa2e7069f81b199164dd59cf96",
    ),
    "onecycle_trace": (
        "shared/schedule_trace.json",
        "a46a5beb2484b3549b6fdb926b885ecbcaf4a61b1cec60dd3750760eafff27de",
    ),
    "profile_inputs": (
        "shared/profile_inputs.npz",
        "f042ae97ddebac87c1023758ccd84d9b2923e5cc6ae3ac878596e84bfde9a1c6",
    ),
    "profile_inputs_receipt": (
        "shared/profile_inputs_receipt.json",
        "6620c5f34c13fd9310a4aaa7d27e0631b9bdf8d05ceb0459863cfed1746596ab",
    ),
    "sample_orders": (
        "shared/sample_orders.json",
        "cbb972a29278617890ac3e51d9f4d3cb954d392192aef3ebe044b8a71a8ec523",
    ),
    "source_stage_receipt": (
        "shared/source_stage_receipt.json",
        "557cd0a61f1129f811dc104398c3c9c7295c3343a1c9a302018fcb588c0c1de7",
    ),
    "source_verification": (
        "shared/source_verification.json",
        "1ccd91ccec85bd157cbd0244ae9032316b88d781a0caf19daf7a027f3a908567",
    ),
}
NATIVE_FILES = (
    "native_gpu_fp16_contract.py",
    "native_gpu_fp16_worker.py",
    "native_gpu_fp16_lifecycle_gate.py",
    "tests/test_native_gpu_fp16_contract.py",
    "BENCHMARK_CONTRACT_R1.md",
)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            hasher.update(block)
    return hasher.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _native_kernel(
    base_bytes: bytes,
    *,
    asset_manifest_sha: str,
    source_manifest_sha: str,
    asset_paths: list[str],
) -> bytes:
    source = base_bytes.decode()
    marker = "\n\ndef sha256_file(path: Path) -> str:\n"
    quoted_asset_paths = ",\n".join(f"    {json.dumps(path)}" for path in sorted(asset_paths))
    constants = (
        f"\nNATIVE_ASSET_DATASET = {json.dumps(SOURCE_DATASET)}\n"
        "EXPECTED_NATIVE_ASSET_MANIFEST_SHA256 = (\n"
        f"    {json.dumps(asset_manifest_sha)}\n"
        ")\n"
        "EXPECTED_NATIVE_SOURCE_MANIFEST_SHA256 = (\n"
        f"    {json.dumps(source_manifest_sha)}\n"
        ")\n"
        "EXPECTED_NATIVE_ASSET_PATHS = {\n"
        f"{quoted_asset_paths}\n"
        "}\n"
    )
    if source.count(marker) != 1:
        raise ValueError("R4 kernel bootstrap insertion point changed")
    source = source.replace(marker, constants + marker)

    functions = [
        "def discover_native_asset_manifest() -> tuple[Path, dict[str, Any]]:",
        "    found = []",
        "    root_depth = len(INPUT.parts)",
        "    for directory, subdirs, files in os.walk(INPUT, followlinks=False):",
        "        current = Path(directory)",
        "        depth = len(current.parts) - root_depth",
        "        subdirs[:] = sorted(subdirs) if depth < 8 else []",
        '        if "assets-manifest.json" not in files:',
        "            continue",
        '        path = current / "assets-manifest.json"',
        '        value = json.loads(path.read_text(encoding="utf-8"))',
        '        if value.get("dataset") == NATIVE_ASSET_DATASET:',
        "            found.append((path, value))",
        "    if len(found) != 1:",
        "        raise RuntimeError(",
        '            f"expected one native overlay asset manifest, found {len(found)}"',
        "        )",
        "    path, manifest = found[0]",
        "    if sha256_file(path) != EXPECTED_NATIVE_ASSET_MANIFEST_SHA256:",
        "        raise RuntimeError(",
        '            "native overlay asset manifest hash differs from its packet pin"',
        "        )",
        '    if manifest.get("private") is not True or manifest.get("version") != 1:',
        '        raise RuntimeError("native overlay asset manifest metadata is unsupported")',
        '    rows = manifest.get("files")',
        "    if (",
        "        not isinstance(rows, list)",
        '        or {row.get("path") for row in rows} != EXPECTED_NATIVE_ASSET_PATHS',
        "    ):",
        "        raise RuntimeError(",
        '            "native overlay asset inventory differs from the sealed path set"',
        "        )",
        "    for row in rows:",
        '        filename = str(row.get("filename", ""))',
        '        if Path(filename).name != filename or not filename.endswith(".wnn"):',
        "            raise RuntimeError(",
        '                "native overlay payload filenames must be flat opaque .wnn names"',
        "            )",
        "        payload = path.parent / filename",
        "        if (",
        '            payload.stat().st_size != int(row["size"])',
        '            or sha256_file(payload) != row["sha256"]',
        "        ):",
        "            raise RuntimeError(",
        "                f\"native overlay payload differs from its manifest: {row.get('path')}\"",
        "            )",
        "    return path.parent, manifest",
        "",
        "def unpack_native_overlay(",
        "    asset_root: Path, manifest: dict[str, Any]",
        ") -> dict[str, Any]:",
        '    for row in manifest["files"]:',
        '        relative = PurePosixPath(str(row["destination"]))',
        "        if (",
        "            relative.is_absolute()",
        "            or not relative.parts",
        '            or any(part in ("", ".", "..") for part in relative.parts)',
        "        ):",
        '            raise RuntimeError("native overlay destination path is unsafe")',
        '        source_path = asset_root / row["filename"]',
        "        destination = REPO.joinpath(*relative.parts)",
        "        destination.parent.mkdir(parents=True, exist_ok=True)",
        "        shutil.copyfile(source_path, destination)",
        "        if (",
        '            destination.stat().st_size != int(row["size"])',
        '            or sha256_file(destination) != row["sha256"]',
        "        ):",
        "            raise RuntimeError(",
        "                f\"extracted native overlay differs from its pin: {row.get('path')}\"",
        "            )",
        '    source_manifest_path = REPO / "native_controls/source_manifest.json"',
        "    if sha256_file(source_manifest_path) != EXPECTED_NATIVE_SOURCE_MANIFEST_SHA256:",
        "        raise RuntimeError(",
        '            "native source manifest hash differs from its root review pin"',
        "        )",
        '    return json.loads(source_manifest_path.read_text(encoding="utf-8"))',
        "",
        "",
    ]
    functions_text = "\n".join(functions)
    unpack_marker = (
        "def unpack_source(asset_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:\n"
    )
    if source.count(unpack_marker) != 1:
        raise ValueError("R4 source unpack insertion point changed")
    source = source.replace(unpack_marker, functions_text + unpack_marker)

    main_marker = "        code_manifest = unpack_source(asset_root, manifest)\n"
    native_main = (
        main_marker
        + "        native_asset_root, native_manifest = discover_native_asset_manifest()\n"
        + "        unpack_native_overlay(native_asset_root, native_manifest)\n"
    )
    if source.count(main_marker) != 1:
        raise ValueError("R4 main bootstrap invocation changed")
    source = source.replace(main_marker, native_main)
    bootstrap_marker = '            "source_file_count": len(code_manifest["files"]),\n'
    bootstrap_fields = (
        bootstrap_marker
        + '            "native_source_manifest_sha256": EXPECTED_NATIVE_SOURCE_MANIFEST_SHA256,\n'
        + '            "native_controls_root": str(REPO / "native_controls"),\n'
        + '            "native_root_admission_path": str(REPO / "native_controls/root_admission.json"),\n'
        + '            "native_kernel_code_sha256": sha256_file(Path(__file__).resolve()),\n'
    )
    if source.count(bootstrap_marker) != 1:
        raise ValueError("R4 bootstrap receipt insertion point changed")
    source = source.replace(bootstrap_marker, bootstrap_fields)
    entry = '[sys.executable, str(REPO / "scripts/loss_pair_gpu_worker.py")]'
    if source.count(entry) != 1:
        raise ValueError("R4 entry point changed")
    source = source.replace(
        entry, '[sys.executable, str(REPO / "scripts/native_gpu_fp16_worker.py")]'
    )
    return source.encode()


def seal(source_root: Path, artifact_root: Path) -> dict[str, Any]:
    package = source_root / PREFIX
    source_assets = package / "source_asset"
    kernel_dir = package / "kernel"
    if source_assets.exists() or kernel_dir.exists():
        raise FileExistsError(
            "R1 packet artifacts already exist; preserve them and use a successor"
        )
    source_assets.mkdir(parents=True)
    kernel_dir.mkdir(parents=True)
    diversity = (
        artifact_root / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    )
    r4_packet = diversity / "cloud_delivery/gpu_loss_pair_r4/gpu_loss_pair.packet.json"
    r4_asset_root = diversity / "cloud_delivery/gpu_loss_pair_r4/source_asset"
    r4_manifest_path = r4_asset_root / "assets-manifest.json"
    r4_kernel = (
        diversity / "cloud_delivery/gpu_loss_pair_r4/kernel/wnn-loss-pair-gpu-20261009-v4.py"
    )
    if (
        file_digest(r4_manifest_path) != R4_ASSET_MANIFEST_SHA256
        or file_digest(r4_kernel) != R4_KERNEL_SHA256
    ):
        raise ValueError("reviewed R4 source packet or kernel bootstrap changed")
    r4_packet_value = json.loads(r4_packet.read_text(encoding="utf-8"))
    r4_assets_manifest = json.loads(r4_manifest_path.read_text(encoding="utf-8"))
    code_rows = [
        row for row in r4_assets_manifest["files"] if row.get("path") == "source-code-manifest.json"
    ]
    if len(code_rows) != 1:
        raise ValueError("R4 source asset manifest lacks its code-manifest payload")
    r4_code_path = r4_asset_root / code_rows[0]["filename"]
    if file_digest(r4_code_path) != R4_CODE_MANIFEST_SHA256:
        raise ValueError("R4 code manifest differs from its reviewed pin")
    r4_code = json.loads(r4_code_path.read_text(encoding="utf-8"))
    r4_code_files = {row["path"]: row for row in r4_code["files"]}
    if r4_code_files["scripts/train_rnn.py"]["sha256"] != R4_TRAINER_SHA256:
        raise ValueError("reviewed R4 trainer source differs from its precision-overlay pin")
    if r4_code_files["scripts/predict_run.py"]["sha256"] != R4_PREDICTOR_SHA256:
        raise ValueError("reviewed R4 predictor source differs from its CLI-consumer pin")
    source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    r5_output = (
        diversity
        / "cloud_delivery/gpu_loss_pair_r5_export/terminal_collect_r1/attempt_001/outputs/out"
    )
    r5_review = diversity / "ROOT_GPU_R5_TERMINAL_REVIEW_R1.json"
    if file_digest(r5_output / "training_contract.json") != R5_CONTRACT_SHA256:
        raise ValueError("actual R5 training contract differs from the reviewed pin")
    if file_digest(r5_review) != R5_REVIEW_SHA256:
        raise ValueError("actual R5 root review differs from the reviewed pin")

    native_payloads: dict[str, tuple[str, bytes]] = {}
    source_files = []
    for filename in NATIVE_FILES:
        local = package / filename
        payload = local.read_bytes()
        if filename.endswith(".py") and filename != "tests/test_native_gpu_fp16_contract.py":
            destination = f"scripts/{filename}"
        else:
            destination = f"native_source/{filename}"
        native_path = f"native_code/{filename}"
        native_payloads[native_path] = (destination, payload)
        source_files.append(
            {
                "path": native_path,
                "destination": destination,
                "size_bytes": len(payload),
                "sha256": digest(payload),
            }
        )
    controls: dict[str, tuple[Path, str]] = {
        "native_controls/training_contract.json": (
            r5_output / "training_contract.json",
            R5_CONTRACT_SHA256,
        ),
        "native_controls/ROOT_GPU_R5_TERMINAL_REVIEW_R1.json": (r5_review, R5_REVIEW_SHA256),
    }
    for relative, expected_sha in R5_SHARED.values():
        controls[f"native_controls/{relative}"] = (r5_output / relative, expected_sha)
    for destination, (source_path, expected_sha) in controls.items():
        if file_digest(source_path) != expected_sha:
            raise ValueError(f"actual R5 role differs from its packet pin: {destination}")
        payload = source_path.read_bytes()
        native_payloads[destination] = (destination, payload)
        source_files.append(
            {
                "path": destination,
                "destination": destination,
                "size_bytes": len(payload),
                "sha256": expected_sha,
            }
        )

    source_manifest = {
        "schema": "wnn.native-gpu-fp16.source-manifest.v1",
        "format_version": 1,
        "source_commit": source_commit,
        "producer": {
            "kernel": "vladkochetov07/wnn-loss-pair-export-recovery-20261009-r1",
            "r5_training_contract_sha256": R5_CONTRACT_SHA256,
            "r5_root_review_sha256": R5_REVIEW_SHA256,
            "base_gpu_code_dataset": R4_DATASET,
            "base_gpu_source_commit": r4_packet_value["source"]["source_commit"],
            "base_gpu_worker_sha256": r4_code_files["scripts/loss_pair_gpu_worker.py"]["sha256"],
            "base_trainer_sha256": r4_code_files["scripts/train_rnn.py"]["sha256"],
            "base_predictor_sha256": r4_code_files["scripts/predict_run.py"]["sha256"],
        },
        "precision_policy": {
            "compute": "PyTorch CUDA FP16 autocast",
            "master_parameters": "FP32",
            "AdamW_accumulators": "FP32",
            "GradScaler_initial_scale": 1024.0,
            "scaler_search": False,
        },
        "files": sorted(source_files, key=lambda row: row["path"]),
    }
    source_manifest_bytes = canonical(source_manifest)
    source_manifest_sha = digest(source_manifest_bytes)
    native_payloads["native_controls/source_manifest.json"] = (
        "native_controls/source_manifest.json",
        source_manifest_bytes,
    )
    pending_admission = {
        "schema": "wnn.native-gpu-fp16.root-admission.v1",
        "state": "pending",
        "source_manifest_sha256": source_manifest_sha,
        "r5_training_contract_sha256": R5_CONTRACT_SHA256,
        "r5_root_review_sha256": R5_REVIEW_SHA256,
        "compute_precision": "fp16",
        "amp_initial_scale": 1024.0,
        "push_allowed": False,
        "profile_allowed": False,
        "conditional_fit_allowed": False,
        "training_contract_sha256": None,
    }
    auth_bytes = canonical(pending_admission)
    native_payloads["native_controls/root_admission.json"] = (
        "native_controls/root_admission.json",
        auth_bytes,
    )
    upload_rows = []
    for path, (destination, payload) in sorted(native_payloads.items()):
        opaque = hashlib.sha256((path + "\0" + digest(payload)).encode()).hexdigest() + ".wnn"
        (source_assets / opaque).write_bytes(payload)
        upload_rows.append(
            {
                "path": path,
                "destination": destination,
                "filename": opaque,
                "sha256": digest(payload),
                "size": len(payload),
            }
        )
    asset_manifest = {
        "dataset": SOURCE_DATASET,
        "version": 1,
        "private": True,
        "files": upload_rows,
    }
    asset_manifest_bytes = canonical(asset_manifest)
    asset_manifest_sha = digest(asset_manifest_bytes)
    (source_assets / "assets-manifest.json").write_bytes(asset_manifest_bytes)
    metadata = {
        "id": SOURCE_DATASET,
        "title": "WNN Native GPU FP16 Assets 20261009 R1",
        "isPrivate": True,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (source_assets / "dataset-metadata.json").write_bytes(canonical(metadata))
    (package / "source_manifest.json").write_bytes(source_manifest_bytes)
    (package / "root_admission.pending.json").write_bytes(auth_bytes)

    base_kernel = _native_kernel(
        r4_kernel.read_bytes(),
        asset_manifest_sha=asset_manifest_sha,
        source_manifest_sha=source_manifest_sha,
        asset_paths=[row["path"] for row in upload_rows],
    )
    kernel_path = kernel_dir / "wnn-native-gpu-fp16-20261009-r1.py"
    kernel_path.write_bytes(base_kernel)
    kernel_metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": kernel_path.name,
        "language": "python",
        "kernel_type": "script",
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "is_private": True,
        "competition_sources": [],
        "dataset_sources": [R4_DATASET, SOURCE_DATASET, FOLD0_DATASET],
        "kernel_sources": [CACHE_KERNEL],
        "model_sources": [],
    }
    metadata_bytes = canonical(kernel_metadata)
    (kernel_dir / "kernel-metadata.json").write_bytes(metadata_bytes)
    packet = {
        "schema": "wnn.native-gpu-fp16.packet.v1",
        "packet_id": KERNEL_ID,
        "status": "sealed_candidate_waiting_root_review_and_admission",
        "authorized_for_execution": False,
        "root_admission_required_before_asset_upload_or_kernel_push": True,
        "source_manifest_sha256": source_manifest_sha,
        "source_asset_dataset": SOURCE_DATASET,
        "source_asset_manifest_sha256": asset_manifest_sha,
        "source_commit": source_commit,
        "producer": {
            "r5_contract_sha256": R5_CONTRACT_SHA256,
            "r5_root_terminal_review_sha256": R5_REVIEW_SHA256,
            "r4_kernel": r4_packet_value["packet_id"],
            "r4_kernel_source_sha256": R4_KERNEL_SHA256,
            "r4_code_dataset": R4_DATASET,
        },
        "kernel": {
            "id": KERNEL_ID,
            "title": KERNEL_TITLE,
            "code_file": kernel_path.name,
            "code_file_sha256": digest(base_kernel),
            "metadata_sha256": digest(metadata_bytes),
            "title_slug_matches_id": True,
            "accelerator": "two Nvidia Tesla T4 GPUs",
            "worker_wall_seconds": 3300,
            "provider_timeout_seconds": 3600,
        },
        "recipe": {
            "precision": "native CUDA FP16 autocast; FP32 master parameters and AdamW accumulators",
            "GradScaler_initial_scale": 1024.0,
            "scaler_search": False,
            "loss_arms": ["WMSE", "Huber delta 1"],
            "control_contract_sha256": R5_CONTRACT_SHA256,
            "same_initialization_membership_order_and_schedule": True,
            "batch_size": 32,
            "chunk_rows": 2000,
            "fit_sequences": 350,
            "evaluation_sequences": 468,
            "updates_per_arm": 220,
            "final_gpu_batch": "30 real sequences; drop_last=False",
        },
        "conditional_fit_headroom": {
            "minimum_host_mem_available_bytes": 4_000_000_000,
            "minimum_bounded_cgroup_headroom_bytes": 4_000_000_000,
            "unbounded_cgroup_requires_explicit_observation_and_host_reserve": True,
            "minimum_working_disk_free_bytes": 4_000_000_000,
            "minimum_free_per_gpu_bytes": 2_000_000_000,
        },
        "source_asset_upload_inventory": [
            {
                "filename": row["filename"],
                "path": row["path"],
                "destination": row["destination"],
                "sha256": row["sha256"],
                "size_bytes": row["size"],
            }
            for row in upload_rows
        ]
        + [
            {
                "filename": name,
                "path": name,
                "sha256": file_digest(source_assets / name),
                "size_bytes": (source_assets / name).stat().st_size,
            }
            for name in ("assets-manifest.json", "dataset-metadata.json")
        ],
        "root_admission": pending_admission,
    }
    packet_bytes = canonical(packet)
    packet["packet_sha256_before_self_pin"] = digest(packet_bytes)
    (package / "native_gpu_fp16.packet.json").write_bytes(canonical(packet))
    return {
        "source_manifest_sha256": source_manifest_sha,
        "source_asset_manifest_sha256": asset_manifest_sha,
        "kernel_sha256": digest(base_kernel),
        "packet_path": str(package / "native_gpu_fp16.packet.json"),
        "source_asset_path": str(source_assets),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[6])
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    result = seal(args.source_root.resolve(), (args.artifact_root or args.source_root).resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
