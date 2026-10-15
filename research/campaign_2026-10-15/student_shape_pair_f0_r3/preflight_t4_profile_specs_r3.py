from __future__ import annotations

import ast
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

# Heterogeneous nested JSON values follow the frozen packet schemas.

ROOT = Path(__file__).resolve().parents[3]
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r3"
PACKET_R2 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r2"
BUNDLE_R2 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r2"
BUNDLE_R3 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r3"
R1 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r1"
CONTRACT = R1 / "CPU_STUDENT_SHAPE_PAIR_F0_SEED3_B16_C2000_FP16_3E_R1_CONTRACT.json"
SOURCE_BINDING = PACKET / "SOURCE_BINDING_R3.json"
INPUT_BINDING = PACKET / "INPUT_BINDING_R3.json"
KERNEL_SEAL = PACKET / "GENERATED_KERNEL_SEAL_R3_SEALED.json"
OUTPUT = PACKET / "OFFLINE_SPEC_PREFLIGHT_R3_SEALED.json"
EXPECTED_SHAPES = ((256, 2), (512, 1))
TEACHER_DATASET = "vladkochetov07/wnn-clean-t3-f0-h384-cp2-20261012-r9"
R9_MANIFEST = MAIN / (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_f0_h390_r9_export_from_owned_checkpoints_20261012_r2/"
    "R9_CHECKPOINT_INPUT_MANIFEST_R2.json"
)
sys.path.insert(0, str(ROOT / "scripts"))
_kaggle_common = importlib.import_module("kaggle_common")
read_spec = _kaggle_common.read_spec


def sha256_file(path: Path) -> str:
    """Hash one local artifact with bounded memory use."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a small, packet-controlled JSON object."""
    if path.stat().st_size > 4_000_000:
        raise ValueError(f"refusing oversized packet JSON: {path}")
    return json.loads(path.read_text())


def argument(argv: list[str], name: str) -> str:
    """Return one value-bearing option from a trainer argument vector."""
    try:
        index = argv.index(name)
    except ValueError as error:
        raise ValueError(f"Missing required trainer option {name}") from error
    if index + 1 >= len(argv):
        raise ValueError(f"Trainer option {name} has no value")
    return argv[index + 1]


def external_map(binding: dict[str, Any]) -> dict[str, object]:
    """Convert the input binding to the generated kernel's mount-map format."""
    return {
        item["logical_path"]: (
            item["filename"]
            if item["dataset"] == "vladkochetov07/wunder-funder"
            else {"dataset": item["dataset"], "filename": item["filename"]}
        )
        for item in binding["external_mounts"]
    }


def kernel_mount_map(kernel_path: Path) -> dict[str, object]:
    """Read the generated kernel's literal mount map without importing it."""
    tree = ast.parse(kernel_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_EXTERNAL_MOUNTS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                return value
    raise ValueError("generated kernel has no literal EXPECTED_EXTERNAL_MOUNTS map")


def main() -> None:
    """Validate both R3 profile specs and the corrected six-file mount packet."""
    contract = read_json(CONTRACT)
    source_binding = read_json(SOURCE_BINDING)
    input_binding = read_json(INPUT_BINDING)
    kernel_seal = read_json(KERNEL_SEAL)
    build_receipt = read_json(PACKET / "OFFLINE_BUNDLE_BUILD_R3.json")
    mount_receipt = read_json(PACKET / "KERNEL_MOUNT_SUCCESSOR_R3.json")
    r9_manifest = read_json(R9_MANIFEST)
    r2_input_binding = read_json(PACKET_R2 / "INPUT_BINDING_R2.json")
    r2_bundle_manifest = read_json(BUNDLE_R2 / "bundle-manifest.json")
    r3_bundle_manifest = read_json(BUNDLE_R3 / "bundle-manifest.json")
    stage = contract["stages"]["f0_seed3_t4_profile"]
    recipe = contract["recipe"]

    jobs: list[dict[str, Any]] = []
    source_paths: list[set[str]] = []
    input_identities: list[set[tuple[str, int, str]]] = []
    for hidden, layers in EXPECTED_SHAPES:
        path = (
            PACKET
            / "jobs"
            / f"student_pair_t4_profile_f0s3_h{hidden}_l{layers}_r2.json"
        )
        spec = read_spec(path)
        argv = spec["argv"]
        expected = {
            "--arch": "gru",
            "--cell": "gru",
            "--hidden": str(hidden),
            "--layers": str(layers),
            "--batch": str(recipe["batch_size"]),
            "--chunk": str(recipe["chunk_rows"]),
            "--seed": str(stage["seed"]),
            "--precision": recipe["precision"],
            "--amp-init-scale": str(int(recipe["optimizer"]["amp_init_scale"])),
            "--epochs": str(stage["epochs"]),
            "--checkpoint-epoch": str(stage["checkpoint_epoch"]),
            "--max-batches": str(stage["max_batches"]),
            "--profile-steps": str(stage["profile_steps"]),
            "--profile-warmup": str(stage["profile_warmup"]),
            "--student-frontier-stage": "f0_seed3_t4_profile",
            "--distill-alpha": str(
                contract["cache"]["historical_student_contract"]["distill_alpha"]
            ),
        }
        for option, value in expected.items():
            if argument(argv, option) != value:
                raise ValueError(
                    f"{path.name}: {option} differs from the frozen profile recipe"
                )
        if spec["kind"] != "gpu" or spec["artifacts"]:
            raise ValueError(
                f"{path.name}: expected profile-only GPU spec with no saved artifacts"
            )
        if spec["limits"] != {
            "cpus": 2,
            "ram_bytes": 8_000_000_000,
            "timeout_seconds": 600,
            "gpu_memory_bytes": 3_500_000_000,
            "gpu_stop_bytes": 5_000_000_000,
        }:
            raise ValueError(
                f"{path.name}: resource caps differ from the reviewed profile plan"
            )
        if argument(argv, "--student-frontier-contract") != str(
            CONTRACT.relative_to(ROOT)
        ):
            raise ValueError(f"{path.name}: contract path differs")
        if argument(argv, "--distill-cache-root") != contract["cache"]["root"]:
            raise ValueError(f"{path.name}: cache root differs")
        jobs.append(spec)
        source_paths.append({item["path"] for item in spec["code"]})
        input_identities.append(
            {(item["path"], item["size"], item["sha256"]) for item in spec["inputs"]}
        )

    if source_paths[0] != source_paths[1]:
        raise ValueError("profile shapes do not share the same code closure")
    expected_external = {
        (item["logical_path"], item["size"], item["sha256"])
        for item in input_binding["external_mounts"]
    }
    if any(identity != expected_external for identity in input_identities):
        raise ValueError("profile job input identities differ from the R3 binding")
    if source_binding["recipe_contract_sha256"] != sha256_file(CONTRACT):
        raise ValueError("R3 source binding has a stale recipe hash")
    if source_binding["input_binding_sha256"] != sha256_file(INPUT_BINDING):
        raise ValueError("R3 source binding has a stale input binding hash")
    if source_binding["consumer_source_sha256"] != sha256_file(
        ROOT / source_binding["consumer_source_path"]
    ):
        raise ValueError("consumer source hash differs from the R3 binding")
    if source_binding["spec_generator_sha256"] != sha256_file(
        ROOT / source_binding["spec_generator_path"]
    ):
        raise ValueError("spec generator source hash differs from the R3 binding")
    if source_binding["bundle_successor_sha256"] != sha256_file(
        ROOT / source_binding["bundle_successor_path"]
    ):
        raise ValueError("bundle successor source hash differs from the R3 binding")
    if source_binding["bundle_successor_receipt_sha256"] != sha256_file(
        PACKET / "OFFLINE_BUNDLE_BUILD_R3.json"
    ):
        raise ValueError("bundle successor receipt hash differs from the R3 binding")
    if source_binding["kernel_mount_successor_receipt_sha256"] != sha256_file(
        PACKET / "KERNEL_MOUNT_SUCCESSOR_R3.json"
    ):
        raise ValueError("kernel mount successor receipt differs from the R3 binding")
    if source_binding["r2_bundle_rebuild_failure_sha256"] != sha256_file(
        PACKET / "R3_BUNDLE_REBUILD_FAILURE_R1.json"
    ):
        raise ValueError("preserved R3 full-rebuild failure receipt changed")
    if source_binding["code_aggregate_sha256"] != jobs[0]["code_sha256"]:
        raise ValueError("profile code aggregate differs from the R3 source binding")
    r2_files = {
        item["path"]: (item["size"], item["sha256"])
        for item in r2_bundle_manifest["files"]
    }
    for spec in jobs:
        code_rows = {
            item["path"]: (item["size"], item["sha256"]) for item in spec["code"]
        }
        if not code_rows or any(
            r2_files.get(path) != identity for path, identity in code_rows.items()
        ):
            raise ValueError(
                "profile code identities differ from the already sealed R2 archive"
            )
    if len(jobs[0]["code"]) != source_binding["code_file_count"]:
        raise ValueError("profile code closure count differs from the source binding")
    if sha256_file(BUNDLE_R3 / "bundle.wnn") != sha256_file(BUNDLE_R2 / "bundle.wnn"):
        raise ValueError("R3 archive is not byte-identical to the sealed R2 archive")
    r2_manifest_without_mounts = dict(r2_bundle_manifest)
    r3_manifest_without_mounts = dict(r3_bundle_manifest)
    r2_manifest_without_mounts.pop("external_mounts")
    r3_manifest_without_mounts.pop("external_mounts")
    if r2_manifest_without_mounts != r3_manifest_without_mounts:
        raise ValueError("R3 bundle manifest changed fields outside external mounts")
    old_mounts = external_map(r2_input_binding)
    new_mounts = external_map(input_binding)
    changed_mount_paths = [
        path for path in old_mounts if old_mounts[path] != new_mounts[path]
    ]
    receipt_mount_paths = [
        row["logical_path"] for row in mount_receipt["changed_fields"]
    ]
    if len(changed_mount_paths) != 2 or set(changed_mount_paths) != set(
        receipt_mount_paths
    ):
        raise ValueError("R3 successor does not record exactly the two norm filenames")
    if r3_bundle_manifest["external_mounts"] != new_mounts:
        raise ValueError("R3 bundle manifest map differs from the frozen input binding")
    if (
        build_receipt["archive_byte_identical_to_r2"] is not True
        or build_receipt["r3_source_archive_sha256"]
        != build_receipt["r2_source_archive_sha256"]
        or mount_receipt["r3_kernel_sha256"] != kernel_seal["kernel"]["patched_sha256"]
    ):
        raise ValueError(
            "R3 transport or kernel mount successor receipt is inconsistent"
        )
    expected_job_hashes = [
        sha256_file(
            PACKET
            / row["path"].removeprefix(
                "research/campaign_2026-10-15/student_shape_pair_f0_r3/"
            )
        )
        for row in source_binding["profile_jobs"]
    ]
    if expected_job_hashes != [
        sha256_file(path) for path in sorted(PACKET.glob("jobs/*.json"))
    ]:
        raise ValueError("R3 profile job hash list differs from the sealed job specs")

    r9_by_name = {item["path"]: item for item in r9_manifest["files"]}
    teacher_mounts = [
        item
        for item in input_binding["external_mounts"]
        if item["dataset"] == TEACHER_DATASET
    ]
    if len(teacher_mounts) != 6:
        raise ValueError("R3 must bind exactly six R9 CP2/provenance files")
    for item in teacher_mounts:
        expected = r9_by_name.get(item["filename"])
        if expected is None or (item["sha256"], item["size"]) != (
            expected["sha256"],
            expected["bytes"],
        ):
            raise ValueError(f"R3 mount differs from R9 manifest: {item['filename']}")
    expected_mount_map = external_map(input_binding)
    bundle_path = ROOT / kernel_seal["bundle"]["archive_path"]
    bundle_manifest_path = ROOT / kernel_seal["bundle"]["manifest_path"]
    kernel_path = ROOT / kernel_seal["kernel"]["path"]
    if kernel_seal["source_binding_sha256"] != sha256_file(SOURCE_BINDING):
        raise ValueError("generated kernel seal has stale R3 source binding hash")
    if kernel_seal["input_binding_sha256"] != sha256_file(INPUT_BINDING):
        raise ValueError("generated kernel seal has stale R3 input binding hash")
    if kernel_seal["bundle"]["archive_sha256"] != sha256_file(bundle_path):
        raise ValueError("R3 bundle archive differs from its seal")
    if kernel_seal["bundle"]["manifest_sha256"] != sha256_file(bundle_manifest_path):
        raise ValueError("R3 bundle manifest differs from its seal")
    if kernel_seal["kernel"]["patched_sha256"] != sha256_file(kernel_path):
        raise ValueError("R3 patched kernel differs from its seal")
    if kernel_mount_map(kernel_path) != expected_mount_map:
        raise ValueError(
            "generated kernel external mounts do not use exact R3 logical/flat names"
        )
    if (
        input_binding["provider_action_performed"]
        or source_binding["provider_action_performed"]
    ):
        raise ValueError("offline R3 packet must not claim provider actions")

    spec_receipts = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "shape": [hidden, layers],
            "trainer_run_name": source_binding["profile_jobs"][index][
                "trainer_run_name"
            ],
            "input_count": len(jobs[index]["inputs"]),
            "code_count": len(jobs[index]["code"]),
            "expected_attempts": int(argument(jobs[index]["argv"], "--max-batches"))
            * (
                int(contract["recipe"]["sequence_rows"])
                // int(contract["recipe"]["chunk_rows"])
            ),
            "expected_complete_sequences": int(
                argument(jobs[index]["argv"], "--max-batches")
            ),
        }
        for index, (hidden, layers) in enumerate(EXPECTED_SHAPES)
        for path in [
            PACKET
            / "jobs"
            / f"student_pair_t4_profile_f0s3_h{hidden}_l{layers}_r2.json"
        ]
    ]
    receipt = {
        "schema": "wnn.student-shape-pair-offline-spec-preflight.r3",
        "status": "passed_offline_no_provider_action",
        "provider_action_performed": False,
        "command": (
            "WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=20% WNN_THREADS=1 "
            "WNN_MEM_MAX=300000000 WNN_RUNTIME_MAX=120s "
            "/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited "
            "/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python "
            "research/campaign_2026-10-15/student_shape_pair_f0_r3/"
            "preflight_t4_profile_specs_r3.py"
        ),
        "working_directory": str(ROOT),
        "preflight_script_sha256": sha256_file(Path(__file__)),
        "source_binding_sha256": sha256_file(SOURCE_BINDING),
        "input_binding_sha256": sha256_file(INPUT_BINDING),
        "kernel_seal_sha256": sha256_file(KERNEL_SEAL),
        "contract_sha256": sha256_file(CONTRACT),
        "r9_manifest_sha256": sha256_file(R9_MANIFEST),
        "r9_mounts": [
            {
                "dataset": item["dataset"],
                "version": 1,
                "filename": item["filename"],
                "logical_path": item["logical_path"],
                "sha256": item["sha256"],
                "size": item["size"],
            }
            for item in teacher_mounts
        ],
        "checks": [
            "exact two-shape H256x2/H512x1 B16/C2000/FP16 five-batch CLI specs",
            "same code closure and exact 22-input identities for both jobs",
            "all 68 frozen job code identities are present with matching hashes and sizes in the already sealed R2 archive manifest; the mutable checkout is not used to rebuild or replace those bytes",
            "R3 source archive is byte-identical to the sealed R2 source archive, with all manifest fields except external_mounts unchanged",
            "exactly two normalization filenames changed in the external map; logical paths, payload hashes, sizes, and dataset identity remain fixed",
            "all six R9 CP2/provenance flat filenames, hashes, sizes, and unchanged logical mount paths",
            "generated kernel AST mount map equals R3 input binding",
            "R3 source binding, build and kernel-mount receipts, bundle archive, generated kernel, and reviewed R1 recipe contract hashes agree",
            "the earlier full local rebuild mismatch remains recorded as infrastructure evidence and is not retried",
            "profile-only specs have empty saved-artifact lists and no provider action",
        ],
        "attempt_semantics": {
            "formula": "max_batches * (sequence_rows // chunk_rows)",
            "max_batches_per_shape": int(stage["max_batches"]),
            "chunk_updates_per_sequence": int(contract["recipe"]["sequence_rows"])
            // int(contract["recipe"]["chunk_rows"]),
            "attempts_per_shape": int(stage["max_batches"])
            * (
                int(contract["recipe"]["sequence_rows"])
                // int(contract["recipe"]["chunk_rows"])
            ),
        },
        "specs": spec_receipts,
    }
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite R3 preflight receipt {OUTPUT}")
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "specs": spec_receipts,
                "r9_mount_count": len(teacher_mounts),
                "kernel_seal_sha256": receipt["kernel_seal_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
