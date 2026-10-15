from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# JSON objects here have heterogeneous, contract-controlled nested values.

ROOT = Path(__file__).resolve().parents[3]
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
R1 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r1"
PACKET = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r2"
UPLOAD = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r2"
CONTRACT = R1 / "CPU_STUDENT_SHAPE_PAIR_F0_SEED3_B16_C2000_FP16_3E_R1_CONTRACT.json"
R9_MANIFEST = MAIN / (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_f0_h390_r9_export_from_owned_checkpoints_20261012_r2/"
    "R9_CHECKPOINT_INPUT_MANIFEST_R2.json"
)
OWNER = "vladkochetov07"
SOURCE_DATASET = f"{OWNER}/wnn-f0-student-shape-t4-profile-20261015-r1"
RAW_DATASET = f"{OWNER}/wunder-funder"
TEACHER_DATASET = f"{OWNER}/wnn-clean-t3-f0-h384-cp2-20261012-r9"


def sha256_file(path: Path) -> str:
    """Hash a local ancestry or recipe file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a small, locally pinned JSON object."""
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    """Write a deterministic packet receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Build the R2 six-file mount correction and unique profile job specs."""
    if UPLOAD.exists():
        raise FileExistsError(f"refusing to overwrite source upload stage: {UPLOAD}")

    r1_binding = read_json(R1 / "INPUT_BINDING_R1.json")
    r9_manifest = read_json(R9_MANIFEST)
    if r9_manifest["dataset_id"] != TEACHER_DATASET:
        raise ValueError("R9 checkpoint manifest names another teacher dataset")
    file_rows = {item["path"]: item for item in r9_manifest["files"]}

    teacher_filename_by_hash: dict[str, str] = {}
    for teacher in read_json(CONTRACT)["cache"]["teachers"]:
        run = teacher["run"]
        for logical_path, digest, filename in (
            (
                teacher["checkpoint"],
                teacher["checkpoint_sha256"],
                f"{run}__model_ep2.pt",
            ),
            (
                teacher["provenance"],
                teacher["provenance_sha256"],
                f"{run}__provenance.json",
            ),
        ):
            source_path = Path(logical_path).name
            item = file_rows.get(filename)
            if item is None or item["sha256"] != digest:
                raise ValueError(
                    f"R9 manifest does not bind exact expected ancestry: {filename}"
                )
            if item["bytes"] != (8_692_775 if "model_ep2" in filename else 159_434):
                raise ValueError(f"R9 manifest size differs for {filename}")
            if not source_path:
                raise ValueError("empty logical ancestry filename")
            teacher_filename_by_hash[digest] = filename

    input_binding = r1_binding
    input_binding["schema"] = "wnn.student-shape-pair-input-binding.r2"
    input_binding["readiness_status"] = (
        "exact_R9_v1_flat_filename_mapping_verified; provider readiness recheck required"
    )
    input_binding["r9_manifest"] = {
        "path": str(R9_MANIFEST),
        "sha256": sha256_file(R9_MANIFEST),
        "dataset": TEACHER_DATASET,
        "version": 1,
    }
    for item in input_binding["external_mounts"]:
        if item["dataset"] == TEACHER_DATASET:
            item["filename"] = teacher_filename_by_hash[item["sha256"]]
    teacher_rows = [
        item
        for item in input_binding["external_mounts"]
        if item["dataset"] == TEACHER_DATASET
    ]
    if len(teacher_rows) != 6 or {item["filename"] for item in teacher_rows} != set(
        teacher_filename_by_hash.values()
    ):
        raise ValueError(
            "R2 teacher mount list is not the exact six-file R9 ancestry set"
        )
    write_json(PACKET / "INPUT_BINDING_R2.json", input_binding)

    profile_jobs: list[dict[str, Any]] = []
    job_paths: list[Path] = []
    for hidden, layers in ((256, 2), (512, 1)):
        r1_path = (
            R1 / "jobs" / f"student_pair_t4_profile_f0s3_h{hidden}_l{layers}_r1.json"
        )
        spec = read_json(r1_path)
        spec["name"] = f"student_pair_t4_profile_f0s3_h{hidden}_l{layers}_r2"
        spec["argv"][spec["argv"].index("--name") + 1] = (
            f"student_pair_profile_f0s3_h{hidden}_l{layers}_t4_r2"
        )
        path = PACKET / "jobs" / f"{spec['name']}.json"
        write_json(path, spec)
        job_paths.append(path)
        profile_jobs.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "name": spec["name"],
                "trainer_run_name": spec["argv"][spec["argv"].index("--name") + 1],
                "shape": [hidden, layers],
                "profile_batches": 5,
                "optimizer_attempts_expected": 50,
                "timeout_seconds": 600,
            }
        )

    r1_source = read_json(R1 / "SOURCE_BINDING_R1.json")
    source_binding = dict(r1_source)
    source_binding.update(
        {
            "schema": "wnn.student-shape-pair-source-binding.r2",
            "input_binding_path": str(
                (PACKET / "INPUT_BINDING_R2.json").relative_to(ROOT)
            ),
            "input_binding_sha256": sha256_file(PACKET / "INPUT_BINDING_R2.json"),
            "spec_generator_path": str(
                (PACKET / "prepare_mountfix_r2.py").relative_to(ROOT)
            ),
            "spec_generator_sha256": sha256_file(Path(__file__).resolve()),
            "profile_jobs": profile_jobs,
            "collection_run_name_mapping": [
                {"job_name": row["name"], "trainer_run_name": row["trainer_run_name"]}
                for row in profile_jobs
            ],
            "r9_mount_correction": {
                "manifest_path": str(R9_MANIFEST),
                "manifest_sha256": sha256_file(R9_MANIFEST),
                "manifest_schema": r9_manifest["schema"],
                "dataset": TEACHER_DATASET,
                "version": 1,
                "corrected_mount_count": 6,
                "sha256_values_unchanged": True,
                "readiness_status": "live READY/version must be rechecked before provider mutation",
                "mounts": [
                    {
                        "logical_path": item["logical_path"],
                        "filename": item["filename"],
                        "sha256": item["sha256"],
                        "size": item["size"],
                    }
                    for item in teacher_rows
                ],
            },
        }
    )

    external_args: list[str] = []
    asset_datasets: set[str] = set()
    for item in input_binding["external_mounts"]:
        if item["dataset"] == RAW_DATASET:
            target = item["filename"]
        else:
            target = f"{item['dataset']}:{item['filename']}"
            asset_datasets.add(item["dataset"])
        external_args.extend(["--external", f"{item['logical_path']}={target}"])

    command = [
        "/usr/bin/python3",
        "scripts/kaggle_bundle.py",
        "--root",
        str(ROOT),
    ]
    for path in job_paths:
        command.extend(["--spec", str(path)])
    command.extend(
        [
            "--destination",
            str(UPLOAD),
            "--dataset",
            SOURCE_DATASET,
            *external_args,
        ]
    )
    for dataset in sorted(asset_datasets):
        command.extend(["--asset-dataset", dataset])
    source_binding["offline_bundle_command"] = command
    write_json(PACKET / "SOURCE_BINDING_R2.json", source_binding)

    patch_plan = read_json(R1 / "KERNEL_DEADLINE_PATCH_PLAN_R1.json")
    patch_plan["schema"] = "wnn.student-shape-pair-kernel-deadline-patch.r2"
    write_json(PACKET / "KERNEL_DEADLINE_PATCH_PLAN_R2.json", patch_plan)
    print(
        json.dumps(
            {
                "status": "prepared_local_mountfix_packet_no_provider_action",
                "r9_manifest_sha256": sha256_file(R9_MANIFEST),
                "r9_dataset": TEACHER_DATASET,
                "teacher_mounts": [
                    {
                        "logical_path": item["logical_path"],
                        "filename": item["filename"],
                        "sha256": item["sha256"],
                        "size": item["size"],
                    }
                    for item in teacher_rows
                ],
                "input_binding_sha256": sha256_file(PACKET / "INPUT_BINDING_R2.json"),
                "job_hashes": [row["sha256"] for row in profile_jobs],
                "source_binding_sha256": sha256_file(PACKET / "SOURCE_BINDING_R2.json"),
                "offline_bundle_command": command,
                "provider_action_performed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
