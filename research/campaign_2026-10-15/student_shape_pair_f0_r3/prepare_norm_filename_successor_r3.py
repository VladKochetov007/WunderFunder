from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PACKET_R2 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r2"
PACKET_R3 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r3"
READBACK = PACKET_R2 / "NORM_ASSET_REMOTE_READBACK_R1.json"
UPLOAD = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r3"
OWNER = "vladkochetov07"
RAW_DATASET = f"{OWNER}/wunder-funder"
NORM_DATASET = f"{OWNER}/wnn-f0-h256l2-cp5-norm-20261015-r1"


def sha256_file(path: Path) -> str:
    """Hash a small packet or staged file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one small frozen packet object."""
    if path.stat().st_size > 4_000_000:
        raise ValueError(f"refusing oversized packet JSON: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def write_json(path: Path, value: object) -> None:
    """Write a deterministic successor packet file without replacing data."""
    if path.exists():
        raise FileExistsError(f"refusing to replace existing packet file: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Bind the two verified remote filenames without changing the recipe."""
    allowed = {"__pycache__", Path(__file__).name}
    if PACKET_R3.exists() and any(
        path.name not in allowed for path in PACKET_R3.iterdir()
    ):
        raise FileExistsError(f"R3 packet is not empty: {PACKET_R3}")
    if UPLOAD.exists():
        raise FileExistsError(f"refusing to replace source upload stage: {UPLOAD}")

    readback = read_json(READBACK)
    if readback["dataset"] != {
        "ref": NORM_DATASET,
        "version": 1,
        "status": "ready",
        "private": True,
        "owner": OWNER,
    }:
        raise ValueError(
            "normalizer remote readback is not the reviewed private READY v1"
        )
    if [row["remote_sha256_verified"] for row in readback["approved_payloads"]] != [
        True,
        True,
    ]:
        raise ValueError("normalizer remote payload hashes were not both verified")

    original_input_binding = read_json(PACKET_R2 / "INPUT_BINDING_R2.json")
    input_binding = copy.deepcopy(original_input_binding)
    changed: list[dict[str, str]] = []
    rows_by_hash = {row["sha256"]: row for row in readback["approved_payloads"]}
    for row in input_binding["external_mounts"]:
        if row["dataset"] != NORM_DATASET:
            continue
        verified = rows_by_hash.get(row["sha256"])
        if verified is None or row["size"] != verified["size"]:
            raise ValueError("normalizer input row differs from remote readback hashes")
        before = row["filename"]
        after = verified["remote_filename"]
        if before != verified["expected_filename"]:
            raise ValueError(
                "R2 normalizer filename does not match its hash-addressed asset"
            )
        row["filename"] = after
        changed.append(
            {
                "logical_path": row["logical_path"],
                "sha256": row["sha256"],
                "size": row["size"],
                "before": before,
                "after": after,
            }
        )
    if len(changed) != 2:
        raise ValueError(
            f"expected exactly two normalizer filename changes, got {len(changed)}"
        )
    comparison = copy.deepcopy(input_binding)
    for row in comparison["external_mounts"]:
        for mount in changed:
            if row["logical_path"] == mount["logical_path"]:
                row["filename"] = mount["before"]
    if comparison != original_input_binding:
        raise ValueError("R3 changed an input-binding field beyond two mount filenames")

    source_binding = copy.deepcopy(read_json(PACKET_R2 / "SOURCE_BINDING_R2.json"))
    source_binding["schema"] = "wnn.student-shape-pair-source-binding.r3"
    (PACKET_R3 / "jobs").mkdir(exist_ok=True)
    for job in source_binding["profile_jobs"]:
        original_path = ROOT / job["path"]
        successor_path = PACKET_R3 / "jobs" / original_path.name
        shutil.copyfile(original_path, successor_path)
        if sha256_file(successor_path) != job["sha256"]:
            raise ValueError(f"copied profile job changed: {original_path.name}")
        job["path"] = str(successor_path.relative_to(ROOT))
    source_binding["input_binding_path"] = str(
        (PACKET_R3 / "INPUT_BINDING_R3.json").relative_to(ROOT)
    )
    source_binding["input_binding_sha256"] = hashlib.sha256(
        json.dumps(input_binding, indent=2, sort_keys=True).encode() + b"\n"
    ).hexdigest()
    source_binding["spec_generator_path"] = str(
        (PACKET_R3 / "prepare_norm_filename_successor_r3.py").relative_to(ROOT)
    )
    source_binding["spec_generator_sha256"] = sha256_file(Path(__file__).resolve())
    source_binding["normalization_mount_correction"] = {
        "dataset": NORM_DATASET,
        "version": 1,
        "remote_readback_path": str(READBACK.relative_to(ROOT)),
        "remote_readback_sha256": sha256_file(READBACK),
        "exact_payload_sha256_and_size_verified": True,
        "only_filename_fields_changed": True,
        "mounts": changed,
    }

    external_args: list[str] = []
    asset_datasets: set[str] = set()
    for row in input_binding["external_mounts"]:
        target = row["filename"]
        if row["dataset"] != RAW_DATASET:
            target = f"{row['dataset']}:{row['filename']}"
            asset_datasets.add(row["dataset"])
        external_args.extend(["--external", f"{row['logical_path']}={target}"])

    command = ["/usr/bin/python3", "scripts/kaggle_bundle.py", "--root", str(ROOT)]
    for job in source_binding["profile_jobs"]:
        command.extend(["--spec", str(ROOT / job["path"])])
    command.extend(
        [
            "--destination",
            str(UPLOAD),
            "--dataset",
            source_binding["kernel_dataset"],
            *external_args,
        ]
    )
    for dataset in sorted(asset_datasets):
        command.extend(["--asset-dataset", dataset])
    source_binding["offline_bundle_command"] = command

    write_json(PACKET_R3 / "INPUT_BINDING_R3.json", input_binding)
    write_json(PACKET_R3 / "SOURCE_BINDING_R3.json", source_binding)
    delta = {
        "schema": "wnn.student-shape-pair-norm-mount-delta.r1",
        "status": "two_filename_fields_corrected_after_remote_readback",
        "provider_mutation_performed_by_this_generator": False,
        "r2_readback_path": str(READBACK.relative_to(ROOT)),
        "r2_readback_sha256": sha256_file(READBACK),
        "input_binding_r2_sha256": sha256_file(PACKET_R2 / "INPUT_BINDING_R2.json"),
        "input_binding_r3_sha256": sha256_file(PACKET_R3 / "INPUT_BINDING_R3.json"),
        "source_binding_r3_sha256": sha256_file(PACKET_R3 / "SOURCE_BINDING_R3.json"),
        "changed_mounts": changed,
        "only_two_mount_filename_fields_changed": True,
        "all_other_mounts_unchanged": True,
        "recipe_contract_sha256": source_binding["recipe_contract_sha256"],
        "profile_job_hashes_unchanged": [
            row["sha256"] for row in source_binding["profile_jobs"]
        ],
        "source_code_commit_unchanged": source_binding["source_tree_commit"],
    }
    write_json(PACKET_R3 / "NORM_MOUNT_DELTA_R3.json", delta)
    print(
        json.dumps(
            {
                "status": delta["status"],
                "changed_mounts": changed,
                "input_binding_sha256": delta["input_binding_r3_sha256"],
                "source_binding_sha256": delta["source_binding_r3_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
