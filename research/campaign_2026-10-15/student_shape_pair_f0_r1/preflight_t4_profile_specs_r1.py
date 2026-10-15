from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PACKET = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r1"
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_common import read_spec, verify_files  # noqa: E402


CONTRACT = PACKET / "CPU_STUDENT_SHAPE_PAIR_F0_SEED3_B16_C2000_FP16_3E_R1_CONTRACT.json"
SOURCE_BINDING = PACKET / "SOURCE_BINDING_R1.json"
INPUT_BINDING = PACKET / "INPUT_BINDING_R1.json"
OUTPUT = PACKET / "OFFLINE_SPEC_PREFLIGHT_R1.json"
EXPECTED_SHAPES = ((256, 2), (512, 1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def argument(argv: list[str], name: str) -> str:
    try:
        index = argv.index(name)
    except ValueError as error:
        raise ValueError(f"Missing required trainer option {name}") from error
    if index + 1 >= len(argv):
        raise ValueError(f"Trainer option {name} has no value")
    return argv[index + 1]


def main() -> None:
    contract = read_json(CONTRACT)
    source_binding = read_json(SOURCE_BINDING)
    input_binding = read_json(INPUT_BINDING)
    parsed_specs: list[dict[str, Any]] = []
    spec_receipts: list[dict[str, Any]] = []
    input_identities: list[set[tuple[str, int, str]]] = []
    code_paths: list[set[str]] = []
    stage = contract["stages"]["f0_seed3_t4_profile"]
    recipe = contract["recipe"]
    for hidden, layers in EXPECTED_SHAPES:
        path = PACKET / "jobs" / f"student_pair_t4_profile_f0s3_h{hidden}_l{layers}_r1.json"
        spec = read_spec(path)
        verify_files(ROOT, spec["code"])
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
                raise ValueError(f"{path.name}: {option} differs from the frozen profile recipe")
        if spec["kind"] != "gpu" or spec["artifacts"]:
            raise ValueError(f"{path.name}: expected GPU profile with no saved artifacts")
        if spec["limits"] != {
            "cpus": 2,
            "ram_bytes": 8_000_000_000,
            "timeout_seconds": 600,
            "gpu_memory_bytes": 3_500_000_000,
            "gpu_stop_bytes": 5_000_000_000,
        }:
            raise ValueError(f"{path.name}: resource caps differ from the reviewed profile plan")
        if argument(argv, "--student-frontier-contract") != str(CONTRACT.relative_to(ROOT)):
            raise ValueError(f"{path.name}: contract path differs")
        if argument(argv, "--distill-cache-root") != contract["cache"]["root"]:
            raise ValueError(f"{path.name}: distillation cache root differs")
        parsed_specs.append(spec)
        input_identities.append(
            {(item["path"], item["size"], item["sha256"]) for item in spec["inputs"]}
        )
        code_paths.append({item["path"] for item in spec["code"]})
        spec_receipts.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "shape": [hidden, layers],
                "trainer_run_name": argument(argv, "--name"),
                "inputs": len(spec["inputs"]),
                "code": len(spec["code"]),
                "expected_attempts": int(argument(argv, "--max-batches")) * 10,
            }
        )
    hashes = {item["sha256"] for item in spec_receipts}
    if len(hashes) != len(spec_receipts):
        raise ValueError("Profile job specs must be distinct")
    if any(item["inputs"] != 22 or item["code"] != 68 for item in spec_receipts):
        raise ValueError("Profile jobs do not share the reviewed 22-input/68-source closure")
    expected_inputs = {
        (item["logical_path"], item["size"], item["sha256"])
        for item in input_binding["external_mounts"]
    }
    if any(identity != expected_inputs for identity in input_identities):
        raise ValueError("Profile job mounts differ from the pinned external input binding")
    if code_paths[0] != code_paths[1]:
        raise ValueError("Profile jobs do not share one exact source closure")
    if source_binding["recipe_contract_sha256"] != sha256_file(CONTRACT):
        raise ValueError("Source binding contract hash is stale")
    if source_binding["consumer_source_sha256"] != sha256_file(
        ROOT / source_binding["consumer_source_path"]
    ):
        raise ValueError("Source binding consumer hash is stale")
    if source_binding["spec_generator_sha256"] != sha256_file(ROOT / source_binding["spec_generator_path"]):
        raise ValueError("Source binding generator hash is stale")
    if [item["sha256"] for item in source_binding["profile_jobs"]] != [
        item["sha256"] for item in spec_receipts
    ]:
        raise ValueError("Source binding job hashes are stale")
    if [item["trainer_run_name"] for item in source_binding["profile_jobs"]] != [
        item["trainer_run_name"] for item in spec_receipts
    ]:
        raise ValueError("Source binding trainer run names do not match worker argv")
    if source_binding["code_aggregate_sha256"] != parsed_specs[0]["code_sha256"]:
        raise ValueError("Source binding code aggregate hash is stale")
    if not input_binding["provider_action_performed"] and not source_binding["provider_action_performed"]:
        status = "passed_offline_no_provider_action"
    else:
        raise ValueError("Offline packet must not record provider actions")

    receipt = {
        "schema": "wnn.student-shape-pair-offline-spec-preflight.r1",
        "status": status,
        "provider_action_performed": False,
        "command": (
            "WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=20% WNN_THREADS=1 "
            "WNN_MEM_MAX=300000000 WNN_RUNTIME_MAX=120s "
            "/home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited "
            "/usr/bin/python3 research/campaign_2026-10-15/student_shape_pair_f0_r1/"
            "preflight_t4_profile_specs_r1.py"
        ),
        "preflight_script_sha256": sha256_file(Path(__file__)),
        "checks": [
            "kaggle_common.read_spec schema, resource bounds, aggregate code hashes, and local source bytes",
            "exact H256x2/H512x1 B16/C2000/FP16 five-batch profile argv",
            "same 22 external input mounts and 68 code sources",
            "contract, consumer, generator, source-binding, and job hashes agree",
            "empty saved-artifact allowlist and no provider action",
        ],
        "contract_sha256": sha256_file(CONTRACT),
        "consumer_sha256": sha256_file(ROOT / source_binding["consumer_source_path"]),
        "generator_sha256": sha256_file(ROOT / source_binding["spec_generator_path"]),
        "code_aggregate_sha256": parsed_specs[0]["code_sha256"],
        "source_binding_sha256": sha256_file(SOURCE_BINDING),
        "input_binding_sha256": sha256_file(INPUT_BINDING),
        "specs": spec_receipts,
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "specs": spec_receipts}, sort_keys=True))


if __name__ == "__main__":
    main()
