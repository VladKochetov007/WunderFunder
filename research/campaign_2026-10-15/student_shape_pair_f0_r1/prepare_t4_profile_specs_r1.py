from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r1"
CONTRACT_PATH = PACKET / "CPU_STUDENT_SHAPE_PAIR_F0_SEED3_B16_C2000_FP16_3E_R1_CONTRACT.json"
BASE_MANIFEST_PATH = MAIN / (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/h382_actual_startup_fixture_r5_sealed_r3/root/"
    "h382_fold0_asset_manifest.json"
)
CACHE_MANIFEST_PATH = PACKET / "private_readback/assets-manifest.json"
NORM_STAGE = ROOT / "private_assets/upload_wnn-f0-h256l2-cp5-norm-20261015-r1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def descriptor(path: str, sha256: str, size: int) -> dict[str, Any]:
    return {"path": path, "sha256": sha256, "size": size}


def base_file(manifest: dict[str, Any], logical_path: str) -> dict[str, Any]:
    for item in manifest["files"]:
        if item["path"] == logical_path:
            return item
    raise ValueError(f"Missing exact source file from base asset manifest: {logical_path}")


def asset_file(manifest: dict[str, Any], filename: str) -> dict[str, Any]:
    for item in manifest["files"]:
        if item["filename"] == filename:
            return item
    raise ValueError(f"Missing exact file from flat asset manifest: {filename}")


def main() -> None:
    contract = read_json(CONTRACT_PATH)
    base = read_json(BASE_MANIFEST_PATH)
    cache = read_json(CACHE_MANIFEST_PATH)
    norm = read_json(NORM_STAGE / "assets-manifest.json")
    order_receipt_path = PACKET / "B16_ORDER_DERIVATION_R1.json"
    old_order_receipt = read_json(order_receipt_path)
    if base.get("dataset") != "vladkochetov07/wnn-c20261004-fold0-assets":
        raise ValueError("Unexpected F0 base asset dataset identity")
    if contract["cache"]["metadata_sha256"] != "1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8":
        raise ValueError("Unexpected F0 T3 cache metadata identity")
    if norm.get("dataset") != "vladkochetov07/wnn-f0-h256l2-cp5-norm-20261015-r1":
        raise ValueError("Unexpected normalization asset identity")

    old_order = old_order_receipt["sample_order"]
    new_order = contract["sample_order"]
    common_order_fields = sorted(set(old_order) & set(new_order))
    order_field_mismatches = {
        field: {"old": old_order[field], "contract": new_order[field]}
        for field in common_order_fields
        if old_order[field] != new_order[field]
    }
    first_batch_matches = (
        old_order_receipt["first_batch"]["positions"]
        == new_order["first_batch_positions"]
        and old_order_receipt["first_batch"]["members"]
        == new_order["first_batch_members"]
    )
    order_reconciliation = {
        "schema": "wnn.student-shape-pair-order-reconciliation.r1",
        "status": "stale_derivation_preserved_contract_is_runtime_authority",
        "provider_action_performed": False,
        "historical_receipt": str(order_receipt_path.relative_to(ROOT)),
        "historical_receipt_sha256": sha256_file(order_receipt_path),
        "historical_recipe_sha256": old_order_receipt["recipe_sha256"],
        "current_contract_sha256": sha256_file(CONTRACT_PATH),
        "old_sampler_source_sha256": old_order_receipt["sampler_source_sha256"],
        "contract_sampler_source_sha256": contract["source_sha256"]["src/wnn/training_sampling.py"],
        "common_sample_order_field_mismatches": order_field_mismatches,
        "first_batch_positions_and_source_qualified_members_match": first_batch_matches,
        "fresh_order_derivation_repeated": False,
        "runtime_gate": "both profile CLIs must pass strict EpochOrderSampler validation against actual F0 source-qualified membership before updates",
    }
    (PACKET / "ORDER_SOURCE_RECONCILIATION_R1.json").write_text(
        json.dumps(order_reconciliation, indent=2, sort_keys=True) + "\n"
    )

    external: list[dict[str, Any]] = []

    def add_raw(path: str, filename: str, size: int, digest: str) -> None:
        external.append(
            {
                **descriptor(path, digest, size),
                "dataset": "vladkochetov07/wunder-funder",
                "filename": filename,
                "version_expected": 1,
            }
        )

    def add_base(path: str) -> None:
        item = base_file(base, path)
        external.append(
            {
                **descriptor(path, item["sha256"], item["size"]),
                "dataset": base["dataset"],
                "filename": item["filename"],
                "version_expected": base["version"],
            }
        )

    def add_flat_asset(path: str, filename: str, manifest: dict[str, Any]) -> None:
        item = asset_file(manifest, filename)
        external.append(
            {
                **descriptor(path, item["sha256"], item["size"]),
                "dataset": "vladkochetov07/wnn-clean-t3-f0-treatment-assets-20261012-r22"
                if manifest is cache
                else norm["dataset"],
                "filename": filename,
                "version_expected": 1,
            }
        )

    add_raw(
        "datasets/train.parquet",
        "train.parquet",
        29_240_219_011,
        "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
    )
    add_raw(
        "datasets/valid.parquet",
        "valid.parquet",
        5_615_509_128,
        "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
    )
    for source in (
        "cache/chain_folds.npz",
        "cache/chains.npz",
        "cache/propcache_c20260929_prop_f0_train.json",
        "cache/propcache_c20260929_prop_f0_train.npy",
        "cache/propcache_c20260929_prop_f0_valid.json",
        "cache/propcache_c20260929_prop_f0_valid.npy",
        "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "research/campaign_2026-10-04/portable_inputs.json",
        "artifacts/runs/c20260929_prop_f0/model.pt",
    ):
        add_base(source)

    teacher_asset_dataset = "vladkochetov07/wnn-clean-t3-f0-h384-cp2-20261012-r9"
    teacher_checkpoint_files = {
        "clean_t3_f0_muonall_h384_e3_s3_r5": (
            "201221c6b4140dda70b078db2a9d7a7246ec40b813f054570d195cecefc61582",
            "740b0d96ee311f215153db62272262e0153ce11f8f91560b03efae8e739f9ed0",
        ),
        "clean_t3_f0_muonall_h384_e3_s4_r5": (
            "714ac9b04cd1be0f63fb5f9f60cce2059e5e9284848470cae9d7198794767df7",
            "f738ef25f1b7d4244d6c900ebe4fba861efc24ddd106ff425fc60d7e3caccf11",
        ),
        "clean_t3_f0_muonall_h384_e3_s5_r5": (
            "763fac7f0aa24507167425e9905f1514a0a28ace48036b1c1c5819c7eb02c8d1",
            "44728c0c60981166f63038f2a6696b6714ce57c52e039e304883ffe7449def18",
        ),
    }
    for teacher in contract["cache"]["teachers"]:
        run = teacher["run"]
        checkpoint_hash, provenance_hash = teacher_checkpoint_files[run]
        if (
            teacher["checkpoint_sha256"] != checkpoint_hash
            or teacher["provenance_sha256"] != provenance_hash
        ):
            raise ValueError(f"T3 teacher ancestry pins differ for {run}")
        add_teacher = (
            (teacher["checkpoint"], checkpoint_hash, 8_692_775),
            (teacher["provenance"], provenance_hash, 159_434),
        )
        for logical_path, digest, size in add_teacher:
            external.append(
                {
                    **descriptor(logical_path, digest, size),
                    "dataset": teacher_asset_dataset,
                    "filename": f"{digest}.wnn",
                    "version_expected": 1,
                }
            )

    cache_root = "cache/clean_t3_f0_h390_t3r5_a9_r2"
    for path, filename in (
        (f"{cache_root}/metadata.json", "pair_r2_cache_metadata.json"),
        (f"{cache_root}/main_homogeneous_train.npy", "pair_r2_main_homogeneous_train.npy"),
        (f"{cache_root}/main_homogeneous_valid.npy", "pair_r2_main_homogeneous_fit_valid.npy"),
    ):
        add_flat_asset(path, filename, cache)

    norm_root = "artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1"
    for path, filename in (
        (f"{norm_root}/model_ep5.pt", "e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda.wnn"),
        (f"{norm_root}/provenance.json", "bdf643043884b1344564b636a2972ad52b8202ee30cf22598b1afd90405dd961.wnn"),
    ):
        add_flat_asset(path, filename, norm)

    provenance_source = MAIN / "artifacts/runs/c20260929_prop_f0/provenance.json"
    provenance_target = ROOT / "artifacts/runs/c20260929_prop_f0/provenance.json"
    if not provenance_source.is_file():
        raise ValueError("Pinned propensity provenance source is unavailable")
    provenance_target.parent.mkdir(parents=True, exist_ok=True)
    if not provenance_target.exists():
        provenance_target.hardlink_to(provenance_source)
    if sha256_file(provenance_target) != contract["propensity"]["provenance_sha256"]:
        raise ValueError("Propensity provenance hardlink differs from the contract")

    code_paths = {
        "scripts/train_rnn.py",
        "scripts/kaggle_common.py",
        "scripts/kaggle_gpu_runner.py",
        "scripts/kaggle_kernel.py",
        "src/sitecustomize.py",
        "research/campaign_2026-10-15/student_shape_pair_f0_r1/"
        "CPU_STUDENT_SHAPE_PAIR_F0_SEED3_B16_C2000_FP16_3E_R1_CONTRACT.json",
        "artifacts/runs/c20260929_prop_f0/provenance.json",
    }
    code_paths.update(str(path.relative_to(ROOT)) for path in (ROOT / "src/wnn").rglob("*.py"))
    code: list[dict[str, Any]] = []
    for relative in sorted(code_paths):
        path = ROOT / relative
        if not path.is_file():
            if relative == "src/sitecustomize.py":
                continue
            raise ValueError(f"Missing source dependency: {relative}")
        item = descriptor(relative, sha256_file(path), path.stat().st_size)
        expected = contract.get("source_sha256", {}).get(relative)
        if expected is not None and item["sha256"] != expected:
            raise ValueError(f"Contract source hash differs: {relative}")
        code.append(item)

    unique_paths = [item["path"] for item in code + external]
    if len(set(unique_paths)) != len(unique_paths):
        raise ValueError("Duplicate code/input path")

    external_map = [
        {
            "logical_path": item["path"],
            "dataset": item["dataset"],
            "filename": item["filename"],
            "sha256": item["sha256"],
            "size": item["size"],
            "version_expected": item["version_expected"],
        }
        for item in external
    ]
    binding = {
        "schema": "wnn.student-shape-pair-input-binding.r1",
        "provider_action_performed": False,
        "readiness_status": "version_and_privacy_recheck_required_before_create_or_push",
        "recipe_contract_sha256": sha256_file(CONTRACT_PATH),
        "sources": {
            "raw_dataset": {
                "dataset": "vladkochetov07/wunder-funder",
                "version_expected": 1,
                "train_sha256": external[0]["sha256"],
                "valid_sha256": external[1]["sha256"],
            },
            "fold0_base_assets": {
                "dataset": base["dataset"],
                "version_expected": base["version"],
                "manifest_path": str(BASE_MANIFEST_PATH),
                "manifest_sha256": sha256_file(BASE_MANIFEST_PATH),
            },
            "clean_t3_cache": {
                "dataset": "vladkochetov07/wnn-clean-t3-f0-treatment-assets-20261012-r22",
                "version_expected": 1,
                "manifest_sha256": sha256_file(CACHE_MANIFEST_PATH),
                "historical_cache_profile": contract["cache"]["profile"],
                "metadata_sha256": contract["cache"]["metadata_sha256"],
            },
            "teacher_cp2_ancestry": {
                "dataset": teacher_asset_dataset,
                "version_expected": 1,
                "teacher_count": len(contract["cache"]["teachers"]),
                "purpose": "strict source hash validation only; the profile does not load teacher weights",
                "checkpoint_and_provenance_hashes": [
                    {
                        "run": teacher["run"],
                        "checkpoint_sha256": teacher["checkpoint_sha256"],
                        "provenance_sha256": teacher["provenance_sha256"],
                    }
                    for teacher in contract["cache"]["teachers"]
                ],
            },
            "normalization_only_asset": {
                "dataset": norm["dataset"],
                "version_expected": 1,
                "manifest_sha256": sha256_file(NORM_STAGE / "assets-manifest.json"),
                "buffers_only": True,
                "checkpoint_sha256": contract["normalization_source"]["checkpoint_sha256"],
                "provenance_sha256": contract["normalization_source"]["provenance_sha256"],
            },
        },
        "external_mounts": external_map,
        "excluded_inputs": [
            "train_targets.npz",
            "valid_targets.npz",
            "teacher checkpoint weights are mounted only for ancestry hashing and are not loaded into student models",
            "all student checkpoints (profile stage writes none)",
        ],
    }
    (PACKET / "INPUT_BINDING_R1.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n"
    )

    code_for_hash = sorted(code, key=lambda item: item["path"])
    code_hash = hashlib.sha256(
        json.dumps(code_for_hash, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_specs = []
    shapes = ((256, 2), (512, 1))
    for hidden, layers in shapes:
        shape_name = f"h{hidden}_l{layers}"
        run_name = f"student_pair_profile_f0s3_{shape_name}_t4_r1"
        argv = [
            "{python}",
            "scripts/train_rnn.py",
            "--name", run_name,
            "--device", "cuda",
            "--arch", "gru",
            "--cell", "gru",
            "--enc", "linear",
            "--hidden", str(hidden),
            "--layers", str(layers),
            "--dropout", "0",
            "--train-seqs", "10607",
            "--train-source", "train+valid",
            "--valid-train-fraction", "1",
            "--valid-repeat", "1",
            "--valid-fold", "0",
            "--n-valid-folds", "4",
            "--full-valid",
            "--batch", "16",
            "--chunk", "2000",
            "--epochs", "3",
            "--checkpoint-policy", "fixed",
            "--checkpoint-epoch", "2",
            "--no-eval-during-fit",
            "--split-manifest", "research/campaign_2026-09-29/artifacts/splits/fold0.json",
            "--student-frontier-contract", str(CONTRACT_PATH.relative_to(ROOT)),
            "--student-frontier-stage", "f0_seed3_t4_profile",
            "--seed", "3",
            "--precision", "fp16",
            "--amp-init-scale", "65536",
            "--distill-cache-root", "cache/clean_t3_f0_h390_t3r5_a9_r2",
            "--distill-cache-variant", "homogeneous",
            "--distill-cache-scale", "1",
            "--distill-alpha", "0.9",
            "--loss", "wmse",
            "--loss-norm", "chunk",
            "--propensity", "c20260929_prop_f0",
            "--prop-power", "2",
            "--prop-eps", "0.03",
            "--valid-true-mask", "0.03",
            "--batch-order", "seeded",
            "--deterministic",
            "--fast",
            "--grad-stats",
            "--optim", "adamw",
            "--lr", "0.0001",
            "--beta2", "0.999",
            "--wd", "0.01",
            "--sched", "onecycle",
            "--warmup-frac", "0.05",
            "--head-lr-mult", "1",
            "--clip", "1",
            "--max-batches", "5",
            "--profile-steps", "5",
            "--profile-warmup", "0",
        ]
        spec = {
            "version": 1,
            "name": f"student_pair_t4_profile_f0s3_{shape_name}_r1",
            "kind": "gpu",
            "argv": argv,
            "code": code,
            "code_sha256": code_hash,
            "inputs": [descriptor(item["path"], item["sha256"], item["size"]) for item in external],
            "artifacts": [],
            "limits": {
                "cpus": 2,
                "ram_bytes": 8_000_000_000,
                "timeout_seconds": 600,
                "gpu_memory_bytes": 3_500_000_000,
                "gpu_stop_bytes": 5_000_000_000,
            },
            "environment": {
                "PYTHONHASHSEED": "0",
                "WNN_DEFERRED_IDENTITY_PATHS": "datasets/train.parquet,datasets/valid.parquet",
                "WNN_LOADER_RAM_GB": "4",
                "WNN_LOADER_WORKERS": "0",
                "WNN_PORTABLE_DATA_MANIFEST": "/kaggle/working/experiment/research/campaign_2026-10-04/portable_inputs.json",
                "WNN_PROJECT_ROOT": "/kaggle/working/experiment",
                "WNN_STARTUP_DIAGNOSTICS": "1",
            },
        }
        job_path = PACKET / "jobs" / f"{spec['name']}.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        output_specs.append(
            {
                "path": str(job_path.relative_to(ROOT)),
                "sha256": sha256_file(job_path),
                "name": spec["name"],
                "shape": [hidden, layers],
                "profile_batches": 5,
                "optimizer_attempts_expected": 50,
                "timeout_seconds": 600,
            }
        )

    bundle_slug = "wnn-f0-student-shape-t4-profile-20261015-r1"
    bundle_destination = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r1"
    asset_datasets = sorted(
        {
            item["dataset"]
            for item in external
            if item["dataset"] != "vladkochetov07/wunder-funder"
        }
    )
    bundle_command = [
        "/usr/bin/python3",
        "scripts/kaggle_bundle.py",
        "--root",
        str(ROOT),
    ]
    for job in output_specs:
        bundle_command.extend(["--spec", str(ROOT / job["path"])])
    bundle_command.extend(["--destination", str(bundle_destination), "--dataset", f"vladkochetov07/{bundle_slug}"])
    for item in external:
        target = (
            item["filename"]
            if item["dataset"] == "vladkochetov07/wunder-funder"
            else f"{item['dataset']}:{item['filename']}"
        )
        bundle_command.extend(["--external", f"{item['path']}={target}"])
    for dataset in asset_datasets:
        bundle_command.extend(["--asset-dataset", dataset])

    kernel_patch = {
        "schema": "wnn.student-shape-pair-kernel-deadline-patch.r1",
        "provider_action_performed": False,
        "generated_source": "scripts/kaggle_kernel.py",
        "source_default_seconds": 43200,
        "required_kernel_process_wall_seconds_max": 1800,
        "search_literal": "KERNEL_PROCESS_WALL_SECONDS_MAX = 43_200",
        "replacement_literal": "KERNEL_PROCESS_WALL_SECONDS_MAX = 1800",
        "required_post_build_assertions": [
            "generated kernel id equals vladkochetov07/wnn-f0-student-shape-t4-profile-20261015-r1-runner",
            "replace exactly one generated KERNEL_PROCESS_WALL_SECONDS_MAX = 43200 with 1800",
            "kernel bootstrap and metadata remain private and GPU-enabled",
            "bundle archive SHA-256 stays equal to bundle-manifest.json archive_sha256",
            "record generated kernel.py and kernel-metadata.json SHA-256 before any push",
        ],
        "reason": "the generic 43,200-second default exceeds this exact profile's 1,800-second outer budget",
    }
    (PACKET / "KERNEL_DEADLINE_PATCH_PLAN_R1.json").write_text(
        json.dumps(kernel_patch, indent=2, sort_keys=True) + "\n"
    )

    source_binding = {
        "schema": "wnn.student-shape-pair-source-binding.r1",
        "provider_action_performed": False,
        "base_commit": "afb2e89e18d04c45853097becdb7264d3dfc386a",
        "spec_generator_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "spec_generator_sha256": sha256_file(Path(__file__).resolve()),
        "recipe_contract_path": str(CONTRACT_PATH.relative_to(ROOT)),
        "recipe_contract_sha256": sha256_file(CONTRACT_PATH),
        "consumer_source_path": "src/wnn/student_frontier.py",
        "consumer_source_sha256": sha256_file(ROOT / "src/wnn/student_frontier.py"),
        "consumer_source_change": [
            "precision validation now uses the explicit contract precision",
            "AMP initial scale is checked against the contract",
            "profile max_batches/profile_steps/profile_warmup are read from the declared stage",
            "relative source_root resolves relative to the runtime root",
        ],
        "code_file_count": len(code),
        "code_aggregate_sha256": code_hash,
        "profile_jobs": output_specs,
        "kernel_dataset": "vladkochetov07/wnn-f0-student-shape-t4-profile-20261015-r1",
        "kernel_id": "vladkochetov07/wnn-f0-student-shape-t4-profile-20261015-r1-runner",
        "bootstrap_outer_deadline_seconds": 1800,
        "asset_datasets": asset_datasets,
        "teacher_cp2_asset_dataset": teacher_asset_dataset,
        "offline_bundle_command": bundle_command,
        "kernel_deadline_patch_plan_path": str(
            (PACKET / "KERNEL_DEADLINE_PATCH_PLAN_R1.json").relative_to(ROOT)
        ),
    }
    (PACKET / "SOURCE_BINDING_R1.json").write_text(
        json.dumps(source_binding, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "generated_offline_only",
                "provider_action_performed": False,
                "contract_sha256": source_binding["recipe_contract_sha256"],
                "consumer_sha256": source_binding["consumer_source_sha256"],
                "code_file_count": len(code),
                "profile_jobs": output_specs,
                "input_count": len(external),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
