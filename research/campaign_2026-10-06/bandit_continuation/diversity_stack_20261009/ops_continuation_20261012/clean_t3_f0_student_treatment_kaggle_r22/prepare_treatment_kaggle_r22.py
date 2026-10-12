from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET = Path(__file__).resolve().parent
R6_PAIR = MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r6_pair_20261012/PAIR_PREPARATION_R2.json"
R5_PACKET = MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r5_teacher_cohort/package_r9"
R5_BUNDLE = R5_PACKET / "bundle_teacher_cohort_r5"
CACHE = MAIN / "cache/clean_t3_f0_h390_t3r5_a9_r2"
SOURCE_ID = "vladkochetov07/wnn-clean-t3-f0-treatment-source-20261012-r22"
ASSET_ID = "vladkochetov07/wnn-clean-t3-f0-treatment-assets-20261012-r22"
KERNEL_ID = "vladkochetov07/wnn-clean-t3-f0-muonall-student-20261012-r22-runner"
SOURCE_TITLE = "wnn-clean-t3-f0-treatment-source-20261012-r22"
ASSET_TITLE = "wnn-clean-t3-f0-treatment-assets-20261012-r22"
KERNEL_TITLE = "wnn-clean-t3-f0-muonall-student-20261012-r22-runner"
SOURCE_STAGE = Path("/home/vlad/development/wunder/.wnn-clean-t3-f0-treatment-r22-source")
ASSET_STAGE = Path("/home/vlad/development/wunder/.wnn-clean-t3-f0-treatment-r22-assets")
KERNEL_STAGE = PACKET / "kernel"
SOURCE_SNAPSHOT = Path("research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    offset = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(stream.fileno(), offset, len(chunk), os.POSIX_FADV_DONTNEED)
            offset += len(chunk)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_paths() -> dict[str, Path]:
    return {
        "bundle": R5_BUNDLE / "bundle.wnn",
        "bundle_manifest": R5_BUNDLE / "bundle-manifest.json",
        "cache_metadata": CACHE / "metadata.json",
        "cache_train": CACHE / "main_homogeneous_train.npy",
        "cache_valid": CACHE / "main_homogeneous_valid.npy",
        "treatment_checkpoint": MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/kaggle_h382_delivery_20261011_r4/provider_output/out/runs/kg_h382r3_f0_muonall_prop0_e4_s3/model_ep3.pt",
        "treatment_provenance": MAIN / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/kaggle_h382_delivery_20261011_r4/provider_output/out/runs/kg_h382r3_f0_muonall_prop0_e4_s3/provenance.json",
    }


def copy_checked(source: Path, destination: Path, *, expected_size: int | None = None, expected_sha: str | None = None) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    size = source.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"source byte count differs: {source}")
    same_device = source.stat().st_dev == destination.parent.stat().st_dev
    if size >= 64 * 1024 * 1024 and same_device:
        os.link(source, destination)
        staged_sha = sha256_file(destination)
        if expected_sha is not None and staged_sha != expected_sha:
            raise ValueError(f"source SHA differs: {source}")
        return {"path": str(destination), "bytes": size, "sha256": staged_sha, "staging_mode": "hardlink_same_filesystem"}
    digest = hashlib.sha256()
    offset = 0
    chunk_size = 64 * 1024 * 1024
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        for chunk in iter(lambda: input_stream.read(chunk_size), b""):
            output_stream.write(chunk)
            digest.update(chunk)
            offset += len(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(input_stream.fileno(), offset - len(chunk), len(chunk), os.POSIX_FADV_DONTNEED)
                os.posix_fadvise(output_stream.fileno(), offset - len(chunk), len(chunk), os.POSIX_FADV_DONTNEED)
    source_sha = digest.hexdigest()
    if expected_sha is not None and source_sha != expected_sha:
        raise ValueError(f"source SHA differs: {source}")
    if destination.stat().st_size != size or sha256_file(destination) != source_sha:
        raise ValueError(f"staged copy differs: {destination}")
    return {"path": str(destination), "bytes": size, "sha256": source_sha, "staging_mode": "streamed_copy_verified"}


def staged_source_descriptors() -> dict[str, dict]:
    raw = json.loads(R6_PAIR.read_text())
    parents = {item["arm"]: item for item in raw["parents"]}
    return {
        "cache": {
            "metadata": {"source": CACHE / "metadata.json", "filename": "pair_r2_cache_metadata.json", "bytes": 323404, "sha256": "1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8"},
            "train": {"source": CACHE / "main_homogeneous_train.npy", "filename": "pair_r2_main_homogeneous_train.npy", "bytes": 1685600128, "sha256": "6a19f7ac45c6e8a02d3b89c0ac4986ffab4885ab7a4ebc5fb4599b3047f4e7b3"},
            "fit_valid": {"source": CACHE / "main_homogeneous_valid.npy", "filename": "pair_r2_main_homogeneous_fit_valid.npy", "bytes": 224800128, "sha256": "d1cd78537f3b6aaa830861ac771cf5c4621f1f2c1f19a73f6affb813c33ca3a2"},
        },
        "parents": {
            "treatment": {
                "checkpoint": {"source": source_paths()["treatment_checkpoint"], "filename": "treatment_parent_model_ep3.pt", "bytes": parents["treatment"]["checkpoint"]["bytes"], "sha256": parents["treatment"]["checkpoint"]["sha256"]},
                "provenance": {"source": source_paths()["treatment_provenance"], "filename": "treatment_parent_provenance.json", "bytes": parents["treatment"]["provenance"]["bytes"], "sha256": parents["treatment"]["provenance"]["sha256"]},
            },
        },
    }


def build_plan(asset_files: dict[str, dict]) -> dict:
    pair = json.loads(R6_PAIR.read_text())
    manifest = json.loads((R5_BUNDLE / "bundle-manifest.json").read_text())
    with tarfile.open(R5_BUNDLE / "bundle.wnn", "r:gz") as archive:
        base_fit = json.load(archive.extractfile("kaggle_specs/fit_clean_t3_f0_muonall_h384_e3_s3_r5.json"))
    base_profile = pair["profile"]["argv_template"]
    treatment_parent = next(item for item in pair["parents"] if item["arm"] == "treatment")
    treatment_stages = [item for item in pair["fit_stages"] if item["arm"] == "treatment"]
    pair_asset_map = {
        "cache/clean_t3_f0_h390_t3r5_a9_r2/metadata.json": {"dataset": ASSET_ID, "filename": "pair_r2_cache_metadata.json"},
        "cache/clean_t3_f0_h390_t3r5_a9_r2/main_homogeneous_train.npy": {"dataset": ASSET_ID, "filename": "pair_r2_main_homogeneous_train.npy"},
        "cache/clean_t3_f0_h390_t3r5_a9_r2/main_homogeneous_valid.npy": {"dataset": ASSET_ID, "filename": "pair_r2_main_homogeneous_fit_valid.npy"},
        "artifacts/runs/kg_h382r3_f0_muonall_prop0_e4_s3/model_ep3.pt": {"dataset": ASSET_ID, "filename": "treatment_parent_model_ep3.pt"},
        "artifacts/runs/kg_h382r3_f0_muonall_prop0_e4_s3/provenance.json": {"dataset": ASSET_ID, "filename": "treatment_parent_provenance.json"},
    }
    base_inputs = [dict(item) for item in base_fit["inputs"]]
    base_input_paths = {item["path"] for item in base_inputs}
    worker_inputs = base_inputs + [
        {"path": logical, "size": asset_files[mapping["filename"]]["size"], "sha256": asset_files[mapping["filename"]]["sha256"]}
        for logical, mapping in pair_asset_map.items()
    ]
    external_mounts = {path: value for path, value in manifest["external_mounts"].items() if path in base_input_paths}
    external_mounts.update(pair_asset_map)
    external_ids = {item["path"]: {"size": item["size"], "sha256": item["sha256"]} for item in base_inputs}
    external_ids.update({item["path"]: {"size": item["size"], "sha256": item["sha256"]} for item in worker_inputs if item["path"] not in external_ids})
    if set(external_mounts) != set(external_ids):
        raise ValueError("every worker input must have exactly one declared mounted source")
    source_manifest_sha = next(item["sha256"] for item in base_fit["code"] if item["path"].endswith("SOURCE_MANIFEST_H390.json"))
    code = base_fit["code"]
    if canonical_sha(sorted(code, key=lambda item: item["path"])) != base_fit.get("code_sha256"):
        raise ValueError("frozen R5 worker code closure does not match its recipe binding")
    source_root = str(SOURCE_SNAPSHOT)
    def normalize_argv(raw: list[str]) -> list[str]:
        values = list(raw)
        for i, value in enumerate(values):
            if value.endswith("/source_snapshot/scripts/cleanfold_exec.py"):
                values[i] = f"{source_root}/scripts/cleanfold_exec.py"
            values[i] = values[i].replace("cache/clean_t3_f0_h390_t3r5_a9_r1", "cache/clean_t3_f0_h390_t3r5_a9_r2")
            values[i] = values[i].replace("t3r5_pair_r6", "t3r5_treatment_r22")
        return values

    profile_argv = normalize_argv(base_profile)
    profile_name = "clean_t3_f0_muonall_treatment_profile_r22"
    profile_argv[profile_argv.index("--name") + 1] = profile_name
    profile_argv[profile_argv.index("--init") + 1] = f"{treatment_parent['run']}@ep{treatment_parent['epoch']}"
    profile = {
        "name": profile_name,
        "argv": profile_argv,
        "max_batches": 30,
        "warmup_complete_batches": 5,
        "measured_complete_batches": 25,
        "required_before_fits": True,
        "local_profile_reference": {"terminal_receipt_sha256": "8eedb01be05ecee2f0086ab4b01856d6a53cf6c290020ae18ec91de21440d339", "contract_sha256": "cae424ce880651db4b618a757ae9f2979fca40a45b28273e3d2b413b32ed2e76", "ram_peak_bytes": 4330741760, "vram_peak_bytes": 2178940928, "scope": "local T4 profile only; does not qualify Kaggle device"},
        "limits": {"cpus": 2, "ram_bytes": 6000000000, "gpu_memory_bytes": 4300000000, "gpu_stop_bytes": 5000000000, "timeout_seconds": 1800},
    }
    stages = []
    for item in treatment_stages:
        name = item["run"].replace("t3r5_pair_r6", "t3r5_treatment_r22")
        argv = normalize_argv(item["argv_template"])
        argv[argv.index("--name") + 1] = name
        if "--resume-state" in argv:
            index = argv.index("--resume-state") + 1
            argv[index] = argv[index].replace("t3r5_pair_r6", "t3r5_treatment_r22")
        stages.append({"name": name, "part": item["part"], "stop_after_epoch_inclusive": item["stop_after_epoch_inclusive"],
                       "argv": argv, "artifacts": [f"artifacts/runs/{name}"],
                       "limits": {"cpus": 2, "ram_bytes": 6000000000, "gpu_memory_bytes": 4300000000, "gpu_stop_bytes": 5000000000, "timeout_seconds": 5200}})
    final_run = stages[-1]["name"]
    export_name = f"{final_run}_full_valid"
    export_argv = ["{python}", f"{source_root}/scripts/cleanfold_exec.py", "scripts/predict_run.py", final_run, "valid",
                   "--epoch", "5", "--output-name", export_name, "--campaign-split-manifest",
                   "research/campaign_2026-09-29/artifacts/splits/fold0.json", "--device", "cuda",
                   "--campaign-full-valid-feature-only", "--inference-precision", "fp32"]
    arms = [{"arm": "treatment", "parent_run": treatment_parent["run"],
             "fit_stages": stages, "export": {"name": f"{export_name}_export", "argv": export_argv,
             "artifacts": [f"cache/pred_{export_name}_valid.npy", f"cache/pred_{export_name}_valid.json"],
             "limits": {"cpus": 2, "ram_bytes": 6000000000, "gpu_memory_bytes": 4300000000, "gpu_stop_bytes": 5000000000, "timeout_seconds": 1800},
             "run": final_run, "epoch": 5, "shape": [1873, 20000, 2], "dtype": "float32", "bytes": 299680128}}]
    treatment_recipe = dict(pair["shared_student_recipe"])
    treatment_recipe["arms"] = [item for item in treatment_recipe["arms"] if item["arm"] == "treatment"]
    treatment_recipe["fixed_fit_membership"] = "10,535 train + 1,405 fit-valid groups; exclude 468 fold-0 evaluation groups"
    treatment_recipe["prior_f0_student_control_not_repeated"] = (
        "the matched combo-parent p1/p2/p3 control is already admitted locally; this cloud cohort runs treatment only"
    )
    teacher_target = {
        "target_formula": "float32(0.8125 * float32_mean([teacher_s3_epoch2, teacher_s4_epoch2, teacher_s5_epoch2]))",
        "cache_root": "cache/clean_t3_f0_h390_t3r5_a9_r2",
        "cache_validation_receipt_sha256": "90528f221fdca114f506871fed06372bb7580a9d2700632abd156aa03830bedf",
        "cache_metadata_adapter_sha256": "c04fc22b78be9869c4e447316a5bd84a100bd4ef469bb244ce6f591974d9c033",
        "fit_membership": {"train_groups": 10535, "fit_valid_groups": 1405, "excluded_evaluation_groups": 468},
        "teacher_ancestry": "three own clean-fold-0 H384 Muon-all/prop2/e3 seed3/4/5 CP2 teachers",
    }
    return {
        "schema": "wnn.clean_f0_student_treatment_kaggle_plan.r2", "status": "prepared_pending_root_release",
        "preparation_id": "clean-t3-f0-matched-student-treatment-kaggle-r22-20261012",
        "supersedes_draft": "clean_t3_f0_student_pair_kaggle_r21; never staged or dispatched",
        "cohort_scope": "treatment-only; reuse the independently trained local control endpoint if its terminal gate passes",
        "source_pair_r2_sha256": sha256_file(R6_PAIR), "source_pair_r2_preparation_id": pair["preparation_id"],
        "source_manifest_sha256": source_manifest_sha, "trainer_sha256": pair["source"]["trainer_sha256"],
        "cleanfold_exec_sha256": pair["source"]["cleanfold_exec_sha256"],
        "fold": 0, "seed": 3, "membership": pair["fold_membership"],
        "recipe": treatment_recipe, "teacher_target": teacher_target,
        "cache": {"metadata": {"sha256": "1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8", "size": 323404, "filename": "pair_r2_cache_metadata.json"},
                   "train": {"sha256": "6a19f7ac45c6e8a02d3b89c0ac4986ffab4885ab7a4ebc5fb4599b3047f4e7b3", "size": 1685600128, "shape": [10535, 20000, 2], "filename": "pair_r2_main_homogeneous_train.npy"},
                   "fit_valid": {"sha256": "d1cd78537f3b6aaa830861ac771cf5c4621f1f2c1f19a73f6affb813c33ca3a2", "size": 224800128, "shape": [1405, 20000, 2], "filename": "pair_r2_main_homogeneous_fit_valid.npy"},
                   "validation_receipt_sha256": "90528f221fdca114f506871fed06372bb7580a9d2700632abd156aa03830bedf",
                   "metadata_adapter_sha256": "c04fc22b78be9869c4e447316a5bd84a100bd4ef469bb244ce6f591974d9c033"},
        "parents": {"treatment": {"run": treatment_parent["run"], "checkpoint": {"sha256": treatment_parent["checkpoint"]["sha256"], "size": treatment_parent["checkpoint"]["bytes"], "filename": "treatment_parent_model_ep3.pt"},
                                   "provenance": {"sha256": treatment_parent["provenance"]["sha256"], "size": treatment_parent["provenance"]["bytes"], "filename": "treatment_parent_provenance.json"}, "epoch": 3, "own_weights": True}},
        "external_mounts": external_mounts, "external_input_identities": external_ids,
        "asset_files": asset_files, "worker_code": code,
        "worker_code_sha256": canonical_sha(sorted(code, key=lambda item: item["path"])),
        "worker_inputs": worker_inputs, "profile": profile, "arms": arms,
        "budgets": {"kernel_wall_ceiling_seconds": 14400, "finalization_reserve_seconds": 300,
                    "single_device_gpu_seconds_upper_bound": 14400, "fit_stage_count": 3, "export_stage_count": 1,
                    "one_worker_at_a_time": True, "profile_batches": 30,
                    "expected_wall_seconds": None, "actual_kaggle_profile": False},
        "root_release": {"required": True, "template_status": "pending", "profile_authorized": False,
                         "fit_authorized": False, "paid_spending_authorized": False,
                         "receipt_schema": "wnn.clean_f0_student_treatment_root_release.r2"},
        "local_control_dependency": {"run": "clean_t3_f0_combo_parent_s3_a9_t3r5_pair_r6_p3", "required_for_paired_grade": True,
                                     "status": "reuse only after root validates its completed CP5 and prediction artifacts"},
        "r2_export_target_correction": {"basis": "R6 p3 writes its `_p3` run; export must target that actual treatment run",
                                         "successor_behavior": "export treatment p3 at epoch5", "science_changed": False},
    }


def main() -> None:
    if SOURCE_STAGE.exists() or ASSET_STAGE.exists():
        raise FileExistsError("refusing to overwrite an existing physical Kaggle stage")
    SOURCE_STAGE.mkdir(parents=True)
    ASSET_STAGE.mkdir(parents=True)
    data = staged_source_descriptors()
    asset_records: dict[str, dict] = {}
    asset_staging_methods: dict[str, str] = {}
    for group in (data["cache"], *data["parents"].values()):
        for descriptor in group.values():
            source = descriptor["source"]
            dest = ASSET_STAGE / descriptor["filename"]
            item = copy_checked(source, dest, expected_size=descriptor["bytes"], expected_sha=descriptor["sha256"])
            asset_records[descriptor["filename"]] = {"filename": descriptor["filename"], "size": item["bytes"], "sha256": item["sha256"]}
            asset_staging_methods[descriptor["filename"]] = item["staging_mode"]
    write_json(ASSET_STAGE / "assets-manifest.json", {"schema": "wnn.clean_f0_student_treatment_flat_assets.r2", "files": sorted(asset_records.values(), key=lambda row: row["filename"])})
    asset_manifest_sha = sha256_file(ASSET_STAGE / "assets-manifest.json")
    write_json(ASSET_STAGE / "dataset-metadata.json", {"id": ASSET_ID, "title": ASSET_TITLE, "isPrivate": True, "licenses": [{"name": "other"}]})

    bundle_manifest = json.loads((R5_BUNDLE / "bundle-manifest.json").read_text())
    bundle_manifest["dataset"] = SOURCE_ID
    plan = build_plan(asset_records)
    bundle_manifest["external_mounts"] = plan["external_mounts"]
    write_json(SOURCE_STAGE / "bundle-manifest.json", bundle_manifest)
    copy_checked(R5_BUNDLE / "bundle.wnn", SOURCE_STAGE / "bundle.wnn", expected_size=246396,
                 expected_sha="88641b3c9fc629a7a2ce8b11cfd98ecf8b53fe943ffb2d7f632e1dfb997fa549")
    source_manifest_sha = sha256_file(SOURCE_STAGE / "bundle-manifest.json")
    write_json(SOURCE_STAGE / "pair-plan.json", plan)
    plan_sha = sha256_file(SOURCE_STAGE / "pair-plan.json")
    write_json(SOURCE_STAGE / "ROOT_TREATMENT_RELEASE_R2.json", {"schema": "wnn.clean_f0_student_treatment_root_release.r2", "status": "pending", "authorized": False,
               "target": {"kernel_id": KERNEL_ID, "source_dataset": SOURCE_ID, "asset_dataset": ASSET_ID},
               "profile_authorized": False, "fit_authorized": False, "quota_budget": None,
               "binding": {"pair_plan_sha256": plan_sha, "asset_manifest_sha256": asset_manifest_sha,
                           "bundle_sha256": "88641b3c9fc629a7a2ce8b11cfd98ecf8b53fe943ffb2d7f632e1dfb997fa549",
                           "bundle_manifest_sha256": source_manifest_sha,
                           "source_manifest_sha256": plan["source_manifest_sha256"],
                           "cache_metadata_sha256": plan["cache"]["metadata"]["sha256"],
                           "cache_train_sha256": plan["cache"]["train"]["sha256"],
                           "cache_valid_sha256": plan["cache"]["fit_valid"]["sha256"],
                           "parent_checkpoint_sha256": {k:v["checkpoint"]["sha256"] for k,v in plan["parents"].items()},
                           "parent_provenance_sha256": {k:v["provenance"]["sha256"] for k,v in plan["parents"].items()}}})
    write_json(SOURCE_STAGE / "dataset-metadata.json", {"id": SOURCE_ID, "title": SOURCE_TITLE, "isPrivate": True, "licenses": [{"name": "other"}]})
    kernel_template = (PACKET / "kernel.py").read_text()
    kernel = kernel_template.replace("__PAIR_PLAN_SHA256__", plan_sha).replace("__ASSET_MANIFEST_SHA256__", asset_manifest_sha).replace("__SOURCE_BUNDLE_MANIFEST_SHA256__", source_manifest_sha)
    unresolved = ("__PAIR_PLAN_SHA256__", "__ASSET_MANIFEST_SHA256__", "__SOURCE_BUNDLE_MANIFEST_SHA256__")
    if any(token in kernel for token in unresolved):
        raise ValueError("unresolved kernel template token")
    KERNEL_STAGE.mkdir(parents=True, exist_ok=True)
    (KERNEL_STAGE / "kernel.py").write_text(kernel)
    write_json(KERNEL_STAGE / "kernel-metadata.json", {"code_file": "kernel.py", "competition_sources": [],
        "dataset_sources": [SOURCE_ID, ASSET_ID, "vladkochetov07/wnn-c20261004-fold0-assets", "vladkochetov07/wunder-funder"],
        "enable_gpu": True, "enable_internet": False, "enable_tpu": False, "id": KERNEL_ID,
        "is_private": True, "kernel_sources": [], "kernel_type": "script", "language": "python",
        "machine_shape": "NvidiaTeslaT4", "title": KERNEL_TITLE})
    plan_receipt = {"schema": "wnn.clean_f0_student_treatment_kaggle_stage_receipt.r2", "status": "prepared_not_uploaded",
        "source_stage": str(SOURCE_STAGE), "asset_stage": str(ASSET_STAGE), "kernel_stage": str(KERNEL_STAGE),
        "source_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in SOURCE_STAGE.iterdir() if p.is_file()},
        "asset_files": {
            **{name: {"bytes": record["size"], "sha256": record["sha256"]} for name, record in asset_records.items()},
            "assets-manifest.json": {"bytes": (ASSET_STAGE / "assets-manifest.json").stat().st_size,
                                     "sha256": asset_manifest_sha},
        },
        "kernel_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in KERNEL_STAGE.iterdir() if p.is_file()},
        "treatment_plan_sha256": plan_sha, "asset_manifest_sha256": asset_manifest_sha,
        "source_bundle_manifest_sha256": source_manifest_sha, "bundle_sha256": sha256_file(SOURCE_STAGE / "bundle.wnn"),
        "builder_script_sha256": sha256_file(Path(__file__)),
        "asset_staging_methods": asset_staging_methods,
        "provider_mutation": False, "profile_or_fit_started": False, "expected_source_data_file_count": 4,
        "expected_flat_asset_file_count": 6, "expected_kernel_code_bytes": (KERNEL_STAGE / "kernel.py").stat().st_size,
        "profile": {"full_batches": 30, "warmup": 5, "measured": 25, "one_device_t4": True},
        "fit_recipe": "treatment-only p1/p2/p3 resume; six epochs, fixed CP5; serial workers; local control is not repeated",
        "unresolved": ["fresh quota and logical hold", "source and flat asset dataset upload/READY version readback", "root release receipt", "independent packet review", "actual Kaggle T4 profile and fit elapsed time"],
    }
    write_json(PACKET / "STAGING_RECEIPT_R2.json", plan_receipt)
    print(json.dumps(plan_receipt, indent=2))


if __name__ == "__main__":
    main()
