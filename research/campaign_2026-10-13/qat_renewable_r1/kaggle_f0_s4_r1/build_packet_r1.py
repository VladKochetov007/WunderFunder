from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKET_DIR = Path(__file__).resolve().parent
DELIVERY_DIR = REPO_ROOT / "research/campaign_2026-10-13/qat_renewable_r1/delivery_r1"
SOURCE_RECIPE = DELIVERY_DIR / "RUN_RECIPE_R1.json"
SOURCE_ANCESTRY = DELIVERY_DIR / "ANCESTRY_VERIFICATION_R1.json"
FULL_SUBSET = DELIVERY_DIR / "subsets/f0_s3/fit_components_100pct.json"
SPLIT_REL = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
PORTABLE_REL = "research/campaign_2026-10-04/portable_inputs.json"
OLD_ASSET_ID = "vladkochetov07/wnn-c20261004-fold0-assets"
RAW_DATASET_ID = "vladkochetov07/wunder-funder"
NEW_ASSET_ID = "vladkochetov07/wnn-qat-f0-s4-assets-r1"
PROFILE_BUNDLE_ID = "vladkochetov07/wnn-qat-f0-s4-profile-r1"
FIT_BUNDLE_ID = "vladkochetov07/wnn-qat-f0-s4-fit-r1"
PROVIDER_TIMEOUT_SECONDS = {"profile": 1_800, "fit": 10_800}
EXPERIMENT_ID = "E-QAT-DEPLOYMENT-20261013-R1"
CACHE_REL = "cache/clean_t3_f0_h390_t3r5_a9_r2"
PARENT_RUN = "clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1"
INIT_RUN = "c20261003_combo_s4_f0"
PARENT_CHECKPOINT_SHA256 = "e6be82ff799f4ea2b62e770e9c340a8cfd8f98780d1c7dec4a76eab5a853d35c"
PARENT_PROVENANCE_SHA256 = "bdd531eb4e1e8ce031fda1b03ecf1c33c133c892768b7d404da30ec69679dd49"
INIT_CHECKPOINT_SHA256 = "0203b109a1f25f3c82794ad3684b83fe4a4bc815f4b5c1bc4bc1ea450dba0a8d"
INIT_PROVENANCE_SHA256 = "c9fb263175f9b13c5b897f8c24d8f2dbe1164415bfd5adcb22f59dc95a72b332"
PROPENSITY_SHA256 = "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PORTABLE_SHA256 = "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c"
PARENT_MEAN_SHA256 = "d3517e22c9ca90cd4cc61ae9bc624f27f849285a66f9c3b79b7238168ab92d9b"
PARENT_STD_SHA256 = "b228595dbbcb774679b4d779bcc188079da504349da481cc5b7b33743034d7ce"


def sha256_file(path: Path) -> str:
    """Hash small source artifacts in bounded blocks; large target arrays are skipped."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, maximum_bytes: int = 5_000_000) -> dict[str, Any]:
    """Read size-bounded metadata used by the packet builder."""
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f"metadata exceeds {maximum_bytes} bytes: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def write_json(path: Path, value: object) -> None:
    """Write stable JSON with an atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def descriptor(path: Path, logical: str | None = None) -> dict[str, Any]:
    """Describe a local code or metadata dependency by exact content identity."""
    return {"path": logical or path.relative_to(REPO_ROOT).as_posix(),
            "size": path.stat().st_size, "sha256": sha256_file(path)}


def add_asset(
    data_root: Path,
    staging: Path,
    logical: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    source: str,
) -> dict[str, Any]:
    """Hardlink one immutable source into the short physical asset staging path."""
    source_path = data_root / logical
    stat = source_path.stat()
    size = stat.st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"source length differs for {logical}: {size} != {expected_size}")
    if size > 64 * 1024**2:
        if expected_sha256 is None:
            raise ValueError(f"large immutable source has no trusted expected hash: {logical}")
        digest = expected_sha256
        verification = "trusted_metadata_expected_hash_pending_local_payload_read"
    else:
        digest = sha256_file(source_path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"source digest differs for {logical}: {digest}")
        verification = "payload_sha256_verified_during_bounded_packet_build"
    filename = f"{digest}.wnn"
    target = staging / filename
    if target.exists():
        target_stat = target.stat()
        if target_stat.st_size != size:
            raise ValueError(f"content-addressed filename collision for {logical}")
        if target_stat.st_dev != stat.st_dev or target_stat.st_ino != stat.st_ino:
            if size > 64 * 1024**2 or sha256_file(target) != digest:
                raise ValueError(f"existing staged source is not the expected hardlink: {logical}")
    else:
        if stat.st_dev != staging.stat().st_dev:
            raise ValueError(f"cross-filesystem staging is forbidden for {logical}")
        os.link(source_path, target)
    target_stat = target.stat()
    if (target_stat.st_dev, target_stat.st_ino) != (stat.st_dev, stat.st_ino):
        raise ValueError(f"immutable source was copied instead of hardlinked: {logical}")
    return {
        "path": logical,
        "size": size,
        "sha256": digest,
        "filename": filename,
        "source": source,
        "sha256_verification": verification,
        "hardlink": True,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def old_asset_entries(data_root: Path) -> dict[str, dict[str, Any]]:
    """Load the already-published F0 asset manifest without reading its payloads."""
    path = data_root / "artifacts/kaggle_assets/wnn-c20261004-fold0-assets/assets-manifest.json"
    manifest = read_json(path, maximum_bytes=1_000_000)
    if manifest.get("dataset") != OLD_ASSET_ID or manifest.get("private") is not True:
        raise ValueError("existing F0 asset dataset identity is not the reviewed private mount")
    return {item["path"]: item for item in manifest["files"]}


def build_new_asset_set(data_root: Path, stage_root: Path, ancestry: dict[str, Any], cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Stage the seed-4 checkpoint/cache ancestry and target cache by hardlink."""
    asset_dir = stage_root / "asset_dataset"
    asset_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    parent_record = ancestry["parent_chain"]["verified_f0_seed4_sibling"]
    if (parent_record["run"] != PARENT_RUN
            or parent_record["checkpoint_sha256"] != PARENT_CHECKPOINT_SHA256
            or parent_record["provenance_sha256"] != PARENT_PROVENANCE_SHA256
            or parent_record["initializer_checkpoint_sha256"] != INIT_CHECKPOINT_SHA256
            or parent_record["initializer_provenance_sha256"] != INIT_PROVENANCE_SHA256):
        raise ValueError("seed-4 sibling checkpoint pins differ from the verified ancestry receipt")

    parent_prov_path = data_root / f"artifacts/runs/{PARENT_RUN}/provenance.json"
    parent_prov = read_json(parent_prov_path, maximum_bytes=2_000_000)
    if sha256_file(parent_prov_path) != PARENT_PROVENANCE_SHA256:
        raise ValueError("seed-4 CP5 provenance no longer matches its ancestry receipt")
    p2_state = parent_prov["resumed_from"]
    if p2_state["next_epoch"] != 4 or p2_state["path"] != f"artifacts/runs/clean_t3_f0_control_s4_a9_t3r5_r3_p2_envr1/training_state.pt":
        raise ValueError("seed-4 CP5 optimizer continuation source differs")
    p2_state_sha = p2_state["sha256"]
    p2_state_path = data_root / p2_state["path"]
    p2_prov_path = p2_state_path.parent / "provenance.json"
    p2_prov = read_json(p2_prov_path, maximum_bytes=2_000_000)
    p1_state = p2_prov["resumed_from"]
    if p1_state["next_epoch"] != 2:
        raise ValueError("seed-4 P2 does not resume the expected P1 epoch boundary")
    p1_state_path = data_root / p1_state["path"]
    p1_prov_path = p1_state_path.parent / "provenance.json"
    p1_prov = read_json(p1_prov_path, maximum_bytes=2_000_000)
    if p1_prov.get("initial_checkpoint_sha256") != INIT_CHECKPOINT_SHA256:
        raise ValueError("seed-4 CP5 P1 does not use the pinned combo initializer")

    fixed_assets = [
        (f"artifacts/runs/{PARENT_RUN}/model_ep5.pt", PARENT_CHECKPOINT_SHA256, 3_983_015, "seed4_cp5_parent"),
        (f"artifacts/runs/{PARENT_RUN}/provenance.json", PARENT_PROVENANCE_SHA256, 166_153, "seed4_cp5_provenance"),
        (f"artifacts/runs/{INIT_RUN}/model_ep3.pt", INIT_CHECKPOINT_SHA256, 3_977_183, "seed4_combo_initializer"),
        (f"artifacts/runs/{INIT_RUN}/provenance.json", INIT_PROVENANCE_SHA256, 159_653, "seed4_combo_provenance"),
        (p2_state["path"], p2_state_sha, None, "seed4_cp5_p2_training_state"),
        (p2_prov_path.relative_to(data_root).as_posix(), None, None, "seed4_cp5_p2_provenance"),
        (p1_state["path"], p1_state["sha256"], None, "seed4_cp5_p1_training_state"),
        (p1_prov_path.relative_to(data_root).as_posix(), None, None, "seed4_cp5_p1_provenance"),
        ("artifacts/runs/c20260929_prop_f0/provenance.json", None, None, "fold0_propensity_provenance"),
    ]
    for logical, expected, size, role in fixed_assets:
        entries.append(add_asset(data_root, asset_dir, logical, expected_sha256=expected,
                                 expected_size=size, source=role))

    verified_files = ancestry["parent_chain"]["verified_files"]
    verified_by_path = {item["path"]: item for item in verified_files}
    seed4_teacher_records = ancestry["parent_chain"]["source_args"]["combo_initializer_seed4"]["distill_provenance"]
    seed4_teachers: dict[str, str] = {}
    for teacher in seed4_teacher_records:
        if teacher.get("epoch") != 3 or teacher.get("split") not in {"train", "valid"}:
            raise ValueError("seed-4 combo teacher initialization record has an unexpected prediction source")
        previous = seed4_teachers.setdefault(teacher["run"], teacher["checkpoint_sha256"])
        if previous != teacher["checkpoint_sha256"]:
            raise ValueError("seed-4 combo teacher checkpoint identity changes across split predictions")
    if len(seed4_teachers) != 2:
        raise ValueError("seed-4 combo initializer does not identify exactly two source teachers")
    for run, checkpoint_sha256 in sorted(seed4_teachers.items()):
        checkpoint_path = f"artifacts/runs/{run}/model_ep3.pt"
        provenance_path = f"artifacts/runs/{run}/provenance.json"
        checkpoint_entry = verified_by_path.get(checkpoint_path)
        provenance_entry = verified_by_path.get(provenance_path)
        if checkpoint_entry is None or provenance_entry is None:
            raise ValueError(f"verified seed-4 combo teacher files are missing: {run}")
        entries.append(add_asset(data_root, asset_dir, checkpoint_path,
                                 expected_sha256=checkpoint_sha256,
                                 expected_size=checkpoint_entry["bytes"], source="combo_teacher_epoch3"))
        entries.append(add_asset(data_root, asset_dir, provenance_path,
                                 expected_sha256=provenance_entry["sha256"],
                                 expected_size=provenance_entry["bytes"], source="combo_teacher_provenance"))

    for teacher in cache["teachers"]:
        checkpoint = teacher["checkpoint"]
        provenance = teacher["provenance"]
        entries.append(add_asset(data_root, asset_dir, checkpoint["path"],
                                 expected_sha256=checkpoint["sha256"], expected_size=checkpoint["bytes"],
                                 source="clean_t3_target_teacher_epoch2"))
        entries.append(add_asset(data_root, asset_dir, provenance["path"],
                                 expected_sha256=provenance["sha256"], expected_size=provenance["bytes"],
                                 source="clean_t3_target_teacher_provenance"))

    entries.append(add_asset(data_root, asset_dir, cache["metadata_path"],
                             expected_sha256=cache["metadata_sha256"],
                             expected_size=cache["metadata_bytes"], source="clean_t3_cache_metadata"))
    for array in cache["payloads"]:
        entries.append(add_asset(data_root, asset_dir, array["path"],
                                 expected_sha256=array["sha256_from_metadata"],
                                 expected_size=array["bytes"], source="clean_t3_cache_array"))

    manifest = {
        "version": 1,
        "dataset": NEW_ASSET_ID,
        "private": True,
        "verification_status": "pending_large_target_payload_hash_and_controller_review",
        "files": [{key: value for key, value in entry.items()
                    if key in {"path", "size", "sha256", "filename"}} for entry in entries],
        "local_staging": entries,
    }
    write_json(asset_dir / "assets-manifest.pending.json", manifest)
    write_json(asset_dir / "dataset-metadata.json", {
        "id": NEW_ASSET_ID,
        "title": NEW_ASSET_ID.split("/")[1],
        "isPrivate": True,
        "licenses": [{"name": "other"}],
    })
    return entries


def npy_header(path: Path) -> tuple[tuple[int, ...], str]:
    """Inspect one NPY header without reading its array payload."""
    with path.open("rb") as source:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(source)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(source)
    return tuple(int(value) for value in shape), str(dtype)


def make_recipes(data_root: Path, ancestry: dict[str, Any], cache: dict[str, Any], new_assets: list[dict[str, Any]], old_assets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Derive matched full-data seed-4 profiles and fits from the reviewed R1 vectors."""
    source_recipe = read_json(SOURCE_RECIPE, maximum_bytes=2_000_000)
    profile_base = source_recipe["profiles"][0]["argv"]
    fit_base = next(item["argv"] for item in source_recipe["fits"]
                    if item["subset"] == "100pct" and item["condition"] == "fp32")
    modes = [
        ("fp32", "none"),
        ("uniform_w4", "recurrent_w4"),
        ("mixed_w4w8", "input_w4_recurrent_w8"),
    ]

    def rewrite(argv: list[str], name: str, qat: str, *, profile: bool) -> list[str]:
        result = list(argv)
        values = {
            "--name": name,
            "--seed": "4",
            "--init": f"{PARENT_RUN}@ep5",
            "--fit-subset-manifest": FULL_SUBSET.relative_to(REPO_ROOT).as_posix(),
            "--qat": qat,
        }
        for option, value in values.items():
            index = result.index(option)
            result[index + 1] = value
        if profile:
            result[result.index("--profile-warmup") + 1] = "2"
            result[result.index("--profile-steps") + 1] = "3"
            if "--grad-stats" not in result:
                result.append("--grad-stats")
        return result

    profiles = []
    fits = []
    for label, qat in modes:
        profile_name = f"qat_deploy_f0_s4_100pct_{label}_profile_r1"
        fit_name = f"qat_deploy_f0_s4_100pct_{label}_r1"
        profiles.append({
            "name": profile_name,
            "condition": label,
            "qat": qat,
            "argv": rewrite(profile_base, profile_name, qat, profile=True),
            "warmup_batches": 2,
            "measured_batches": 3,
            "updates_per_batch": 20,
            "warmup_optimizer_updates": 40,
            "timed_optimizer_updates": 60,
            "total_batches": 5,
            "total_optimizer_updates": 100,
            "profile_acceptance": "PROFILE_UPDATE_EVIDENCE valid_for_throughput=true; 60 timed finite FP32 optimizer updates; GradStats has 100 total optimizer steps and zero nonfinite steps",
        })
        fits.append({
            "name": fit_name,
            "condition": label,
            "qat": qat,
            "argv": rewrite(fit_base, fit_name, qat, profile=False),
            "timeout_seconds": 10800,
            "checkpoint": f"artifacts/runs/{fit_name}/model_ep0.pt",
            "recovery": "training_state.pt at completed epoch only; an interrupted mid-epoch attempt is preserved and retried only with a new run identity",
        })

    old_required = [
        "cache/chain_folds.npz", "cache/chains.npz", "cache/train_targets.npz", "cache/valid_targets.npz",
        SPLIT_REL, "artifacts/runs/c20260929_prop_f0/model.pt",
        "cache/propcache_c20260929_prop_f0_train.json", "cache/propcache_c20260929_prop_f0_train.npy",
        "cache/propcache_c20260929_prop_f0_valid.json", "cache/propcache_c20260929_prop_f0_valid.npy",
    ]
    external: list[dict[str, Any]] = []
    for logical in old_required:
        item = old_assets.get(logical)
        if item is None:
            raise ValueError(f"required F0 external asset is absent from the pinned dataset: {logical}")
        external.append({
            "path": logical, "dataset": OLD_ASSET_ID, "filename": item["filename"],
            "size": item["size"], "sha256": item["sha256"],
            "sha256_source": "existing private fold0-assets assets-manifest.json",
        })
    portable = read_json(data_root / PORTABLE_REL, maximum_bytes=1_000_000)
    for split in ("train", "valid"):
        logical = f"datasets/{split}.parquet"
        item = portable["sources"][logical]
        external.append({
            "path": logical, "dataset": RAW_DATASET_ID, "filename": f"{split}.parquet",
            "size": item["original_identity"]["bytes"], "sha256": item["sha256"],
            "sha256_source": PORTABLE_REL,
        })
    for item in new_assets:
        external.append({
            "path": item["path"], "dataset": NEW_ASSET_ID, "filename": item["filename"],
            "size": item["size"], "sha256": item["sha256"],
            "sha256_source": item["source"],
            "sha256_verification": item["sha256_verification"],
        })

    external_by_path = {item["path"]: item for item in external}
    if len(external_by_path) != len(external):
        raise ValueError("external input paths are not unique")
    input_manifest = {
        "schema": "qat-kaggle-input-identities-r1",
        "training_source_revision": "173cce08e9c7f13caccd1c3090419dd044aa702d",
        "experiment_id": EXPERIMENT_ID,
        "external_inputs": external,
        "portable_registry_path": PORTABLE_REL,
        "portable_registry_sha256": PORTABLE_SHA256,
        "split_manifest_path": SPLIT_REL,
        "split_manifest_sha256": SPLIT_SHA256,
        "fit_subset_path": FULL_SUBSET.relative_to(REPO_ROOT).as_posix(),
        "parent_checkpoint_path": f"artifacts/runs/{PARENT_RUN}/model_ep5.pt",
        "parent_provenance_path": f"artifacts/runs/{PARENT_RUN}/provenance.json",
        "parent": {
            "run": PARENT_RUN, "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "checkpoint_bytes": 3_983_015, "provenance_sha256": PARENT_PROVENANCE_SHA256,
            "initializer": f"{INIT_RUN}@ep3", "initializer_sha256": INIT_CHECKPOINT_SHA256,
            "propensity_sha256": PROPENSITY_SHA256,
            "mean_sha256": PARENT_MEAN_SHA256, "std_sha256": PARENT_STD_SHA256,
            "n_in": 211, "aux_width": 23,
            "interpretation": "--init loads the exact raw CP5 model tensors and mean/std buffers; no optimizer or averaging state is inherited; no --inherit-init-preprocessing flag is used because the recipe explicitly matches the parent model arguments.",
        },
        "teacher_cache": {
            "root": CACHE_REL,
            "metadata_path": f"{CACHE_REL}/metadata.json",
            "metadata_sha256": cache["metadata_sha256"],
            "teachers": cache["teachers"],
            "payloads": cache["payloads"],
            "target_formula": "float32(0.8125 * float32_mean(three_epoch2_predictions))",
        },
    }
    write_json(PACKET_DIR / "INPUT_IDENTITIES_R1.json", input_manifest)
    recipe = {
        "schema": "qat-kaggle-full-f0-seed4-recipe-r1",
        "status": "frozen_preparation_no_fit_or_profile_run",
        "experiment_id": EXPERIMENT_ID,
        "training_source_revision": "173cce08e9c7f13caccd1c3090419dd044aa702d",
        "source_recipe": SOURCE_RECIPE.relative_to(REPO_ROOT).as_posix(),
        "source_recipe_sha256": sha256_file(SOURCE_RECIPE),
        "parent_run": PARENT_RUN,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "fit_subset_manifest": FULL_SUBSET.relative_to(REPO_ROOT).as_posix(),
        "full_membership": {"components": 3148, "train_groups": 10535, "fit_valid_groups": 1405, "fixed_evaluation_groups": 468},
        "common_recipe": {
            "precision": "fp32", "optimizer": "fresh AdamW state for each condition",
            "learning_rate": 0.0001, "weight_decay": 0.01, "schedule": "onecycle",
            "epochs": 1, "batch": 16, "chunk": 1000, "batch_order": "random", "seed": 4,
            "ema_teacher": 0, "swa_frac": 0, "checkpoint": "fixed raw model_ep0.pt",
            "fit_eval": "disabled; root evaluator owns full fixed F0 evaluation and native archive export",
        },
        "quantization": {
            "fp32": "qat none",
            "uniform_w4": "recurrent_w4: GRU weight_ih and weight_hh Q4, split head.0.weight Q8",
            "mixed_w4w8": "input_w4_recurrent_w8: GRU weight_ih Q4, weight_hh Q8, split head.0.weight Q8",
            "training": "FP32 weight-only QAT; recurrent activations are not quantized",
        },
        "profile_gate": profiles,
        "fits": fits,
        "cloud_external_inputs": external,
        "collection": {
            "profile": "Kaggle kernel terminal outputs include runner summary, per-job result and worker.log; collect all before removing/replacing a private version.",
            "fit": "Each completed run directory is declared for artifact collection and contains model_ep0.pt, model.pt, training_state.pt, provenance.json and history.json; checksum every received file against runner result artifacts.",
            "native_evaluation": "root-owned: export every raw fit checkpoint and the matched FP32 PTQ controls to native archives, then run the frozen callback/full-trajectory evaluator; framework output is not QAT quality evidence.",
        },
    }
    write_json(PACKET_DIR / "RUN_RECIPE_S4_R1.json", recipe)
    return {"profiles": profiles, "fits": fits, "external": external,
            "input_manifest": input_manifest, "recipe": recipe}


def build_specs(recipes: dict[str, Any], stage_root: Path) -> tuple[list[Path], list[Path]]:
    """Create schema-valid runner specs and two offline private bundles."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from kaggle_bundle import build_bundle
    from kaggle_common import canonical_sha256, read_spec

    code_paths = [
        "scripts/kaggle_common.py", "scripts/kaggle_gpu_runner.py", "scripts/kaggle_kernel.py",
        "scripts/kaggle_qat_kernel_r1.py", "scripts/train_rnn.py", "scripts/qat_kaggle_preflight_r1.py",
        *sorted(path.relative_to(REPO_ROOT).as_posix() for path in (REPO_ROOT / "src/wnn").rglob("*.py")),
    ]
    code = [descriptor(REPO_ROOT / name) for name in code_paths]
    local_inputs = [
        descriptor(PACKET_DIR / "INPUT_IDENTITIES_R1.json"),
        descriptor(PACKET_DIR / "RUN_RECIPE_S4_R1.json"),
        descriptor(SOURCE_ANCESTRY),
        descriptor(FULL_SUBSET),
        descriptor(REPO_ROOT / PORTABLE_REL),
    ]
    external = recipes["external"]
    inputs = sorted(local_inputs + [{"path": item["path"], "size": item["size"], "sha256": item["sha256"]}
                                    for item in external], key=lambda item: item["path"])
    deferred_paths = ",".join(sorted(item["path"] for item in external))
    base_environment = {
        "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "WNN_THREADS": "2", "WNN_PROJECT_ROOT": ".", "WNN_LOADER_RAM_GB": "4",
        "WNN_LOADER_WORKERS": "0", "WNN_PORTABLE_DATA_MANIFEST": PORTABLE_REL,
        "WNN_DEFERRED_IDENTITY_PATHS": deferred_paths,
    }
    def make_spec(name: str, argv: list[str], *, kind: str, artifacts: list[str],
                  limits: dict[str, int], environment: dict[str, str]) -> dict[str, Any]:
        return {
            "version": 1, "name": name, "kind": kind,
            "argv": ["{python}", *argv], "code": code,
            "code_sha256": canonical_sha256(sorted(code, key=lambda item: item["path"])),
            "inputs": inputs, "artifacts": artifacts, "limits": limits,
            "environment": environment,
        }

    def gpu_limits(ram: int, timeout: int) -> dict[str, int]:
        return {"cpus": 2, "ram_bytes": ram, "timeout_seconds": timeout,
                "gpu_memory_bytes": 3_500_000_000, "gpu_stop_bytes": 4_500_000_000}

    spec_dir = PACKET_DIR / "specs"
    spec_dir.mkdir(exist_ok=True)
    preflight_env = {**base_environment,
                     "WNN_QAT_INPUT_IDENTITIES_R1": (PACKET_DIR / "INPUT_IDENTITIES_R1.json").relative_to(REPO_ROOT).as_posix()}
    preflight_spec = make_spec(
        "qat_f0_s4_actual_input_preflight_r1",
        ["scripts/qat_kaggle_preflight_r1.py", "--repo-root", ".", "--identity-manifest",
         (PACKET_DIR / "INPUT_IDENTITIES_R1.json").relative_to(REPO_ROOT).as_posix(),
         "--output", "artifacts/qat_kaggle_f0_s4_r1/input_preflight.json"],
        kind="actual_data_identity_preflight", artifacts=["artifacts/qat_kaggle_f0_s4_r1/input_preflight.json"],
        limits=gpu_limits(6 * 1024**3, 3600), environment=preflight_env)
    preflight_path = spec_dir / "preflight.spec.json"
    write_json(preflight_path, preflight_spec)

    profile_paths: list[Path] = [preflight_path]
    for item in recipes["profiles"]:
        spec = make_spec(item["name"], ["scripts/train_rnn.py", *item["argv"][2:]],
                         kind="full_shape_qat_profile", artifacts=[],
                         limits=gpu_limits(8 * 1024**3, 3600), environment=base_environment)
        path = spec_dir / f"{item['name']}.spec.json"
        write_json(path, spec)
        profile_paths.append(path)

    fit_paths: list[Path] = [preflight_path]
    for item in recipes["fits"]:
        spec = make_spec(item["name"], ["scripts/train_rnn.py", *item["argv"][2:]],
                         kind="full_epoch_matched_qat_fit",
                         artifacts=[f"artifacts/runs/{item['name']}"],
                         limits=gpu_limits(8 * 1024**3, item["timeout_seconds"]),
                         environment=base_environment)
        path = spec_dir / f"{item['name']}.spec.json"
        write_json(path, spec)
        fit_paths.append(path)

    for path in set(profile_paths + fit_paths):
        read_spec(path)
    external_args: list[str] = []
    for item in external:
        if item["dataset"] == RAW_DATASET_ID:
            external_args.append(f"{item['path']}={item['filename']}")
        else:
            external_args.append(f"{item['path']}={item['dataset']}:{item['filename']}")
    asset_datasets = [OLD_ASSET_ID, NEW_ASSET_ID]
    build_bundle(REPO_ROOT, profile_paths, stage_root / "profile_upload", PROFILE_BUNDLE_ID,
                 external_args, asset_datasets)
    build_bundle(REPO_ROOT, fit_paths, stage_root / "fit_upload", FIT_BUNDLE_ID,
                 external_args, asset_datasets)
    template_path = REPO_ROOT / "scripts/kaggle_qat_kernel_r1.py"
    template_sha256 = sha256_file(template_path)
    entrypoints = {}
    for slug, bundle_dir in (("profile", stage_root / "profile_upload"),
                             ("fit", stage_root / "fit_upload")):
        manifest = read_json(bundle_dir / "bundle-manifest.json", maximum_bytes=2_000_000)
        source = template_path.read_text(encoding="utf-8")
        source = source.replace('BUNDLE_SLUG = ""', f"BUNDLE_SLUG = {manifest['dataset'].split('/')[1]!r}")
        source = source.replace('EXPECTED_ARCHIVE_SHA256 = ""',
                                f"EXPECTED_ARCHIVE_SHA256 = {manifest['archive_sha256']!r}")
        source = source.replace("EXPECTED_EXTERNAL_MOUNTS = None",
                                f"EXPECTED_EXTERNAL_MOUNTS = {manifest['external_mounts']!r}")
        source = source.replace(
            "KERNEL_PROCESS_WALL_SECONDS_MAX = 43_200  # Packet builder binds this to the Kaggle push timeout.",
            f"KERNEL_PROCESS_WALL_SECONDS_MAX = {PROVIDER_TIMEOUT_SECONDS[slug]}",
        )
        kernel_path = bundle_dir / "kernel/kernel.py"
        kernel_path.write_text(source, encoding="utf-8")
        entrypoints[slug] = {
            "dataset": manifest["dataset"],
            "archive_sha256": manifest["archive_sha256"],
            "template_path": template_path.relative_to(REPO_ROOT).as_posix(),
            "template_sha256": template_sha256,
            "kernel_path": str(kernel_path),
            "kernel_bytes": kernel_path.stat().st_size,
            "kernel_sha256": sha256_file(kernel_path),
            "provider_timeout_seconds": PROVIDER_TIMEOUT_SECONDS[slug],
            "runner_finalization_reserve_seconds": 300,
            "execution_barrier": "run the sole preflight spec in its own runner invocation; require its complete terminal result and input_preflight.json pass receipt before one second invocation admits the three candidate jobs",
        }
    write_json(PACKET_DIR / "ENTRYPOINT_BINDING_R1.json", entrypoints)
    return profile_paths, fit_paths


def main() -> None:
    """Build a private offline packet; this command never contacts Kaggle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--refresh-staged-hash-receipt", action="store_true",
                        help="refresh PACKET_R1.json from the existing bounded receipt without reading asset payloads")
    args = parser.parse_args()
    stage_root = args.stage_root.resolve()
    if args.refresh_staged_hash_receipt:
        packet = refresh_packet_from_hash_receipt(stage_root)
        print(json.dumps(packet, indent=2, sort_keys=True))
        return
    if args.data_root is None:
        parser.error("--data-root is required unless --refresh-staged-hash-receipt is set")
    data_root = args.data_root.resolve(strict=True)
    if any(part in {"tmp", "shm"} for part in stage_root.parts):
        raise ValueError("packet staging must use the allocated physical filesystem, not RAM tmpfs")
    stage_root.mkdir(parents=True, exist_ok=True)
    if os.stat(data_root).st_dev != os.stat(stage_root).st_dev:
        raise ValueError("large immutable assets must be hardlinked on one physical filesystem")
    if shutil.disk_usage(stage_root).free < 20 * 1024**3:
        raise ValueError("staging violates the 20 GiB physical-disk reserve")

    ancestry = read_json(SOURCE_ANCESTRY)
    cache = ancestry["teacher_cache"]
    old_assets = old_asset_entries(data_root)
    new_assets = build_new_asset_set(data_root, stage_root, ancestry, cache)
    for payload, expected_shape in zip(cache["payloads"], ((10535, 20000, 2), (1405, 20000, 2)), strict=True):
        source = data_root / payload["path"]
        actual_shape, dtype = npy_header(source)
        if actual_shape != expected_shape or dtype != "float32" or source.stat().st_size != payload["bytes"]:
            raise ValueError(f"source producer NPY header differs: {payload['path']}")
    recipes = make_recipes(data_root, ancestry, cache, new_assets, old_assets)
    profile_specs, fit_specs = build_specs(recipes, stage_root)

    upload_metadata = {}
    for slug, bundle_dir, expected_id in (
        ("profile", stage_root / "profile_upload", PROFILE_BUNDLE_ID),
        ("fit", stage_root / "fit_upload", FIT_BUNDLE_ID),
    ):
        metadata = read_json(bundle_dir / "dataset-metadata.json", maximum_bytes=100_000)
        kernel = read_json(bundle_dir / "kernel/kernel-metadata.json", maximum_bytes=100_000)
        bootstrap = bundle_dir / "kernel/kernel.py"
        if metadata.get("id") != expected_id or metadata.get("title") != expected_id.split("/")[1]:
            raise ValueError(f"{slug} title-derived dataset slug differs")
        if kernel.get("id") != expected_id + "-runner" or kernel.get("is_private") is not True:
            raise ValueError(f"{slug} kernel metadata identity differs")
        if bootstrap.stat().st_size >= 1_000_000:
            raise ValueError(f"{slug} Kaggle bootstrap exceeds the 1 MB envelope")
        upload_metadata[slug] = {
            "dataset_id": metadata["id"], "dataset_title": metadata["title"],
            "kernel_id": kernel["id"], "bootstrap_bytes": bootstrap.stat().st_size,
            "bootstrap_sha256": sha256_file(bootstrap),
            "specs": [path.relative_to(REPO_ROOT).as_posix() for path in (profile_specs if slug == "profile" else fit_specs)],
        }
    packet = {
        "schema": "qat-kaggle-f0-s4-packet-r1",
        "status": "offline_packet_prepared_large_array_hash_and_upload_pending",
        "training_source_revision": "173cce08e9c7f13caccd1c3090419dd044aa702d",
        "data_root": str(data_root), "staging_root": str(stage_root),
        "private_asset_dataset": NEW_ASSET_ID,
        "profile_bundle": PROFILE_BUNDLE_ID, "fit_bundle": FIT_BUNDLE_ID,
        "profile_gate_before_fit": True,
        "uploads_or_provider_mutations_performed": False,
        "large_target_arrays_hashed_locally": False,
        "upload_metadata": upload_metadata,
    }
    write_json(PACKET_DIR / "PACKET_R1.json", packet)
    print(json.dumps(packet, indent=2, sort_keys=True))


def refresh_packet_from_hash_receipt(stage_root: Path) -> dict[str, Any]:
    """Bind the existing asset hash receipt without rereading the large payloads."""
    packet_path = PACKET_DIR / "PACKET_R1.json"
    binding_path = PACKET_DIR / "ENTRYPOINT_BINDING_R1.json"
    receipt_path = stage_root / "STAGED_ASSET_HASH_VERIFICATION_R1.json"
    manifest_path = stage_root / "asset_dataset/assets-manifest.json"
    packet = read_json(packet_path)
    binding = read_json(binding_path, maximum_bytes=100_000)
    receipt = read_json(receipt_path, maximum_bytes=2_000_000)
    manifest = read_json(manifest_path, maximum_bytes=2_000_000)
    if (receipt.get("schema") != "qat-kaggle-staged-asset-verification-r1"
            or receipt.get("status") != "pass"
            or receipt.get("dataset") != NEW_ASSET_ID
            or manifest.get("dataset") != NEW_ASSET_ID
            or manifest.get("private") is not True
            or receipt.get("hardlinks_preserved") is not True):
        raise ValueError("staged hash receipt or final asset manifest does not match the private packet")
    expected = {item["path"]: item for item in manifest.get("files", [])}
    observed = {item["path"]: item for item in receipt.get("files", [])}
    if (len(expected) != 22 or len(receipt.get("files", [])) != len(expected)
            or set(expected) != set(observed)):
        raise ValueError("staged hash receipt does not cover the exact 22-file asset manifest")
    verified_bytes = 0
    for path, expected_item in expected.items():
        item = observed[path]
        if (item.get("filename") != expected_item.get("filename")
                or item.get("bytes") != expected_item.get("size")
                or item.get("expected_sha256") != expected_item.get("sha256")
                or item.get("observed_sha256") != expected_item.get("sha256")
                or item.get("status") != "match"):
            raise ValueError(f"staged receipt identity differs from the sealed manifest: {path}")
        staged = stage_root / "asset_dataset" / expected_item["filename"]
        if not staged.is_file() or staged.stat().st_size != expected_item["size"]:
            raise ValueError(f"staged asset is missing or its length changed after receipt: {path}")
        verified_bytes += int(expected_item["size"])
    if (receipt.get("files_verified") != len(expected)
            or receipt.get("bytes_verified") != verified_bytes
            or receipt.get("block_bytes") != 8 * 1024**2):
        raise ValueError("staged hash receipt totals do not match the sealed manifest")

    upload_metadata = packet.get("upload_metadata", {})
    for slug in ("profile", "fit"):
        bound = binding.get(slug)
        metadata = upload_metadata.get(slug)
        if not isinstance(bound, dict) or not isinstance(metadata, dict):
            raise ValueError(f"missing packet entrypoint binding for {slug}")
        if metadata.get("dataset_id") != bound.get("dataset"):
            raise ValueError(f"dataset slug differs from generated {slug} entrypoint")
        metadata.update({
            "archive_sha256": bound["archive_sha256"],
            "bootstrap_bytes": bound["kernel_bytes"],
            "bootstrap_sha256": bound["kernel_sha256"],
            "provider_timeout_seconds": bound["provider_timeout_seconds"],
            "runner_finalization_reserve_seconds": bound["runner_finalization_reserve_seconds"],
            "execution_barrier": bound["execution_barrier"],
        })
    packet.update({
        "entrypoint_binding_path": binding_path.relative_to(REPO_ROOT).as_posix(),
        "status": "offline_packet_prepared_staged_hashes_verified_provider_review_pending",
        "large_target_arrays_hashed_locally": True,
        "staged_asset_hash_verification": {
            "status": "pass",
            "files_verified": receipt["files_verified"],
            "bytes_verified": receipt["bytes_verified"],
            "block_bytes": receipt["block_bytes"],
            "hardlinks_preserved": receipt["hardlinks_preserved"],
            "receipt_path": str(receipt_path),
            "upload_directory_file_count": len(list((stage_root / "asset_dataset").iterdir())),
        },
        "uploads_or_provider_mutations_performed": False,
        "provider_timeout_alignment": {
            slug: {
                "seconds": binding[slug]["provider_timeout_seconds"],
                "bootstrap_deadline_matches": True,
                "runner_finalization_reserve_seconds": binding[slug]["runner_finalization_reserve_seconds"],
                "push_command": f"kaggle kernels push --path {stage_root / (slug + '_upload/kernel')} --timeout {binding[slug]['provider_timeout_seconds']}",
            }
            for slug in ("profile", "fit")
        },
    })
    write_json(packet_path, packet)
    return packet


if __name__ == "__main__":
    main()
