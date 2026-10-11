from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKET = Path(__file__).resolve().parent
REL_PACKET = PACKET.relative_to(Path(__file__).resolve().parents[6])
DATASET = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r1"
FOLD0_ASSETS = "vladkochetov07/wnn-c20261004-fold0-assets"
FOLD0 = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
PORTABLE = "research/campaign_2026-10-04/portable_inputs.json"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PROP_SHA = "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def descriptor(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    return {"path": name, "size": path.stat().st_size, "sha256": sha256(path)}


def external_descriptor(path: str, filename: str, size: int, digest: str) -> dict[str, Any]:
    return {"path": path, "size": size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze code-only clean-fold T3 F0 packet inputs and teacher specs.")
    parser.add_argument("--root", type=Path, required=True, help="repository checkout used for bounded bundle validation")
    parser.add_argument("--bundle-destination", type=Path, required=True)
    parser.add_argument("--student-parent-stage", type=Path, default=Path("/tmp/wnn-clean-t3-f0-student-parents-r1"))
    parser.add_argument("--evidence-root", type=Path, default=Path("/home/vlad/development/wunder/wnn_connectome_starterpack"))
    args = parser.parse_args()
    root = args.root.resolve()
    evidence_root = args.evidence_root.resolve()

    source_manifest_path = PACKET / "source_snapshot/SOURCE_MANIFEST_H390.json"
    old_manifest = json.loads(source_manifest_path.read_text())
    old_sources = old_manifest["sources"]
    current_sources: dict[str, str] = {}
    for path in sorted((PACKET / "source_snapshot").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        current_sources[path.relative_to(PACKET / "source_snapshot").as_posix()] = sha256(path)

    allowed_delta = {
        "scripts/predict_run.py": "added clean-fold fit/evaluation selectors, full-valid feature-only scorer-aligned student export, and only records a fit-subset digest when that manifest is supplied",
        "src/wnn/distillation_cache.py": "added narrow homogeneous strong-T3 clean-fold cache profile validation",
        "scripts/build_cleanfold_t3_cache.py": "new own-source fold-0 cache builder; train/valid fit membership only",
        "scripts/cleanfold_exec.py": "new source-snapshot child runner with fixed import root",
    }
    source_deltas = []
    old_by_path = {name: digest for name, digest in old_sources.items()}
    for name, digest in sorted(current_sources.items()):
        if name in allowed_delta:
            source_deltas.append({"path": name, "prior_h390_sha256": old_by_path.get(name), "current_sha256": digest, "reason": allowed_delta[name]})
        elif old_by_path.get(name) != digest:
            raise ValueError(f"unexpected H390 source drift at {name}")
    if set(allowed_delta) - set(current_sources):
        raise ValueError("an allowed source delta is absent from the source snapshot")

    input_manifest = json.loads((PACKET / "reference_inputs/h382_fold0_asset_manifest.json").read_text())
    portable = json.loads((PACKET / "reference_inputs/portable_inputs_fold0.json").read_text())
    split = json.loads((PACKET / "reference_inputs/fold0.json").read_text())
    f1_reference_provenance_path = evidence_root / "artifacts/runs/vt_f1_dsq_dm_t3_a9_e6/provenance.json"
    f1_reference_cache_path = evidence_root / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/h391_f1_h384_t3_student_r1/actual_producer_receipts_r1/dc_98411230_metadata.json"
    f1_reference_provenance = json.loads(f1_reference_provenance_path.read_text())
    f1_reference_cache = json.loads(f1_reference_cache_path.read_text())
    f1_distillation_source = evidence_root / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/h391_f1_h384_t3_student_r1/child_source_v1/src/wnn/distillation.py"
    packet_distillation_source = PACKET / "source_snapshot/src/wnn/distillation.py"
    expected_f1_coefficients = [
        ("strong", "vt_f1_muonall_h384_e3", 0.1),
        ("strong", "vt_f1_muonall_h384_e3_s4", 0.1),
        ("strong", "vt_f1_muonall_h384_e3_s5", 0.1),
        ("full_finetuned", "vt_f1_muonall_h384_e3_a1", 0.2),
        ("raw_wmse", "vt_f1_muonall_h384_e3_s5_a2", 0.25),
        ("samefold_baseline", "vt_f1_muonall_h384_e3_a3", 0.0625),
    ]
    observed_f1_coefficients = [
        (row.get("family"), row.get("run"), float(row.get("coefficient", -1)))
        for row in f1_reference_cache.get("source_coefficients", [])
    ]
    f1_selected_cache = f1_reference_provenance.get("distillation_cache", {})
    if (
        sha256(f1_reference_cache_path) != "a2f82d8dbc7bbbe2c7cf36e18e3b7be0b41f887aa14a742b43f616b060856779"
        or observed_f1_coefficients != expected_f1_coefficients
        or f1_reference_cache.get("family_order") != ["strong", "full_finetuned", "raw_wmse", "samefold_baseline"]
        or f1_reference_cache.get("homogeneous_family") != "strong"
        or abs(float(f1_reference_cache.get("coefficient_sum", 0)) - 0.8125) > 1e-8
        or f1_selected_cache.get("variant") != "homogeneous"
        or f1_selected_cache.get("homogeneous_family") != "strong"
        or f1_selected_cache.get("metadata_sha256") != "a2f82d8dbc7bbbe2c7cf36e18e3b7be0b41f887aa14a742b43f616b060856779"
        or f1_reference_provenance.get("args", {}).get("distill_alpha") != 0.9
        or sha256(f1_distillation_source) != sha256(packet_distillation_source)
    ):
        raise ValueError("F1 H390 homogeneous strong-T3 reference metadata or effective student recipe differs")
    if sha256(PACKET / "reference_inputs/fold0.json") != SPLIT_SHA:
        raise ValueError("fold-0 split digest mismatch")
    if portable.get("split_manifest_sha256") != SPLIT_SHA:
        raise ValueError("portable registry is not bound to fold-0 split")

    needed = {
        "artifacts/runs/c20260929_prop_f0/model.pt",
        "cache/chain_folds.npz",
        "cache/chains.npz",
        "cache/propcache_c20260929_prop_f0_train.json",
        "cache/propcache_c20260929_prop_f0_train.npy",
        "cache/propcache_c20260929_prop_f0_valid.json",
        "cache/propcache_c20260929_prop_f0_valid.npy",
        "cache/train_targets.npz",
        "cache/valid_targets.npz",
        FOLD0,
        PORTABLE,
    }
    asset_by_path = {item["path"]: item for item in input_manifest["files"]}
    if needed - set(asset_by_path):
        raise ValueError(f"fold-0 asset manifest misses dependencies: {sorted(needed - set(asset_by_path))}")

    inputs = []
    external: list[str] = []
    for name in sorted(needed):
        item = asset_by_path[name]
        inputs.append(external_descriptor(name, item["filename"], item["size"], item["sha256"]))
        external.append(f"{name}={FOLD0_ASSETS}:{item['filename']}")
    for name in ("datasets/train.parquet", "datasets/valid.parquet"):
        identity = portable["sources"][name]
        inputs.append(external_descriptor(name, name.rsplit("/", 1)[1], identity["original_identity"]["bytes"], identity["sha256"]))
        external.append(f"{name}={name.rsplit('/', 1)[1]}")
    inputs.sort(key=lambda item: item["path"])

    source_files = [source_manifest_path] + sorted((PACKET / "source_snapshot").rglob("*.py"))
    source_files = [p for p in source_files if "__pycache__" not in p.parts]
    code_names = {p.relative_to(root).as_posix() for p in source_files}
    code_names.update({"scripts/kaggle_common.py", "scripts/kaggle_gpu_runner.py", "scripts/kaggle_kernel.py"})
    code = [descriptor(root, name) for name in sorted(code_names)]
    sys.path.insert(0, str(root / "scripts"))
    from kaggle_common import canonical_sha256  # noqa: PLC0415

    code_sha = canonical_sha256(sorted(code, key=lambda item: item["path"]))

    teachers = []
    for seed in (3, 4, 5):
        name = f"clean_t3_f0_muonall_h384_e3_s{seed}_r1"
        argv = [
            "{python}", f"{REL_PACKET.as_posix()}/source_snapshot/scripts/cleanfold_exec.py", "scripts/train_rnn.py",
            "--name", name, "--seed", str(seed),
            "--device", "cuda", "--arch", "gru", "--cell", "gru", "--hidden", "384", "--layers", "2", "--dropout", "0",
            "--train-source", "train+valid", "--train-seqs", "10607", "--valid-seqs", "600", "--valid-fold", "0",
            "--no-eval-during-fit", "--split-manifest", FOLD0,
            "--epochs", "3", "--checkpoint-policy", "fixed", "--checkpoint-epoch", "2", "--save-training-state",
            "--batch", "32", "--chunk", "1000", "--loss", "wmse", "--loss-norm", "chunk",
            "--lr", "0.001", "--wd", "0.01", "--optim", "muon", "--muon-scope", "all", "--sched", "onecycle", "--precision", "fp16",
            "--propensity", "c20260929_prop_f0", "--prop-power", "2", "--prop-eps", "0.03", "--prop-cache",
            "--valid-true-mask", "0.03", "--aux-struct", "0.3", "--aux-next", "0.3", "--since-weight", "32", "--since-floor", "0.25",
            "--fast", "--sync-free", "--fused-adam", "--pin-memory",
        ]
        spec = {
            "version": 1,
            "name": name,
            "kind": "gpu",
            "argv": argv,
            "code": code,
            "code_sha256": code_sha,
            "inputs": inputs,
            "artifacts": [
                f"artifacts/runs/{name}/model_ep2.pt",
                f"artifacts/runs/{name}/model.pt",
                f"artifacts/runs/{name}/provenance.json",
                f"artifacts/runs/{name}/history.json",
            ],
            "limits": {"cpus": 2, "ram_bytes": 16000000000, "gpu_memory_bytes": 10000000000, "gpu_stop_bytes": 12000000000, "timeout_seconds": 5200},
            "environment": {
                "WNN_PROJECT_ROOT": "/kaggle/working/experiment",
                "WNN_PORTABLE_DATA_MANIFEST": f"/kaggle/working/experiment/{PORTABLE}",
                "WNN_DEFERRED_IDENTITY_PATHS": "datasets/train.parquet,datasets/valid.parquet",
                "WNN_LOADER_WORKERS": "0",
                "WNN_LOADER_RAM_GB": "4",
                "PYTHONHASHSEED": "0",
            },
        }
        spec["artifacts"].append(f"artifacts/runs/{name}/training_state.pt")
        teachers.append(spec)
        write_json(PACKET / "teacher_fit_templates" / f"{name}.json", spec)

    profile_specs = []
    for slot in (1, 2):
        profile_name = f"clean_t3_f0_muonall_h384_e3_s3_t4_profile{slot}_r1"
        profile_argv = list(teachers[0]["argv"])
        profile_argv[profile_argv.index("--name") + 1] = profile_name
        profile_argv.remove("--save-training-state")
        profile_argv.extend(["--profile-warmup", "5", "--profile-steps", "25"])
        profile_spec = {
            **teachers[0],
            "name": profile_name,
            "argv": profile_argv,
            "artifacts": [],
            "limits": {
                "cpus": 2,
                "ram_bytes": 8000000000,
                "gpu_memory_bytes": 10000000000,
                "gpu_stop_bytes": 12000000000,
                "timeout_seconds": 240,
            },
        }
        profile_specs.append(profile_spec)
        write_json(PACKET / "teacher_profile_specs" / f"{profile_name}.json", profile_spec)

    recipe = {
        "schema": "wnn.cleanfold_t3_f0_recipe.r1",
        "status": "frozen_recipe_unadmitted",
        "source_reference": "own F1 provenance vt_f1_muonall_h384_e3/{_s4,_s5}; only fold/name/seed/own-fold propensity bindings changed",
        "fold": 0,
        "split_manifest": {"path": FOLD0, "sha256": SPLIT_SHA, "train_groups": 10535, "valid_fit_groups": 1405, "evaluation_valid_groups": 468},
        "ordered_fit_membership_sha256": {"train": "98a3e8da248f52e307af61efaeab845a76cf8e89e005ea5630c0af9a133508a9", "valid": "0ef790f8c27723b8f0ca1ebe045c2dcbebb4c4531a17952985802382c9568d75"},
        "teacher_recipe": {"architecture": "GRU(384, layers=2)", "seeds": [3, 4, 5], "epochs": 3, "fixed_checkpoint_epoch": 2, "optimizer": "Muon, scope=all", "schedule": "OneCycle over the full 3 epochs", "precision": "fp16", "batch": 32, "chunk": 1000, "loss": "wMSE/chunk", "lr": 0.001, "weight_decay": 0.01, "propensity_checkpoint_sha256": PROP_SHA, "propensity_power": 2, "propensity_epsilon": 0.03, "prop_cache": True, "valid_true_mask": 0.03, "aux_struct": 0.3, "aux_next": 0.3, "since_weight": 32, "since_floor": 0.25, "eval_during_fit": False, "initializer": None},
        "teacher_parent_reference": {
            str(seed): {
                "provenance_path": f"reference_receipts/vt_f1_muonall_h384_e3{'_s'+str(seed) if seed != 3 else ''}.provenance.json",
                "provenance_sha256": sha256(PACKET / f"reference_receipts/vt_f1_muonall_h384_e3{'_s'+str(seed) if seed != 3 else ''}.provenance.json"),
                "epoch2_checkpoint_sha256": json.loads((PACKET / f"reference_receipts/vt_f1_muonall_h384_e3{'_s'+str(seed) if seed != 3 else ''}.provenance.json").read_text())["checkpoint_sha256"]["model_ep2.pt"],
                "historical_fit_seconds_f1_reference_only": json.loads((PACKET / f"reference_receipts/vt_f1_muonall_h384_e3{'_s'+str(seed) if seed != 3 else ''}.provenance.json").read_text())["fitting_seconds"],
                "allowed_recipe_delta": ["run name", "seed for s4/s5", "fold 1 to fold 0", "own propensity F1 to F0 and its hash"],
            } for seed in (3, 4, 5)
        },
        "f1_homogeneous_reference": {
            "student_run": "vt_f1_dsq_dm_t3_a9_e6",
            "student_provenance_path": "artifacts/runs/vt_f1_dsq_dm_t3_a9_e6/provenance.json",
            "student_provenance_sha256": sha256(f1_reference_provenance_path),
            "actual_cache_metadata_path": "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/h391_f1_h384_t3_student_r1/actual_producer_receipts_r1/dc_98411230_metadata.json",
            "actual_cache_metadata_sha256": "a2f82d8dbc7bbbe2c7cf36e18e3b7be0b41f887aa14a742b43f616b060856779",
            "homogeneous_operation_source_path": "source_snapshot/src/wnn/distillation.py::combine_teacher_chunk",
            "homogeneous_operation_source_sha256": sha256(packet_distillation_source),
            "f1_reference_operation_source_path": "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/h391_f1_h384_t3_student_r1/child_source_v1/src/wnn/distillation.py",
            "f1_reference_operation_source_sha256": sha256(f1_distillation_source),
            "family_order": ["strong", "full_finetuned", "raw_wmse", "samefold_baseline"],
            "source_coefficients": [
                {"family": "strong", "run": "vt_f1_muonall_h384_e3", "epoch": 2, "coefficient": 0.1},
                {"family": "strong", "run": "vt_f1_muonall_h384_e3_s4", "epoch": 2, "coefficient": 0.1},
                {"family": "strong", "run": "vt_f1_muonall_h384_e3_s5", "epoch": 2, "coefficient": 0.1},
                {"family": "full_finetuned", "run": "vt_f1_muonall_h384_e3_a1", "epoch": 2, "coefficient": 0.2},
                {"family": "raw_wmse", "run": "vt_f1_muonall_h384_e3_s5_a2", "epoch": 2, "coefficient": 0.25},
                {"family": "samefold_baseline", "run": "vt_f1_muonall_h384_e3_a3", "epoch": 2, "coefficient": 0.0625}
            ],
            "selected_variant": "homogeneous",
            "homogeneous_family": "strong",
            "operation_order": "np.add.reduce(three strong epoch-2 float32 predictions, dtype=float32) / float32(3), then multiply by float32(0.8125); no clipping during cache construction",
            "interpretation": "The six-source metadata vector totals 0.8125; homogeneous selects only the unweighted strong-family mean and scales it by that total. Fold-0 cache is an exact reduced representation of this selected output, not the six-source main blend.",
            "effective_fit_args_from_provenance": f1_reference_provenance["args"]
        },
        "cache_target": {"profile": "cleanfold_homogeneous_strong_t3_v1", "formula": "float32(0.8125 * float32_mean(three epoch-2 predictions))", "strong_family_source_coefficients": [0.1, 0.1, 0.1], "strong_family_coefficient_subtotal": 0.3, "homogeneous_scale": 0.8125, "student_alpha": 0.9, "training_rows_only": True, "evaluation_predictions_written": 0, "estimated_float32_prediction_target_bytes": 1910400000},
        "student_recipe": {"architecture": "GRU(256, layers=2)", "seed": 3, "epochs": 6, "checkpoint_epoch": 5, "schedule": "single OneCycle across all six epochs; exact training_state/RNG/optimizer/scheduler continuation", "batch": 16, "chunk": 2000, "optimizer": "AdamW", "lr": 0.0001, "weight_decay": 0.01, "precision": "fp16", "loss": "wMSE/chunk", "distill_alpha": 0.9, "propensity_checkpoint_sha256": PROP_SHA, "propensity_power": 2, "propensity_epsilon": 0.03, "prop_cache": True, "valid_true_mask": 0.03, "aux_struct": 0.3, "aux_next": 0.3, "since_weight": 0, "normalization": "fit on clean-fold training rows; init preprocessing is not inherited", "heldout_evaluation": "feature-only prediction over all 1,873 valid groups in valid_targets row-group order; the frozen scorer applies its fixed 468-group fold-0 mask"},
        "student_parent_arms": [
            {"arm": "control", "run": "c20261003_combo_s3_f0", "epoch": 3, "checkpoint_sha256": "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e", "provenance_sha256": "9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f", "checkpoint_bytes": 3977183, "propensity_power": 2, "role": "same-method own combo initializer"},
            {"arm": "treatment", "run": "kg_h382r3_f0_muonall_prop0_e4_s3", "epoch": 3, "checkpoint_sha256": "359d337db09f0f8e306a225b050884741f006190523b2e940d36ef87d35a25be", "provenance_sha256": "b220b6e822c9e4779d508655c56ecacb4c42399a001a817d30de8fd3bd2610b7", "checkpoint_bytes": 3978087, "propensity_power": 0, "role": "collected own H382 family-treatment initializer; source R4 rescue dataset v1"},
        ],
        "recipe_fidelity_vs_weight_identity": "Recipe-level replication on fold 0; regenerated teacher weights will not be byte-identical to F1. Existing parent checkpoint identities are pinned and are not to be replaced.",
    }
    write_json(PACKET / "CLEANFOLD_T3_F0_RECIPE_R1.json", recipe)

    bindings = {
        "schema": "wnn.cleanfold_t3_f0_source_bindings.r1",
        "base_commit": "e4f044e56eed0224d9273b9bbd43241561ec36c7",
        "source_manifest_path": "source_snapshot/SOURCE_MANIFEST_H390.json",
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_snapshot_sha256_by_path": current_sources,
        "allowed_deltas_from_h390": source_deltas,
        "packet_builder_sha256": sha256(PACKET / "prepare_packet_r1.py"),
        "code_package": {"worker_sources": len(code), "paths": [item["path"] for item in code], "aggregate_sha256": code_sha},
        "data_manifest_sha256": sha256(PACKET / "reference_inputs/h382_fold0_asset_manifest.json"),
        "portable_input_registry_sha256": sha256(PACKET / "reference_inputs/portable_inputs_fold0.json"),
        "split_sha256": SPLIT_SHA,
        "teacher_spec_sha256": {},
        "teacher_profile_spec_sha256": {},
        "fit_bundle_gate": "Full teacher templates are not in the current bundle; future fit bundle requires exact-profile receipt hashes and root reseal.",
    }
    for spec in teachers:
        path = PACKET / "teacher_fit_templates" / f"{spec['name']}.json"
        bindings["teacher_spec_sha256"][spec["name"]] = sha256(path)
    for spec in profile_specs:
        path = PACKET / "teacher_profile_specs" / f"{spec['name']}.json"
        bindings["teacher_profile_spec_sha256"][spec["name"]] = sha256(path)
    write_json(PACKET / "SOURCE_BINDINGS_CLEAN_T3_F0_R1.json", bindings)

    args_list: list[str] = [str(root / "scripts/kaggle_bundle.py"), "--root", str(root)]
    for spec in profile_specs:
        args_list += ["--spec", str(PACKET / "teacher_profile_specs" / f"{spec['name']}.json")]
    args_list += ["--destination", str(args.bundle_destination.resolve()), "--dataset", DATASET, "--asset-dataset", FOLD0_ASSETS]
    for item in external:
        args_list += ["--external", item]
    write_json(PACKET / "TEACHER_PROFILE_BUNDLE_BUILD_COMMAND_R1.json", {"argv": [sys.executable, *args_list], "status": "profile_only_bundle_prepared_not_provider_uploaded", "bundle_dataset_id": DATASET, "included_jobs": [spec["name"] for spec in profile_specs], "long_teacher_fits_included": False, "historical_asset_remote_version": "unverified_read_only_gate"})
    completed = subprocess.run([sys.executable, *args_list], check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)

    parent_stage = args.student_parent_stage.resolve()
    parent_items = []
    parent_sources = {
        "control": {
            "run": "c20261003_combo_s3_f0",
            "source_paths": [
                evidence_root / "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt",
                evidence_root / "artifacts/runs/c20261003_combo_s3_f0/provenance.json",
            ],
            "staged_paths": [parent_stage / "control/model_ep3.pt", parent_stage / "control/provenance.json"],
            "expected_checkpoint_sha256": "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e",
            "expected_provenance_sha256": "9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f",
        },
        "treatment": {
            "run": "kg_h382r3_f0_muonall_prop0_e4_s3",
            "source_paths": [
                evidence_root / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/kaggle_h382_delivery_20261011_r4/provider_output/out/runs/kg_h382r3_f0_muonall_prop0_e4_s3/model_ep3.pt",
                evidence_root / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/kaggle_h382_delivery_20261011_r4/provider_output/out/runs/kg_h382r3_f0_muonall_prop0_e4_s3/provenance.json",
            ],
            "staged_paths": [parent_stage / "treatment/model_ep3.pt", parent_stage / "treatment/provenance.json"],
            "expected_checkpoint_sha256": "359d337db09f0f8e306a225b050884741f006190523b2e940d36ef87d35a25be",
            "expected_provenance_sha256": "b220b6e822c9e4779d508655c56ecacb4c42399a001a817d30de8fd3bd2610b7",
        },
    }
    for arm, record in parent_sources.items():
        if not all(path.is_file() for path in record["staged_paths"]):
            raise FileNotFoundError(f"staged {arm} student parent is absent: {parent_stage}")
        source_model, source_provenance = record["source_paths"]
        staged_model, staged_provenance = record["staged_paths"]
        observed = {
            "checkpoint_sha256": sha256(staged_model),
            "provenance_sha256": sha256(staged_provenance),
            "checkpoint_bytes": staged_model.stat().st_size,
            "provenance_bytes": staged_provenance.stat().st_size,
            "source_checkpoint_sha256": sha256(source_model),
            "source_provenance_sha256": sha256(source_provenance),
        }
        if (observed["checkpoint_sha256"] != record["expected_checkpoint_sha256"]
                or observed["provenance_sha256"] != record["expected_provenance_sha256"]
                or observed["source_checkpoint_sha256"] != observed["checkpoint_sha256"]
                or observed["source_provenance_sha256"] != observed["provenance_sha256"]):
            raise ValueError(f"staged {arm} parent bytes do not match the bound own checkpoint: {observed}")
        parent_items.append({
            "arm": arm,
            "run": record["run"],
            "epoch": 3,
            "staged_directory": str(parent_stage),
            "files": [
                {"path": str(staged_model), "filename": f"{record['run']}_model_ep3.pt", "bytes": observed["checkpoint_bytes"], "sha256": observed["checkpoint_sha256"]},
                {"path": str(staged_provenance), "filename": f"{record['run']}_provenance.json", "bytes": observed["provenance_bytes"], "sha256": observed["provenance_sha256"]},
            ],
            "source_files": [str(source_model), str(source_provenance)],
            "source_version": "own verified producer output; no teammate weights",
        })
    parent_manifest = {
        "schema": "wnn.cleanfold_t3_student_parent_assets.r1",
        "status": "locally_staged_private_asset_no_upload",
        "future_private_dataset_id": "vladkochetov07/wnn-clean-t3-f0-student-parents-20261011-r1",
        "remote_version": "not_created_or_uploaded",
        "parents": parent_items,
        "consumer_contract": "both exact epoch-3 own F0 GRU256x2 dense float checkpoints; student --init and train_rnn.validate_campaign_checkpoint; never replace by a different same-name run",
    }
    write_json(PACKET / "STUDENT_PARENT_ASSET_R1.json", parent_manifest)

    student_common = [
        "--seed", "3", "--device", "cuda", "--arch", "gru", "--cell", "gru", "--hidden", "256", "--layers", "2", "--dropout", "0",
        "--train-source", "train+valid", "--train-seqs", "10607", "--valid-seqs", "600", "--valid-fold", "0",
        "--no-eval-during-fit", "--split-manifest", FOLD0,
        "--epochs", "6", "--checkpoint-policy", "fixed", "--checkpoint-epoch", "5", "--save-training-state",
        "--batch", "16", "--chunk", "2000", "--loss", "wmse", "--loss-norm", "chunk",
        "--lr", "0.0001", "--wd", "0.01", "--optim", "adamw", "--sched", "onecycle", "--precision", "fp16",
        "--propensity", "c20260929_prop_f0", "--prop-power", "2", "--prop-eps", "0.03", "--prop-cache",
        "--valid-true-mask", "0.03", "--aux-struct", "0.3", "--aux-next", "0.3", "--since-weight", "0", "--since-floor", "0.25",
        "--distill-cache-root", "cache/clean_t3_f0_h390_r1", "--distill-cache-variant", "homogeneous", "--distill-alpha", "0.9",
        "--fast", "--sync-free", "--fused-adam", "--pin-memory",
    ]
    student_stages = []
    for arm, parent_name in (("control", "c20261003_combo_s3_f0"), ("treatment", "kg_h382r3_f0_muonall_prop0_e4_s3")):
        names = [f"clean_t3_f0_{arm}_s3_a9_e6_r1_p{part}" for part in (1, 2, 3)]
        for part, stop_after in ((1, 1), (2, 3), (3, 5)):
            name = names[part - 1]
            argv = ["{python}", f"{REL_PACKET.as_posix()}/source_snapshot/scripts/cleanfold_exec.py", "scripts/train_rnn.py", "--name", name, *student_common, "--init", f"{parent_name}@ep3", "--stop-after-epoch", str(stop_after)]
            depends = ["teacher_cache_complete"]
            if part > 1:
                argv += ["--resume-state", f"/kaggle/working/experiment/artifacts/runs/{names[part - 2]}/training_state.pt"]
                depends.append(names[part - 2] + ":training_state.pt")
            artifacts = [f"artifacts/runs/{name}/training_state.pt", f"artifacts/runs/{name}/provenance.json", f"artifacts/runs/{name}/history.json"]
            if part == 3:
                artifacts += [f"artifacts/runs/{name}/model_ep5.pt", f"artifacts/runs/{name}/model.pt"]
            student_stages.append({"arm": arm, "part": part, "name": name, "epochs": 6, "stop_after_epoch_inclusive": stop_after, "schedule_continuity": "same six-epoch OneCycle; resume full optimizer/scheduler/scaler/model/loader/RNG state", "depends_on": depends, "argv": argv, "artifacts": artifacts, "boundary_contract": "stop_after_epoch is a zero-based completed-epoch index; resumptions keep --epochs 6 and --checkpoint-epoch 5 unchanged"})
    student_exports = []
    for arm in ("control", "treatment"):
        name = f"clean_t3_f0_{arm}_s3_a9_e6_r1_p3"
        argv = ["{python}", f"{REL_PACKET.as_posix()}/source_snapshot/scripts/cleanfold_exec.py", "scripts/predict_run.py", name, "valid", "--epoch", "5", "--campaign-split-manifest", FOLD0, "--campaign-full-valid-feature-only", "--inference-precision", "fp32"]
        student_exports.append({"arm": arm, "name": f"pred_{name}_valid", "depends_on": [f"{name}:model_ep5.pt", f"{name}:provenance.json"], "argv": argv, "artifacts": [f"cache/pred_{name}_valid.npy", f"cache/pred_{name}_valid.json"], "expected_shape": [1873, 20000, 2], "dtype": "float32", "row_order": "valid_targets.npz row_group order", "feature_only": True, "grader_scope": "the fixed source-qualified fold-0 evaluation mask selects its 468 evaluation groups; no labels are loaded by prediction"})
    write_json(PACKET / "STUDENT_STAGE_TEMPLATES_R1.json", {
        "schema": "wnn.cleanfold_t3_student_stages.r1",
        "status": "command_frozen_dynamic_state_hashes_pending",
        "student_common_recipe_argv_without_name_parent_resume_boundary": student_common,
        "stages": student_stages,
        "final_student_exports": student_exports,
        "training_state_fields": ["model", "optimizer", "scheduler", "scaler", "EMA/SWA", "loader", "auxiliary state", "progress/history", "Python/NumPy/Torch/CUDA RNG"],
        "resume_contract_source": "source_snapshot/scripts/train_rnn.py::resume_contract and source_snapshot/src/wnn/training_state.py::capture_training_state/restore_training_state",
        "no_independent_short_schedules": True,
    })

    exports = []
    for seed in (3, 4, 5):
        name = f"clean_t3_f0_muonall_h384_e3_s{seed}_r1"
        for source in ("train", "valid"):
            argv = ["{python}", f"{REL_PACKET.as_posix()}/source_snapshot/scripts/cleanfold_exec.py", "scripts/predict_run.py", name, source, "--epoch", "2", "--campaign-split-manifest", FOLD0, "--inference-precision", "fp32"]
            exports.append({"name": f"pred_{name}_{source}", "depends_on": [f"{name}:model_ep2.pt", f"{name}:provenance.json"], "argv": argv, "artifacts": [f"cache/pred_{name}_{source}.npy", f"cache/pred_{name}_{source}.json"], "expected_scope": f"campaign split {source} fitting members only; feature-only dataset path"})
    cache_step = {"name": "build_clean_t3_f0_h390_r1", "depends_on": [f"pred_clean_t3_f0_muonall_h384_e3_s{seed}_r1_{source}" for seed in (3, 4, 5) for source in ("train", "valid")], "argv": ["{python}", f"{REL_PACKET.as_posix()}/source_snapshot/scripts/cleanfold_exec.py", "scripts/build_cleanfold_t3_cache.py", "--output", "cache/clean_t3_f0_h390_r1", "--chunk-sequences", "4", "--min-free-gb", "20", "--output-budget-gb", "2.5"], "artifacts": ["cache/clean_t3_f0_h390_r1/metadata.json", "cache/clean_t3_f0_h390_r1/main_homogeneous_train.npy", "cache/clean_t3_f0_h390_r1/main_homogeneous_valid.npy"]}
    write_json(PACKET / "TEACHER_EXPORT_AND_CACHE_QUEUE_R1.json", {"schema": "wnn.cleanfold_t3_teacher_exports_cache.r1", "status": "commands_frozen_dynamic_output_hashes_pending", "prediction_jobs": exports, "cache_build": cache_step, "gate": "No cache/student fit until all six teacher exports match source-bound F0 fitting memberships and the builder validates their source checkpoint, provenance, propensity and split hashes."})

    queue = {
        "schema": "wnn.cleanfold_t3_f0_stage_inventory.r1",
        "status": "profile_only_bundle_sealed; full-fit bundle withheld pending exact device profiles and root reseal/admission",
        "completed_counts": {"exact_f0_z4_t3_teachers": 0, "teacher_fit_templates": 3, "teacher_profile_specs": 2, "teacher_profile_bundle_built_offline": 1, "student_parent_checkpoints_staged": 2, "student_parent_pairs_reviewed_by_root": 0, "prediction_exports": 0, "target_caches": 0, "student_arm_fits": 0, "student_evaluation_exports": 0},
        "required_work_counts": {"teacher_profiles": 2, "teacher_fits": 3, "teacher_prediction_exports": 6, "one_train_only_cache_build": 1, "student_fits": 6, "student_final_exports": 2},
        "cohorts": [
            {"id": "teacher-device-profile", "workers": [s["name"] for s in profile_specs], "fit_count": 0, "concurrency": "two separate profile workers exercise both available T4s when runner inventory supports it", "profile_steps": 25, "profile_warmup": 5, "per_worker_timeout_seconds": 240, "required_receipt": ["T4 model/UUID", "actual worker argv/source bundle hashes", "25 timed updates completed after five warmups", "peak GPU bytes", "peak RSS bytes", "median batches/sec", "no OOM or resource stop"], "gate": "Required receipt must be hash-bound into a newly generated fit bundle before any full fit spec is executable; current archive contains profiles only."},
            {"id": "teacher-fits", "workers": [s["name"] for s in teachers], "fit_count": 3, "concurrency": "future separate bundle; omitted from current profile-only archive", "per_worker_timeout_seconds": 5200, "observed_reference_fit_seconds_f1_only": {str(seed): recipe["teacher_parent_reference"][str(seed)]["historical_fit_seconds_f1_reference_only"] for seed in (3, 4, 5)}, "additional_prediction_export_time": "unmeasured", "profile_gate": "not met; root must bind both actual T4 profile receipts, reassess peak memory and fit timing, then reseal templates before release"},
            {"id": "teacher-exports-and-cache", "workers": 7, "fit_count": 0, "exports": 6, "cache_builds": 1, "observed_runtime": "unmeasured", "dependency": "three collected epoch-2 teacher checkpoints and provenance"},
            {"id": "student-p1", "workers": [student_stages[0]["name"], student_stages[3]["name"]], "timeout_seconds_per_worker": 5200, "boundary_epoch": 1, "actual_profile": "required before root admits this cohort"},
            {"id": "student-p2", "workers": [student_stages[1]["name"], student_stages[4]["name"]], "timeout_seconds_per_worker": 5200, "boundary_epoch": 3, "actual_profile": "required before root admits this cohort", "dependency": "exact training_state.pt from same arm p1 and all unchanged parent/cache/data bindings"},
            {"id": "student-p3", "workers": [student_stages[2]["name"], student_stages[5]["name"]], "timeout_seconds_per_worker": 5200, "boundary_epoch": 5, "actual_profile": "required before root admits this cohort", "dependency": "exact training_state.pt from same arm p2"},
        ],
        "historical_fit_cost_reference": {"teachers_f1_seconds_total": 8508.7, "one_student_f1_seconds": 9619.7, "two_student_f1_seconds_total": 19239.4, "fit_only_total_seconds": 27748.1, "status": "F1 historical recipe fit durations only; not F0 T4 profile or full-pipeline ETA; excludes all exports, cache build, provider startup, transfers, profiles and retries"},
        "hard_blockers": ["two actual T4 profile receipts for H384/B32/C1000 with source/input identity and both devices recorded; this profile-only archive has no full-fit job", "read-only verify exact current remote versions/READY status for raw training data, fold0 assets and R4 rescue dataset; local receipts pin bytes but fold0 remote version is unverified", "root-owned fresh quota/cohort reservations and admission after profile; a future fit bundle must bind the receipt hashes", "actual H256 B16/C2000 student profile and schedule-continuation profile before admitting student cohorts", "all postfit checkpoint, prediction, target-cache and resume-state digests are naturally unknown until their producers complete", "fold1 H382 treatment counterpart is absent, so this F0 cohort is a screen and cannot close crossfold qualification alone"],
        "not_blocked_by": ["historical seed-99 identity; this recipe comparison uses explicit seed 3", "missing source: exact own H384 F1 teacher sources and H256 F1 student receipt are present", "student parents: both exact own F0 checkpoint/provenance pairs are locally staged and content-bound"],
    }
    write_json(PACKET / "STAGE_QUEUE_R1.json", queue)

    profile_gate = {
        "schema": "wnn.cleanfold_t3_t4_profile_gate.r1",
        "status": "pending_actual_device_profiles",
        "profile_bundle_archive_sha256": sha256(args.bundle_destination.resolve() / "bundle.wnn"),
        "profile_spec_sha256": {spec["name"]: sha256(PACKET / "teacher_profile_specs" / f"{spec['name']}.json") for spec in profile_specs},
        "full_fit_specs_in_profile_bundle": False,
        "profile_contract": {"architecture": "GRU(384x2)", "batch": 32, "chunk": 1000, "seed": 3, "warmup_steps": 5, "timed_steps": 25, "timeout_seconds": 240, "save_training_state": False, "loader_workers": 0},
        "pass_requirements": ["both profile specs exit 0", "both emit a PROFILE record after 25 measured batches and 5 warmups", "both resolve to actual T4 devices and record device UUID/model", "peak GPU memory remains below each spec's 12,000,000,000-byte stop bound", "no OOM/resource stop and RSS remains below each spec's 8,000,000,000-byte cap", "measured throughput and full recipe reference timings support the proposed 5,200-second per-fit bound; otherwise the root must choose a different bounded cohort contract"],
        "required_future_fit_reseal": ["bind both profile terminal result/log hashes", "bind observed device identities and peak GPU/RSS measurements", "bind source bundle and input identity receipts", "reassess timeout from actual T4 profile plus recorded F1 fit times", "generate a new bundle containing the three full teacher fits and training_state.pt outputs", "root-owned current quota, reservation and admission receipt"],
        "no_profile_means_no_fit": True,
    }
    write_json(PACKET / "PROFILE_GATE_R1.json", profile_gate)

    readme = f"""# Clean-fold T3 F0 parent-interaction packet R1\n\nThis packet freezes the minimum useful H390 parent interaction: three independently trained own H384 Muon-all strong-family teachers, the exact fold-0 fitting-only homogeneous target, then paired same-seed H256 students initialized from the existing clean F0 combo control or collected own H382 prop0 parent. F0 is a screen; a matched fold-1 counterpart is still required before comparative promotion.\n\nThe executable `{DATASET}` archive in `bundle_teacher_profile_r1/` contains two 25-step T4 profile jobs only. It cannot start a full teacher fit. The three full-fit source specs are retained separately as non-executable templates under `teacher_fit_templates/`, and request `training_state.pt` along with both fixed-epoch-2 checkpoints and provenance. `PROFILE_GATE_R1.json` specifies exact terminal receipts and requires a newly generated, root-reviewed fit bundle to bind both actual T4 profile receipts. No upload, provider call, profile, model fit, queue mutation or quota mutation occurred during packet preparation.\n\nThe source entrypoint is `research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r1/source_snapshot/scripts/cleanfold_exec.py` inside the archive; each spec calls that exact path. It fixes `PYTHONPATH` to this snapshot's `src/` before importing the frozen trainer.\n\n`CLEANFOLD_T3_F0_RECIPE_R1.json` pins the F1 reference student's effective arguments and actual cache metadata SHA. That six-source metadata vector totals .8125, while the selected `homogeneous` target is exactly `.8125 * float32_mean(three strong-family epoch-2 predictions)`. The F0 cache preserves the same float32 reduction/division-then-multiply order and records the original .1 coefficient per strong teacher separately from the .8125 homogeneous scale.\n\n`TEACHER_EXPORT_AND_CACHE_QUEUE_R1.json` freezes six feature-only train/valid exports restricted to fitting members and the training-only cache. `STUDENT_STAGE_TEMPLATES_R1.json` freezes six-epoch OneCycle continuation at completed zero-based epochs 1, 3 and 5; all segments retain `--epochs 6` and resume full optimizer/scheduler/scaler/model/loader/RNG state. The two final student exports read features only for all 1,873 valid groups, preserve valid_targets row-group order, and emit (1873, 20000, 2) float32 predictions; the frozen scorer applies its fixed 468-group fold-0 evaluation mask. Student cohorts still require actual B16/C2000 device/continuation profiles, output hashes and root reseal.\n\nThe fold-0 fit uses 10,535 train groups plus 1,405 valid fit groups; the 468 valid evaluation groups are excluded from teacher fitting, cache construction and student fitting. Both exact own F0 student parent checkpoint/provenance pairs are byte-pinned in `STUDENT_PARENT_ASSET_R1.json` and staged locally at `{parent_stage}`; neither has been uploaded. The H382 treatment parent retains its own propensity-zero ancestry; the H390 student refits with F0 propensity power 2.\n\nThe measured F1 reference teacher fits total 8,508.7 GPU worker-seconds, and the existing F1 student fit is 9,619.7 seconds. Those are historical timings, not F0 T4 throughput. The full proposed fit-only sequence totals 27,748.1 worker-seconds before exports, cache construction, profiles, startup, transfers and retries; this packet makes no 10,800-second completion claim. Root owns live quota, profile review and cohort admission. Exact remaining blockers and work counts are in `STAGE_QUEUE_R1.json`.\n"""
    (PACKET / "README_R1.md").write_text(readme)

    # Bind the final protocol documents after construction; dynamic output hashes stay explicit nulls.
    bindings = json.loads((PACKET / "SOURCE_BINDINGS_CLEAN_T3_F0_R1.json").read_text())
    for path in sorted(PACKET.glob("*.json")):
        if path.name != "SOURCE_BINDINGS_CLEAN_T3_F0_R1.json":
            bindings.setdefault("packet_document_sha256", {})[path.name] = sha256(path)
    for directory in ("teacher_fit_templates", "teacher_profile_specs"):
        for path in sorted((PACKET / directory).glob("*.json")):
            bindings.setdefault("packet_document_sha256", {})[path.relative_to(PACKET).as_posix()] = sha256(path)
    bindings["student_parent_asset_sha256"] = sha256(PACKET / "STUDENT_PARENT_ASSET_R1.json")
    bindings["teacher_profile_bundle_archive_sha256"] = sha256(args.bundle_destination.resolve() / "bundle.wnn")
    bindings["profile_gate_sha256"] = sha256(PACKET / "PROFILE_GATE_R1.json")
    bindings["packet_tool_sha256"] = {
        name: sha256(PACKET / name)
        for name in ("prepare_packet_r1.py", "preflight_actual_producers.py", "validate_profile_bundle_r1.py")
    }
    write_json(PACKET / "SOURCE_BINDINGS_CLEAN_T3_F0_R1.json", bindings)


if __name__ == "__main__":
    main()
