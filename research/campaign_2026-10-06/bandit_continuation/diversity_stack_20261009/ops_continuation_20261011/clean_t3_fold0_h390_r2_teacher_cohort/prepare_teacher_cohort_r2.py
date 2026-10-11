"""Freeze the R2 clean-fold teacher cohort without contacting Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[6]
REL_PACKET = PACKET.relative_to(REPO).as_posix()
R1_REL = "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r1"
R1_ARCHIVE_SHA = "0ba1b1f7b99dc5dabd892c7d99aa030733465aebc73bb652455c9bc527e0d65f"
R1_SOURCE_SHA = "fb32325fa1010615c12a45e67e81edfd18141e288c620a5c73664d45beb397ea"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PROP_SHA = "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"
DATASET = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r2"
ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
FOLD0 = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
PORTABLE = "research/campaign_2026-10-04/portable_inputs.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def descriptor(path: Path, relative: str) -> dict[str, Any]:
    return {"path": relative, "size": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-r1-packet", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    r1 = args.profile_r1_packet.resolve()
    bundle_destination = args.destination.resolve()
    authoring_files = {
        "prepare_teacher_cohort_r2.py",
        "validate_teacher_cohort_r2.py",
        "kernel_support",
    }
    if {item.name for item in PACKET.iterdir()} - authoring_files:
        raise SystemExit(
            "R2 packet contains generated artifacts; refusing to overwrite identified content"
        )
    if r1.name != "clean_t3_fold0_h390_r1":
        raise SystemExit("unexpected R1 profile packet identity")

    r1_bundle = r1 / "bundle_teacher_profile_r1"
    r1_manifest = json.loads((r1_bundle / "bundle-manifest.json").read_text())
    if r1_manifest["archive_sha256"] != R1_ARCHIVE_SHA:
        raise SystemExit("reviewed R1 archive hash changed")
    r1_source_manifest = r1 / "source_snapshot/SOURCE_MANIFEST_H390.json"
    if sha256(r1_source_manifest) != R1_SOURCE_SHA:
        raise SystemExit("reviewed R1 source-manifest hash changed")
    if sha256(r1 / "reference_inputs/fold0.json") != SPLIT_SHA:
        raise SystemExit("fold-0 split hash changed")
    r1_actual_preflight = r1 / "ACTUAL_PRODUCER_PREFLIGHT_R2.json"
    preflight = json.loads(r1_actual_preflight.read_text())
    if (
        preflight.get("status") != "pass_static_actual_producer_fixtures_only"
        or preflight.get("split_manifest_sha256") != SPLIT_SHA
        or preflight.get("source_snapshot_manifest_sha256") != R1_SOURCE_SHA
    ):
        raise SystemExit("R1 actual-producer receipt does not bind the expected source and split")

    # Preserve the reviewed source snapshot exactly; campaign support stays outside it.
    shutil.copytree(r1 / "source_snapshot", PACKET / "source_snapshot")
    if sha256(PACKET / "source_snapshot/SOURCE_MANIFEST_H390.json") != R1_SOURCE_SHA:
        raise SystemExit("copied source snapshot changed")

    gate_path = f"{REL_PACKET}/CONDITIONAL_FIT_ADMISSION_R2.json"
    gate = {
        "schema": "wnn.kaggle.full_teacher_fit_admission.r1",
        "status": "pending_actual_t4_profile_and_root_release",
        "release_authorized": False,
        "profile_kernel": "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r1-runner",
        "profile_archive_sha256": R1_ARCHIVE_SHA,
        "profile_receipt_sha256": None,
        "root_release_receipt_sha256": None,
        "required_before_execution": [
            "R1 terminal T4 profile receipt with usable fit and export measurements",
            "independent full-cohort timing/resource feasibility decision, or a separately frozen staged successor",
            "fresh quota/session, exact private input version/readiness/hash and output-capacity gates",
            "root release bound to the exact future archive and kernel configuration",
        ],
        "note": "Immutable pending gate. This archive must refuse before external mounts or GPU workers; any later release needs a new identified packet and resealed archive.",
    }
    write_json(PACKET / "CONDITIONAL_FIT_ADMISSION_R2.json", gate)
    gate_sha = sha256(PACKET / "CONDITIONAL_FIT_ADMISSION_R2.json")

    # Keep the same external inputs and owner/version paths as the actual R1 bundle.
    external_map = r1_manifest["external_mounts"]
    external_args: list[str] = []
    for logical, mapping in sorted(external_map.items()):
        if isinstance(mapping, str):
            external_args.append(f"{logical}={mapping}")
        else:
            if mapping.get("dataset") != ASSET_DATASET:
                raise SystemExit(f"unexpected asset dataset in R1 external map: {logical}")
            external_args.append(f"{logical}={mapping['dataset']}:{mapping['filename']}")
    input_files = {row["path"]: row for row in r1_manifest["files"]}
    external_bindings = []
    for logical, mapping in sorted(external_map.items()):
        row = input_files[logical]
        external_bindings.append({**row, "mount": mapping})
    write_json(
        PACKET / "INPUT_BINDING_R2.json",
        {
            "schema": "wnn.kaggle.cleanfold_inputs_binding.r2",
            "source_profile_archive_sha256": R1_ARCHIVE_SHA,
            "external_inputs": external_bindings,
            "split_manifest_sha256": SPLIT_SHA,
            "own_propensity_checkpoint_sha256": PROP_SHA,
            "remote_asset_version_and_ready": "not verified by packet builder; root must verify current exact version before any dispatch",
        },
    )
    trainer_sha = next(
        row["sha256"]
        for row in r1_manifest["files"]
        if row["path"] == f"{R1_REL}/source_snapshot/scripts/train_rnn.py"
    )
    write_json(
        PACKET / "SOURCE_BINDING_R2.json",
        {
            "schema": "wnn.kaggle.cleanfold_source_binding.r2",
            "copied_source_manifest_sha256": R1_SOURCE_SHA,
            "r1_profile_bundle_sha256": R1_ARCHIVE_SHA,
            "trainer_sha256": trainer_sha,
            "source_copy_relation": "byte-identical copy of reviewed R1 source_snapshot; no trainer or predictor edits",
            "actual_producer_preflight_path": f"{R1_REL}/ACTUAL_PRODUCER_PREFLIGHT_R2.json",
            "actual_producer_preflight_sha256": sha256(r1_actual_preflight),
            "actual_producer_preflight_scope": preflight["scope"],
        },
    )
    shutil.copyfile(r1_actual_preflight, PACKET / "ACTUAL_PRODUCER_PREFLIGHT_R1_RECEIPT.json")

    # Rebind every R1 code-closure path to this isolated R2 packet without changing bytes.
    code: list[dict[str, Any]] = []
    for old in r1_manifest["files"]:
        old_path = old["path"]
        if old_path in external_map:
            continue
        new_path = (
            old_path.replace(R1_REL + "/", REL_PACKET + "/", 1)
            if old_path.startswith(R1_REL + "/")
            else old_path
        )
        local = REPO / new_path
        if (
            not local.is_file()
            or local.stat().st_size != old["size"]
            or sha256(local) != old["sha256"]
        ):
            raise SystemExit(f"R1 sealed code closure mismatch at {new_path}")
        code.append({"path": new_path, "size": old["size"], "sha256": old["sha256"]})
    driver_path = f"{REL_PACKET}/kernel_support/teacher_cohort_r2_driver.py"
    code.append(descriptor(PACKET / "kernel_support/teacher_cohort_r2_driver.py", driver_path))
    gate_descriptor = descriptor(PACKET / "CONDITIONAL_FIT_ADMISSION_R2.json", gate_path)
    code.append(gate_descriptor)
    code.sort(key=lambda row: row["path"])

    r1_template = json.loads(
        (r1 / "teacher_fit_templates/clean_t3_f0_muonall_h384_e3_s3_r1.json").read_text()
    )
    inputs = r1_template["inputs"]
    if {row["path"] for row in inputs} != set(external_map):
        raise SystemExit("R1 recipe inputs differ from R1 external mounts")
    sys.path.insert(0, str(REPO / "scripts"))
    from kaggle_bundle import build_bundle  # noqa: PLC0415
    from kaggle_common import canonical_sha256  # noqa: PLC0415

    code_sha = canonical_sha256(code)
    specs: list[dict[str, Any]] = []
    teacher_runs = []
    for seed in (3, 4, 5):
        run = f"clean_t3_f0_muonall_h384_e3_s{seed}_r2"
        name = f"fit_{run}"
        template = json.loads(
            (r1 / f"teacher_fit_templates/clean_t3_f0_muonall_h384_e3_s{seed}_r1.json").read_text()
        )
        template["name"] = name
        template["code"] = code
        template["code_sha256"] = code_sha
        template["argv"][1] = f"{REL_PACKET}/source_snapshot/scripts/cleanfold_exec.py"
        template["argv"][template["argv"].index("--name") + 1] = run
        template["artifacts"] = [
            f"artifacts/runs/{run}/model_ep2.pt",
            f"artifacts/runs/{run}/model.pt",
            f"artifacts/runs/{run}/provenance.json",
            f"artifacts/runs/{run}/history.json",
            f"artifacts/runs/{run}/training_state.pt",
        ]
        template["limits"] = {
            "cpus": 2,
            "ram_bytes": 16000000000,
            "gpu_memory_bytes": 10000000000,
            "gpu_stop_bytes": 12000000000,
            "timeout_seconds": 5200,
        }
        template["environment"]["WNN_LOADER_WORKERS"] = "0"
        write_json(PACKET / "jobs" / f"{name}.json", template)
        specs.append(template)
        teacher_runs.append(
            {
                "seed": seed,
                "run_id": run,
                "fit_spec": f"kaggle_specs/{name}.json",
                "required_artifacts": template["artifacts"],
            }
        )

    prediction_jobs = []
    for seed in (3, 4, 5):
        run = f"clean_t3_f0_muonall_h384_e3_s{seed}_r2"
        for kind, split in (
            ("fit_train", "train"),
            ("fit_valid", "valid"),
            ("full_valid", "valid"),
        ):
            name = f"export_{run}_{kind}"
            output_name = f"{run}_{kind}"
            argv = [
                "{python}",
                f"{REL_PACKET}/source_snapshot/scripts/cleanfold_exec.py",
                "scripts/predict_run.py",
                run,
                split,
                "--epoch",
                "2",
                "--output-name",
                output_name,
                "--campaign-split-manifest",
                FOLD0,
                "--device",
                "cuda",
                "--inference-precision",
                "fp32",
            ]
            if kind == "full_valid":
                argv.append("--campaign-full-valid-feature-only")
            rows = {"fit_train": 10535, "fit_valid": 1405, "full_valid": 1873}[kind]
            base = f"cache/pred_{output_name}_{split}"
            export = {
                "version": 1,
                "name": name,
                "kind": "gpu",
                "argv": argv,
                "code": code,
                "code_sha256": code_sha,
                "inputs": inputs,
                "artifacts": [f"{base}.npy", f"{base}.json"],
                "limits": {
                    "cpus": 2,
                    "ram_bytes": 8000000000,
                    "gpu_memory_bytes": 5000000000,
                    "gpu_stop_bytes": 6000000000,
                    "timeout_seconds": 1800,
                },
                "environment": {
                    "WNN_PROJECT_ROOT": "/kaggle/working/experiment",
                    "WNN_PORTABLE_DATA_MANIFEST": f"/kaggle/working/experiment/{PORTABLE}",
                    "WNN_DEFERRED_IDENTITY_PATHS": "datasets/train.parquet,datasets/valid.parquet",
                    "WNN_LOADER_WORKERS": "0",
                    "WNN_LOADER_RAM_GB": "4",
                    "PYTHONHASHSEED": "0",
                },
            }
            write_json(PACKET / "jobs" / f"{name}.json", export)
            specs.append(export)
            prediction_jobs.append(
                {
                    "spec": f"kaggle_specs/{name}.json",
                    "teacher_run_id": run,
                    "kind": kind,
                    "split": split,
                    "output_name": output_name,
                    "npy": {"path": f"{base}.npy", "shape": [rows, 20000, 2], "dtype": "float32"},
                    "sidecar": f"{base}.json",
                    "checkpoint": f"artifacts/runs/{run}/model_ep2.pt",
                }
            )

    build_bundle(
        REPO,
        [PACKET / "jobs" / f"{spec['name']}.json" for spec in specs],
        bundle_destination,
        DATASET,
        external_args,
        [ASSET_DATASET],
    )
    manifest_path = bundle_destination / "bundle-manifest.json"
    bundle_manifest = json.loads(manifest_path.read_text())
    archive_sha = bundle_manifest["archive_sha256"]
    kernel_path = bundle_destination / "kernel/kernel.py"
    kernel = kernel_path.read_text()
    kernel = kernel.replace(
        "KERNEL_PROCESS_WALL_SECONDS_MAX = 6_000", "KERNEL_PROCESS_WALL_SECONDS_MAX = 14_400"
    )
    kernel = kernel.replace("6,000-second", "14,400-second")
    gate_constant = f"R2_GATE_RELATIVE_PATH = {gate_path!r}\nR2_GATE_SHA256 = {gate_sha!r}\n"
    constants_marker = "RUNNER_FINALIZATION_RESERVE_SECONDS = 300\nBOOTSTRAP_STARTED_MONOTONIC = time.monotonic()\n"
    if constants_marker not in kernel:
        raise SystemExit("generated bootstrap constants do not match expected template")
    kernel = kernel.replace(constants_marker, constants_marker + gate_constant, 1)
    mount_marker = '    require_remaining(kernel_deadline, "bundle extraction")\n    for logical, mapping in manifest["external_mounts"].items():\n'
    gate_check = (
        '    require_remaining(kernel_deadline, "bundle extraction")\n'
        "    gate_file = root / R2_GATE_RELATIVE_PATH\n"
        "    if not gate_file.is_file() or hashlib.sha256(gate_file.read_bytes()).hexdigest() != R2_GATE_SHA256:\n"
        '        raise ValueError("R2 conditional fit admission receipt is missing or changed")\n'
        "    gate_receipt = json.loads(gate_file.read_text())\n"
        '    if gate_receipt.get("status") != "admitted" or gate_receipt.get("release_authorized") is not True or not gate_receipt.get("root_release_receipt_sha256"):\n'
        '        raise PermissionError("R2 fit admission is pending; refusing before external mounts or GPU workers")\n'
        '    for logical, mapping in manifest["external_mounts"].items():\n'
    )
    if mount_marker not in kernel:
        raise SystemExit("generated bootstrap mount boundary was not found")
    kernel = kernel.replace(mount_marker, gate_check, 1)
    command_marker = '    command = [\n        sys.executable,\n        str(root / "scripts/kaggle_gpu_runner.py"),'
    start = kernel.index(command_marker)
    end_marker = '    finally:\n        release_mounts(root, manifest["external_mounts"])'
    end = kernel.index(end_marker, start) + len(end_marker)
    driver = f"{REL_PACKET}/kernel_support/teacher_cohort_r2_driver.py"
    orchestration = (
        f'    command = [sys.executable, str(root / {driver!r}), "--root", str(root)]\n'
        '    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"\n'
        "    environment = os.environ.copy()\n"
        '    environment["WNN_KERNEL_DEADLINE_MONOTONIC"] = repr(kernel_deadline)\n'
        "    try:\n"
        '        require_remaining(kernel_deadline, "teacher cohort driver launch")\n'
        "        remaining = remaining_seconds(kernel_deadline)\n"
        "        process = subprocess.Popen(command, cwd=root, env=environment, start_new_session=True)\n"
        "        try:\n"
        "            returncode = process.wait(timeout=remaining)\n"
        "        except subprocess.TimeoutExpired as error:\n"
        "            terminate_process_tree(process)\n"
        '            raise TimeoutError("teacher cohort exceeded the bounded 14,400-second kernel-process budget") from error\n'
        "        if returncode != 0:\n"
        "            raise subprocess.CalledProcessError(returncode, command)\n"
        "        if remaining_seconds(kernel_deadline) <= 0:\n"
        '            raise TimeoutError("teacher cohort completed outside its wall ceiling")\n'
        "    finally:\n"
        '        release_mounts(root, manifest["external_mounts"])'
    )
    kernel = kernel[:start] + orchestration + kernel[end:]
    compile(kernel, str(kernel_path), "exec")
    kernel_path.write_text(kernel)
    bootstrap_sha = sha256(kernel_path)

    write_json(
        PACKET / "KERNEL_CONFIGURATION_R2.json",
        {
            "schema": "wnn.kaggle.teacher_cohort_kernel_config.r2",
            "dataset": DATASET,
            "kernel_id": json.loads(
                (bundle_destination / "kernel/kernel-metadata.json").read_text()
            )["id"],
            "kernel_title": json.loads(
                (bundle_destination / "kernel/kernel-metadata.json").read_text()
            )["title"],
            "profile_r1_archive_sha256": R1_ARCHIVE_SHA,
            "profile_r1_source_manifest_sha256": R1_SOURCE_SHA,
            "bundle_archive_sha256": archive_sha,
            "bundle_manifest_sha256": sha256(manifest_path),
            "bootstrap_sha256": bootstrap_sha,
            "bootstrap_bytes": kernel_path.stat().st_size,
            "wall_ceiling_seconds": 14400,
            "two_device_gpu_seconds_upper_bound": 28800,
            "actual_fit_seconds": None,
            "actual_export_seconds": None,
            "actual_total_eta_seconds": None,
            "full_cohort_fit_admission": False,
        },
    )
    plan = {
        "schema": "wnn.kaggle.clean_t3_f0_teacher_cohort_plan.r2",
        "status": "frozen_preparation_not_admitted",
        "profile_parent": R1_REL,
        "profile_archive_sha256": R1_ARCHIVE_SHA,
        "source_snapshot_manifest_sha256": R1_SOURCE_SHA,
        "fold0_manifest_sha256": SPLIT_SHA,
        "own_propensity_checkpoint_sha256": PROP_SHA,
        "membership": {
            "train_fit_groups": 10535,
            "valid_fit_groups": 1405,
            "valid_eval_groups_excluded_from_fit": 468,
        },
        "teacher_recipe": {
            "architecture": "GRU(384,layers=2)",
            "seeds": [3, 4, 5],
            "epochs": 3,
            "fixed_checkpoint_epoch": 2,
            "optimizer": "Muon-all",
            "propensity_power": 2,
            "batch": 32,
            "chunk": 1000,
            "precision": "fp16",
            "initializer": None,
        },
        "fit_stage": teacher_runs,
        "export_stage": prediction_jobs,
        "cache_exports_use_only_fit_membership": True,
        "full_valid_exports_separate_feature_only": True,
        "wall_ceiling_seconds": 14400,
        "two_device_gpu_seconds_upper_bound": 28800,
        "fit_eta_seconds": None,
        "export_eta_seconds": None,
        "total_eta_seconds": None,
        "admission": "root-only after R1 actual T4 profile, quota/session, exact input READY/version/hash, output capacity and independent review",
    }
    write_json(PACKET / "TEACHER_COHORT_PLAN_R2.json", plan)
    write_json(
        PACKET / "COLLECTION_CONTRACT_R2.json",
        {
            "schema": "wnn.kaggle.teacher_cohort_collection.r2",
            "status": "required_at_terminal_not_collected",
            "output_roots": [
                "results/teacher_fits",
                "results/teacher_exports",
                "results/teacher_cohort_pipeline.json",
            ],
            "fit_run_artifacts": [
                "model_ep2.pt",
                "model.pt",
                "training_state.pt",
                "provenance.json",
                "history.json",
            ],
            "fit_worker_files": ["worker.log", "result.json", "input_identity.json"],
            "prediction_specs": prediction_jobs,
            "export_worker_files": ["worker.log", "result.json", "input_identity.json"],
            "terminal_checks": [
                "provider terminal status with kernel stdout/stderr and worker inventory",
                "3 fit and 9 export worker results return zero and report complete",
                "copy each NPY and JSON sidecar without overwrite; bind both copied hashes/bytes to that same fresh worker result",
                "verify NPY shape/dtype and scan all outputs for finite values",
                "bind every sidecar to source run, model_ep2 checkpoint, split hash, fit scope and group order",
                "full-valid arrays have 1873 rows and official scorer order; fit-cache arrays use only 10535/1405 fitting groups",
                "retain original checkpoints, full training state, provenance, history, logs, worker receipts and stage heartbeats",
            ],
        },
    )
    write_json(
        PACKET / "ROOT_RELEASE_GATES_R2.json",
        {
            "schema": "wnn.root_conditional_fit_release.r1",
            "status": "pending",
            "authorized": False,
            "binding": {
                "dataset": DATASET,
                "bundle_archive_sha256": archive_sha,
                "bootstrap_sha256": bootstrap_sha,
                "gate_sha256": gate_sha,
            },
            "required": [
                "usable R1 actual T4 profile",
                "measured R2 fit/export resource feasibility",
                "fresh quota/session",
                "exact READY input version and hashes",
                "root admission within current reservation ledger",
            ],
            "eta_seconds": None,
            "wall_ceiling_seconds": 14400,
            "two_device_gpu_seconds_upper_bound": 28800,
        },
    )
    (PACKET / "README_R2.md").write_text(
        f"""# Clean T3 fold-0 teacher cohort R2

Three fresh H384 GRU teachers use Muon-all, fold-0 own propensity power 2, seeds 3/4/5, 3 epochs and fixed epoch-2 checkpoints. The source snapshot is byte-identical to reviewed R1 `{R1_SOURCE_SHA}`; profile archive `{R1_ARCHIVE_SHA}`; fold-0 split `{SPLIT_SHA}`; own propensity checkpoint `{PROP_SHA}`.

The 10,535 train-fit and 1,405 valid-fit groups exclude all 468 fold-0 valid evaluation groups. Six fit-membership cache exports are separate from three all-1,873-row full-valid feature-only scorer exports.

Private dataset `{DATASET}` creates title-derived kernel ID `{json.loads((bundle_destination / "kernel/kernel-metadata.json").read_text())["id"]}`. Its embedded admission gate is pending and fails before external mounts or GPU worker launch. No upload or execution was performed by this builder.

The 14,400-second wall and 28,800 two-device GPU-second figures are ceilings, not timing estimates. Actual profile results, fit/export ETA and full-cohort feasibility remain unknown; admission is false until root’s live gates pass.
"""
    )
    if kernel_path.stat().st_size >= 1_000_000:
        raise SystemExit("Kaggle bootstrap exceeds the conservative 1 MB source envelope")
    print(
        json.dumps(
            {
                "packet": str(PACKET),
                "bundle": str(bundle_destination),
                "archive_sha256": archive_sha,
                "manifest_sha256": sha256(manifest_path),
                "bootstrap_sha256": bootstrap_sha,
                "bootstrap_bytes": kernel_path.stat().st_size,
                "gate_sha256": gate_sha,
                "fit_jobs": 3,
                "export_jobs": 9,
                "admission": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
