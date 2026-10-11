"""Freeze the R3 clean-fold teacher cohort with a paired-T4 profile gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PACKET_REL = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort"
)
R2_REL = PACKET_REL.replace("_r3_teacher_cohort", "_r2_teacher_cohort")
PROFILE_REL = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_f0_startup_profile_r3"
)
R2_ARCHIVE_SHA = "5f2ea48be58e4057e23800d187dbbe5c9f2ea8905fd2c6a6283969804aeffdd6"
PROFILE_SOURCE_SHA = "41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa"
PROFILE_ARCHIVE_SHA = "986304b0278ea2ea3c5b903984edd5b58fda6e04e497a6bd10f06a58a635e9e1"
PROFILE_RESULT_SHA = "a7cb249776485d259442c749ad3f82584da9386ab316320a731dd77b34479a68"
PROFILE_LOG_SHA = "220ee9cc858203bdc4b97d473132cc284a5a409ea6c2695c6fa681a3e8a87352"
PROFILE_HANDOFF_SHA = "46eee3e280bbb1c436bb868c3e49696101726303c449be00b28a5c505d39afb1"
SOURCE_DELTA_SHA = "912205eba101c8539c372206e5ae0e56f2d0a39f8997b271189b3d2f7d601406"
PRODUCER_PREFLIGHT_SHA = "fc16040084e041a8e458f80463f9096bcd9a3c946928580b55aa4468cde35afc"
R2_SOURCE_SHA = "fb32325fa1010615c12a45e67e81edfd18141e288c620a5c73664d45beb397ea"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PROP_SHA = "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"
DATASET = "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r3"
ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
PORTABLE = "research/campaign_2026-10-04/portable_inputs.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def describe(root: Path, path: str) -> dict[str, Any]:
    local = root / path
    return {"path": path, "size": local.stat().st_size, "sha256": sha256(local)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--profile-evidence-root",
        type=Path,
        default=Path("/home/vlad/development/wunder/wnn_connectome_starterpack"),
    )
    parser.add_argument("--bundle-destination", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    evidence_root = args.profile_evidence_root.resolve(strict=True)
    packet = root / PACKET_REL
    r2 = root / R2_REL
    profile = root / PROFILE_REL
    profile_evidence = evidence_root / PROFILE_REL
    bundle = args.bundle_destination.resolve()
    packet.mkdir(parents=True, exist_ok=True)
    allowed = {
        "prepare_teacher_cohort_r3.py",
        "validate_teacher_cohort_r3.py",
        "kernel_support",
    }
    if {p.name for p in packet.iterdir()} - allowed:
        raise SystemExit("R3 packet already contains generated files; refusing overwrite")
    r2_manifest_path = r2 / "bundle_teacher_cohort_r2/bundle-manifest.json"
    r2_manifest = json.loads(r2_manifest_path.read_text())
    if r2_manifest.get("archive_sha256") != R2_ARCHIVE_SHA:
        raise SystemExit("sealed R2 teacher packet changed")
    profile_source_manifest = profile / "source_snapshot/SOURCE_MANIFEST_H390.json"
    if sha256(profile_source_manifest) != PROFILE_SOURCE_SHA:
        raise SystemExit("measured R3 startup-profile source changed")
    profile_manifest = json.loads((profile / "bundle_profile_r3/bundle-manifest.json").read_text())
    if profile_manifest.get("archive_sha256") != PROFILE_ARCHIVE_SHA:
        raise SystemExit("measured R3 startup-profile archive changed")
    actual_result = (
        profile_evidence / "terminal_collection_root_r1/attempts/attempt_001/outputs/results/"
        "clean_t3_f0_muonall_h384_e3_s3_t4_startup_profile_r3/result.json"
    )
    actual_log = actual_result.with_name("worker.log")
    actual_handoff = profile_evidence / "terminal_collection_root_r1/ROOT_TERMINAL_HANDOFF_R2.json"
    for path, expected in (
        (actual_result, PROFILE_RESULT_SHA),
        (actual_log, PROFILE_LOG_SHA),
        (actual_handoff, PROFILE_HANDOFF_SHA),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"actual one-T4 profile receipt changed: {path.name}")
    profile_result = json.loads(actual_result.read_text())
    terminal_receipt = json.loads(actual_handoff.read_text())
    if (
        profile_result.get("status") != "complete"
        or profile_result.get("returncode") != 0
        or profile_result.get("device", {}).get("name") != "Tesla T4"
        or terminal_receipt.get("collection_complete") is not True
    ):
        raise SystemExit("measured single-T4 profile result/collection is incomplete")
    log_lines = actual_log.read_text(errors="replace").splitlines()

    def log_record(marker: str) -> dict[str, Any]:
        matches = [line[len(marker) :].strip() for line in log_lines if line.startswith(marker)]
        if len(matches) != 1:
            raise SystemExit(f"actual profile log must contain exactly one {marker.strip()} record")
        return json.loads(matches[0])

    update_counts = log_record("PROFILE_UPDATE_COUNTS ")["cumulative"]
    profile_metrics = log_record("PROFILE ")
    grad_stats = log_record("GRAD_STATS ")
    if (
        update_counts.get("completed_batches") != 30
        or update_counts.get("optimizer_attempts") != 600
        or update_counts.get("optimizer_completed_updates") != 599
        or update_counts.get("amp_skipped_updates") != 1
        or grad_stats.get("nonfinite_steps") != 1
        or not isinstance(profile_metrics.get("step_wall"), (int, float))
    ):
        raise SystemExit("actual single-T4 profile metrics do not match the sealed reference")

    shutil.copytree(profile / "source_snapshot", packet / "source_snapshot")
    if sha256(packet / "source_snapshot/SOURCE_MANIFEST_H390.json") != PROFILE_SOURCE_SHA:
        raise SystemExit("copied profile source snapshot changed")
    source_delta_src = profile / "SOURCE_DELTA_R3.json"
    if sha256(source_delta_src) != SOURCE_DELTA_SHA:
        raise SystemExit("actual profile source-delta receipt changed")
    shutil.copyfile(source_delta_src, packet / "SOURCE_DELTA_R3.json")
    producer_receipt_src = r2 / "ACTUAL_PRODUCER_PREFLIGHT_R1_RECEIPT.json"
    if sha256(producer_receipt_src) != PRODUCER_PREFLIGHT_SHA:
        raise SystemExit("actual-producer preflight receipt changed")
    shutil.copyfile(producer_receipt_src, packet / "ACTUAL_PRODUCER_PREFLIGHT_R2_RECEIPT.json")

    profile_reference = {
        "schema": "wnn.kaggle.teacher_profile_reference.r3",
        "scope": "one-worker startup/fit-shape profile only; not a paired-device or cohort qualification",
        "archive_sha256": PROFILE_ARCHIVE_SHA,
        "source_manifest_sha256": PROFILE_SOURCE_SHA,
        "terminal_handoff_sha256": PROFILE_HANDOFF_SHA,
        "result_sha256": PROFILE_RESULT_SHA,
        "worker_log_sha256": PROFILE_LOG_SHA,
        "terminal_collection_note": terminal_receipt.get("incident"),
        "observed": {
            "device": "Tesla T4",
            "worker_wall_seconds": 397.85713796599975,
            "peak_ram_bytes": 4802109440,
            "peak_gpu_bytes": 3499098112,
            "warmup_complete_batches": 5,
            "measured_complete_batches": 25,
            "optimizer_attempts": update_counts["optimizer_attempts"],
            "optimizer_completed_updates": update_counts["optimizer_completed_updates"],
            "amp_skipped_updates": update_counts["amp_skipped_updates"],
            "profile_step_wall_ms": profile_metrics["step_wall"],
            "profile_gpu_total_ms": profile_metrics["gpu_total"],
            "profile_loader_wait_ms": profile_metrics["loader_wait"],
            "grad_stats_nonfinite_steps": grad_stats["nonfinite_steps"],
        },
        "limitations": [
            "one physical T4 only; does not qualify concurrent hashing, host memory, or dual-device throughput",
            "no checkpoint save/reload was exercised",
            "--grad-stats adds diagnostic synchronization; observed time is not an unbiased full-fit estimate",
            "the 60-second faulthandler messages are periodic stack heartbeats, not timeout failures",
        ],
    }
    write_json(packet / "PROFILE_REFERENCE_R3.json", profile_reference)
    guard = {
        "schema": "wnn.kaggle.paired_t4_profile_guard.r1",
        "profile_jobs": [
            f"profile_clean_t3_f0_muonall_h384_e3_s{seed}_paired_t4_r3" for seed in (3, 4)
        ],
        "profile_batches": {"warmup_complete_batches": 5, "measured_complete_batches": 5},
        "require_exactly_two_distinct_t4_devices": True,
        "require_runner_concurrency": 2,
        "per_worker_peak_ram_bytes_max": 6000000000,
        "sum_profile_worker_peak_ram_bytes_max": 12000000000,
        "per_worker_peak_gpu_bytes_max": 5000000000,
        "require_profile_log_counters": {
            "completed_batches_min": 10,
            "optimizer_completed_updates_min": 1,
            "attempts_equal_completed_plus_amp_skipped": True,
            "finite_profile_metrics": True,
            "finite_gradient_summary": True,
        },
        "decision": "Any missing, failed, serialized, non-T4, resource-over-limit, or non-finite profile blocks all full fits in this kernel. No serial fallback is allowed after a failed profile gate; root may create a separately identified serial successor.",
    }
    write_json(packet / "PROFILE_GUARD_CONTRACT_R3.json", guard)

    original_profile_spec = json.loads(
        (
            profile
            / "teacher_profile_specs/clean_t3_f0_muonall_h384_e3_s3_t4_startup_profile_r3.json"
        ).read_text()
    )
    old_profile_prefix = f"{PROFILE_REL}/"
    code: list[dict[str, Any]] = []
    external_map = r2_manifest["external_mounts"]
    external_names = set(external_map)
    external_rows = [row for row in r2_manifest["files"] if row["path"] in external_names]
    if len(external_rows) != len(external_names):
        raise SystemExit("R2 input map is missing expected hash/size records")
    write_json(
        packet / "INPUT_BINDING_R3.json",
        {
            "schema": "wnn.kaggle.clean_t3_f0_inputs_binding.r3",
            "source_profile_bundle_sha256": PROFILE_ARCHIVE_SHA,
            "source_prior_teacher_bundle_sha256": R2_ARCHIVE_SHA,
            "external_inputs": [
                {**row, "mount": external_map[row["path"]]}
                for row in sorted(external_rows, key=lambda item: item["path"])
            ],
            "split_manifest_sha256": SPLIT_SHA,
            "own_propensity_checkpoint_sha256": PROP_SHA,
            "remote_dataset_asset_version_and_readiness": "not verified by packet builder; root must verify the exact current READY versions and hashes before dispatch",
        },
    )
    source_document = json.loads((packet / "source_snapshot/SOURCE_MANIFEST_H390.json").read_text())
    trainer_sha = source_document["sources"]["scripts/train_rnn.py"]
    predictor_sha = source_document["sources"]["scripts/predict_run.py"]
    write_json(
        packet / "SOURCE_BINDING_R3.json",
        {
            "schema": "wnn.kaggle.clean_t3_f0_source_binding.r3",
            "source_manifest_sha256": PROFILE_SOURCE_SHA,
            "source_manifest_parent_sha256": R2_SOURCE_SHA,
            "source_delta_receipt_sha256": SOURCE_DELTA_SHA,
            "source_delta_scope": "startup hash/split telemetry, profile-only optimizer counters, and opt-in stack heartbeat; full input hash and fitting semantics retained",
            "trainer_sha256": trainer_sha,
            "predictor_sha256": predictor_sha,
            "actual_producer_preflight_sha256": PRODUCER_PREFLIGHT_SHA,
            "actual_producer_preflight_parent_source_sha256": R2_SOURCE_SHA,
            "profile_archive_sha256": PROFILE_ARCHIVE_SHA,
            "no_source_identity_bypass": True,
        },
    )
    for row in original_profile_spec["code"]:
        old_path = row["path"]
        new_path = old_path.replace(old_profile_prefix, f"{PACKET_REL}/", 1)
        if new_path in external_names:
            continue
        descriptor = describe(root, new_path)
        if descriptor["size"] != row["size"] or descriptor["sha256"] != row["sha256"]:
            raise SystemExit(f"R3 source/code closure differs from actual profile: {new_path}")
        code.append(descriptor)
    driver_path = f"{PACKET_REL}/kernel_support/teacher_cohort_r3_driver.py"
    code.extend(
        [
            describe(root, driver_path),
            describe(root, f"{PACKET_REL}/PROFILE_REFERENCE_R3.json"),
            describe(root, f"{PACKET_REL}/PROFILE_GUARD_CONTRACT_R3.json"),
            describe(root, f"{PACKET_REL}/SOURCE_DELTA_R3.json"),
            describe(root, f"{PACKET_REL}/SOURCE_BINDING_R3.json"),
            describe(root, f"{PACKET_REL}/INPUT_BINDING_R3.json"),
            describe(root, f"{PACKET_REL}/ACTUAL_PRODUCER_PREFLIGHT_R2_RECEIPT.json"),
        ]
    )
    gate_rel = f"{PACKET_REL}/CONDITIONAL_FIT_ADMISSION_R3.json"
    gate = {
        "schema": "wnn.kaggle.conditional_teacher_fit_admission.r3",
        "status": "pending_root_review_and_release",
        "release_authorized": False,
        "profile_reference_sha256": sha256(packet / "PROFILE_REFERENCE_R3.json"),
        "profile_guard_contract_sha256": sha256(packet / "PROFILE_GUARD_CONTRACT_R3.json"),
        "root_release_receipt_sha256": None,
        "required": [
            "root signs a receipt bound to the exact R3 archive/kernel and this gate before dispatch",
            "both profile workers complete on distinct actual Tesla T4 UUIDs with concurrency 2",
            "both profiles report 10 completed batches, positive completed optimizer updates, consistent attempt counters and finite metrics",
            "both per-worker and combined observed RAM/VRAM remain within PROFILE_GUARD_CONTRACT_R3.json",
            "fresh Kaggle quota/session and exact private input READY/version/hash checks pass",
            "the profile-gated 3-fit/9-export cohort remains within the root-admitted wall and GPU-second reservation",
        ],
        "note": "The builder cannot authorize this gate. The kernel runs full teacher fits only after a root-bound release is embedded and every live profile guard passes.",
    }
    write_json(packet / "CONDITIONAL_FIT_ADMISSION_R3.json", gate)
    code.append(describe(root, gate_rel))
    code.sort(key=lambda row: row["path"])

    sys.path.insert(0, str(root / "scripts"))
    from kaggle_bundle import build_bundle  # noqa: PLC0415
    from kaggle_common import canonical_sha256  # noqa: PLC0415

    code_sha = canonical_sha256(code)
    external_inputs = {
        row["path"]: row for row in r2_manifest["files"] if row["path"] in external_names
    }
    if set(external_inputs) != external_names:
        raise SystemExit("R2 external input path list is incomplete")
    profile_inputs = {row["path"]: row for row in original_profile_spec["inputs"]}
    if profile_inputs != external_inputs:
        raise SystemExit("R3 profile inputs differ from the previously reviewed cohort inputs")
    template_path = r2 / "jobs/fit_clean_t3_f0_muonall_h384_e3_s3_r2.json"
    fit_specs: list[dict[str, Any]] = []
    fit_plan = []
    for seed in (3, 4, 5):
        run = f"clean_t3_f0_muonall_h384_e3_s{seed}_r3"
        spec = json.loads(template_path.read_text())
        spec["name"] = f"fit_{run}"
        spec["argv"][spec["argv"].index("--name") + 1] = run
        spec["argv"][spec["argv"].index("--seed") + 1] = str(seed)
        spec["argv"][1] = f"{PACKET_REL}/source_snapshot/scripts/cleanfold_exec.py"
        spec["code"] = code
        spec["code_sha256"] = code_sha
        spec["artifacts"] = [
            f"artifacts/runs/{run}/model_ep2.pt",
            f"artifacts/runs/{run}/model.pt",
            f"artifacts/runs/{run}/provenance.json",
            f"artifacts/runs/{run}/history.json",
            f"artifacts/runs/{run}/training_state.pt",
        ]
        spec["limits"] = {
            "cpus": 2,
            "ram_bytes": 8000000000,
            "gpu_memory_bytes": 6000000000,
            "gpu_stop_bytes": 7000000000,
            "timeout_seconds": 5200,
        }
        spec["environment"].update(
            {
                "WNN_PORTABLE_DATA_MANIFEST": f"/kaggle/working/experiment/{PORTABLE}",
                "PYTHONUNBUFFERED": "1",
                "WNN_LOADER_WORKERS": "0",
                "WNN_STARTUP_DIAGNOSTICS": "1",
                "WNN_STARTUP_STACK_DUMP_SECONDS": "60",
            }
        )
        write_json(packet / "jobs" / f"{spec['name']}.json", spec)
        fit_specs.append(spec)
        fit_plan.append(
            {
                "seed": seed,
                "run_id": run,
                "fit_spec": f"kaggle_specs/{spec['name']}.json",
                "artifacts": spec["artifacts"],
            }
        )

    profile_specs = []
    profile_names = []
    for seed in (3, 4):
        name = f"profile_clean_t3_f0_muonall_h384_e3_s{seed}_paired_t4_r3"
        spec = json.loads(json.dumps(original_profile_spec))
        spec["name"] = name
        spec["argv"][1] = f"{PACKET_REL}/source_snapshot/scripts/cleanfold_exec.py"
        spec["argv"][spec["argv"].index("--name") + 1] = name
        spec["argv"][spec["argv"].index("--seed") + 1] = str(seed)
        spec["argv"][spec["argv"].index("--profile-steps") + 1] = "5"
        spec["code"] = code
        spec["code_sha256"] = code_sha
        spec["artifacts"] = []
        spec["limits"] = {
            "cpus": 2,
            "ram_bytes": 6000000000,
            "gpu_memory_bytes": 5000000000,
            "gpu_stop_bytes": 6000000000,
            "timeout_seconds": 1800,
        }
        spec["environment"].update(
            {
                "WNN_PORTABLE_DATA_MANIFEST": f"/kaggle/working/experiment/{PORTABLE}",
                "WNN_STARTUP_DIAGNOSTICS": "1",
                "WNN_STARTUP_STACK_DUMP_SECONDS": "60",
                "WNN_PROFILE_STAGE": "paired_t4_before_teacher_fits",
            }
        )
        write_json(packet / "jobs" / f"{name}.json", spec)
        profile_specs.append(spec)
        profile_names.append(name)

    export_specs = []
    export_plan = []
    r2_export = json.loads(
        (r2 / "jobs/export_clean_t3_f0_muonall_h384_e3_s3_r2_full_valid.json").read_text()
    )
    for seed in (3, 4, 5):
        run = f"clean_t3_f0_muonall_h384_e3_s{seed}_r3"
        for kind, split in (
            ("fit_train", "train"),
            ("fit_valid", "valid"),
            ("full_valid", "valid"),
        ):
            name = f"export_{run}_{kind}"
            output_name = f"{run}_{kind}"
            rows = {"fit_train": 10535, "fit_valid": 1405, "full_valid": 1873}[kind]
            argv = [
                "{python}",
                f"{PACKET_REL}/source_snapshot/scripts/cleanfold_exec.py",
                "scripts/predict_run.py",
                run,
                split,
                "--epoch",
                "2",
                "--output-name",
                output_name,
                "--campaign-split-manifest",
                "research/campaign_2026-09-29/artifacts/splits/fold0.json",
                "--device",
                "cuda",
                "--inference-precision",
                "fp32",
            ]
            if kind == "full_valid":
                argv.append("--campaign-full-valid-feature-only")
            base = f"cache/pred_{output_name}_{split}"
            spec = {
                "version": 1,
                "name": name,
                "kind": "gpu",
                "argv": argv,
                "code": code,
                "code_sha256": code_sha,
                "inputs": r2_export["inputs"],
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
                    "WNN_STARTUP_DIAGNOSTICS": "1",
                },
            }
            write_json(packet / "jobs" / f"{name}.json", spec)
            export_specs.append(spec)
            export_plan.append(
                {
                    "spec": f"kaggle_specs/{name}.json",
                    "run_id": run,
                    "kind": kind,
                    "split": split,
                    "output_name": output_name,
                    "npy": {"path": f"{base}.npy", "shape": [rows, 20000, 2], "dtype": "float32"},
                    "sidecar": f"{base}.json",
                    "checkpoint": f"artifacts/runs/{run}/model_ep2.pt",
                }
            )

    specs = profile_specs + fit_specs + export_specs
    external_args: list[str] = []
    asset_datasets: set[str] = set()
    for logical, mapping in sorted(external_map.items()):
        if isinstance(mapping, str):
            external_args.append(f"{logical}={mapping}")
        else:
            external_args.append(f"{logical}={mapping['dataset']}:{mapping['filename']}")
            asset_datasets.add(mapping["dataset"])
    build_bundle(
        root,
        [packet / "jobs" / f"{spec['name']}.json" for spec in specs],
        bundle,
        DATASET,
        external_args,
        sorted(asset_datasets),
    )
    manifest_path = bundle / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    kernel_path = bundle / "kernel/kernel.py"
    kernel = kernel_path.read_text()
    kernel = kernel.replace(
        "KERNEL_PROCESS_WALL_SECONDS_MAX = 6_000", "KERNEL_PROCESS_WALL_SECONDS_MAX = 14_400"
    )
    gate_path = f"{PACKET_REL}/CONDITIONAL_FIT_ADMISSION_R3.json"
    gate_sha = sha256(packet / "CONDITIONAL_FIT_ADMISSION_R3.json")
    constants = f"R3_GATE_RELATIVE_PATH = {gate_path!r}\nR3_GATE_SHA256 = {gate_sha!r}\n"
    marker = "RUNNER_FINALIZATION_RESERVE_SECONDS = 300\nBOOTSTRAP_STARTED_MONOTONIC = time.monotonic()\n"
    if marker not in kernel:
        raise SystemExit("unexpected Kaggle bootstrap template")
    kernel = kernel.replace(marker, marker + constants, 1)
    mount_marker = '    require_remaining(kernel_deadline, "bundle extraction")\n    for logical, mapping in manifest["external_mounts"].items():\n'
    guarded = (
        '    require_remaining(kernel_deadline, "bundle extraction")\n'
        "    gate_file = root / R3_GATE_RELATIVE_PATH\n"
        "    if not gate_file.is_file() or hashlib.sha256(gate_file.read_bytes()).hexdigest() != R3_GATE_SHA256:\n"
        '        raise ValueError("R3 conditional admission receipt is missing or changed")\n'
        "    gate_receipt = json.loads(gate_file.read_text())\n"
        '    if gate_receipt.get("status") != "admitted" or gate_receipt.get("release_authorized") is not True or not gate_receipt.get("root_release_receipt_sha256"):\n'
        '        raise PermissionError("R3 conditional admission is pending; refusing before external mounts or workers")\n'
        '    for logical, mapping in manifest["external_mounts"].items():\n'
    )
    if mount_marker not in kernel:
        raise SystemExit("bootstrap external-mount boundary not found")
    kernel = kernel.replace(mount_marker, guarded, 1)
    command_marker = '    command = [\n        sys.executable,\n        str(root / "scripts/kaggle_gpu_runner.py"),'
    start = kernel.index(command_marker)
    end_marker = '    finally:\n        release_mounts(root, manifest["external_mounts"])'
    end = kernel.index(end_marker, start) + len(end_marker)
    driver = f"{PACKET_REL}/kernel_support/teacher_cohort_r3_driver.py"
    orchestration = (
        f'    command = [sys.executable, str(root / {driver!r}), "--root", str(root)]\n'
        '    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"\n'
        "    environment = os.environ.copy()\n"
        '    environment["WNN_KERNEL_DEADLINE_MONOTONIC"] = repr(kernel_deadline)\n'
        "    try:\n"
        '        require_remaining(kernel_deadline, "teacher cohort driver launch")\n'
        "        process = subprocess.Popen(command, cwd=root, env=environment, start_new_session=True)\n"
        "        try:\n"
        "            returncode = process.wait(timeout=remaining_seconds(kernel_deadline))\n"
        "        except subprocess.TimeoutExpired as error:\n"
        "            terminate_process_tree(process)\n"
        '            raise TimeoutError("R3 profile-gated teacher cohort exceeded 14,400 seconds") from error\n'
        "        if returncode != 0:\n"
        "            raise subprocess.CalledProcessError(returncode, command)\n"
        "        if remaining_seconds(kernel_deadline) <= 0:\n"
        '            raise TimeoutError("R3 cohort completed outside the admitted wall ceiling")\n'
        "    finally:\n"
        '        release_mounts(root, manifest["external_mounts"])'
    )
    kernel = kernel[:start] + orchestration + kernel[end:]
    compile(kernel, str(kernel_path), "exec")
    kernel_path.write_text(kernel)
    archive_sha = manifest["archive_sha256"]
    manifest["archive_sha256"] = sha256(bundle / "bundle.wnn")
    if archive_sha != manifest["archive_sha256"]:
        raise SystemExit("kernel bootstrap sealing must not mutate the opaque bundle archive")
    write_json(
        packet / "KERNEL_CONFIGURATION_R3.json",
        {
            "schema": "wnn.kaggle.teacher_cohort_kernel_config.r3",
            "dataset": DATASET,
            "kernel_id": json.loads((bundle / "kernel/kernel-metadata.json").read_text())["id"],
            "kernel_title": json.loads((bundle / "kernel/kernel-metadata.json").read_text())[
                "title"
            ],
            "bundle_archive_sha256": manifest["archive_sha256"],
            "bundle_manifest_sha256": sha256(manifest_path),
            "bootstrap_sha256": sha256(kernel_path),
            "bootstrap_bytes": kernel_path.stat().st_size,
            "profile_source_manifest_sha256": PROFILE_SOURCE_SHA,
            "single_t4_reference_archive_sha256": PROFILE_ARCHIVE_SHA,
            "wall_ceiling_seconds": 14400,
            "two_device_gpu_seconds_upper_bound": 28800,
            "full_cohort_fit_admission": False,
            "fit_eta_seconds": None,
            "export_eta_seconds": None,
            "total_eta_seconds": None,
        },
    )
    write_json(
        packet / "TEACHER_COHORT_PLAN_R3.json",
        {
            "schema": "wnn.kaggle.clean_t3_f0_teacher_cohort_plan.r3",
            "status": "frozen_preparation_not_admitted",
            "profile_parent": PROFILE_REL,
            "single_t4_profile_reference_sha256": sha256(packet / "PROFILE_REFERENCE_R3.json"),
            "source_snapshot_manifest_sha256": PROFILE_SOURCE_SHA,
            "split_manifest_sha256": SPLIT_SHA,
            "own_propensity_checkpoint_sha256": PROP_SHA,
            "fit_membership": {"train": 10535, "valid": 1405, "excluded_valid_eval": 468},
            "recipe": {
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
            "profile_stage": {
                "jobs": profile_names,
                "workers": 2,
                "distinct_devices_required": 2,
                "warmup_complete_batches_per_worker": 5,
                "measured_complete_batches_per_worker": 5,
                "profile_only_no_checkpoint": True,
            },
            "fit_stage": fit_plan,
            "export_stage": export_plan,
            "fit_concurrency": 2,
            "max_wall_seconds": 14400,
            "max_two_device_gpu_seconds": 28800,
            "actual_fit_seconds": None,
            "actual_export_seconds": None,
            "actual_total_eta_seconds": None,
            "admission": "Root must replace the pending conditional gate with an exact packet-bound conditional release. The driver starts no full fits unless both paired T4 profiles meet the sealed guard.",
        },
    )
    write_json(
        packet / "COLLECTION_CONTRACT_R3.json",
        {
            "schema": "wnn.kaggle.teacher_cohort_collection.r3",
            "status": "required_at_terminal_not_collected",
            "stages": ["paired_profiles", "three_teacher_fits", "nine_prediction_exports"],
            "profile_outputs": [
                "results/paired_profiles/summary.json",
                "per-worker result.json",
                "worker.log",
                "input_identity.json",
                "preflight_inventory.json",
            ],
            "fit_outputs": [
                "model_ep2.pt",
                "model.pt",
                "training_state.pt",
                "provenance.json",
                "history.json",
                "worker.log",
                "result.json",
                "input_identity.json",
            ],
            "export_outputs": [
                "fit_train and fit_valid cache arrays/sidecars",
                "full_valid feature-only arrays/sidecars with 1873 rows",
                "worker.log",
                "result.json",
                "input_identity.json",
            ],
            "terminal_checks": [
                "pairwise profile device UUID and resource-gate receipt",
                "3 fit and 9 export workers complete with zero return code",
                "copy both NPY and sidecar per fresh worker receipt",
                "verify shape, dtype, finiteness, source run, checkpoint, split and membership ancestry",
                "preserve provider terminal stdout/stderr and complete worker logs",
            ],
        },
    )
    write_json(
        packet / "ROOT_RELEASE_GATES_R3.json",
        {
            "schema": "wnn.root_conditional_fit_release.r3",
            "status": "pending",
            "authorized": False,
            "binding": {
                "dataset": DATASET,
                "bundle_archive_sha256": manifest["archive_sha256"],
                "bundle_manifest_sha256": sha256(manifest_path),
                "bootstrap_sha256": sha256(kernel_path),
                "gate_sha256": gate_sha,
            },
            "requirements": [
                "fresh provider quota/session and exact private dataset readiness/version",
                "actual two-T4 paired profiles pass the embedded resource/update guard",
                "three-fit/9-export worker bounds and output capacity admitted",
            ],
            "wall_ceiling_seconds": 14400,
            "two_device_gpu_seconds_upper_bound": 28800,
            "all_eta_seconds": None,
        },
    )
    (packet / "README_R3.md").write_text(
        f"""# Clean T3 fold-0 teacher cohort R3\n\nThis identified packet runs a short paired-device profile stage, then conditionally runs three fresh H384 Muon-all teachers (seeds 3/4/5, 3 epochs, fixed epoch-2 checkpoint) and their nine required exports. The paired stage uses the actual R3 diagnostic source closure `{PROFILE_SOURCE_SHA}` and five warmup plus five measured complete train batches on each T4. It must return two distinct T4 UUIDs, positive completed optimizer counts, finite metrics, and stay within the per-worker and combined RAM/VRAM guard. The runner refuses a serial profile before starting any worker. If any profile fails, the driver writes a refusal receipt and starts no fit; there is no automatic serial fallback.\n\nThe previous one-T4 profile measured 5+25 complete batches, 599 completed optimizer updates, 1 AMP-skipped update, 4,802,109,440B peak worker RAM, and 3,499,092,112B peak GPU memory. It does not establish two-device concurrency or a full-fit estimate. Its `--grad-stats` instrumentation adds synchronization. Full-cohort fit/export ETA therefore remains null.\n\nFold-0 fitting is source-qualified: 10,535 train-fit and 1,405 valid-fit groups; all 468 fold-0 valid evaluation groups are excluded. Fit-membership predictions (10,535/1,405 rows) are separate from full-valid feature-only predictions (1,873 rows). Each run retains fixed CP2, final model, full training state, provenance/history and worker receipts.\n\nThe private dataset is `{DATASET}` and title-derived kernel ID is `{json.loads((bundle / "kernel/kernel-metadata.json").read_text())["id"]}`. The embedded root admission gate is pending; this builder performed no upload, provider mutation, or fit. The kernel wall ceiling is 14,400 seconds and the two-device GPU-second ceiling is 28,800 seconds, not an ETA or admission.\n"""
    )
    if kernel_path.stat().st_size >= 1_000_000:
        raise SystemExit("bootstrap exceeds the conservative 1MB source envelope")
    print(
        json.dumps(
            {
                "packet": str(packet),
                "bundle": str(bundle),
                "archive_sha256": manifest["archive_sha256"],
                "manifest_sha256": sha256(manifest_path),
                "bootstrap_sha256": sha256(kernel_path),
                "profile_jobs": 2,
                "fit_jobs": len(fit_specs),
                "export_jobs": len(export_specs),
                "admission": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
