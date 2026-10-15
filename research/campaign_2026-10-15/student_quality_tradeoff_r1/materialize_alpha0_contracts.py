#!/usr/bin/env python3
"""Materialize source-bound alpha-zero continuation contracts without launching fits."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any  # JSON records have heterogeneous values by design.

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
SOURCE_ROOT = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    / "ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort/source_snapshot"
)
TEMPLATE_DIR = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    / "ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/student_fit_r2"
)
ATTEMPT_DIR = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts"
MAIN_CAMPAIGN_DIR = ROOT / "research/campaign_2026-10-15/student_quality_tradeoff_r1"
OUTPUT_DIR = Path(__file__).resolve().parent
SPLIT_PATH = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
RUNTIME_DIR = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime"

STAGE_SPECS = (
    {
        "part": "p1",
        "epochs": [0, 1],
        "stop_after_epoch": 1,
        "a9_run": "clean_t3_f0_control_s3_a9_t3r5_r2_p1",
        "a9_attempt": "clean-t3-f0-control-s3-a9-r2-p1-20261012.json",
        "a0_run": "student_f0_warm_s3_alpha0_r1_p1",
        "a0_attempt": "student-f0-warm-s3-alpha0-r1-p1-20261015",
        "profile": "student_f0_warm_s3_alpha0_r1_p1",
        "resume_run": None,
        "curve_epoch": 1,
    },
    {
        "part": "p2",
        "epochs": [2, 3],
        "stop_after_epoch": 3,
        "a9_run": "clean_t3_f0_control_s3_a9_t3r5_r2_p2_envr1",
        "a9_attempt": "clean-t3-f0-control-s3-a9-r2-p2-envr1-20261012.json",
        "a0_run": "student_f0_warm_s3_alpha0_r1_p2_envr1",
        "a0_attempt": "student-f0-warm-s3-alpha0-r1-p2-envr1-20261015",
        "profile": "student_f0_warm_s3_alpha0_r1_p2_envr1",
        "resume_run": "student_f0_warm_s3_alpha0_r1_p1",
        "curve_epoch": 3,
    },
    {
        "part": "p3",
        "epochs": [4, 5],
        "stop_after_epoch": 5,
        "a9_run": "clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1",
        "a9_attempt": "clean-t3-f0-control-s3-a9-r2-p3-envr1-20261012.json",
        "a0_run": "student_f0_warm_s3_alpha0_r1_p3_envr1",
        "a0_attempt": "student-f0-warm-s3-alpha0-r1-p3-envr1-20261015",
        "profile": "student_f0_warm_s3_alpha0_r1_p3_envr1",
        "resume_run": "student_f0_warm_s3_alpha0_r1_p2_envr1",
        "curve_epoch": 5,
    },
)

WORKER_RAM_BYTES = 5_500_000_000
OUTER_RAM_BYTES = 500_000_000
ALLOCATOR_BYTES = 4_300_000_000
VRAM_CEILING_BYTES = 5_000_000_000
TRAINER_SHA256 = "a2b5440ff4b8e5179db01cc047102ef634522327374fced31ca5fdf8afcd17a6"
WRAPPER_SHA256 = "246586bfd4be3a4e3d95ac026860bb835611f09958193b75798124c9500455ab"
PREDICTOR_SHA256 = "30647bb74da64ddab4ccaa5e2916338bbb4826466b47cd68ec61593269554dd0"
SOURCE_MANIFEST_SHA256 = "41a5452c752bb10bee28af2eb2e06e7a224da8d72f307b4d5bec6fb5b3b236fa"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash canonical JSON for a recipe or structured source receipt."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Load an object-valued JSON file."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a stable, reviewable JSON artifact."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def assert_sha256(path: Path, expected: str, label: str) -> str:
    """Require a source file to match its frozen digest."""
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def verify_frozen_sources() -> dict[str, Any]:
    """Verify the frozen source snapshot and bounded inputs used by the recipe."""
    source_manifest = SOURCE_ROOT / "SOURCE_MANIFEST_H390.json"
    assert_sha256(source_manifest, SOURCE_MANIFEST_SHA256, "source manifest")
    manifest = read_json(source_manifest)
    source_hashes: dict[str, str] = {}
    for relative_path, expected_hash in manifest["sources"].items():
        path = SOURCE_ROOT / relative_path
        source_hashes[relative_path] = assert_sha256(path, expected_hash, relative_path)
    if source_hashes["scripts/train_rnn.py"] != TRAINER_SHA256:
        raise ValueError("frozen trainer digest differs from the packet binding")
    if source_hashes["scripts/cleanfold_exec.py"] != WRAPPER_SHA256:
        raise ValueError("frozen execution wrapper digest differs from the packet binding")
    if source_hashes["scripts/predict_run.py"] != PREDICTOR_SHA256:
        raise ValueError("frozen predictor digest differs from the packet binding")

    split_sha = assert_sha256(
        ROOT / SPLIT_PATH,
        "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d",
        "fold-0 split",
    )
    parent_checkpoint = ROOT / "artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"
    parent_provenance = ROOT / "artifacts/runs/c20261003_combo_s3_f0/provenance.json"
    parent_checkpoint_sha = assert_sha256(
        parent_checkpoint,
        "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e",
        "warm parent",
    )
    parent_provenance_sha = assert_sha256(
        parent_provenance,
        "9a9d17eeb4b844d85bb5ca132e0085d0cf6dd1387d3d3de4c56d8e3bb8be7a6f",
        "parent provenance",
    )
    cache_metadata = ROOT / "cache/clean_t3_f0_h390_t3r5_a9_r2/metadata.json"
    cache_metadata_sha = assert_sha256(
        cache_metadata,
        "1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8",
        "clean-T3 cache metadata",
    )

    refs = {
        "p1_contract": TEMPLATE_DIR / "CONTROL_P1_FIT_CONTRACT_R1.json",
        "successor_contract": TEMPLATE_DIR / "CONTROL_TREATMENT_SUCCESSOR_STAGES_CONTRACT_R1.json",
        "successor_launch": TEMPLATE_DIR / "CONTROL_TREATMENT_SUCCESSOR_LAUNCH_TEMPLATES_R1.json",
        "p1_launch": TEMPLATE_DIR / "CONTROL_P1_LAUNCH_MANIFEST_R1.json",
        "env_correction": TEMPLATE_DIR
        / "CONTROL_TREATMENT_ENV_PREFIXED_SUCCESSOR_BINDINGS_R1.json",
        "grade_report": TEMPLATE_DIR
        / "f0_grade_r1/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1.grade.json",
    }
    expected_refs = {
        "p1_contract": "17350c0616debacd3507ad25bc627ba2f7087ddb17e0398cedf7674d696aaf79",
        "successor_contract": "957a49c882e4745404c0ec7ff545f5835249291e089cd0888b23aaefb2923a8e",
        "successor_launch": "6f4c7912449a45ed1e5a2baaa2a0f9c9b5fedd727b4d62099543cd47d70a664a",
        "p1_launch": "",
        "env_correction": "",
        "grade_report": "c518f7cdb5f3e5148f72de152c3aaecd308ff97c83aa74169fdaa95442e085a0",
    }
    ref_hashes = {name: sha256_file(path) for name, path in refs.items()}
    for name, expected in expected_refs.items():
        if expected and ref_hashes[name] != expected:
            raise ValueError(f"frozen reference {name} changed: {ref_hashes[name]}")

    attempt_hashes: dict[str, str] = {}
    for stage in STAGE_SPECS:
        attempt_path = ATTEMPT_DIR / stage["a9_attempt"]
        attempt_hashes[stage["part"]] = sha256_file(attempt_path)
    return {
        "source_snapshot_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "frozen_source_file_count": len(source_hashes),
        "frozen_source_file_sha256": source_hashes,
        "fold0_split_sha256": split_sha,
        "parent_checkpoint_sha256": parent_checkpoint_sha,
        "parent_provenance_sha256": parent_provenance_sha,
        "teacher_cache_metadata_sha256": cache_metadata_sha,
        "reference_contract_sha256": ref_hashes,
        "a9_attempt_receipt_sha256": attempt_hashes,
    }


def replace_option(argv: list[str], option: str, value: str) -> None:
    """Replace the single value following a required command-line option."""
    matches = [index for index, token in enumerate(argv) if token == option]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {option} in argv; found {len(matches)}")
    argv[matches[0] + 1] = value


def actual_a9_command(stage: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Read the exact successful alpha-.9 worker vector for this stage."""
    attempt = read_json(ATTEMPT_DIR / stage["a9_attempt"])
    if attempt.get("exit_status") != 0 or attempt.get("status") != "finished":
        raise ValueError(
            f"alpha-.9 source attempt is not a successful terminal receipt: {stage['part']}"
        )
    argv = list(attempt["command"])
    if argv.count("--distill-alpha") != 1 or argv[argv.index("--distill-alpha") + 1] != "0.9":
        raise ValueError(f"alpha-.9 command pin is invalid: {stage['part']}")
    if argv[argv.index("--name") + 1] != stage["a9_run"]:
        raise ValueError(f"alpha-.9 run name differs from terminal command: {stage['part']}")
    if argv[argv.index("--stop-after-epoch") + 1] != str(stage["stop_after_epoch"]):
        raise ValueError(f"stage stop epoch differs from terminal command: {stage['part']}")
    if stage["resume_run"] is None:
        if "--resume-state" in argv:
            raise ValueError("P1 reference unexpectedly resumes a predecessor")
    else:
        resume = argv[argv.index("--resume-state") + 1]
        expected_resume_fragment = (
            f"artifacts/runs/{stage['a9_run'].rsplit('_p', 1)[0]}_p{int(stage['part'][1]) - 1}"
        )
        if expected_resume_fragment not in resume and "_envr1/training_state.pt" not in resume:
            raise ValueError(
                f"alpha-.9 predecessor path is unexpected in {stage['part']}: {resume}"
            )
    return attempt, argv


def alpha0_command(stage: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Clone a successful A9 argv and apply only alpha0's run and resume paths."""
    attempt, argv = actual_a9_command(stage)
    original_argv = list(argv)
    replace_option(argv, "--name", stage["a0_run"])
    replace_option(argv, "--distill-alpha", "0.0")
    if stage["resume_run"] is not None:
        replace_option(
            argv,
            "--resume-state",
            f"artifacts/runs/{stage['resume_run']}/training_state.pt",
        )
    if argv[argv.index("--distill-alpha") + 1] != "0.0":
        raise ValueError("alpha0 coefficient was not applied")
    allowed_indices = {
        original_argv.index("--name") + 1,
        original_argv.index("--distill-alpha") + 1,
    }
    if stage["resume_run"] is not None:
        allowed_indices.add(original_argv.index("--resume-state") + 1)
    changed_indices = {
        index
        for index, (old, new) in enumerate(zip(original_argv, argv, strict=True))
        if old != new
    }
    if changed_indices != allowed_indices:
        raise ValueError(f"unexpected A9-to-alpha0 argv diff in {stage['part']}: {changed_indices}")
    return attempt, argv


def canonical_recipe(packet: dict[str, Any]) -> dict[str, Any]:
    """Return the fully specified label-only continuation semantics."""
    recipe = copy.deepcopy(packet["common_recipe"])
    recipe.update(
        {
            "distill_alpha": 0.0,
            "main_target": "clip(y, -2, 2)",
            "teacher_target_coefficient": 0.0,
            "teacher_cache_path_retained_for_argv_parity": packet["cache"]["root"],
            "teacher_target_exposure": "zero coefficient; cache remains configured identically",
            "evaluation_precision": "fp32",
        }
    )
    return recipe


def stage_contract(
    packet: dict[str, Any],
    source_receipt: dict[str, Any],
    stage: dict[str, Any],
    recipe: dict[str, Any],
) -> dict[str, Any]:
    """Build one explicit alpha-zero stage contract from its actual A9 receipt."""
    a9_attempt, a9_argv = actual_a9_command(stage)
    a0_attempt, a0_argv = alpha0_command(stage)
    trainer_source_index = a0_argv.index(str(SOURCE_ROOT / "scripts/cleanfold_exec.py"))
    trainer_argv = a0_argv[trainer_source_index:]
    output = f"artifacts/runs/{stage['a0_run']}"
    profile = stage["profile"]
    attempt_id = stage["a0_attempt"]
    return {
        "schema": f"wnn.student-warmparent-alpha0-{stage['part']}-fit-contract.r2",
        "status": "PREPARED_CODE_ONLY_REQUIRES_REVIEW_AND_FRESH_ROOT_RELEASE",
        "fit_authorized": False,
        "stage_id": f"alpha0_{stage['part']}",
        "arm": "warmparent_label_only_alpha0",
        "run_name": stage["a0_run"],
        "attempt_id": attempt_id,
        "profile_id": profile,
        "source_revision": {
            "main_checkout_commit_at_materialization": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "trainer_sha256": TRAINER_SHA256,
            "wrapper_sha256": WRAPPER_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_manifest_inventory_verified": True,
        },
        "science": {
            "question": packet["question"],
            "claim_scope": packet["claim_scope"],
            "exposure": packet["membership"]["exposure"],
            "only_objective_coefficient_change_vs_a9": True,
            "alpha0_target": "clip(y, -2, 2)",
            "shared_teacher_cache_remains_bound_but_coefficient_is_zero": True,
        },
        "recipe": recipe,
        "recipe_sha256": canonical_sha256(recipe),
        "source_bindings": {
            "parent_checkpoint_sha256": packet["parent"]["checkpoint_sha256"],
            "parent_provenance_sha256": packet["parent"]["provenance_sha256"],
            "split_manifest_sha256": packet["membership"]["split_sha256"],
            "fit_train_order_sha256": packet["membership"]["seed3_fit_loader_order_sha256"],
            "fit_valid_order_sha256": packet["membership"]["fit_valid_order_sha256"],
            "teacher_cache_metadata_sha256": packet["cache"]["metadata_sha256"],
            "frozen_a9_attempt_receipt_sha256": source_receipt["a9_attempt_receipt_sha256"][
                stage["part"]
            ],
            "frozen_a9_run_name": stage["a9_run"],
            "a9_command_sha256": canonical_sha256(a9_argv),
            "alpha0_command_sha256": canonical_sha256(a0_argv),
        },
        "stage": {
            "part": stage["part"],
            "epoch_indices": stage["epochs"],
            "stop_after_epoch_inclusive": stage["stop_after_epoch"],
            "resume_run": stage["resume_run"],
            "resume_state_path": None
            if stage["resume_run"] is None
            else f"artifacts/runs/{stage['resume_run']}/training_state.pt",
            "expected_next_epoch_at_boundary": stage["stop_after_epoch"] + 1,
            "expected_optimizer_updates": 14920,
            "full_schedule_optimizer_updates": 44760,
            "worker_argv": a0_argv,
            "trainer_argv": trainer_argv,
            "output_directory": output,
            "expected_outputs": [
                f"{output}/training_state.pt",
                f"{output}/provenance.json",
                f"{output}/history.json",
                f"{output}/model_ep{stage['curve_epoch']}.pt",
            ],
            "a9_source_worker_argv": a9_argv,
            "a9_source_run_directory": f"artifacts/runs/{stage['a9_run']}",
            "actual_a9_attempt_receipt": str(ATTEMPT_DIR / stage["a9_attempt"]),
        },
        "resources": {
            "worker": {
                "cpu_cores": 1,
                "ram_reserve_bytes": WORKER_RAM_BYTES,
                "allocator_bytes": ALLOCATOR_BYTES,
                "gpu_workers": 1,
                "vram_ceiling_bytes": VRAM_CEILING_BYTES,
                "loader_workers": 0,
                "timeout_seconds": 3600,
                "swap_bytes": 0,
            },
            "outer_controller": {
                "cpu_cores": 1,
                "ram_bytes": OUTER_RAM_BYTES,
                "runtime_seconds": 380,
            },
            "admission": {
                "fresh_aggregate_observation_required": True,
                "parent_current_plus_reserve_max_bytes": 18_000_000_000,
                "cpu_total_max": 5,
                "shared_gpu_lock_required": True,
                "gpu_must_be_idle": True,
                "caps_may_not_change": True,
            },
        },
        "launch_gates": {
            "independent_delta_review_required": True,
            "root_release_required_for_this_stage": True,
            "output_absence_recheck_required": True,
            "resume_state_loader_and_membership_validation_required": stage["part"] != "p1",
            "launch_authorized": False,
        },
    }


def update_launch_argv(argv: list[str], stage: dict[str, Any], contract_path: Path) -> list[str]:
    """Create a fresh controller argv whose worker is the exact alpha-zero vector."""
    result = list(argv)
    replace_option(result, "--attempt", stage["a0_attempt"])
    replace_option(result, "--profile", stage["profile"])
    replace_option(result, "--contract", str(contract_path))
    contract_index = result.index("--contract") + 1
    result[contract_index] = str(contract_path)
    separator = result.index("--")
    _, worker = alpha0_command(stage)
    del result[separator + 1 :]
    result.extend(worker)
    if result[separator + 1 :] != worker:
        raise ValueError("controller tail differs from the alpha-zero worker vector")
    return result


def materialize() -> dict[str, Any]:
    """Generate exact fit, launch, and post-fit command packets; never execute them."""
    packet = read_json(OUTPUT_DIR / "MATCHED_WARMPARENT_ALPHA0_PACKET_R1.json")
    source_receipt = verify_frozen_sources()
    recipe = canonical_recipe(packet)
    recipe_sha = canonical_sha256(recipe)

    contracts: dict[str, dict[str, Any]] = {}
    for stage in STAGE_SPECS:
        contract = stage_contract(packet, source_receipt, stage, recipe)
        contract["recipe_sha256"] = recipe_sha
        contracts[stage["part"]] = contract
        write_json(OUTPUT_DIR / f"ALPHA0_{stage['part'].upper()}_FIT_CONTRACT_R2.json", contract)

    p1_launch_ref = read_json(TEMPLATE_DIR / "CONTROL_P1_LAUNCH_MANIFEST_R1.json")
    successor_launch_ref = read_json(
        TEMPLATE_DIR / "CONTROL_TREATMENT_SUCCESSOR_LAUNCH_TEMPLATES_R1.json"
    )
    launch_stages: list[dict[str, Any]] = []
    for stage in STAGE_SPECS:
        contract_output_path = OUTPUT_DIR / f"ALPHA0_{stage['part'].upper()}_FIT_CONTRACT_R2.json"
        contract_sha = sha256_file(contract_output_path)
        contract_path = MAIN_CAMPAIGN_DIR / contract_output_path.name
        if stage["part"] == "p1":
            base = copy.deepcopy(p1_launch_ref)
            controller_base = base["controller_argv"]
            controller_prefix = controller_base[: controller_base.index("--")]
            env = base["limited_environment"]
            caps = base["resource_caps"]
            worker_argv = contracts["p1"]["stage"]["worker_argv"]
        else:
            base = next(
                item
                for item in successor_launch_ref["stages"]
                if item["arm"] == "control" and item["part"] == stage["part"]
            )
            controller_base = base["controller_argv_after_fresh_release"]
            controller_prefix = controller_base[: controller_base.index("--")]
            env = p1_launch_ref["limited_environment"]
            caps = base["resource_caps"]
            worker_argv = contracts[stage["part"]]["stage"]["worker_argv"]
        launch_template_argv = update_launch_argv(controller_prefix + ["--"], stage, contract_path)
        contract_relpath = str(contract_path)
        stage_template = {
            "stage_id": f"alpha0_{stage['part']}",
            "part": stage["part"],
            "run_name": stage["a0_run"],
            "attempt_id": stage["a0_attempt"],
            "systemd_unit": f"wnn-bandit-{stage['a0_attempt']}.service",
            "profile_id": stage["profile"],
            "contract_path": contract_relpath,
            "contract_sha256": contract_sha,
            "worker_argv": worker_argv,
            "controller_argv_after_fresh_release": launch_template_argv,
            "limited_wrapper": str(ROOT / "scripts/limited"),
            "outer_limited_environment": env,
            "resource_caps": caps,
            "output_directory": str(ROOT / "artifacts/runs" / stage["a0_run"]),
            "output_absence_at_materialization": "NOT_CHECKED; require absence check immediately before launch",
            "resume_state_path": None
            if stage["resume_run"] is None
            else str(ROOT / "artifacts/runs" / stage["resume_run"] / "training_state.pt"),
            "resume_state_sha256": None,
            "fresh_state_and_membership_validation_required": stage["part"] != "p1",
            "fresh_root_release_required": True,
            "launch_authorized": False,
        }
        if worker_argv != launch_template_argv[launch_template_argv.index("--") + 1 :]:
            raise ValueError(f"controller argv is not bound to the worker for {stage['part']}")
        launch_stages.append(stage_template)

    launch_manifest = {
        "schema": "wnn.student-warmparent-alpha0-exact-launch-templates.r2",
        "status": "PREPARED_NO_FIT_NO_LAUNCH",
        "fit_authorized": False,
        "purpose": "Exact source-bound one-GPU sequential commands; root release and fresh admission remain required per stage.",
        "source_reference": {
            "original_successful_attempts": source_receipt["a9_attempt_receipt_sha256"],
            "frozen_trainer_sha256": TRAINER_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "resource_policy": {
            "one_fit_stage_at_a_time": True,
            "worker_reserve_bytes": WORKER_RAM_BYTES,
            "outer_controller_ram_bytes": OUTER_RAM_BYTES,
            "cpu_cores_per_worker": 1,
            "gpu_workers": 1,
            "vram_ceiling_bytes": VRAM_CEILING_BYTES,
            "swap_bytes": 0,
            "fresh_root_admission_required_for_every_launch": True,
        },
        "stages": launch_stages,
    }
    write_json(OUTPUT_DIR / "ALPHA0_LAUNCH_TEMPLATES_R2.json", launch_manifest)

    predictor_path = str(SOURCE_ROOT / "scripts/predict_run.py")
    grader_path = ROOT / "scripts/kaggle_warm_cohort/grade_fold0_run.py"
    grader_sha = sha256_file(grader_path)
    snapshots: list[dict[str, Any]] = []
    for arm, run_key in (("alpha0", "a0_run"), ("alpha09", "a9_run")):
        for stage in STAGE_SPECS:
            run = stage[run_key]
            epoch = stage["curve_epoch"]
            grade_dir = str(MAIN_CAMPAIGN_DIR / f"f0_grade_{arm}_r2" / f"ep{epoch + 1}")
            export_argv = [
                str(ROOT / ".venv/bin/python"),
                predictor_path,
                run,
                "valid",
                "--epoch",
                str(epoch),
                "--campaign-split-manifest",
                SPLIT_PATH,
                "--device",
                "cuda",
                "--inference-precision",
                "fp32",
                "--campaign-full-valid-feature-only",
            ]
            grade_argv = [
                "env",
                "WNN_SLICE=wnn-campaign.slice",
                "WNN_CPU_QUOTA=100%",
                "WNN_THREADS=1",
                "WNN_MEM_MAX=1500000000",
                "WNN_RUNTIME_MAX=120s",
                "scripts/limited",
                str(ROOT / ".venv/bin/python"),
                str(grader_path),
                "--root",
                str(ROOT),
                "--run",
                run,
                "--epoch",
                str(epoch),
                "--out-dir",
                grade_dir,
            ]
            snapshots.append(
                {
                    "arm": arm,
                    "run_name": run,
                    "checkpoint_epoch_zero_based": epoch,
                    "after_epochs": epoch + 1,
                    "checkpoint_sha256": None
                    if arm == "alpha0"
                    else next(
                        item["sha256"]
                        for item in packet["existing_alpha09_curve_checkpoints"]
                        if item["after_epoch"] == epoch + 1
                    ),
                    "export_argv": export_argv,
                    "export_environment": {
                        "PYTHONPATH": str(SOURCE_ROOT / "src"),
                        "WNN_PROJECT_ROOT": str(ROOT),
                        "WNN_GPU_MEMORY_BYTES": str(ALLOCATOR_BYTES),
                        "WNN_LOADER_RAM_GB": "4",
                        "WNN_LOADER_WORKERS": "0",
                        "WNN_THREADS": "1",
                    },
                    "export_resource_gate": {
                        "one_gpu_worker": True,
                        "worker_reserve_bytes": WORKER_RAM_BYTES,
                        "outer_controller_ram_bytes": OUTER_RAM_BYTES,
                        "outer_controller_runtime_seconds": 120,
                        "vram_ceiling_bytes": VRAM_CEILING_BYTES,
                        "shared_gpu_lock": True,
                        "fresh_parent_current_plus_reserve_le_bytes": 18_000_000_000,
                        "root_release_required": True,
                    },
                    "prediction_path": str(ROOT / "cache" / f"pred_{run}_valid.npy"),
                    "prediction_sidecar_path": str(ROOT / "cache" / f"pred_{run}_valid.json"),
                    "preflight": "Never overwrite an existing path without matching run/epoch/split/row-order/precision and content pins.",
                    "grade_argv": grade_argv,
                    "grade_environment": {
                        "WNN_SLICE": "wnn-campaign.slice",
                        "WNN_CPU_QUOTA": "100%",
                        "WNN_THREADS": "1",
                        "WNN_MEM_MAX": "1500000000",
                        "WNN_RUNTIME_MAX": "120s",
                    },
                    "grade_limits": {
                        "cpu_cores": 1,
                        "ram_max_bytes": 1_500_000_000,
                        "runtime_seconds": 120,
                    },
                    "metric": "global_wp_clip2_v1",
                    "evaluation_panel": "source-qualified fold-0 valid evaluation groups; exposed development panel",
                    "prediction_label_mask": "predictor reads features only; grader applies the frozen official mask",
                }
            )
    postfit = {
        "schema": "wnn.student-warmparent-alpha0-postfit-export-grade-commands.r1",
        "status": "PREPARED_COMMANDS_ONLY_NO_EXPORT_NO_GRADE",
        "source_pins": {
            "frozen_predictor_sha256": PREDICTOR_SHA256,
            "frozen_trainer_sha256": TRAINER_SHA256,
            "fold0_split_sha256": packet["membership"]["split_sha256"],
            "grader_sha256_at_materialization": grader_sha,
        },
        "protocol": {
            "snapshot_order": [
                "alpha0 ep2",
                "alpha09 ep2",
                "alpha0 ep4",
                "alpha09 ep4",
                "alpha0 ep6",
                "alpha09 ep6",
            ],
            "export_feature_only": True,
            "export_all_1873_valid_groups_in_target_row_order": True,
            "no_labels_or_masks_read_by_export": True,
            "inference_precision": "fp32",
            "fixed_endpoint_epoch": 5,
            "intermediate_points_descriptive_only": True,
            "fresh_resource_admission_and_output_preflight_per_export": True,
        },
        "snapshots": snapshots,
        "a9_fixed_endpoint_existing_grade": {
            "grade_report_path": str(
                TEMPLATE_DIR / "f0_grade_r1/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1.grade.json"
            ),
            "grade_sha256": packet["arms"][1]["endpoint_grade_sha256"],
            "wp": packet["arms"][1]["endpoint_wp_epoch5"],
            "reuse_only_after_exact_prediction_and_source_revalidation": True,
        },
        "no_fit_authorization": True,
        "no_public_score_forecast": True,
    }
    write_json(OUTPUT_DIR / "ALPHA0_POSTFIT_EXPORT_GRADE_COMMANDS_R1.json", postfit)

    outputs = [
        "ALPHA0_P1_FIT_CONTRACT_R2.json",
        "ALPHA0_P2_FIT_CONTRACT_R2.json",
        "ALPHA0_P3_FIT_CONTRACT_R2.json",
        "ALPHA0_LAUNCH_TEMPLATES_R2.json",
        "ALPHA0_POSTFIT_EXPORT_GRADE_COMMANDS_R1.json",
    ]
    receipt = {
        "schema": "wnn.student-warmparent-alpha0-sourcebound-materialization-validation.r2",
        "status": "PASS_SOURCE_PINS_AND_COMMAND_CONSTRUCTION; NO_FIT_NO_LAUNCH",
        "validated_utc_date": "2026-10-15",
        "fit_authorized": False,
        "source_validation": source_receipt,
        "recipe_sha256": recipe_sha,
        "stages": {
            stage["part"]: {
                "a9_reference_run": stage["a9_run"],
                "alpha0_run": stage["a0_run"],
                "a9_worker_argv_sha256": canonical_sha256(actual_a9_command(stage)[1]),
                "alpha0_worker_argv_sha256": canonical_sha256(alpha0_command(stage)[1]),
                "distill_alpha": 0.0,
                "unique_run_and_resume_paths": True,
            }
            for stage in STAGE_SPECS
        },
        "assertions": {
            "frozen_source_manifest_and_every_listed_source_file_match": True,
            "all_three_a9_reference_attempts_terminal_exit_zero": True,
            "a0_argv_differs_only_in_alpha_name_and_expected_resume_path": True,
            "all_launch_templates_tail_matches_embedded_worker_argv": True,
            "each_stage_requires_fresh_root_release_and_output_absence_recheck": True,
            "postfit_commands_bind_same_three_snapshots_per_arm_and_fp32": True,
            "no_fit_export_or_grade_executed": True,
        },
        "materialized_files_sha256": {name: sha256_file(OUTPUT_DIR / name) for name in outputs},
        "builder_sha256": sha256_file(Path(__file__)),
        "frozen_trainer_source_sha256": TRAINER_SHA256,
    }
    write_json(OUTPUT_DIR / "ALPHA0_SOURCEBOUND_VALIDATION_R2.json", receipt)
    return receipt


if __name__ == "__main__":
    validation = materialize()
    print(json.dumps({"status": validation["status"], "stages": validation["stages"]}, indent=2))
