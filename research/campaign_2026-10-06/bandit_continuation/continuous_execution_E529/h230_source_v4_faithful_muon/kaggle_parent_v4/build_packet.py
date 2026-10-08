from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# These packet builders consume nested JSON manifests with schema checks at each boundary.
PACKET_ROOT = Path(__file__).resolve().parent
V2_ROOT = PACKET_ROOT.parent / "kaggle_parent_v2"
REPO_ROOT = next(
    (parent for parent in PACKET_ROOT.parents if (parent / "specs/README.md").is_file()),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError("could not locate the project root")

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from kaggle_bundle import build_bundle  # noqa: E402
from kaggle_common import canonical_sha256, sha256_file, write_json  # noqa: E402

CONTINUOUS_ROOT = Path("research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529")
V4_ROOT = PACKET_ROOT.parent
V2_TRANSPORT_PATH = V2_ROOT / "H230_KAGGLE_PARENT_FIT_TRANSPORT_V2_20261008.json"
V3_TRANSPORT_PATH = (
    PACKET_ROOT.parent
    / "kaggle_parent_v3/H230_KAGGLE_PARENT_FIT_TRANSPORT_V3_20261008.json"
)
V2_SPEC_PATH = V2_ROOT / "spec/h230-seed9-parent-fit-v1.json"
V2_PROFILE_RECEIPT = (
    REPO_ROOT
    / CONTINUOUS_ROOT
    / "H230_STRONG_PARENT_SEED9_REAL_DATA_PROFILE_TERMINAL_RECEIPT_V1_20261008.json"
)
V4_SOURCE_MANIFEST = (
    REPO_ROOT
    / CONTINUOUS_ROOT
    / "h230_source_v4_faithful_muon/H230_FAITHFUL_MUON_SOURCE_MANIFEST_V4_20261008.json"
)
RECIPE_PATH = (
    REPO_ROOT
    / CONTINUOUS_ROOT
    / "h230_source_v4_faithful_muon/H230_STRONG_PARENT_SEED9_RECIPE_V2_20261008.json"
)
PORTABLE_INPUTS = REPO_ROOT / "research/campaign_2026-10-04/portable_inputs.json"
CACHE_UPLOAD_AUTH = (
    REPO_ROOT
    / CONTINUOUS_ROOT
    / "H230_MAIN_SOURCE_CACHE_ROOT_PRIVATE_UPLOAD_AUTHORIZATION_20261008_R1.json"
)
CACHE_UPLOAD_RECEIPT = (
    REPO_ROOT
    / CONTINUOUS_ROOT
    / "H230_MAIN_SOURCE_CACHE_PRIVATE_UPLOAD_TERMINAL_RECEIPT_V1_20261008.json"
)
V4_BUNDLE_DATASET = "vladkochetov07/h230-seed9-parent-fit-v4"
V4_SPEC_NAME = "h230_seed9_parent_profilefit_v4"
V4_ATTEMPT_ID = "h230-strong-parent-seed9-kaggle-profilefit-v4-20261008"
V4_FIT_RUN = "bd26_h230_strong_parent_seed9_muon_kaggle_v4"
V4_PROFILE_RUN = "bd26_h230_strong_parent_seed9_muon_kaggle_v4_profile"
PLAN_REL = (
    CONTINUOUS_ROOT / "h230_source_v4_faithful_muon/kaggle_parent_v4/profile_fit_plan_v1.json"
).as_posix()
SEQUENCE_REL = (
    CONTINUOUS_ROOT / "h230_source_v4_faithful_muon/kaggle_parent_v4/profile_fit_sequence_v1.json"
).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_once(path: Path, value: object) -> None:
    if path.exists():
        if _read_json(path) != value:
            raise FileExistsError(f"V4 frozen JSON output exists with different content: {path}")
        return
    write_json(path, value)


def _hash_json(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arg_value(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _replace_arg(argv: list[str], name: str, value: str) -> list[str]:
    result = list(argv)
    result[result.index(name) + 1] = value
    return result


def _remove_flag(argv: list[str], flag: str) -> list[str]:
    result = list(argv)
    if flag in result:
        result.remove(flag)
    return result


def _external_arguments(external_mounts: dict[str, object]) -> list[str]:
    result = []
    for logical, mapping in sorted(external_mounts.items()):
        if isinstance(mapping, str):
            source = mapping
        else:
            source = f"{mapping['dataset']}:{mapping['filename']}"
        result.append(f"{logical}={source}")
    return result


def _copy_v2_payload(destination: Path, v2_spec: dict[str, Any]) -> list[dict[str, Any]]:
    source_root = V2_ROOT / "payload_root"
    destination.mkdir(parents=True, exist_ok=True)
    existing = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    allowed_existing = {item["path"] for item in v2_spec["code"]}
    allowed_existing.update(
        {
            "scripts/h230_profile_then_fit.py",
            "scripts/kaggle_sequence.py",
            "research/campaign_2026-10-04/portable_inputs.json",
            (
                "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
                "H230_MAIN_SOURCE_CACHE_ROOT_PRIVATE_UPLOAD_AUTHORIZATION_20261008_R1.json"
            ),
            (
                "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
                "H230_MAIN_SOURCE_CACHE_PRIVATE_UPLOAD_TERMINAL_RECEIPT_V1_20261008.json"
            ),
            (
                "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
                "h230_source_v4_faithful_muon/kaggle_parent_v4/profile_fit_plan_v1.json"
            ),
            (
                "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
                "h230_source_v4_faithful_muon/kaggle_parent_v4/profile_fit_sequence_v1.json"
            ),
        }
    )
    if existing - allowed_existing:
        raise FileExistsError(
            "V4 payload staging directory contains unexpected files: "
            f"{sorted(existing - allowed_existing)}"
        )
    code = []
    for item in sorted(v2_spec["code"], key=lambda value: value["path"]):
        source = source_root / item["path"]
        target = destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != item["size"] or sha256_file(target) != item["sha256"]:
                raise FileExistsError(f"V4 frozen source target changed: {item['path']}")
        else:
            shutil.copyfile(source, target)
        if target.stat().st_size != item["size"] or sha256_file(target) != item["sha256"]:
            raise ValueError(f"V2 frozen source member changed during V4 staging: {item['path']}")
        code.append(dict(item))
    return code


def _copy_member(payload_root: Path, relative: str, source: Path) -> dict[str, Any]:
    target = payload_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        source_sha = sha256_file(source)
        if target.stat().st_size != source.stat().st_size or sha256_file(target) != source_sha:
            raise FileExistsError(
                f"V4 payload member already exists with different bytes: {relative}"
            )
        return {"path": relative, "size": target.stat().st_size, "sha256": source_sha}
    shutil.copyfile(source, target)
    return {"path": relative, "size": target.stat().st_size, "sha256": sha256_file(target)}


def _build_argv(v2_spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    base = v2_spec["argv"][1:]
    fit_argv = _replace_arg(base, "--name", V4_FIT_RUN)
    profile_argv = _remove_flag(
        _replace_arg(base, "--name", V4_PROFILE_RUN), "--save-training-state"
    )
    profile_argv.extend(["--grad-stats", "--profile-warmup", "1", "--profile-steps", "3"])
    return profile_argv, fit_argv


def _expected_fit_args() -> dict[str, object]:
    return {
        "name": V4_FIT_RUN,
        "seed": 9,
        "device": "cuda",
        "arch": "gru",
        "cell": "gru",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "valid_fold": -1,
        "train_source": "train+valid",
        "include_heldout_valid": True,
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "no_eval_during_fit": True,
        "save_training_state": True,
        "batch": 16,
        "chunk": 1000,
        "loss": "wmse",
        "loss_norm": "chunk",
        "lr": 0.001,
        "wd": 0.01,
        "optim": "muon",
        "sched": "onecycle",
        "warmup_frac": 0.05,
        "precision": "fp16",
        "muon_scope": "hidden",
        "muon_gatewise": "none",
        "muon_lr_mult": 3.0,
        "muon_momentum": 0.95,
        "muon_ortho_dtype_policy": "native_bf16_if_supported_else_fp32_v1",
        "propensity": "c20260929_prop_f0",
        "prop_power": 2.0,
        "prop_eps": 0.03,
        "valid_true_mask": 0.03,
        "aux_struct": 0.3,
        "aux_next": 0.3,
        "ema_teacher": 1.0,
        "ema_decay": 0.999,
        "eval_ema": True,
        "since_weight": 32.0,
        "since_floor": 0.25,
        "distill_cache_root": "cache/bd26_h230_fold0_teacher_family_all_rows_v1",
        "distill_cache_all_data": True,
        "distill_cache_variant": "source",
        "distill_alpha": 0.5,
        "distill_boot_h": 8,
        "distill_family_head": False,
        "distill_family_weight": 0.0,
    }


def _build_plan(v2_spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_argv, fit_argv = _build_argv(v2_spec)
    plan = {
        "schema": "h230-profile-gated-fit-plan-v1",
        "profile_argv": profile_argv,
        "fit_argv": fit_argv,
        "profile_run_name": V4_PROFILE_RUN,
        "fit_run_name": V4_FIT_RUN,
        "expected": {
            "device_name_substring": "T4",
            "device_capability": [7, 5],
            "cuda_bf16_supported": False,
            "orthogonalization_policy": "native_bf16_if_supported_else_fp32_v1",
            "orthogonalization_dtype": "float32",
            "muon_scope": "hidden",
            "muon_gatewise": "none",
            "muon_lr_multiplier": 3.0,
            "model_precision": "fp16",
            "distill_cache_variant": "source",
            "train_sequences": 10606,
            "valid_sequences": 1873,
            "profile_batch_count": 4,
            "gradient_observations": 80,
            "batches_per_epoch": 779,
            "epochs": 4,
            "model_shape": {
                "raw_feature_width": 112,
                "event_feature_width": 99,
                "model_input_width": 211,
                "model_output_width": 23,
            },
            "profile_args": {
                "name": V4_PROFILE_RUN,
                "seed": 9,
                "batch": 16,
                "chunk": 1000,
                "epochs": 4,
                "profile_warmup": 1,
                "profile_steps": 3,
                "save_training_state": False,
            },
            "fit_args": _expected_fit_args(),
            "source_manifest_sha256": _hash_json(V4_SOURCE_MANIFEST),
            "recipe_sha256": _hash_json(RECIPE_PATH),
        },
        "runtime_gate": {
            "allocator_memory_bytes": 4300000000,
            "worker_timeout_seconds": 9000,
            "timing_margin_multiplier": 1.15,
            "minimum_fit_setup_reserve_seconds": 120,
            "extra_fit_contract_reserve_seconds": 180,
            "checkpoint_reserve_seconds": 300,
        },
        "portable_registry_path": "research/campaign_2026-10-04/portable_inputs.json",
        "profile_log_path": f"artifacts/runs/{V4_PROFILE_RUN}/profile.log",
        "fit_log_path": f"artifacts/runs/{V4_FIT_RUN}/fit.log",
        "gate_receipt_path": "artifacts/h230_seed9_parent_v4_profile_fit_gate.json",
    }
    sequence = {
        "version": 1,
        "steps": [
            {
                "name": "profile_then_conditional_fit",
                "argv": [
                    "{python}",
                    "scripts/h230_profile_then_fit.py",
                    "--plan",
                    PLAN_REL,
                ],
            }
        ],
        "report": "artifacts/h230_seed9_parent_v4_sequence.json",
    }
    return plan, sequence


def _build_spec(
    v2_spec: dict[str, Any], code: list[dict[str, Any]], plan: dict[str, Any]
) -> dict[str, Any]:
    spec = json.loads(json.dumps(v2_spec))
    spec.update(
        {
            "name": V4_SPEC_NAME,
            "kind": "profile_gated_fit",
            "argv": ["{python}", "scripts/kaggle_sequence.py", "--plan", SEQUENCE_REL],
            "code": sorted(code, key=lambda item: item["path"]),
            "code_sha256": canonical_sha256(sorted(code, key=lambda item: item["path"])),
            "limits": {**v2_spec["limits"], "timeout_seconds": 9000},
            "artifacts": [
                f"artifacts/runs/{V4_PROFILE_RUN}",
                f"artifacts/runs/{V4_FIT_RUN}",
                str(plan["gate_receipt_path"]),
                "artifacts/h230_seed9_parent_v4_sequence.json",
            ],
            "environment": {
                **v2_spec["environment"],
                "WNN_PORTABLE_DATA_MANIFEST": "research/campaign_2026-10-04/portable_inputs.json",
                "WNN_DEFERRED_IDENTITY_PATHS": ",".join(
                    [
                        "datasets/train.parquet",
                        "datasets/valid.parquet",
                        "cache/bd26_h230_fold0_teacher_family_all_rows_v1/main_source_train.npy",
                        "cache/bd26_h230_fold0_teacher_family_all_rows_v1/main_source_valid.npy",
                    ]
                ),
            },
        }
    )
    return spec


def _patch_transport_deadlines(payload_root: Path, code: list[dict[str, Any]]) -> None:
    """Raise only the sealed transport's kernel and worker time ceilings for V4."""
    substitutions = {
        "scripts/kaggle_kernel.py": [
            ("KERNEL_PROCESS_WALL_SECONDS_MAX = 6_000", "KERNEL_PROCESS_WALL_SECONDS_MAX = 10_000"),
            ("6,000-second", "10,000-second"),
        ],
        "scripts/kaggle_gpu_runner.py": [
            ("WORKER_TIMEOUT_CEILING_SECONDS = 5_200", "WORKER_TIMEOUT_CEILING_SECONDS = 9_000"),
        ],
    }
    for relative, replacements in substitutions.items():
        path = payload_root / relative
        text = path.read_text(encoding="utf-8")
        for old, new in replacements:
            if old not in text:
                raise ValueError(f"V3 transport deadline pin not found in {relative}: {old}")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
        item = next((entry for entry in code if entry["path"] == relative), None)
        if item is None:
            raise ValueError(f"transport deadline source is not in sealed code closure: {relative}")
        item["size"] = path.stat().st_size
        item["sha256"] = sha256_file(path)


def build() -> None:
    bundle_root = PACKET_ROOT / "bundle"
    spec_dir = PACKET_ROOT / "spec"
    expected_spec_path = spec_dir / "h230-seed9-parent-fit-v4.json"
    if bundle_root.exists():
        if not bundle_root.is_dir() or any(bundle_root.iterdir()):
            raise FileExistsError("V4 bundle output exists; preserve and identify a successor")
        bundle_root.rmdir()
    if spec_dir.exists():
        existing_specs = {path.name for path in spec_dir.iterdir() if path.is_file()}
        if existing_specs - {expected_spec_path.name}:
            raise FileExistsError("V4 spec directory contains unexpected prior outputs")

    v2_transport = _read_json(V2_TRANSPORT_PATH)
    v2_spec = _read_json(V2_SPEC_PATH)
    payload_root = PACKET_ROOT / "payload_root"
    code = _copy_v2_payload(payload_root, v2_spec)
    _patch_transport_deadlines(payload_root, code)

    new_members = [
        ("scripts/kaggle_sequence.py", REPO_ROOT / "scripts/kaggle_sequence.py"),
        (
            "research/campaign_2026-10-04/portable_inputs.json",
            PORTABLE_INPUTS,
        ),
        (
            "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/H230_MAIN_SOURCE_CACHE_ROOT_PRIVATE_UPLOAD_AUTHORIZATION_20261008_R1.json",
            CACHE_UPLOAD_AUTH,
        ),
        (
            "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/H230_MAIN_SOURCE_CACHE_PRIVATE_UPLOAD_TERMINAL_RECEIPT_V1_20261008.json",
            CACHE_UPLOAD_RECEIPT,
        ),
        (
            "scripts/h230_profile_then_fit.py",
            PACKET_ROOT.parent / "kaggle_parent_v3/payload_root/scripts/h230_profile_then_fit.py",
        ),
    ]
    for relative, source in new_members:
        code.append(_copy_member(payload_root, relative, source))

    plan, sequence = _build_plan(v2_spec)
    plan_relative = "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v4_faithful_muon/kaggle_parent_v4/profile_fit_plan_v1.json"
    sequence_relative = "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h230_source_v4_faithful_muon/kaggle_parent_v4/profile_fit_sequence_v1.json"
    _write_json_once(PACKET_ROOT / "profile_fit_plan_v1.json", plan)
    _write_json_once(PACKET_ROOT / "profile_fit_sequence_v1.json", sequence)
    code.append(_copy_member(payload_root, plan_relative, PACKET_ROOT / "profile_fit_plan_v1.json"))
    code.append(
        _copy_member(payload_root, sequence_relative, PACKET_ROOT / "profile_fit_sequence_v1.json")
    )

    spec = _build_spec(v2_spec, code, plan)
    spec_path = PACKET_ROOT / "spec/h230-seed9-parent-fit-v4.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_once(spec_path, spec)

    v2_manifest = _read_json(V2_ROOT / "bundle/bundle-manifest.json")
    external = _external_arguments(v2_manifest["external_mounts"])
    build_bundle(
        payload_root,
        [spec_path],
        bundle_root,
        V4_BUNDLE_DATASET,
        external,
        [
            v2_transport["transports"]["existing_fold_assets_dataset"],
            v2_transport["transports"]["new_cache_asset_dataset"],
        ],
    )
    kernel_metadata_path = bundle_root / "kernel/kernel-metadata.json"
    kernel_metadata = _read_json(kernel_metadata_path)
    kernel_metadata["title"] = kernel_metadata["id"].split("/", 1)[1]
    write_json(kernel_metadata_path, kernel_metadata)

    cache_manifest = _read_json(V2_ROOT / "cache_asset_upload/assets-manifest.json")
    upload_receipt = _read_json(CACHE_UPLOAD_RECEIPT)
    cache_file_stats = []
    for item in cache_manifest["files"]:
        source = V2_ROOT / "cache_asset_upload" / item["filename"]
        stat = source.stat()
        cache_file_stats.append(
            {
                "filename": item["filename"],
                "sha256": item["sha256"],
                "size": stat.st_size,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "hardlink_count": stat.st_nlink,
                "local_bytes_rehashed": False,
            }
        )

    packet = {
        "schema": "h230-kaggle-parent-profile-gated-fit-transport-v4",
        "packet_id": "H230_STRONG_PARENT_SEED9_KAGGLE_PROFILEFIT_V4_20261008",
        "status": "prepared_for_root_review_no_code_upload_kernel_push_or_fit",
        "fit_authorized": False,
        "profile_contract": "profile-only first stage; the fitted source argv is executed only if every sealed T4/FP32/shape/gradient/resource/runtime gate passes and root writes the separate fit admission.",
        "attempt_id": V4_ATTEMPT_ID,
        "fit_run_name": V4_FIT_RUN,
        "profile_run_name": V4_PROFILE_RUN,
        "supersedes": {
            "path": str(V3_TRANSPORT_PATH.relative_to(REPO_ROOT)),
            "sha256": _hash_json(V3_TRANSPORT_PATH),
            "reason": "V4 is a resource-only successor to V3. It preserves V3 scientific argv, source pins, profile measurements, and gates while extending the transport deadlines to 9,000 worker seconds and a 10,000-second kernel-process/provider hold. V3 remains unchanged.",
        },
        "recipe_path": str(RECIPE_PATH.relative_to(REPO_ROOT)),
        "recipe_sha256": _hash_json(RECIPE_PATH),
        "source_manifest_path": str(V4_SOURCE_MANIFEST.relative_to(REPO_ROOT)),
        "source_manifest_sha256": _hash_json(V4_SOURCE_MANIFEST),
        "profile_receipt_path": str(V2_PROFILE_RECEIPT.relative_to(REPO_ROOT)),
        "profile_receipt_sha256": _hash_json(V2_PROFILE_RECEIPT),
        "cache_private_upload_authorization": {
            "path": str(CACHE_UPLOAD_AUTH.relative_to(REPO_ROOT)),
            "sha256": _hash_json(CACHE_UPLOAD_AUTH),
        },
        "cache_private_upload_receipt": {
            "path": str(CACHE_UPLOAD_RECEIPT.relative_to(REPO_ROOT)),
            "sha256": _hash_json(CACHE_UPLOAD_RECEIPT),
            "verified_at_utc": upload_receipt["verified_at_utc"],
            "creation_result": upload_receipt["remote_dataset"]["creation_result"],
            "canonical_dataset_id": upload_receipt["remote_dataset"]["canonical_id"],
            "remote_dataset_numeric_id": upload_receipt["remote_dataset"]["dataset_id"],
            "remote_inventory": upload_receipt["remote_dataset"]["files"],
        },
        "cache_asset_manifest": {
            "path": str(
                (V2_ROOT / "cache_asset_upload/assets-manifest.json").relative_to(REPO_ROOT)
            ),
            "sha256": _hash_json(V2_ROOT / "cache_asset_upload/assets-manifest.json"),
            "file_stat_pins": cache_file_stats,
        },
        "code_members": len(code),
        "source_57_members_byte_identical_to_v4": True,
        "profile_and_fit_contract": {
            "profile_argv": plan["profile_argv"],
            "fit_argv": plan["fit_argv"],
            "profile_plan_sha256": _hash_json(PACKET_ROOT / "profile_fit_plan_v1.json"),
            "sequence_plan_sha256": _hash_json(PACKET_ROOT / "profile_fit_sequence_v1.json"),
            "profile_checks": [
                "T4 device identity and compute capability 7.5",
                "Muon orthogonalization policy resolves to float32, independently of fp16 model autocast",
                "V4 constants plus parsed args derive raw112/event99/model-input211/model-output23",
                "1 warmup + 3 timed B16/C1000 batches produce 80 finite gradient observations",
                "exact all-data membership 10,606 train + 1,873 supplied valid and zero eval groups",
                "allocator allocated/reserved peaks remain <=4.3GB",
                "timed intervals finite and projected profile+fit <=9,000 seconds",
            ],
            "projection": plan["runtime_gate"],
            "raw_input_hash_handoff": "The generic runner defers only raw parquet and main-source cache arrays. The profile wrapper verifies raw train/valid against portable_inputs.json once before importing Torch; the frozen trainer's resume-contract reuses the existing stat-keyed digest cache in the same worker process. The unchanged all-data cache validator hashes and validates source arrays before both profile and fit.",
            "profile_fit_state": "Profile is executed with a unique output name and no --save-training-state. On gate pass the same sealed train_rnn module main() is invoked again with the original seed and full fit argv; it reconstructs normalization/model/optimizer/scheduler/scaler from scratch. Profile updates are discarded and never warm-start the fit.",
        },
        "limits": spec["limits"],
        "cloud_runtime": {
            "worker_timeout_seconds": 9000,
            "kernel_process_wall_seconds_max": 10000,
            "provider_hold_seconds": 10000,
            "runner_finalization_reserve_seconds": 300,
            "bootstrap_start_reserve_seconds": 700,
            "deadline_source": "sealed bundle scripts/kaggle_kernel.py and scripts/kaggle_gpu_runner.py",
            "effective_timeout_policy": "worker timeout is clamped to the minimum of the 9,000-second declaration and remaining 10,000-second kernel budget minus the 300-second finalization reserve; the 700-second start reserve is included in the provider/kernel hold.",
        },
        "input_identity_policy": "Raw train/valid and main-source train/valid arrays have exact sizes in the controller input identity receipt; raw parquet content is streamed through the frozen portable registry once before profile, while the frozen trainer's mandatory cache validator streams and verifies the deferred source arrays before profile and fit. All other declared inputs are SHA-256 verified by the generic worker runner before launch.",
        "non_authorizations": [
            "No code dataset upload was performed.",
            "No kernel push was performed.",
            "No fit or profile was launched by this builder.",
            "Root must review this exact packet and issue a separate fit authorization before any conditional fit can run.",
        ],
        "builder": {
            "path": str(Path(__file__).relative_to(REPO_ROOT)),
            "sha256": _hash_json(Path(__file__).resolve()),
            "v2_builder_sha256_recorded_before_ruff": v2_transport["builder_sha256"],
            "v2_current_builder_sha256": _hash_json(V2_ROOT / "build_packet.py"),
            "v2_builder_digest_note": "V2 transport retained the pre-Ruff builder digest; V2 source files and bundle remain unchanged. V4 records both old receipt value and current builder bytes.",
        },
        "bundle": {
            "dataset": V4_BUNDLE_DATASET,
            "archive_path": str((bundle_root / "bundle.wnn").relative_to(REPO_ROOT)),
            "archive_sha256": _hash_json(bundle_root / "bundle.wnn"),
            "manifest_sha256": _hash_json(bundle_root / "bundle-manifest.json"),
            "kernel_metadata_path": str(kernel_metadata_path.relative_to(REPO_ROOT)),
            "kernel_metadata_sha256": _hash_json(kernel_metadata_path),
            "kernel_id": kernel_metadata["id"],
            "kernel_title": kernel_metadata["title"],
        },
    }
    packet_path = PACKET_ROOT / "H230_KAGGLE_PARENT_FIT_TRANSPORT_V4_20261008.json"
    _write_json_once(packet_path, packet)
    readme_path = PACKET_ROOT / "README.md"
    readme = (
        "# H230 seed-9 parent Kaggle transport (V4)\n\n"
        "This identified successor preserves the V3 science and V4 57-file source closure. Its worker performs a real T4 profile first and conditionally continues into a fresh four-epoch fit only after the sealed FP32-Muon, finite-gradient, exact-shape, membership, allocator and 9,000-second projection gates pass. Root review and a separate fit authorization are required before upload or launch.\n\n"
        "The source-cache dataset was uploaded privately under its separate authorization; this V4 code bundle has not been uploaded and its kernel has not been pushed.\n"
    )
    if readme_path.exists():
        if readme_path.read_text(encoding="utf-8") != readme:
            raise FileExistsError("V4 README exists with changed bytes")
    else:
        readme_path.write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {
                "packet": str(packet_path.relative_to(REPO_ROOT)),
                "packet_sha256": _hash_json(packet_path),
                "spec": str(spec_path.relative_to(REPO_ROOT)),
                "spec_sha256": _hash_json(spec_path),
                "code_members": len(code),
                "archive_sha256": packet["bundle"]["archive_sha256"],
                "kernel_id": packet["bundle"]["kernel_id"],
                "kernel_title": packet["bundle"]["kernel_title"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    build()
