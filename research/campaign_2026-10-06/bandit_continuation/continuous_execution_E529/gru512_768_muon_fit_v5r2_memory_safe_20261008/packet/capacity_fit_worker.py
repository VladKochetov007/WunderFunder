"""Fit the already-profiled GRU pair on two independently initialized T4 workers."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import cast

import capacity_worker as runtime

PACKET_ID = "E529_H232_GRU512_768_MUON_SEED9_FIT_V5R2_20261008"
RECIPE_NAME = "GRU512_768_MUON_FIT_RECIPE_V5R2.json"
FIT_AUTH_NAME = "ROOT_FIT_AUTHORIZATION.json"
FIT_REVIEW_NAME = "ROOT_FIT_SOURCE_REVIEW.json"
PROFILE_EVIDENCE_MANIFEST = "producer_evidence/v4/manifest.json"
RUN_ROOT = Path("/kaggle/working/gru_capacity_v5r2_project")
RESULT_ROOT = Path("/kaggle/working/gru_capacity_v5r2_runner_results")
KERNEL_CAP_SECONDS = 26_000
ARM_CAPS_SECONDS = {"muon512": 15_500, "muon768": 22_500}
CPU_THREADS_PER_ARM = 2
ARM_DEFINITIONS = {
    "muon512": {"hidden": 512, "optimizer": "muon"},
    "muon768": {"hidden": 768, "optimizer": "muon"},
}
RUN_NAMES = {
    "muon512": "bd26_gru512_full_s9_f0_muon_seeded_b8_amp4096_capacityv5r2_fit_20261008",
    "muon768": "bd26_gru768_full_s9_f0_muon_seeded_b8_amp4096_capacityv5r2_fit_20261008",
}
EXPECTED_PROFILE_EVIDENCE = {
    "muon512": {
        "step_wall_ms": 1845.01,
        "updates": 230,
        "skips": 0,
        "peak_gpu_used_mb": 1039,
        "profile_log": "producer_evidence/v4/muon512-profile.log",
        "gpu_samples": "producer_evidence/v4/muon512-profile-gpu-samples.json",
    },
    "muon768": {
        "step_wall_ms": 2861.57,
        "updates": 230,
        "skips": 0,
        "peak_gpu_used_mb": 1663,
        "profile_log": "producer_evidence/v4/muon768-profile.log",
        "gpu_samples": "producer_evidence/v4/muon768-profile-gpu-samples.json",
    },
}
INPUT_NAMES = (
    "datasets/train.parquet",
    "datasets/valid.parquet",
    "cache/chain_folds.npz",
    "cache/chains.npz",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
)
RUNTIME_COPIES = (
    "capacity_worker.py",
    "train_entry.py",
    "verified_digest_handoff.py",
)
FINALIZATION_RESERVE_SECONDS = 600


def _sha256_file(path: Path) -> str:
    """Hash a packet receipt or source file in bounded memory."""
    return runtime.sha256_file(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a finite machine-readable receipt atomically."""
    runtime.write_json(path, payload)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path.name}")
    return payload


def _verify_profile_argv(packet_root: Path, arm_id: str, hidden: int) -> dict[str, object]:
    """Bind reused profile evidence to the exact executed trainer flags."""
    receipt = _read_json(
        packet_root / "producer_evidence/v4" / f"{arm_id}-profile-argv.json"
    )
    argv = receipt.get("argv")
    if (
        receipt.get("arm") != arm_id
        or receipt.get("phase") != "profile"
        or not isinstance(argv, list)
        or not all(isinstance(token, str) for token in argv)
    ):
        raise ValueError(f"V4 executed argv receipt is malformed for {arm_id}")
    expected = {
        "--device": "cuda",
        "--arch": "gru",
        "--cell": "gru",
        "--enc": "linear",
        "--hidden": str(hidden),
        "--layers": "2",
        "--dropout": "0",
        "--seed": "9",
        "--loss": "wmse",
        "--loss-norm": "chunk",
        "--train-source": "train+valid",
        "--train-seqs": "10607",
        "--valid-fold": "0",
        "--epochs": "4",
        "--checkpoint-policy": "fixed",
        "--checkpoint-epoch": "3",
        "--batch": "8",
        "--batch-order": "seeded",
        "--chunk": "2000",
        "--lr": "0.001",
        "--wd": "0.01",
        "--optim": "muon",
        "--muon-scope": "rnn",
        "--sched": "onecycle",
        "--precision": "fp16",
        "--amp-init-scale": "4096.0",
        "--profile-warmup": "3",
        "--profile-steps": "20",
        "--muon-ortho-dtype-policy": "native_bf16_if_supported_else_fp32_v1",
    }
    if "--no-eval-during-fit" not in argv or "--profile-trace-output" in argv:
        raise ValueError(f"V4 profile argv differs from the frozen profile contract for {arm_id}")
    for option, value in expected.items():
        if argv.count(option) != 1:
            raise ValueError(f"V4 profile argv option count changed: {option}")
        if argv[argv.index(option) + 1] != value:
            raise ValueError(f"V4 profile argv option changed: {option}")
    return {
        "model_precision": "fp16",
        "amp_init_scale": 4096.0,
        "profile_warmup_outer_batches": 3,
        "profile_timed_outer_batches": 20,
        "batch": 8,
    }


def _verify_profile_evidence(packet_root: Path) -> dict[str, object]:
    """Verify V4 producer logs against the root-reviewed successful B8 profiles."""
    manifest_path = packet_root / PROFILE_EVIDENCE_MANIFEST
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "wnn-gru-capacity-v4-profile-evidence-v1":
        raise ValueError("profile evidence manifest schema changed")
    if manifest.get("root_review_sha256") != runtime.sha256_file(
        packet_root / "producer_evidence/v4/root_posthoc_profile_review.json"
    ):
        raise ValueError("profile evidence is detached from its root review")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("profile evidence file map is malformed")
    for relative, descriptor in files.items():
        target = packet_root / relative
        if not isinstance(descriptor, dict) or set(descriptor) != {"bytes", "sha256"}:
            raise ValueError("profile evidence descriptor is malformed")
        if (
            not target.is_file()
            or target.stat().st_size != descriptor["bytes"]
            or _sha256_file(target) != descriptor["sha256"]
        ):
            raise ValueError(f"verified V4 producer evidence changed: {Path(relative).name}")

    report = _read_json(packet_root / "producer_evidence/v4/root_posthoc_profile_review.json")
    if report.get("status") != "posthoc_profile_evidence_verified_consumer_failure_not_scientific_negative":
        raise ValueError("root V4 review does not confirm the profile-only recovery scope")
    if report.get("fit_authorized") is not False:
        raise ValueError("V4 profile evidence must remain separate from fit authorization")
    if report.get("all_collected_manifest_files_verified") != 117:
        raise ValueError("root review does not bind the complete V4 terminal collection")
    terminal = _read_json(packet_root / "producer_evidence/v4/cohort_terminal_receipt.json")
    input_receipt = packet_root / "producer_evidence/v4/input_identity_receipt.json"
    if terminal.get("input_receipt_sha256") != _sha256_file(input_receipt):
        raise ValueError("V4 shared-input identity receipt differs from terminal provenance")

    evidence: dict[str, object] = {}
    for arm_id, expected in EXPECTED_PROFILE_EVIDENCE.items():
        argv_evidence = _verify_profile_argv(
            packet_root,
            arm_id,
            512 if arm_id == "muon512" else 768,
        )
        log_path = packet_root / str(expected["profile_log"])
        precision = None
        profile = None
        update = None
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPTIMIZER_COMPUTE_PRECISION "):
                precision = json.loads(line.removeprefix("OPTIMIZER_COMPUTE_PRECISION "))
            elif line.startswith("PROFILE "):
                profile = json.loads(line.removeprefix("PROFILE "))
            elif line.startswith("PROFILE_UPDATE_EVIDENCE "):
                update = json.loads(line.removeprefix("PROFILE_UPDATE_EVIDENCE "))
        if not all(isinstance(value, dict) for value in (precision, profile, update)):
            raise ValueError(f"V4 profile record is incomplete for {arm_id}")
        if profile.get("step_wall") != expected["step_wall_ms"]:
            raise ValueError(f"V4 measured B8 timing changed for {arm_id}")
        required_update = {
            "optimizer_attempts": expected["updates"],
            "optimizer_completed_updates": expected["updates"],
            "amp_skipped_updates": expected["skips"],
            "grad_scaler_enabled": True,
            "loss_checks": expected["updates"],
            "loss_finite": True,
            "gradient_finite": True,
            "valid_for_throughput": True,
        }
        if any(update.get(key) != value for key, value in required_update.items()):
            raise ValueError(f"V4 strict profile contract changed for {arm_id}")
        runtime._validate_precision_policy(
            {"optimizer": "muon"},
            {"optimizer_precision": precision},
            {"compute_capability": "7.5"},
        )
        precision_group = precision["groups"][0]
        if (
            precision_group.get("native_bf16_selected") is not False
            or precision_group.get("selected_dtype") != "float32"
        ):
            raise ValueError(f"V4 T4 orthogonalization dtype evidence changed for {arm_id}")
        samples = _read_json(packet_root / str(expected["gpu_samples"]))
        if samples.get("gpu_peak_used_mb") != expected["peak_gpu_used_mb"]:
            raise ValueError(f"V4 measured peak GPU use changed for {arm_id}")
        evidence[arm_id] = {
            "step_wall_ms": profile["step_wall"],
            "optimizer_attempts": update["optimizer_attempts"],
            "optimizer_completed_updates": update["optimizer_completed_updates"],
            "amp_skipped_updates": update["amp_skipped_updates"],
            **argv_evidence,
            "grad_scaler_enabled": update["grad_scaler_enabled"],
            "orthogonalization_policy": precision["policy"],
            "orthogonalization_dtype": precision_group["selected_dtype"],
            "native_bf16_selected": precision_group["native_bf16_selected"],
            "device_capability": precision_group["device_capability"],
            "gpu_peak_used_mb": samples["gpu_peak_used_mb"],
        }
    return {
        "manifest_sha256": _sha256_file(manifest_path),
        "root_review_sha256": manifest["root_review_sha256"],
        "profiles": evidence,
    }


def _validate_authorization(
    packet_root: Path,
    recipe: dict[str, object],
    auth: dict[str, object],
    review: dict[str, object],
    profile_evidence: dict[str, object],
) -> tuple[list[str], int]:
    """Fail closed unless root approved this exact fit-only packet and arm set."""
    if auth.get("schema") != "wnn-gru-capacity-fit-authorization-v1":
        raise ValueError("fit authorization schema is missing or unsupported")
    if auth.get("packet_id") != PACKET_ID or recipe.get("packet_id") != PACKET_ID:
        raise ValueError("fit authorization names a different packet")
    if auth.get("fit_authorized") is not True:
        raise PermissionError("fit authorization is pending")
    if review.get("status") != "approved" or review.get("packet_id") != PACKET_ID:
        raise PermissionError("root fit source review is pending")
    expected_hashes = {
        "recipe_sha256": _sha256_file(packet_root / RECIPE_NAME),
        "source_archive_sha256": _sha256_file(packet_root / "source/source_archive.tar"),
        "source_manifest_sha256": _sha256_file(
            packet_root / "source/source_archive_manifest.json"
        ),
        "profile_evidence_manifest_sha256": profile_evidence["manifest_sha256"],
        "root_review_sha256": _sha256_file(packet_root / FIT_REVIEW_NAME),
    }
    if any(auth.get(key) != value for key, value in expected_hashes.items()):
        raise PermissionError("fit authorization does not bind the exact packet and evidence")
    if auth.get("root_source_review_sha256") != expected_hashes["root_review_sha256"]:
        raise PermissionError("fit authorization review digest is inconsistent")
    arms = auth.get("fit_arm_ids")
    if not isinstance(arms, list) or not arms or not all(isinstance(arm, str) for arm in arms):
        raise PermissionError("fit authorization arm list is malformed")
    if len(arms) != len(set(arms)) or not set(arms).issubset(ARM_DEFINITIONS):
        raise PermissionError("fit authorization contains duplicate or unknown arms")
    if not set(arms).issubset({"muon512", "muon768"}):
        raise PermissionError("fit authorization contains a non-primary arm")
    if set(arms) != {"muon512", "muon768"}:
        raise PermissionError("V5R2 requires the complete matched Muon512/Muon768 pair")
    if auth.get("selected_batch") != 8:
        raise PermissionError("fit authorization must select the already-profiled B8 batch")
    worker_seconds = auth.get("max_worker_seconds")
    if isinstance(worker_seconds, bool) or not isinstance(worker_seconds, int) or worker_seconds < 1:
        raise PermissionError("fit authorization worker limit is malformed")
    if worker_seconds > KERNEL_CAP_SECONDS:
        raise PermissionError("fit authorization exceeds the packet worker cap")
    profile_rows = cast(dict[str, dict[str, object]], profile_evidence["profiles"])
    projected: dict[str, float] = {}
    for arm_id in arms:
        fit_seconds = float(profile_rows[arm_id]["step_wall_ms"]) / 1000.0 * 1492 * 4 * 1.20
        projected[arm_id] = round(fit_seconds, 3)
        required_seconds = fit_seconds + 60 + 420 + 300
        if required_seconds > min(ARM_CAPS_SECONDS[arm_id], worker_seconds):
            raise PermissionError("actual B8 profile projection exceeds authorized arm deadline")
    if auth.get("profile_projection_seconds") != projected:
        raise PermissionError("fit authorization projection differs from actual V4 profiles")
    return arms, worker_seconds


def _write_shared_binding(
    project_root: Path,
    prepared: dict[str, object],
) -> tuple[Path, str]:
    """Create one digest/stat binding reused by both independent arm workers."""
    receipt_path = Path(str(prepared["input_identity_receipt"]))
    sources: dict[str, object] = {}
    for row in cast(list[dict[str, object]], prepared["raw_sources"]):
        logical = str(row["logical_name"])
        sources[logical] = {
            "mount_path": str(row["path"]),
            "bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
            "identity": row["identity"],
        }
    assets = cast(dict[str, dict[str, object]], prepared["asset_sources"])
    for logical, row in assets.items():
        sources[logical] = {
            "mount_path": str(row["path"]),
            "bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
            "identity": row["identity"],
        }
    if set(sources) != set(INPUT_NAMES):
        raise ValueError("verified input preparation changed the six-file source closure")
    binding = {
        "schema": "wnn-gru-capacity-shared-input-binding-v1",
        "input_identity_receipt_sha256": _sha256_file(receipt_path),
        "sources": sources,
        "reuse_contract": {
            "project_inputs_are_read_only_symlinks": True,
            "cohort_identity_verification_count": 1,
            "per_arm_binding": "stat identity and symlink target only; no raw copy or rehash",
            "parsed_loader_shared": False,
            "os_page_cache_may_be_shared": True,
        },
    }
    binding_path = project_root / "fit_input_binding.json"
    _write_json(binding_path, binding)
    return binding_path, _sha256_file(binding_path)


def _copy_runtime_sources(packet_root: Path, project_root: Path) -> dict[str, str]:
    """Expose the reviewed fit entry modules under the runner's sealed project root."""
    script_root = project_root / "scripts"
    hashes: dict[str, str] = {}
    for name in RUNTIME_COPIES:
        source = packet_root / name
        if not source.is_file():
            raise FileNotFoundError(f"packet runtime dependency is missing: {name}")
        destination = script_root / name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"runner source already exists: {name}")
        shutil.copyfile(source, destination)
        hashes[f"scripts/{name}"] = _sha256_file(destination)
    return hashes


def _make_fit_context(
    project_root: Path,
    split_manifest: dict[str, object],
    source_hashes: dict[str, str],
    input_binding_path: Path,
    input_binding_sha256: str,
    recipe_sha256: str,
) -> tuple[Path, str]:
    """Bind the frozen split, source tree, and shared input identity for both arms."""
    evaluation_groups = list(map(int, split_manifest["evaluation"]["valid"]))
    if len(evaluation_groups) != 468 or evaluation_groups != sorted(set(evaluation_groups)):
        raise ValueError("evaluation groups differ from the frozen 468-member contract")
    context = {
        "schema": "wnn-gru-capacity-fit-context-v1",
        "packet_id": PACKET_ID,
        "recipe_sha256": recipe_sha256,
        "split_manifest_sha256": runtime.sha256_file(
            project_root / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
        ),
        "source_hashes": source_hashes,
        "evaluation_groups": evaluation_groups,
        "input_binding_path": str(input_binding_path),
        "input_binding_sha256": input_binding_sha256,
        "input_identity_receipt_path": str(project_root / "input_identity_receipt.json"),
    }
    path = project_root / "fit_context.json"
    _write_json(path, context)
    return path, _sha256_file(path)


def _code_descriptor(
    project_root: Path, path: str, expected_sha256: str
) -> dict[str, object]:
    target = project_root / path
    if not target.is_file():
        raise FileNotFoundError(f"sealed source member is missing: {path}")
    observed_sha256 = _sha256_file(target)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"sealed source member changed after archive extraction: {path}")
    return {"path": path, "size": target.stat().st_size, "sha256": observed_sha256}


def _make_arm_spec(
    project_root: Path,
    arm_id: str,
    arm_seconds: int,
    input_binding_path: Path,
    input_binding_sha256: str,
    fit_context_path: Path,
    fit_context_sha256: str,
) -> dict[str, object]:
    """Bind one GPU process to exact source, fit context and read-only mount identities."""
    context = _read_json(project_root / "fit_context.json")
    source_hashes = context.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("fit context is missing source hash bindings")
    archive_codes = [
        _code_descriptor(project_root, str(path), str(source_hashes[path]))
        for path in sorted(source_hashes)
    ]
    code = sorted(archive_codes, key=lambda item: str(item["path"]))
    sys.path.insert(0, str(project_root / "scripts"))
    from kaggle_common import canonical_sha256

    spec = {
        "version": 1,
        "name": arm_id,
        "kind": "gru_capacity_fit",
        "argv": [
            "{python}",
            "scripts/capacity_arm_worker.py",
            "--arm",
            arm_id,
            "--batch",
            "8",
            "--run-name",
            RUN_NAMES[arm_id],
        ],
        "code": code,
        "code_sha256": canonical_sha256(code),
        "inputs": [
            *[
                {
                    "path": logical,
                    "size": (project_root / logical).stat().st_size,
                    "sha256": _source_sha_from_context(project_root, logical),
                }
                for logical in INPUT_NAMES
            ],
            {
                "path": "fit_context.json",
                "size": fit_context_path.stat().st_size,
                "sha256": fit_context_sha256,
            },
            {
                "path": "fit_input_binding.json",
                "size": input_binding_path.stat().st_size,
                "sha256": input_binding_sha256,
            },
        ],
        "artifacts": [
            f"artifacts/runs/{RUN_NAMES[arm_id]}",
            f"capacity_receipts/{arm_id}",
            "input_identity_receipt.json",
            "input_cache_advice",
        ],
        "limits": {
            "cpus": CPU_THREADS_PER_ARM,
            "ram_bytes": 8 * 1024**3,
            "timeout_seconds": arm_seconds,
            "gpu_memory_bytes": 6 * 1024**3,
            "gpu_stop_bytes": 12 * 1024**3,
        },
        "environment": {
            "WNN_DEFERRED_IDENTITY_PATHS": ",".join(INPUT_NAMES),
            "WNN_PROJECT_ROOT": str(project_root),
            "WNN_INPUT_IDENTITY_RECEIPT": str(project_root / "input_identity_receipt.json"),
            "WNN_INPUT_BINDING_PATH": str(input_binding_path),
            "WNN_INPUT_BINDING_SHA256": input_binding_sha256,
            "WNN_FIT_CONTEXT_PATH": str(fit_context_path),
            "WNN_FIT_CONTEXT_SHA256": fit_context_sha256,
            "WNN_PACKET_ROOT": str(project_root / "scripts"),
            "PYTHONHASHSEED": "9",
        },
    }
    return spec


def _capacity_preflight_specs(arm_ids: list[str]) -> list[dict[str, object]]:
    """Describe both independent T4 workers using the measured host CPU budget."""
    return [
        {
            "name": arm_id,
            "limits": {
                "cpus": CPU_THREADS_PER_ARM,
                "ram_bytes": 8 * 1024**3,
                "gpu_stop_bytes": 12 * 1024**3,
            },
        }
        for arm_id in arm_ids
    ]


def _source_sha_from_context(project_root: Path, logical: str) -> str:
    portable = _read_json(project_root / "research/campaign_2026-10-04/portable_inputs.json")
    row = portable["sources"][logical]
    return str(row["sha256"])


def _write_runner_specs(
    project_root: Path,
    arm_ids: list[str],
    input_binding_path: Path,
    input_binding_sha256: str,
    fit_context_path: Path,
    fit_context_sha256: str,
    arm_caps: dict[str, int],
) -> list[Path]:
    """Materialize strict read-only runner specifications for exactly authorized arms."""
    paths = []
    for arm_id in arm_ids:
        spec = _make_arm_spec(
            project_root,
            arm_id,
            arm_caps[arm_id],
            input_binding_path,
            input_binding_sha256,
            fit_context_path,
            fit_context_sha256,
        )
        path = project_root / "fit_specs" / f"{arm_id}.json"
        _write_json(path, spec)
        paths.append(path)
    return paths


def run(packet_root: Path) -> dict[str, object]:
    """Run the root-approved fit arm set with one shared input verification pass."""
    started = time.monotonic()
    terminal_path = Path("/kaggle/working/gru_capacity_v5r2_terminal.json")
    outcome: dict[str, object] = {
        "packet_id": PACKET_ID,
        "status": "starting",
        "fit_started": False,
        "profile_rerun": False,
        "parsed_loader_shared": False,
    }
    try:
        recipe_path = packet_root / RECIPE_NAME
        auth_path = packet_root / FIT_AUTH_NAME
        review_path = packet_root / FIT_REVIEW_NAME
        recipe = _read_json(recipe_path)
        auth = _read_json(auth_path)
        review = _read_json(review_path)
        profile_evidence = _verify_profile_evidence(packet_root)
        outcome.update(
            {
                "recipe_sha256": _sha256_file(recipe_path),
                "fit_authorization_sha256": _sha256_file(auth_path),
                "profile_evidence_manifest_sha256": profile_evidence["manifest_sha256"],
            }
        )
        if review.get("status") != "approved":
            outcome["status"] = "source_review_pending"
            outcome["worker_seconds"] = round(time.monotonic() - started, 3)
            _write_json(terminal_path, outcome)
            return outcome
        if auth.get("fit_authorized") is not True:
            outcome["status"] = "fit_authorization_pending"
            outcome["worker_seconds"] = round(time.monotonic() - started, 3)
            _write_json(terminal_path, outcome)
            return outcome
        arm_ids, authorized_seconds = _validate_authorization(
            packet_root, recipe, auth, review, profile_evidence
        )
        if time.monotonic() - started >= authorized_seconds - FINALIZATION_RESERVE_SECONDS:
            raise TimeoutError("authorized worker deadline expired before source preparation")

        phase_start = time.monotonic()
        print("GRU_CAPACITY_PHASE " + json.dumps({"phase": "source_verify", "status": "running", "arm": "cohort", "elapsed_s": round(phase_start - started, 1)}), flush=True)
        source_hashes = runtime._safe_extract_source(packet_root, RUN_ROOT)
        source_hashes.update(_copy_runtime_sources(packet_root, RUN_ROOT))
        source_extract_seconds = time.monotonic() - phase_start
        preflight_started = time.monotonic()
        sys.path.insert(0, str(RUN_ROOT / "scripts"))
        import kaggle_gpu_runner

        host = kaggle_gpu_runner.host_inventory()
        devices = kaggle_gpu_runner.gpu_inventory()
        if len(devices) < len(arm_ids) or len(devices) > 2 or any(
            "T4" not in str(row["name"]) for row in devices
        ):
            raise RuntimeError("actual worker does not expose enough distinct Tesla T4 devices")
        if len({row["uuid"] for row in devices}) != len(devices) or len(
            {row["index"] for row in devices}
        ) != len(devices):
            raise RuntimeError("physical GPU inventory contains duplicate identities")
        preflight_specs = _capacity_preflight_specs(arm_ids)
        admitted_concurrency = kaggle_gpu_runner.admit_specs(
            preflight_specs,
            host,
            devices,
            maximum_preexisting_gpu_used_bytes=512 * 1024**2,
        )
        if admitted_concurrency < len(arm_ids):
            raise RuntimeError("actual host cannot admit the approved independent GPU workers")
        host_gpu_preflight_seconds = time.monotonic() - preflight_started
        inventory = {"host": host, "gpu_devices": devices}
        prepared = runtime._prepare_project(
            packet_root,
            RUN_ROOT,
            verified_source_hashes=source_hashes,
        )
        split_path = RUN_ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
        split_manifest = _read_json(split_path)
        if runtime.sha256_file(split_path) != recipe["inputs"]["split_manifest"]["sha256"]:
            raise ValueError("prepared clean-fold split differs from the frozen recipe")
        expected_fit = recipe["fit_membership"]
        if (
            len(split_manifest["fit"]["train"]) != expected_fit["train_groups"]
            or len(split_manifest["fit"]["valid"]) != expected_fit["valid_fit_groups"]
        ):
            raise ValueError("prepared clean-fold fitting membership changed")

        input_binding_path, input_binding_sha = _write_shared_binding(RUN_ROOT, prepared)
        fit_context_path, fit_context_sha = _make_fit_context(
            RUN_ROOT,
            split_manifest,
            source_hashes,
            input_binding_path,
            input_binding_sha,
            _sha256_file(recipe_path),
        )
        auth_arm_caps = {
            arm_id: min(ARM_CAPS_SECONDS[arm_id], authorized_seconds)
            for arm_id in arm_ids
        }
        specs = _write_runner_specs(
            RUN_ROOT,
            arm_ids,
            input_binding_path,
            input_binding_sha,
            fit_context_path,
            fit_context_sha,
            auth_arm_caps,
        )
        specs_sha = {path.name: _sha256_file(path) for path in specs}
        outcome.update(
            {
                "status": "fit_prepared",
                "fit_started": False,
                "arm_ids": arm_ids,
                "selected_batch": 8,
                "device_inventory": inventory,
                "input_identity_receipt_sha256": _sha256_file(
                    Path(str(prepared["input_identity_receipt"]))
                ),
                "shared_input_binding_sha256": input_binding_sha,
                "source_archive_sha256": _sha256_file(
                    packet_root / "source/source_archive.tar"
                ),
                "source_archive_extract_seconds": round(source_extract_seconds, 3),
                "host_gpu_preflight_seconds": round(host_gpu_preflight_seconds, 3),
                "input_hash_bytes_this_attempt": prepared["input_hash_bytes_this_attempt"],
                "phase_timings": prepared["phase_timings"],
                "fit_spec_sha256": specs_sha,
                "shared_read_contract": {
                    "mounted_raw_and_cache_files_copied": False,
                    "cohort_identity_verification_passes": 1,
                    "per_arm_identity_check": "exact stat and symlink target only",
                    "parsed_loader_shared": False,
                    "model_and_optimizer_instances": "fresh independent process per arm",
                    "os_page_cache": "may be reused by the host; no page-hit claim",
                },
            }
        )
        phase_timings = {
            **cast(dict[str, object], prepared["phase_timings"]),
            "source_archive_extract_and_verify_seconds": round(source_extract_seconds, 3),
            "host_gpu_preflight_seconds": round(host_gpu_preflight_seconds, 3),
            "runner_input_validation": "not_started",
            "arm_timings": {},
        }
        _write_json(Path("/kaggle/working/gru_capacity_v5r2_phase_timings.json"), phase_timings)
        if time.monotonic() - started >= authorized_seconds - FINALIZATION_RESERVE_SECONDS:
            raise TimeoutError("shared source verification exhausted the authorized fit budget")

        remaining_kernel = os.environ.get("WNN_KERNEL_DEADLINE_MONOTONIC")
        if remaining_kernel is None or not math.isfinite(float(remaining_kernel)):
            raise ValueError("kernel deadline handoff is missing or malformed")
        kernel_deadline = float(remaining_kernel)
        if kernel_deadline - time.monotonic() < max(auth_arm_caps.values()) + FINALIZATION_RESERVE_SECONDS:
            raise TimeoutError("remaining kernel deadline cannot fit the approved arm deadline")
        outcome["fit_started"] = True
        outcome["status"] = "fit_started"
        outcome["runner_started_monotonic"] = time.monotonic()
        print("GRU_CAPACITY_PHASE " + json.dumps({"phase": "fit", "status": "running", "arm": "cohort", "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        runner_started = time.monotonic()
        kaggle_gpu_runner.run(
            RUN_ROOT,
            specs,
            RESULT_ROOT,
            stagger_seconds=30,
            required_concurrency=len(arm_ids),
            maximum_preexisting_gpu_used_bytes=512 * 1024**2,
            stop_on_worker_failure=False,
        )
        runner_summary = _read_json(RESULT_ROOT / "summary.json")
        outcome["runner_summary"] = runner_summary
        job_receipts = {}
        for arm_id in arm_ids:
            result_path = RESULT_ROOT / arm_id / "result.json"
            failure_path = RESULT_ROOT / arm_id / "failure.json"
            if result_path.is_file():
                job_receipts[arm_id] = _read_json(result_path)
            elif failure_path.is_file():
                job_receipts[arm_id] = _read_json(failure_path)
            else:
                job_receipts[arm_id] = {"status": "missing_terminal_runner_receipt"}
        outcome["runner_jobs"] = job_receipts
        fit_results = {}
        for arm_id in arm_ids:
            receipt_path = RUN_ROOT / "capacity_receipts" / arm_id / "fit_terminal.json"
            if receipt_path.is_file():
                fit_results[arm_id] = _read_json(receipt_path)
        outcome["fit_results"] = fit_results
        phase_timings["runner_wall_seconds"] = round(time.monotonic() - runner_started, 3)
        phase_timings["arm_timings"] = {
            arm_id: fit_results.get(arm_id, {}).get("phase_timings", {})
            for arm_id in arm_ids
        }
        try:
            pair_validation = runtime._validate_pair_consistency(fit_results)
        except (KeyError, TypeError, ValueError) as error:
            pair_validation = {"status": "fail", "error_type": type(error).__name__}
        outcome["paired_fit_consistency"] = pair_validation
        complete = (
            all(item.get("status") == "complete" for item in fit_results.values())
            and len(fit_results) == len(arm_ids)
            and pair_validation.get("status") in {"pass", "single_arm"}
        )
        if len(arm_ids) == 1 and fit_results.get(arm_ids[0], {}).get("status") == "complete":
            pair_validation = {"status": "single_arm", "arms": arm_ids}
            outcome["paired_fit_consistency"] = pair_validation
            complete = True
        outcome["status"] = "fit_complete" if complete else "fit_partial"
        outcome["worker_seconds"] = round(time.monotonic() - started, 3)
        phase_timings["terminal_status"] = outcome["status"]
        phase_timings["total_worker_seconds"] = outcome["worker_seconds"]
        _write_json(Path("/kaggle/working/gru_capacity_v5r2_phase_timings.json"), phase_timings)
        _write_json(terminal_path, outcome)
        print("GRU_CAPACITY_PHASE " + json.dumps({"phase": "terminal", "status": "pass" if complete else "fail", "arm": "cohort", "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        return outcome
    except BaseException as error:
        previous_status = outcome.get("status")
        outcome["status_before_exception"] = previous_status
        outcome["status"] = "worker_error"
        outcome["error_type"] = type(error).__name__
        outcome["worker_seconds"] = round(time.monotonic() - started, 3)
        try:
            _write_json(terminal_path, outcome)
        except OSError:
            pass
        print("GRU_CAPACITY_PHASE " + json.dumps({"phase": "terminal", "status": "fail", "arm": "cohort", "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        raise


def main() -> None:
    """Execute the fit-only, profile-reuse packet from its materialized root."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.packet_root.resolve(strict=True))
    if result.get("status") not in {
        "authorization_pending",
        "source_review_pending",
        "fit_authorization_pending",
        "fit_complete",
    }:
        raise RuntimeError("fit-only worker did not reach a complete terminal state")


if __name__ == "__main__":
    main()
