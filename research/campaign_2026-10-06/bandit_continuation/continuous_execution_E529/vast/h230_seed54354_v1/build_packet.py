from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

ATTEMPT_ID = "h230-vast-strong-parent-seed54354-muon-v4-20261008"
RUN_NAME = "bd26_h230_strong_parent_seed54354_muon_vast_v4"
PROFILE_NAME = f"{RUN_NAME}_profile"
PREDICTION_NAME = "h230_seed54354_muon_vast_v4_ep3"
RECIPE_FILENAME = "H230_STRONG_PARENT_SEED54354_RECIPE_V2_20261008.json"
PLAN_FILENAME = "H230_STRONG_PARENT_SEED54354_PROFILE_PLAN_V2_20261008.json"
SOURCE_MANIFEST_FILENAME = "H230_FAITHFUL_MUON_SOURCE_MANIFEST_V4_20261008.json"
IMAGE_REFERENCE = (
    "vastai/pytorch@sha256:450710d89268fcd55632f28fdd80508ee920a53075b3f8e9f828bdc154e3d856"
)
PACKET_DIRECTORY = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    """Hash a small source or receipt incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write stable JSON for source and transport contracts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def copy_pinned(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy only an existing source file whose exact digest is pinned."""
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"source is absent or linked: {source}")
    actual = sha256(source)
    if actual != expected_sha256:
        raise ValueError(f"source digest mismatch: {source}: {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def update_argv(argv: list[str], *, run_name: str, seed: int) -> list[str]:
    """Apply the declared identity-only change to a frozen V4 argv list."""
    values = list(argv)
    for index, value in enumerate(values[:-1]):
        if value == "--name":
            values[index + 1] = run_name
        elif value == "--seed":
            values[index + 1] = str(seed)
    if "--name" not in values or "--seed" not in values:
        raise ValueError("frozen V4 argv lacks its explicit name or seed")
    return values


def normalized_scientific_argv(argv: list[str]) -> list[str]:
    """Remove only run-name and seed identities for exact recipe comparison."""
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value in {"--name", "--seed"}:
            if index + 1 >= len(argv):
                raise ValueError(f"identity flag is missing a value: {value}")
            index += 2
            continue
        normalized.append(value)
        index += 1
    return normalized


def load_inputs(root: Path) -> list[dict[str, object]]:
    """Bind six supplied inputs, three cache files, and the own propensity model."""
    campaign = root / "research/campaign_2026-10-06/bandit_continuation"
    identity_path = campaign / "wide_baseline_heterogeneous_H230/H230_LIVE_INPUT_IDENTITY_V1.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    portable = identity["expected"]["portable_files"]

    source_root = campaign / "continuous_execution_E529/h230_source_v4_faithful_muon"
    cache_manifest_path = source_root / "kaggle_parent_v2/cache_asset_upload/assets-manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    prop_recipe_path = source_root / "H230_STRONG_PARENT_SEED9_RECIPE_V2_20261008.json"
    prop_recipe = json.loads(prop_recipe_path.read_text(encoding="utf-8"))
    prop = prop_recipe["propensity_dependency"]

    records: dict[str, dict[str, object]] = {}
    for relative in sorted(portable):
        entry = portable[relative]
        records[relative] = {
            "path": relative,
            "sha256": entry["sha256"],
            "size_bytes": entry["bytes"],
            "basis": "H230 live input identity receipt",
        }
    for item in cache_manifest["files"]:
        relative = str(item["path"])
        records[relative] = {
            "path": relative,
            "sha256": str(item["sha256"]),
            "size_bytes": int(item["size"]),
            "basis": "root-validated immutable H230 cache asset manifest",
        }
    records[str(prop["checkpoint_path"])] = {
        "path": str(prop["checkpoint_path"]),
        "sha256": str(prop["checkpoint_sha256"]),
        "size_bytes": int((root / str(prop["checkpoint_path"])).stat().st_size),
        "basis": "own clean-fold propensity checkpoint pinned in H230 recipe",
    }
    if len(records) != 10:
        raise ValueError(f"expected ten inputs, found {len(records)}")

    for relative, item in records.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"frozen input is absent or linked: {relative}")
        if path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"frozen input size differs from receipt: {relative}")
    return [records[path] for path in sorted(records)]


def copy_project(root: Path, project: Path) -> tuple[dict[str, str], dict[str, object]]:
    """Copy the exact 57 V4 files and separately pinned execution adapters."""
    base = root / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    source_root = base / "h230_source_v4_faithful_muon"
    payload_root = source_root / "kaggle_parent_v4/payload_root"
    manifest_path = source_root / SOURCE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("file_count") != 57 or len(manifest.get("files", [])) != 57:
        raise ValueError("V4 manifest is not the independently reviewed 57-file source")
    source_pins: dict[str, str] = {}
    for item in manifest["files"]:
        relative = str(item["path"])
        copy_pinned(payload_root / relative, project / relative, str(item["sha256"]))
        if (project / relative).stat().st_size != int(item["bytes"]):
            raise ValueError(f"copied V4 source byte count mismatch: {relative}")
        source_pins[relative] = str(item["sha256"])

    source_manifest_path = (
        "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
        + SOURCE_MANIFEST_FILENAME
    )
    copy_pinned(manifest_path, project / source_manifest_path, sha256(manifest_path))
    source_pins[source_manifest_path] = sha256(manifest_path)

    extras = {
        "scripts/h230_profile_then_fit.py": payload_root / "scripts/h230_profile_then_fit.py",
        "scripts/kaggle_common.py": payload_root / "scripts/kaggle_common.py",
        "research/campaign_2026-10-04/portable_inputs.json": payload_root
        / "research/campaign_2026-10-04/portable_inputs.json",
        "scripts/predict_run.py": root / "scripts/predict_run.py",
    }
    for relative, source in extras.items():
        digest = sha256(source)
        copy_pinned(source, project / relative, digest)
        source_pins[relative] = digest

    return source_pins, {
        "manifest": manifest,
        "manifest_path": source_manifest_path,
        "manifest_sha256": sha256(manifest_path),
        "payload_root": str(payload_root.relative_to(root)),
    }


def create_recipe_and_plan(
    root: Path, project: Path, source_sha256: str
) -> tuple[dict[str, object], dict[str, object]]:
    """Derive seed54354 identity from the reviewed seed9 V4 contract."""
    base = root / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
    source_root = base / "h230_source_v4_faithful_muon"
    recipe = json.loads(
        (source_root / "H230_STRONG_PARENT_SEED9_RECIPE_V2_20261008.json").read_text()
    )
    source_plan_path = source_root / "kaggle_parent_v4/profile_fit_plan_v1.json"
    plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))

    recipe["schema"] = "h230-strong-parent-recipe-v1"
    recipe["contract_id"] = "H230_STRONG_PARENT_SEED54354_VAST_RECIPE_V2_20261008"
    recipe["status"] = "prepared_pending_root_source_and_resource_review"
    recipe["launch_authorized"] = False
    recipe["recipe_type"] = (
        "independently trained all-supplied-data seed-54354 strong-parent method-compatible fallback"
    )
    recipe["scientific_recipe"]["seed"] = 54354
    old_run = str(recipe["fit_attempt"]["run_name"])
    recipe["fit_attempt"] = {
        "attempt_id": ATTEMPT_ID,
        "run_name": RUN_NAME,
        "output_path": f"artifacts/runs/{RUN_NAME}",
        "output_absence_checked_on_owned_worker": "required immediately before profile/fit",
        "status": "prepared_not_launched",
        "launch_authorized": False,
        "argv": update_argv(plan["fit_argv"], run_name=RUN_NAME, seed=54354),
        "identity_delta_from_seed9_v4": {
            "old_seed": 9,
            "new_seed": 54354,
            "old_run_name": old_run,
            "new_run_name": RUN_NAME,
            "scientific_flags_changed": False,
        },
        "fresh_fit_preconditions": [
            "Root independently approves the exact Vast packet, live offer, all-in quote, and account-scope-bound durable admission.",
            "The owned worker is verified as one RTX 3090, CUDA capability 8.6, and native BF16 supported before the real-data profile.",
            "The profile gate must pass its exact 80 finite-gradient observations and fit projection before the conditional fresh fit.",
            "The fit uses new seed-54354 initialization and does not load teammate or prior parent model weights.",
        ],
    }
    recipe["metadata_revision"] = {
        "basis": "identity-only successor of the independently reviewed seed9 V4 recipe",
        "parent_recipe_path": str(
            (source_root / "H230_STRONG_PARENT_SEED9_RECIPE_V2_20261008.json").relative_to(root)
        ),
        "parent_recipe_sha256": sha256(
            source_root / "H230_STRONG_PARENT_SEED9_RECIPE_V2_20261008.json"
        ),
        "changes": [
            "Set the independent random seed to 54354.",
            "Set unique Vast run, profile, prediction, and attempt identities.",
            "Require the same V4 scientific argv, with no optimizer, target, architecture, data-membership, or checkpoint-policy change.",
        ],
        "source_manifest_sha256": source_sha256,
        "launch_authorized": False,
    }
    recipe_path = (
        project
        / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
        / RECIPE_FILENAME
    )
    write_json(recipe_path, recipe)
    recipe_sha256 = sha256(recipe_path)

    for key in ("profile_argv", "fit_argv"):
        argv = plan[key]
        plan[key] = update_argv(
            argv,
            run_name=PROFILE_NAME if key == "profile_argv" else RUN_NAME,
            seed=54354,
        )
        source_run_name = (
            str(source_plan["profile_run_name"])
            if key == "profile_argv"
            else str(source_plan["fit_run_name"])
        )
        source_identity_argv = update_argv(source_plan[key], run_name=source_run_name, seed=9)
        if normalized_scientific_argv(plan[key]) != normalized_scientific_argv(
            source_identity_argv
        ):
            raise ValueError(f"scientific argv changed beyond run/seed identity: {key}")
    plan["profile_run_name"] = PROFILE_NAME
    plan["fit_run_name"] = RUN_NAME
    plan["profile_log_path"] = f"artifacts/runs/{PROFILE_NAME}/profile.log"
    plan["fit_log_path"] = f"artifacts/runs/{RUN_NAME}/fit.log"
    plan["gate_receipt_path"] = "artifacts/h230_seed54354_profile_fit_gate.json"
    plan["expected"]["profile_args"]["name"] = PROFILE_NAME
    plan["expected"]["profile_args"]["seed"] = 54354
    plan["expected"]["fit_args"]["name"] = RUN_NAME
    plan["expected"]["fit_args"]["seed"] = 54354
    plan["expected"]["device_name_substring"] = "3090"
    plan["expected"]["device_capability"] = [8, 6]
    plan["expected"]["cuda_bf16_supported"] = True
    plan["expected"]["orthogonalization_dtype"] = "bfloat16"
    plan["expected"]["recipe_sha256"] = recipe_sha256
    plan["expected"]["source_manifest_sha256"] = source_sha256
    plan["runtime_gate"]["worker_timeout_seconds"] = 9000
    plan_path = (
        project
        / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529"
        / PLAN_FILENAME
    )
    write_json(plan_path, plan)
    return recipe, plan


def build_tree(root: Path, build_root: Path) -> tuple[Path, dict[str, object]]:
    """Create a fresh packet tree from the frozen source and H230 input receipts."""
    if build_root.exists():
        raise FileExistsError(f"packet build directory already exists: {build_root}")
    project = build_root / "project"
    project.mkdir(parents=True)
    source_pins, source_receipt = copy_project(root, project)
    source_sha256 = str(source_receipt["manifest_sha256"])
    recipe, plan = create_recipe_and_plan(root, project, source_sha256)
    recipe_path = project / str(
        Path("research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529")
        / RECIPE_FILENAME
    )
    plan_path = project / str(
        Path("research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529")
        / PLAN_FILENAME
    )
    source_pins[str(recipe_path.relative_to(project))] = sha256(recipe_path)
    source_pins[str(plan_path.relative_to(project))] = sha256(plan_path)

    inputs = load_inputs(root)
    spec = {
        "schema": "h230-vast-worker-spec-v1",
        "attempt_id": ATTEMPT_ID,
        "recipe_id": recipe["contract_id"],
        "source_manifest_path": source_receipt["manifest_path"],
        "source_manifest_sha256": source_receipt["manifest_sha256"],
        "source_pins": source_pins,
        "inputs": inputs,
        "recipe_path": str(recipe_path.relative_to(project)),
        "recipe_sha256": sha256(recipe_path),
        "profile_plan_path": str(plan_path.relative_to(project)),
        "profile_plan_sha256": sha256(plan_path),
        "profile_argv": plan["profile_argv"],
        "fit_argv": plan["fit_argv"],
        "profile_run_name": PROFILE_NAME,
        "profile_gate_path": str(plan["gate_receipt_path"]),
        "fit_run_name": RUN_NAME,
        "prediction_output_name": PREDICTION_NAME,
        "expected_device": {
            "device_name_substring": "3090",
            "device_capability": [8, 6],
            "cuda_bf16_supported": True,
        },
        "expected_image": IMAGE_REFERENCE,
        "resources": {
            "profile_fit_timeout_seconds": 9000,
            "prediction_export_timeout_seconds": 2400,
            "worker_ram_bytes": 28_000_000_000,
            "worker_vram_bytes": 18_000_000_000,
            "minimum_available_ram_bytes": 6_000_000_000,
            "environment": {
                "WNN_THREADS": "4",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "NUMEXPR_NUM_THREADS": "4",
                "WNN_LOADER_WORKERS": "0",
                "WNN_GPU_DUTY": "1",
                "WNN_GPU_MEMORY_BYTES": "4300000000",
                "WNN_WORKER_RAM_BYTES": "28000000000",
                "WNN_WORKER_VRAM_BYTES": "18000000000",
            },
        },
        "artifact_project_paths": [
            "artifacts/h230_seed54354_profile_fit_gate.json",
            f"artifacts/runs/{PROFILE_NAME}",
            f"artifacts/runs/{RUN_NAME}",
            f"cache/pred_{PREDICTION_NAME}_train.npy",
            f"cache/pred_{PREDICTION_NAME}_train.json",
            f"cache/pred_{PREDICTION_NAME}_valid.npy",
            f"cache/pred_{PREDICTION_NAME}_valid.json",
            str(recipe_path.relative_to(project)),
            str(plan_path.relative_to(project)),
        ],
    }
    write_json(build_root / "worker_spec.json", spec)
    shutil.copyfile(PACKET_DIRECTORY / "worker.py", build_root / "worker.py")
    shutil.copyfile(PACKET_DIRECTORY / "run.sh", build_root / "run.sh")
    os.chmod(build_root / "run.sh", 0o755)
    return build_root, spec


def add_deterministic(archive: tarfile.TarFile, base: Path, source: Path) -> None:
    """Add one regular file or directory with normalized ownership and timestamps."""
    name = source.relative_to(base).as_posix()
    info = archive.gettarinfo(str(source), arcname=name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if info.issym() or info.islnk() or not (info.isfile() or info.isdir()):
        raise ValueError(f"packet contains an unsafe member: {source}")
    if info.isfile():
        with source.open("rb") as stream:
            archive.addfile(info, stream)
    else:
        archive.addfile(info)


def build_archive(build_root: Path, archive_path: Path) -> None:
    """Build a deterministic, link-free tar with one root-level run.sh."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise FileExistsError(f"refusing to replace an existing packet: {archive_path}")
    with tarfile.open(archive_path, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(
            build_root.rglob("*"), key=lambda item: item.relative_to(build_root).as_posix()
        ):
            add_deterministic(archive, build_root, path)


def create_transport_contract(
    root: Path, archive_path: Path, spec: dict[str, object]
) -> dict[str, object]:
    """Write the source-bound, non-authorizing Vast transport contract."""
    archive_hash = sha256(archive_path)
    input_bytes = sum(int(item["size_bytes"]) for item in spec["inputs"])
    total_upload_bytes = input_bytes + archive_path.stat().st_size
    if total_upload_bytes > 41_500_000_000:
        raise ValueError("packet and staged inputs exceed the declared upload ceiling")
    identity_path = (
        root
        / "research/campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/"
        / "H230_LIVE_INPUT_IDENTITY_V1.json"
    )
    source_manifest_path = (
        root
        / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
        / "h230_source_v4_faithful_muon/H230_FAITHFUL_MUON_SOURCE_MANIFEST_V4_20261008.json"
    )
    predictor_path = root / "scripts/predict_run.py"
    prediction_payload_bytes = (10_606 + 1_873) * 20_000 * 2 * 4
    prediction_header_bytes = 2 * 128
    prediction_bytes = prediction_payload_bytes + prediction_header_bytes
    prior_contract_path = (
        PACKET_DIRECTORY / "H230_SEED54354_VAST_TRANSPORT_CONTRACT_V3_20261008.json"
    )
    prefetch_paths = {"datasets/train.parquet", "datasets/valid.parquet"}
    prefetch_inputs = [item for item in spec["inputs"] if item["path"] in prefetch_paths]
    ssh_inputs = [item for item in spec["inputs"] if item["path"] not in prefetch_paths]
    prefetch_helper = root / "scripts/kaggle_vast_prefetch.py"
    prefetch_remote = root / "scripts/kaggle_vast_prefetch_remote.py"
    supervisor_path = root / "scripts/vast_campaign.py"
    ssh_upload_bytes = sum(int(item["size_bytes"]) for item in ssh_inputs)
    packet_and_ssh_bytes = archive_path.stat().st_size + ssh_upload_bytes
    full_staged_bytes = sum(int(item["size_bytes"]) for item in spec["inputs"])
    return {
        "schema": "h230-vast-seed54354-transport-contract-v4",
        "contract_id": "H230_VAST_STRONG_PARENT_SEED54354_V4_20261008",
        "revision": {
            "supersedes_path": str(prior_contract_path.relative_to(root)),
            "supersedes_sha256": sha256(prior_contract_path),
            "changes": [
                "Fix prediction file-size validation to include the NPY header using memmap offset plus payload nbytes.",
                "Stream the full prediction file and require its actual SHA-256 to match the producer sidecar.",
                "Add real open_memmap valid, truncated, trailing-byte, and sidecar-hash fixtures.",
                "Capture a fail-closed pinned image interpreter/CUDA/device startup receipt before dependency setup or fitting.",
                "Use a new attempt/run identity; preserve the same scientific source, inputs, and normalized scientific argv.",
                "Use the existing signed raw-Parquet prefetch path and retain all ten worker inputs in the frozen manifest.",
            ],
            "science_changed": False,
        },
        "status": "prepared_pending_root_offer_and_launch_review",
        "provider_create_authorized": False,
        "profile_authorized": False,
        "fit_authorized": False,
        "prediction_export_authorized": False,
        "attempt_id": ATTEMPT_ID,
        "campaign_label": "wnn-e529-h230-seed54354-muon-v4-20261008",
        "scientific_identity": {
            "recipe_path_in_packet": str(spec["recipe_path"]),
            "recipe_sha256": spec["recipe_sha256"],
            "profile_plan_path_in_packet": str(spec["profile_plan_path"]),
            "profile_plan_sha256": spec["profile_plan_sha256"],
            "profile_run_name": spec["profile_run_name"],
            "fit_run_name": spec["fit_run_name"],
            "exact_argv": {
                "profile": spec["profile_argv"],
                "fit": spec["fit_argv"],
                "train_prediction": [
                    "scripts/predict_run.py",
                    str(spec["fit_run_name"]),
                    "train",
                    "--epoch",
                    "3",
                    "--output-name",
                    str(spec["prediction_output_name"]),
                    "--inference-precision",
                    "fp32",
                ],
                "valid_prediction": [
                    "scripts/predict_run.py",
                    str(spec["fit_run_name"]),
                    "valid",
                    "--epoch",
                    "3",
                    "--output-name",
                    str(spec["prediction_output_name"]),
                    "--inference-precision",
                    "fp32",
                ],
            },
            "source_manifest_sha256": spec["source_manifest_sha256"],
            "science_delta_from_seed9_v4": {
                "only_change": "fresh independent random seed 54354",
                "seed9_profile_and_fit_args_otherwise_preserved": True,
                "no_teammate_weights_or_code": True,
                "prediction_and_run_names_are_non_scientific_ids": True,
            },
            "method_claim_boundary": (
                "Method-compatible own-weight strong-parent fallback, not an exact reference "
                "reproduction. The 2GB teacher prediction cache was trained on clean-fold fit "
                "groups only despite complete train+valid prediction coverage. The fit uses all "
                "supplied train and valid rows, including exposed groups; no held-out grade is claimed."
            ),
        },
        "source": {
            "v4_manifest_path": str(source_manifest_path.relative_to(root)),
            "v4_manifest_sha256": spec["source_manifest_sha256"],
            "v4_source_file_count": 57,
            "v4_source_pins": spec["source_pins"],
            "profile_helper": {
                "path_in_packet": "project/scripts/h230_profile_then_fit.py",
                "sha256": spec["source_pins"]["scripts/h230_profile_then_fit.py"],
            },
            "prediction_runner": {
                "path": str(predictor_path.relative_to(root)),
                "sha256": spec["source_pins"]["scripts/predict_run.py"],
                "path_in_packet": "project/scripts/predict_run.py",
            },
            "execution_adapters": {
                "worker_path": str((PACKET_DIRECTORY / "worker.py").relative_to(root)),
                "worker_sha256": sha256(PACKET_DIRECTORY / "worker.py"),
                "run_script_path": str((PACKET_DIRECTORY / "run.sh").relative_to(root)),
                "run_script_sha256": sha256(PACKET_DIRECTORY / "run.sh"),
            },
            "prefetch_and_supervisor_primitives": {
                "prefetch_helper_path": str(prefetch_helper.relative_to(root)),
                "prefetch_helper_sha256": sha256(prefetch_helper),
                "prefetch_remote_program_path": str(prefetch_remote.relative_to(root)),
                "prefetch_remote_program_sha256": sha256(prefetch_remote),
                "supervisor_path": str(supervisor_path.relative_to(root)),
                "supervisor_sha256": sha256(supervisor_path),
                "binding_note": "V4 contract pins these existing repository primitives; root must verify the live source hashes at admission.",
            },
            "input_identity_receipt": {
                "path": str(identity_path.relative_to(root)),
                "sha256": sha256(identity_path),
            },
            "local_data_profile_authorization_is_evidence_only": {
                "path": "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/H230_STRONG_PARENT_REALDATA_PROFILE_ROOT_AUTHORIZATION_20261008_R1.json",
                "sha256": sha256(
                    root
                    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/"
                    / "H230_STRONG_PARENT_REALDATA_PROFILE_ROOT_AUTHORIZATION_20261008_R1.json"
                ),
                "authorizes_this_vast_attempt": False,
            },
        },
        "packet": {
            "path": str(archive_path.relative_to(root)),
            "sha256": archive_hash,
            "size_bytes": archive_path.stat().st_size,
            "format": "deterministic uncompressed tar; one root run.sh; no symlinks",
        },
        "inputs": spec["inputs"],
        "input_count": len(spec["inputs"]),
        "input_bytes": input_bytes,
        "total_upload_bytes": total_upload_bytes,
        "maximum_upload_bytes": 41_500_000_000,
        "staging_transport": {
            "all_worker_inputs_remain_required": True,
            "frozen_worker_input_count": len(spec["inputs"]),
            "total_staged_input_bytes": full_staged_bytes,
            "signed_prefetch_required_at_launch": True,
            "prefetch_supervisor_flags": ["--prefetch-plan", "--prefetch-admission"],
            "prefetch_remote_inputs": [
                {
                    "remote_path": str(item["path"]),
                    "sha256": str(item["sha256"]),
                    "size_bytes": int(item["size_bytes"]),
                }
                for item in prefetch_inputs
            ],
            "prefetch_bytes": sum(int(item["size_bytes"]) for item in prefetch_inputs),
            "remaining_ssh_inputs": [
                {
                    "path": str(item["path"]),
                    "sha256": str(item["sha256"]),
                    "size_bytes": int(item["size_bytes"]),
                }
                for item in ssh_inputs
            ],
            "remaining_ssh_input_bytes": ssh_upload_bytes,
            "packet_plus_remaining_ssh_bytes": packet_and_ssh_bytes,
            "private_plan_rule": "Root supplies the signed URL plan only through the supervisor's private runtime file; never commit or log the URLs.",
            "worker_manifest_rule": "Supervisor must still pass the complete ten-input manifest; its verified prefetch result excludes only the two raw Parquets from SSH transfer.",
            "fallback_rule": "If the admitted prefetch pair is unavailable or invalid, do not launch a full raw-Parquet SSH fallback under this V4 transport.",
        },
        "upload_timeout_seconds": 5400,
        "remote_input_verification": (
            "Vast supervisor verifies each remote SHA before atomic rename; the worker checks "
            "exact manifest, names, and sizes; the V4 raw portability and distillation-cache "
            "validators remain mandatory before fit."
        ),
        "execution": {
            "image": IMAGE_REFERENCE,
            "torch": "2.14.0+cu130",
            "cuda_runtime": "13.0",
            "image_preflight_receipt": "image_preflight.json",
            "image_preflight_required_before_dependency_setup_or_fit": True,
            "container_startup_timing_source": "provider lifecycle timestamps; worker records interpreter/import/CUDA-init timing but cannot observe image-pull duration",
            "gpu_gate": spec["expected_device"],
            "requested_gpu": "1x RTX 3090",
            "requested_disk_gb": 120,
            "worker_runtime_seconds": 18_000,
            "worker_setup_timeout_seconds": 450,
            "profile_then_fit_cap_seconds": 9_000,
            "train_prediction_cap_seconds": 2_400,
            "valid_prediction_cap_seconds": 2_400,
            "loader_workers": 0,
            "cpu_threads": 4,
            "worker_ram_bytes": 28_000_000_000,
            "allocator_bytes": 4_300_000_000,
            "observed_total_vram_bytes": 18_000_000_000,
            "minimum_host_available_ram_bytes": 6_000_000_000,
            "expected_prediction_payload_bytes": prediction_payload_bytes,
            "expected_prediction_file_bytes_including_npy_headers": prediction_bytes,
            "prediction_format": "full train and valid target-index order, FP32 N x 20000 x 2, fixed zero-based epoch 3",
            "exact_expected_artifacts": [
                "artifact_bundle.tar",
                "result.json",
                "worker.log",
                "image_preflight.json",
            ],
            "maximum_artifact_bytes": 3_500_000_000,
        },
        "disk_peak_estimate": {
            "staged_final_inputs_bytes": input_bytes,
            "largest_single_temporary_upload_bytes": max(
                int(item["size_bytes"]) for item in spec["inputs"]
            ),
            "prediction_outputs_bytes": prediction_bytes,
            "artifact_bundle_max_bytes": 3_500_000_000,
            "image_and_writable_layer_reserve_bytes": 30_000_000_000,
            "checkpoints_logs_scratch_reserve_bytes": 2_000_000_000,
            "estimated_peak_bytes": (
                input_bytes
                + max(int(item["size_bytes"]) for item in spec["inputs"])
                + prediction_bytes
                + 3_500_000_000
                + 30_000_000_000
                + 2_000_000_000
            ),
            "requested_disk_bytes": 120_000_000_000,
            "headroom_bytes_at_estimate": 120_000_000_000
            - input_bytes
            - max(int(item["size_bytes"]) for item in spec["inputs"])
            - prediction_bytes
            - 3_500_000_000
            - 30_000_000_000
            - 2_000_000_000,
            "assumptions": [
                "Remote input transfers are sequential; only the largest .part and final input coexist during staging.",
                "Artifact bundle is uncompressed and retained once; training outputs are members, not a second full copy.",
                "Image/writable layer has a conservative 30 GB reserve and must be checked against fresh offer disk accounting.",
            ],
        },
        "cost_request": {
            "maximum_all_in_usd": "2.80",
            "not_a_live_quote": True,
            "fresh_offer_required": True,
            "last_observed_3090_example": {
                "offer_id": 45059202,
                "offer_snapshot_sha256": "89ec77a159d662d67321e18162f1795960798a8b268411ea32fb8faa56fbdba6",
                "dph_total_usd": "0.174444",
                "storage_usd_per_gb_month": "0.20",
                "warning": "Reference observation only; root must re-search and recompute every fee before admission.",
            },
            "suggested_quote_component_caps_usd": {
                "compute_storage_usd": "1.40",
                "upload_usd": "0.90",
                "download_usd": "0.20",
                "cleanup_usd": "0.30",
            },
            "quote_component_caps_sum_usd": "2.80",
            "quote_method": (
                "Recompute compute plus 120 GB storage through the bounded startup/runtime/cleanup deadline; "
                "cover two maximum upload attempts and a cold image pull at the fresh offer inbound fee; "
                "cover 3.5 GB artifact download at its fresh outbound fee. Do not create if fresh charges "
                "exceed any component cap or the $2.80 total."
            ),
        },
        "fresh_root_requirements": [
            "Refresh Vast account scope, current credit, exact rentable verified offer and all network/storage rates.",
            "Verify 120 GB requested disk meets the fresh offer and disk-peak estimate.",
            "Seal an exact source/input/packet-bound frozen intent, admission, dispatch receipt, and empty destination.",
            "Recheck quota, ownership baseline, worker resources, one-GPU exclusivity, and all-in quote immediately before create.",
            "Do not touch or adopt any existing instance; only the durable creation receipt may establish ownership.",
        ],
    }


def main() -> None:
    """Build the packet and non-authorizing transport handoff."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PACKET_DIRECTORY / "h230_seed54354_v4.tar")
    parser.add_argument("--build-dir", type=Path, default=PACKET_DIRECTORY / "build_v4")
    parser.add_argument(
        "--contract-output",
        type=Path,
        default=PACKET_DIRECTORY / "H230_SEED54354_VAST_TRANSPORT_CONTRACT_V4_20261008.json",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[6]
    build_root = args.build_dir.resolve()
    if build_root.exists() or args.output.exists():
        raise FileExistsError("refusing to overwrite an existing packet build or archive")
    _, spec = build_tree(root, build_root)
    archive_path = args.output.resolve()
    if not archive_path.is_relative_to(root):
        raise ValueError("packet output must remain inside the project root")
    build_archive(build_root, archive_path)
    contract = create_transport_contract(root, archive_path, spec)
    contract_path = args.contract_output.resolve()
    if contract_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing contract: {contract_path}")
    write_json(contract_path, contract)
    print(
        json.dumps(
            {
                "packet_path": str(archive_path.relative_to(root)),
                "packet_sha256": contract["packet"]["sha256"],
                "packet_bytes": contract["packet"]["size_bytes"],
                "contract_path": str(contract_path.relative_to(root)),
                "contract_sha256": sha256(contract_path),
                "input_count": contract["input_count"],
                "input_bytes": contract["input_bytes"],
                "laptop_ssh_upload_bytes": contract["staging_transport"][
                    "packet_plus_remaining_ssh_bytes"
                ],
                "prefetch_bytes": contract["staging_transport"]["prefetch_bytes"],
                "provider_create_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
