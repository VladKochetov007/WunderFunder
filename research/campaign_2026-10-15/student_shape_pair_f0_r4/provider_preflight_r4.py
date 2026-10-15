from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PACKET = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r4"
PACKET_R3 = ROOT / "research/campaign_2026-10-15/student_shape_pair_f0_r3"
BUNDLE_R3 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r3"
BUNDLE_R4 = ROOT / "private_assets/upload_wnn-f0-student-shape-t4-profile-20261015-r4"
SOURCE_DATASET = "vladkochetov07/wnn-f0-student-shape-t4-profile-20261015-r1"
NORM_DATASET = "vladkochetov07/wnn-f0-h256l2-cp5-norm-20261015-r1"
KERNEL_ID = f"{SOURCE_DATASET}-runner"
AUTHORIZED_OVERLAP_KERNEL = "vladkochetov07/casmi2026-reference-audit-001"
OWNER = "vladkochetov07"
MINIMUM_GPU_SECONDS = 5_400
HELPER_PATH = PACKET_R3 / "provider_preflight_r3.py"
R9_MANIFEST = ROOT / (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_f0_h390_r9_export_from_owned_checkpoints_20261012_r2/"
    "R9_CHECKPOINT_INPUT_MANIFEST_R2.json"
)


def sha256_file(path: Path) -> str:
    """Hash a small packet or bundle file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, maximum: int = 4_000_000) -> dict[str, Any]:
    """Read bounded provider metadata only."""
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise ValueError(f"refusing oversized or empty metadata: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def helper_module():
    """Reuse the reviewed R3 read-only Kaggle API primitives."""
    spec = importlib.util.spec_from_file_location("student_shape_r3_provider", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the reviewed Kaggle provider helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_receipt(path: Path, payload: object) -> None:
    """Create one read-only preflight receipt without overwriting evidence."""
    if path.exists():
        raise FileExistsError(f"preflight receipt already exists: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_files(bundle: Path) -> dict[str, int]:
    """Return only the two root dataset files intended for Kaggle versioning."""
    return {
        "bundle.wnn": (bundle / "bundle.wnn").stat().st_size,
        "bundle-manifest.json": (bundle / "bundle-manifest.json").stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("source-version", "kernel-push"), required=True)
    args = parser.parse_args()
    phase = args.phase
    started = datetime.now(UTC)

    api_helpers = helper_module()
    api = api_helpers.KaggleApi()
    api.authenticate()
    source_binding = read_json(PACKET / "SOURCE_BINDING_R4.json")
    input_binding = read_json(PACKET / "INPUT_BINDING_R4.json")
    seal = read_json(PACKET / "GENERATED_KERNEL_SEAL_R4.json")
    offline = read_json(PACKET / "OFFLINE_SPEC_PREFLIGHT_R4.json")
    r3_terminal = read_json(PACKET_R3 / "PROVIDER_TERMINAL_R3.json")
    r3_source_receipt = read_json(PACKET_R3 / "PROVIDER_PREFLIGHT_KERNEL_PUSH_R3.json")
    norm_readback = read_json(
        PACKET_R3.parent / "student_shape_pair_f0_r2/NORM_ASSET_REMOTE_READBACK_R1.json"
    )
    if (
        source_binding.get("kernel_dataset") != SOURCE_DATASET
        or source_binding.get("kernel_id") != KERNEL_ID
    ):
        raise ValueError("R4 source binding names a different provider dataset or kernel")
    if source_binding.get("bootstrap_outer_deadline_seconds") != 1800:
        raise ValueError("R4 source binding outer deadline differs from the reviewed profile cap")
    if seal.get("source_binding_sha256") != sha256_file(PACKET / "SOURCE_BINDING_R4.json"):
        raise ValueError("R4 generated kernel seal does not bind the current source binding")
    if seal.get("input_binding_sha256") != sha256_file(PACKET / "INPUT_BINDING_R4.json"):
        raise ValueError("R4 generated kernel seal does not bind the current input binding")
    if offline.get("status") != "passed_offline_no_provider_action":
        raise ValueError("R4 offline preflight is not a passing sealed packet")
    if seal.get("status") != "offline_r4_bundle_and_two_target_mounts_pinned_no_provider_action":
        raise ValueError("R4 generated-kernel seal is not the expected offline seal")
    if r3_terminal.get("status") != "terminal_infrastructure_failure_before_cuda_update":
        raise ValueError("R3 terminal record is not the preserved pre-update failure")
    if (
        r3_source_receipt.get("private_targets", {}).get("profile_source_dataset", {}).get("state")
        != "ready"
    ):
        raise ValueError("R3 provider record does not bind the existing READY profile source")
    if (
        norm_readback.get("dataset", {}).get("ref") != NORM_DATASET
        or norm_readback.get("dataset", {}).get("private") is not True
    ):
        raise ValueError("CP5 normalization readback identity differs")

    external = input_binding["external_mounts"]
    mounted: dict[str, dict[str, int]] = {}
    versions: dict[str, int] = {}
    for item in external:
        dataset = item["dataset"]
        expected_version = int(item["version_expected"])
        if dataset in versions and versions[dataset] != expected_version:
            raise ValueError(f"profile bundle mixes dataset versions for {dataset}")
        versions[dataset] = expected_version
        mounted.setdefault(dataset, {})[item["filename"]] = int(item["size"])

    # Verify every exact external dataset ID/version used by the profile jobs.
    dataset_statuses: dict[str, str] = {}
    dataset_rows: dict[str, dict[str, object]] = {}
    for ref, version in sorted(versions.items()):
        model = api_helpers.exact_dataset(api, ref)
        if model is None:
            raise ValueError(f"required mounted dataset is absent: {ref}")
        row = api_helpers.dataset_inventory_row(model)
        if row["private"] is not True or row["current_version"] != version:
            raise ValueError(
                f"mounted dataset is not private at the bound version {version}: {ref}"
            )
        state = str(api.dataset_status(ref))
        if state.upper().rsplit(".", maxsplit=1)[-1] != "READY":
            raise ValueError(f"mounted dataset is not READY: {ref} ({state})")
        live = api_helpers.require_files(api, ref, version, mounted[ref])
        dataset_statuses[ref] = state
        dataset_rows[ref] = {
            **row,
            "status": state,
            "version": version,
            "bound_files_verified_by_remote_name_and_size": {
                name: live[name] for name in sorted(mounted[ref])
            },
        }

    # Recheck the complete small CP5 normalization asset readback.
    norm_inventory = {
        name: int(size) for name, size in norm_readback["remote_file_inventory"].items()
    }
    norm_live = api_helpers.require_files(api, NORM_DATASET, 1, norm_inventory, exact=True)
    if norm_live != norm_inventory:
        raise ValueError("normalization asset live READY v1 inventory differs from its readback")

    source_model = api_helpers.exact_dataset(api, SOURCE_DATASET)
    if source_model is None:
        raise ValueError("R3 private profile source dataset disappeared")
    source_row = api_helpers.dataset_inventory_row(source_model)
    expected_current = 1 if phase == "source-version" else 2
    if source_row["private"] is not True or source_row["current_version"] != expected_current:
        raise ValueError(
            f"profile source dataset must be private current version {expected_current}"
        )
    source_status = str(api.dataset_status(SOURCE_DATASET))
    if source_status.upper().rsplit(".", maxsplit=1)[-1] != "READY":
        raise ValueError(f"profile source dataset is not READY: {source_status}")
    expected_source_files = source_files(BUNDLE_R3 if phase == "source-version" else BUNDLE_R4)
    source_live = api_helpers.require_files(
        api, SOURCE_DATASET, expected_current, expected_source_files
    )

    # The R3 kernel is an existing terminal ERROR. R4 updates its exact versioned target.
    kernel_model = api_helpers.exact_kernel(api, KERNEL_ID)
    if kernel_model is None:
        raise ValueError(
            "expected existing R3 kernel slug is absent; do not create a second kernel"
        )
    kernel_row = api_helpers.kernel_inventory_row(kernel_model)
    prior_kernel_state = api_helpers.kernel_status(api, kernel_model)
    if prior_kernel_state["state"] not in api_helpers.EXIT_STATUSES:
        raise ValueError(f"existing profile kernel is not terminal: {prior_kernel_state}")
    if prior_kernel_state["state"] != "ERROR":
        raise ValueError("exact existing R3 kernel no longer has its recorded ERROR state")

    # The list endpoint omits privacy/accelerator fields for this exact private
    # kernel. Pull its exact metadata and code, then compare them to the frozen R3
    # source before trusting those fields.
    readback_stamp = started.strftime("%Y%m%dT%H%M%SZ")
    kernel_readback = (
        ROOT / "private_assets" / f"readback_student_shape_t4_r4_{phase}_{readback_stamp}"
    )
    if kernel_readback.exists():
        raise FileExistsError(f"kernel metadata readback path already exists: {kernel_readback}")
    api.kernels_pull(KERNEL_ID, str(kernel_readback), metadata=True, quiet=True)
    pulled_metadata_path = kernel_readback / "kernel-metadata.json"
    pulled_metadata = read_json(pulled_metadata_path, maximum=64_000)
    frozen_r3_metadata = read_json(BUNDLE_R3 / "kernel/kernel-metadata.json", maximum=64_000)
    code_file = pulled_metadata.get("code_file")
    if not isinstance(code_file, str) or Path(code_file).name != code_file:
        raise ValueError("pulled kernel metadata contains an unsafe code filename")
    expected_remote_code_file = f"{KERNEL_ID.rsplit('/', maxsplit=1)[-1]}.py"
    if code_file != expected_remote_code_file:
        raise ValueError("exact pulled kernel metadata has an unexpected provider code filename")
    pulled_code_path = kernel_readback / code_file
    frozen_code_path = BUNDLE_R3 / "kernel/kernel.py"
    if not pulled_code_path.is_file() or not frozen_code_path.is_file():
        raise ValueError("exact pulled kernel code or frozen R3 source file is absent")
    if pulled_code_path.stat().st_size <= 0 or pulled_code_path.stat().st_size > 1_000_000:
        raise ValueError("pulled kernel code exceeds the sealed bootstrap size bound")
    kernel_metadata_identity = {
        key: pulled_metadata.get(key)
        for key in (
            "id",
            "is_private",
            "enable_gpu",
            "enable_tpu",
            "kernel_type",
            "code_file",
            "dataset_sources",
        )
    }
    if pulled_metadata.get("id") != KERNEL_ID:
        raise ValueError("pulled kernel metadata identifies a different slug")
    if pulled_metadata.get("is_private") is not True:
        raise ValueError("pulled exact kernel metadata does not confirm private visibility")
    if (
        pulled_metadata.get("enable_gpu") is not True
        or pulled_metadata.get("enable_tpu") is not False
    ):
        raise ValueError("pulled exact kernel metadata does not confirm the frozen GPU-only target")
    frozen_source_ids = sorted(frozen_r3_metadata.get("dataset_sources", []))
    pulled_source_ids = sorted(pulled_metadata.get("dataset_sources", []))
    if pulled_source_ids != frozen_source_ids:
        raise ValueError("exact remote kernel dataset sources differ from frozen R3 bindings")
    for key in ("id", "is_private", "enable_gpu", "enable_tpu", "kernel_type"):
        if pulled_metadata.get(key) != frozen_r3_metadata.get(key):
            raise ValueError(f"exact remote kernel metadata differs from frozen R3 field {key}")
    if sha256_file(pulled_code_path) != sha256_file(frozen_code_path):
        raise ValueError("exact remote kernel code differs from the frozen R3 bootstrap")

    overlap_model = api_helpers.exact_kernel(api, AUTHORIZED_OVERLAP_KERNEL)
    overlap_observation: dict[str, object] = {
        "ref": AUTHORIZED_OVERLAP_KERNEL,
        "visible_in_exact_query": overlap_model is not None,
    }
    if overlap_model is not None:
        overlap_status = api_helpers.kernel_status(api, overlap_model)
        overlap_stamp = started.strftime("%Y%m%dT%H%M%SZ")
        overlap_readback = (
            ROOT / "private_assets" / f"readback_student_shape_t4_r4_overlap_{overlap_stamp}"
        )
        if overlap_readback.exists():
            raise FileExistsError(
                f"overlap metadata readback path already exists: {overlap_readback}"
            )
        api.kernels_pull(
            AUTHORIZED_OVERLAP_KERNEL, str(overlap_readback), metadata=True, quiet=True
        )
        overlap_metadata_path = overlap_readback / "kernel-metadata.json"
        overlap_metadata = read_json(overlap_metadata_path, maximum=64_000)
        if overlap_metadata.get("id") != AUTHORIZED_OVERLAP_KERNEL:
            raise ValueError("overlap kernel metadata identifies a different provider target")
        if not isinstance(overlap_metadata.get("enable_gpu"), bool):
            raise ValueError("overlap kernel metadata omitted its accelerator flag")
        if not isinstance(overlap_metadata.get("enable_tpu"), bool):
            raise ValueError("overlap kernel metadata omitted its TPU flag")
        overlap_observation.update(
            {
                "status": overlap_status,
                "private": overlap_metadata.get("is_private"),
                "enable_gpu": overlap_metadata["enable_gpu"],
                "enable_tpu": overlap_metadata["enable_tpu"],
                "metadata_readback_path": str(overlap_metadata_path),
                "metadata_sha256": sha256_file(overlap_metadata_path),
            }
        )

    # Read typed quota last, immediately before the single version/push action.
    quota = api.quota_view()
    quota_refresh = getattr(quota, "quota_refresh_time", None)
    gpu = api_helpers.quota_lane(getattr(quota, "gpu_quota", None))
    tpu = api_helpers.quota_lane(getattr(quota, "tpu_quota", None))
    remaining = gpu["remaining_seconds"]
    if remaining is None or remaining < MINIMUM_GPU_SECONDS:
        raise ValueError(f"typed GPU quota {remaining} is below the 5,400-second profile gate")

    ended = datetime.now(UTC)
    command = (
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(BUNDLE_R4),
            "-m",
            "R4 restores the two frozen fold0 split target-cache mounts",
        ]
        if phase == "source-version"
        else ["kaggle", "kernels", "push", "-p", str(BUNDLE_R4 / "kernel")]
    )
    receipt = {
        "schema": "wnn.student-shape-pair-provider-preflight.r4",
        "phase": phase,
        "status": f"pass_for_{phase.replace('-', '_')}_only",
        "provider_action_performed": False,
        "start_utc": started.isoformat(),
        "end_utc": ended.isoformat(),
        "source_binding_sha256": sha256_file(PACKET / "SOURCE_BINDING_R4.json"),
        "input_binding_sha256": sha256_file(PACKET / "INPUT_BINDING_R4.json"),
        "offline_preflight_sha256": sha256_file(PACKET / "OFFLINE_SPEC_PREFLIGHT_R4.json"),
        "preflight_script_sha256": sha256_file(Path(__file__)),
        "r3_provider_helper_sha256": sha256_file(HELPER_PATH),
        "minimum_required_gpu_seconds": MINIMUM_GPU_SECONDS,
        "typed_quota": {
            "refresh_time": quota_refresh.isoformat()
            if hasattr(quota_refresh, "isoformat")
            else str(quota_refresh),
            "gpu": gpu,
            "tpu": tpu,
        },
        "mounted_datasets": dataset_rows,
        "normalization_asset": {
            "dataset": NORM_DATASET,
            "version": 1,
            "status": dataset_statuses[NORM_DATASET],
            "file_sizes": norm_inventory,
        },
        "profile_source_dataset": {
            "dataset": SOURCE_DATASET,
            "private": source_row["private"],
            "current_version_before_action": source_row["current_version"],
            "state": source_status,
            "required_file_sizes": expected_source_files,
            "observed_bound_file_sizes": {
                name: source_live[name] for name in sorted(expected_source_files)
            },
            "required_version_after_action": 2,
        },
        "existing_kernel": {
            **kernel_row,
            "id": KERNEL_ID,
            "prior_terminal_status": prior_kernel_state,
            "next_version_expected": 2,
            "list_response_flags_are_authoritative": False,
            "pulled_exact_metadata_is_authoritative": True,
            "known_concurrent_kernel_observation": overlap_observation,
            "pulled_metadata_readback_path": str(pulled_metadata_path),
            "pulled_metadata_sha256": sha256_file(pulled_metadata_path),
            "pulled_metadata_identity": kernel_metadata_identity,
            "pulled_code_sha256": sha256_file(pulled_code_path),
            "frozen_r3_code_sha256": sha256_file(frozen_code_path),
        },
        "action_command": command,
        "checks": [
            "all bound external dataset IDs are private READY at exact version1 and every mounted filename has its declared byte length",
            "fold0 train/valid target-cache assets are the exact pinned hash-named files in fold0 asset version1",
            "CP5 normalization v1 direct remote inventory matches its prior byte-verified readback",
            "existing profile source v1 is READY with the prior R3 bundle files before versioning, or new R4 v2 is READY with the R4 archive/manifest before pushing",
            "exact R3 kernel is terminal ERROR and pulled private/GPU-only metadata and code match the frozen R3 source; R4 will version it rather than create a duplicate slug",
            "the exact known CASMI kernel status and accelerator metadata are recorded as provider concurrency context; source versioning and this bounded profile may overlap under root allocation authority",
            "typed GPU quota is read last and retains at least 5,400 seconds for two 1,800-second profiles plus reserved margin",
        ],
    }
    suffix = "SOURCE_VERSION" if phase == "source-version" else "KERNEL_PUSH"
    write_receipt(PACKET / f"PROVIDER_PREFLIGHT_{suffix}_R4.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "phase": phase,
                "gpu_quota_remaining_seconds": remaining,
                "source_dataset_version": source_row["current_version"],
                "kernel_status": prior_kernel_state["state"],
                "command": command,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
