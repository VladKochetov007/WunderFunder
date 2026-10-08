#!/usr/bin/env python3
"""Build an export-only successor from the frozen E540 V8/V9 artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_bundle import build_bundle  # noqa: E402
from kaggle_common import canonical_sha256, sha256_file, write_json  # noqa: E402

LINEAGE = ROOT / "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540"
PACKET_DIR = LINEAGE / "v11_export_only"
PACKET_ID = "E540_FP32_V11_FIXED_CHECKPOINT_EXPORT_ONLY"
KERNEL_DATASET = "vladkochetov07/e540-v11-binding-fix-20261008"
ASSET_DATASET = "vladkochetov07/e540-v10-artifact-assets-20261008"
SPEC_NAME = "e540-v11-fixed-checkpoint-export-20261008"
RECIPE_PATH = PACKET_DIR / "RECIPE_V11_EXPORT_ONLY.json"
BINDING_PATH = PACKET_DIR / "E540_V11_ARTIFACT_BINDING.json"
AUTH_PATH = PACKET_DIR / "E540_V11_ROOT_EXPORT_AUTHORIZATION.json"
CARD_PATH = PACKET_DIR / "EXPERIMENT_CARD_FP32_V11_EXPORT_ONLY.json"
SUBSET_PATH = LINEAGE / "E540_FIT_SUBSET_V5.json"
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
PORTABLE_PATH = ROOT / "research/campaign_2026-10-04/portable_inputs.json"
V8_TERMINAL_PATH = LINEAGE / "E540_V8_TERMINAL_OPERATOR_RECEIPT_20261008_R1.json"
SOURCE_EVIDENCE_DIR = PACKET_DIR / "source_evidence"
RUN_NAMES = {
    "linear": "e540v8-linear-s540-fp32",
    "ple": "e540v9-ple-s540-fp32",
}
ARTIFACTS = [
    "e540_v11_export",
    f"artifacts/runs/{RUN_NAMES['linear']}",
    f"artifacts/runs/{RUN_NAMES['ple']}",
    f"cache/pred_{RUN_NAMES['linear']}-eval_valid.npy",
    f"cache/pred_{RUN_NAMES['linear']}-eval_valid.json",
    f"cache/pred_{RUN_NAMES['ple']}-eval_valid.npy",
    f"cache/pred_{RUN_NAMES['ple']}-eval_valid.json",
]


def descriptor(path: Path) -> dict[str, object]:
    """Describe a small declared source input without reading model data."""
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def source_paths() -> list[Path]:
    """List the exact project code closure for the export-only worker."""
    return [
        Path("scripts/build_e540_kaggle_v11_export_packet.py"),
        Path("scripts/e540_cli_preflight.py"),
        Path("scripts/e540_export_validation.py"),
        Path("scripts/e540_kaggle_v11_export_worker.py"),
        Path("scripts/kaggle_assets.py"),
        Path("scripts/kaggle_bundle.py"),
        Path("scripts/kaggle_common.py"),
        Path("scripts/kaggle_gpu_runner.py"),
        Path("scripts/kaggle_kernel.py"),
        Path("scripts/predict_run.py"),
        Path("scripts/prepare_e540_fit_subset.py"),
        Path("scripts/train_rnn.py"),
        Path("src/wnn/__init__.py"),
        *sorted(
            path.relative_to(ROOT)
            for path in (ROOT / "src/wnn").rglob("*.py")
            if path.name != "__init__.py"
        ),
    ]


def local_inputs() -> list[Path]:
    """Return the exact recipe, identity, and terminal-evidence files."""
    evidence = sorted(SOURCE_EVIDENCE_DIR.glob("*.json"))
    if not evidence:
        raise ValueError("V11 packet is missing its frozen source evidence")
    return [
        BINDING_PATH,
        RECIPE_PATH,
        CARD_PATH,
        AUTH_PATH,
        SUBSET_PATH,
        SPLIT_PATH,
        PORTABLE_PATH,
        V8_TERMINAL_PATH,
        *evidence,
    ]


def _expected_authorization(
    code_sha: str, recipe_sha: str, binding_sha: str, binding: dict[str, Any]
) -> dict[str, object]:
    return {
        "schema": "wnn.e540-export-only-authorization.v1",
        "owner": "root",
        "status": "pending_root_review",
        "authorized": False,
        "export_only": True,
        "fit_allowed": False,
        "provider_launch_authorized": False,
        "packet_id": PACKET_ID,
        "code_manifest_sha256": code_sha,
        "recipe_sha256": recipe_sha,
        "artifact_binding_sha256": binding_sha,
        "subset_manifest_sha256": sha256_file(SUBSET_PATH),
        "kernel_process_wall_seconds_max": 6_000,
        "worker_wall_seconds_max": 5_200,
        "linear_epoch0_sha256": binding["assets"]["linear_epoch0"]["sha256"],
        "ple_epoch0_sha256": binding["assets"]["ple_epoch0"]["sha256"],
        "linear_provenance_sha256": binding["assets"]["linear_provenance"]["sha256"],
        "ple_provenance_sha256": binding["assets"]["ple_provenance"]["sha256"],
        "linear_history_sha256": binding["assets"]["linear_history"]["sha256"],
        "ple_history_sha256": binding["assets"]["ple_history"]["sha256"],
        "v9_terminal_receipt_sha256": binding["source_fits"]["ple"]["terminal_receipt_sha256"],
        "authorization_id": None,
        "prediction_membership_count": 468,
        "prediction_arms": [RUN_NAMES["linear"], RUN_NAMES["ple"]],
        "fit_arms": [],
        "profile_arms": [],
        "root_review_receipt_sha256": None,
    }


def _prepare_authorization(expected: dict[str, object]) -> dict[str, object]:
    if AUTH_PATH.is_file():
        current = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
        if current.get("status") == "approved" and current.get("authorized") is True:
            for key, value in expected.items():
                if key in {
                    "status",
                    "authorized",
                    "provider_launch_authorized",
                    "authorization_id",
                    "root_review_receipt_sha256",
                }:
                    continue
                if current.get(key) != value:
                    raise ValueError(f"approved V11 authorization differs in {key}")
            return current
    write_json(AUTH_PATH, expected)
    return expected


def _validate_frozen_inputs(binding: dict[str, Any], recipe: dict[str, Any]) -> None:
    subset_sha = sha256_file(SUBSET_PATH)
    split_sha = sha256_file(SPLIT_PATH)
    portable = json.loads(PORTABLE_PATH.read_text(encoding="utf-8"))
    valid_source = binding.get("valid_source")
    original = (
        portable.get("sources", {}).get("datasets/valid.parquet", {}).get("original_identity", {})
    )
    if (
        binding.get("packet_id") != PACKET_ID
        or binding.get("schema") != "wnn.e540-v11-artifact-binding.v1"
        or recipe.get("packet_id") != PACKET_ID
        or recipe.get("kind") != "export_only"
        or recipe.get("fit_arms") != []
        or recipe.get("profile_arms") != []
        or recipe.get("prediction_arms") != [RUN_NAMES["linear"], RUN_NAMES["ple"]]
        or binding.get("subset_manifest_sha256") != subset_sha
        or binding.get("split_manifest_sha256") != split_sha
        or binding.get("selected_component_count") != 512
        or binding.get("evaluation_group_count") != 468
        or not isinstance(valid_source, dict)
        or original.get("bytes") != valid_source.get("size")
        or portable.get("sources", {}).get("datasets/valid.parquet", {}).get("sha256")
        != valid_source.get("sha256")
        or "recipe_sha256" in binding
    ):
        raise ValueError("V11 frozen packet identities or source membership changed")
    if set(binding.get("assets", {})) != {
        "linear_model",
        "linear_epoch0",
        "linear_provenance",
        "linear_history",
        "linear_control_manifest",
        "ple_model",
        "ple_epoch0",
        "ple_provenance",
        "ple_history",
    }:
        raise ValueError("V11 packet does not bind the exact frozen V8/V9 artifact pair")
    assets = binding["assets"]
    for asset in assets.values():
        if (
            not isinstance(asset, dict)
            or not isinstance(asset.get("logical_path"), str)
            or not isinstance(asset.get("upload_filename"), str)
            or not isinstance(asset.get("sha256"), str)
            or not isinstance(asset.get("size"), int)
        ):
            raise ValueError("V11 frozen artifact binding has a malformed asset descriptor")
    ready_path = SOURCE_EVIDENCE_DIR / "E540_V10_PRIVATE_ASSETS_READY_20261008_R1.json"
    readiness = json.loads(ready_path.read_text(encoding="utf-8"))
    expected_files = {
        asset["upload_filename"]: {
            "sha256": asset["sha256"],
            "size_bytes": asset["size"],
        }
        for asset in assets.values()
    }
    observed_files = {
        filename: entry
        for filename, entry in readiness.get("files", {}).items()
        if filename != "assets-manifest.json"
    }
    if (
        readiness.get("dataset_id") != ASSET_DATASET
        or readiness.get("readiness") != "owner file inventory and full dataset download succeeded"
        or readiness.get("exact_downloaded_file_set_and_hashes_verified") is not True
        or observed_files != expected_files
    ):
        raise ValueError("V11 reused private asset dataset differs from the frozen checkpoints")


def build_packet(output_dir: Path) -> dict[str, object]:
    """Build a source-only successor packet without loading checkpoints or data."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite V11 packet directory: {output_dir}")
    binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    _validate_frozen_inputs(binding, recipe)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    source_files = source_paths()
    code_descriptors = [descriptor(ROOT / path) for path in source_files]
    code_sha = canonical_sha256(sorted(code_descriptors, key=lambda item: item["path"]))
    source_hashes = {path.as_posix(): sha256_file(ROOT / path) for path in source_files}
    recipe_sha = sha256_file(RECIPE_PATH)
    binding_sha = sha256_file(BINDING_PATH)
    authorization = _prepare_authorization(
        _expected_authorization(code_sha, recipe_sha, binding_sha, binding)
    )

    local_files = local_inputs()
    local_descriptors = [descriptor(path) for path in local_files]
    external = ["datasets/valid.parquet=valid.parquet"]
    external_descriptors: list[dict[str, object]] = [
        {
            "path": "datasets/valid.parquet",
            "size": binding["valid_source"]["size"],
            "sha256": binding["valid_source"]["sha256"],
        }
    ]
    for asset in binding["assets"].values():
        filename = asset["upload_filename"]
        logical = asset["logical_path"]
        external.append(f"{logical}={ASSET_DATASET}:{filename}")
        external_descriptors.append(
            {"path": logical, "size": asset["size"], "sha256": asset["sha256"]}
        )
    if {item["path"] for item in local_descriptors} & {
        item["path"] for item in external_descriptors
    }:
        raise ValueError("V11 local and mounted input paths collide")

    spec = {
        "version": 1,
        "name": SPEC_NAME,
        "kind": "fixed_checkpoint_pair_prediction_export",
        "argv": ["{python}", "scripts/e540_kaggle_v11_export_worker.py"],
        "code": code_descriptors,
        "code_sha256": code_sha,
        "inputs": [*local_descriptors, *external_descriptors],
        "artifacts": ARTIFACTS,
        "limits": {
            "cpus": 2,
            "ram_bytes": 8 * 1024**3,
            "timeout_seconds": 5_200,
            "gpu_memory_bytes": 4_300_000_000,
            "gpu_stop_bytes": 5_000_000_000,
        },
        "environment": {
            "PYTHONHASHSEED": "0",
            "WNN_THREADS": "2",
            "WNN_LOADER_RAM_GB": "1",
            "WNN_LOADER_WORKERS": "0",
            "WNN_DEFERRED_IDENTITY_PATHS": "datasets/valid.parquet",
            "WNN_E540_PACKET_ID": PACKET_ID,
            "WNN_E540_AUTHORIZATION_FILENAME": AUTH_PATH.name,
            "WNN_E540_AUTHORIZATION_PATH": AUTH_PATH.relative_to(ROOT).as_posix(),
            "WNN_E540_CODE_MANIFEST_SHA256": code_sha,
            "WNN_E540_RECIPE_SHA256": recipe_sha,
            "WNN_E540_ARTIFACT_BINDING_SHA256": binding_sha,
            "WNN_E540_SUBSET_SHA256": sha256_file(SUBSET_PATH),
            "WNN_E540_KERNEL_WALL_SECONDS": "6000",
            "WNN_E540_WORKER_WALL_SECONDS": "5200",
            "WNN_E540_LINEAR_EPOCH0_SHA256": binding["assets"]["linear_epoch0"]["sha256"],
            "WNN_E540_PLE_EPOCH0_SHA256": binding["assets"]["ple_epoch0"]["sha256"],
            "WNN_E540_LINEAR_PROVENANCE_SHA256": binding["assets"]["linear_provenance"]["sha256"],
            "WNN_E540_PLE_PROVENANCE_SHA256": binding["assets"]["ple_provenance"]["sha256"],
            "WNN_E540_LINEAR_HISTORY_SHA256": binding["assets"]["linear_history"]["sha256"],
            "WNN_E540_PLE_HISTORY_SHA256": binding["assets"]["ple_history"]["sha256"],
            "WNN_E540_V9_TERMINAL_RECEIPT_SHA256": binding["source_fits"]["ple"][
                "terminal_receipt_sha256"
            ],
        },
    }
    spec_path = output_dir / f"{SPEC_NAME}.spec.json"
    write_json(spec_path, spec)
    build_bundle(
        ROOT,
        [spec_path],
        output_dir / "bundle",
        KERNEL_DATASET,
        external,
        [ASSET_DATASET],
    )
    packet = {
        "schema": "wnn.e540-kaggle-export-cohort.v1",
        "packet_id": PACKET_ID,
        "parent_packet_id": "E540_FP32_V10_FIXED_CHECKPOINT_EXPORT_ONLY",
        "status": "prepared_pending_root_source_and_final_byte_review",
        "fit_admitted": False,
        "provider_launch_authorized": authorization.get("provider_launch_authorized") is True,
        "grade_owner": "root independent grader",
        "generated_from": {
            "source_checkout": str(ROOT),
            "git_head_before_packet_commit": subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "source_files_hash_bound": source_hashes,
            "code_manifest_sha256": code_sha,
        },
        "recipe": {"path": str(RECIPE_PATH.relative_to(ROOT)), "sha256": recipe_sha},
        "artifact_binding": {
            "path": str(BINDING_PATH.relative_to(ROOT)),
            "sha256": binding_sha,
            "linear_epoch0_sha256": binding["assets"]["linear_epoch0"]["sha256"],
            "ple_epoch0_sha256": binding["assets"]["ple_epoch0"]["sha256"],
            "reused_without_refit": True,
        },
        "data": {
            "valid_source": binding["valid_source"],
            "subset_manifest_sha256": sha256_file(SUBSET_PATH),
            "split_manifest_sha256": sha256_file(SPLIT_PATH),
            "evaluation_group_count": 468,
            "only_valid_parquet_mounted": True,
            "no_target_npz_mount": True,
            "no_full_parquet_footer_or_target_npz_read": True,
        },
        "authorization": {
            "path": str(AUTH_PATH.relative_to(ROOT)),
            "sha256": sha256_file(AUTH_PATH),
            "status": authorization["status"],
            "owner": "root",
            "provider_launch_authorized": authorization.get("provider_launch_authorized") is True,
        },
        "execution": {
            "kernel_process_wall_seconds_max": 6_000,
            "worker_wall_seconds_max": 5_200,
            "runner_finalization_reserve_seconds": 300,
            "profile_arms": [],
            "fit_arms": [],
            "prediction_arms": ["linear", "ple"],
            "grade_performed": False,
            "provider_upload_performed": False,
            "provider_push_performed": False,
        },
        "bundle": {
            "archive_sha256": sha256_file(output_dir / "bundle/bundle.wnn"),
            "manifest_sha256": sha256_file(output_dir / "bundle/bundle-manifest.json"),
            "bootstrap_sha256": sha256_file(output_dir / "bundle/kernel/kernel.py"),
            "kernel_metadata_sha256": sha256_file(
                output_dir / "bundle/kernel/kernel-metadata.json"
            ),
        },
        "private_asset_dataset_reused": ASSET_DATASET,
    }
    write_json(output_dir / "E540_KAGGLE_PACKET_V11.json", packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_packet(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
