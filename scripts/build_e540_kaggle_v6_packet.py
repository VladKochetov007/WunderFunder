#!/usr/bin/env python3
"""Build the V6 E540 packet from frozen small inputs without reading raw data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_bundle import build_bundle  # noqa: E402
from kaggle_common import canonical_sha256, sha256_file  # noqa: E402

LINEAGE = ROOT / "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540"
PACKET_ID = "E540_AMP4096_V6"
SPEC_NAME = "e540-v6-argv-preflight-20261008"
AUTH_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/"
    "E540_V6_ROOT_CONDITIONAL_AUTHORIZATION.json"
)
SUBSET_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/E540_FIT_SUBSET_V5.json"
)
SPLIT_RELATIVE = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
PORTABLE_RELATIVE = Path("research/campaign_2026-10-04/portable_inputs.json")
ASSET_MANIFEST_RELATIVE = Path(
    "research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json"
)
ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
ASSET_MANIFEST_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
EXTERNAL_NAMES = (
    "datasets/train.parquet",
    "datasets/valid.parquet",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
)
ARTIFACTS = [
    "e540_cohort",
    "artifacts/runs/e540v6-linear-s540-amp4096",
    "artifacts/runs/e540v6-ple-s540-amp4096",
    "cache/pred_e540v6-linear-s540-amp4096-eval_valid.npy",
    "cache/pred_e540v6-linear-s540-amp4096-eval_valid.json",
    "cache/pred_e540v6-ple-s540-amp4096-eval_valid.npy",
    "cache/pred_e540v6-ple-s540-amp4096-eval_valid.json",
]


def descriptor(path: Path, logical_name: str | None = None) -> dict[str, object]:
    """Hash one small source or metadata file without loading it into memory."""
    return {
        "path": logical_name or path.relative_to(ROOT).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git_base_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_paths() -> list[Path]:
    return [
        Path("scripts/build_e540_kaggle_v6_packet.py"),
        Path("scripts/e540_cli_preflight.py"),
        Path("scripts/e540_kaggle_worker.py"),
        Path("scripts/kaggle_common.py"),
        Path("scripts/kaggle_bundle.py"),
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


def _local_input_paths() -> list[Path]:
    return [
        SPLIT_RELATIVE,
        PORTABLE_RELATIVE,
        ASSET_MANIFEST_RELATIVE,
        Path("cache/chains.npz"),
        Path("cache/chain_folds.npz"),
        SUBSET_RELATIVE,
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/RECIPE_V2.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/RECIPE_V3_AMP4096.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/RECIPE_V4_AMP4096.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/RECIPE_V5_AMP4096.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/RECIPE_V6_AMP4096.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/EXPERIMENT_CARD_V2.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/EXPERIMENT_CARD_AMP4096_V3.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/EXPERIMENT_CARD_AMP4096_V4.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/EXPERIMENT_CARD_AMP4096_V5.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/EXPERIMENT_CARD_AMP4096_V6.json"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/README_V4.md"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/README_V5.md"
        ),
        Path(
            "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/README_V6.md"
        ),
        AUTH_RELATIVE,
    ]


def _expected_external(portable: dict[str, object]) -> list[dict[str, object]]:
    sources = portable["sources"]
    return [
        {
            "path": name,
            "size": int(sources[name]["original_identity"]["bytes"]),
            "sha256": str(sources[name]["sha256"]),
        }
        for name in EXTERNAL_NAMES
    ]


def _ensure_pending_authorization(
    path: Path,
    *,
    code_manifest_sha256: str,
    recipe_sha256: str,
    subset_sha256: str,
) -> dict[str, object]:
    """Create only a fail-closed template; root owns approval and authorization ID."""
    expected = {
        "schema": "wnn.e540-conditional-fit-authorization.v1",
        "owner": "root",
        "status": "pending_root_review",
        "authorized": False,
        "conditional_fit_after_profile_gate": True,
        "packet_id": PACKET_ID,
        "code_manifest_sha256": code_manifest_sha256,
        "recipe_sha256": recipe_sha256,
        "subset_manifest_sha256": subset_sha256,
        "kernel_process_wall_seconds_max": 6000,
        "worker_wall_seconds_max": 5200,
        "authorization_id": None,
        "provider_launch_authorized": False,
        "profile_gate": {
            "both_arms_pass": True,
            "amp_skipped_updates_per_arm": 0,
            "allocator_reserved_bytes_max": 4_300_000_000,
            "projected_worker_seconds_max": 5200,
        },
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("status") == "approved" or existing.get("authorized") is True:
            for key, value in expected.items():
                if (
                    key not in {"status", "authorized", "authorization_id"}
                    and existing.get(key) != value
                ):
                    raise ValueError(f"existing approved root authorization differs in {key}")
            return existing
        path.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return expected
    path.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return expected


def build_packet(output_dir: Path, *, dataset: str) -> dict[str, object]:
    """Build the bundle from frozen manifests; never open raw or target NPZ data."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite V6 output: {output_dir}")
    if not dataset.startswith("vladkochetov07/"):
        raise ValueError("E540 private packet dataset must belong to the authorized account")

    v5_packet_path = LINEAGE / "kaggle_v5_final_20261008/E540_KAGGLE_PACKET_V5.json"
    v5_packet = json.loads(v5_packet_path.read_text(encoding="utf-8"))
    subset_path = ROOT / SUBSET_RELATIVE
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    subset_sha = sha256_file(subset_path)
    if (
        v5_packet["membership"]["subset_manifest_sha256"] != subset_sha
        or subset.get("selected_component_count") != 512
        or subset.get("evaluation_group_count") != 468
        or subset.get("selected_component_sha256")
        != v5_packet["membership"]["selected_component_sha256"]
        or subset.get("evaluation_membership_sha256")
        != v5_packet["membership"]["evaluation_membership_sha256"]
    ):
        raise ValueError("frozen V5 subset identity or 512/468 membership changed")

    portable_path = ROOT / PORTABLE_RELATIVE
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    external_descriptors = _expected_external(portable)
    if external_descriptors != v5_packet["external_input_files"]:
        raise ValueError("V6 external mounted-input identities changed from V5")
    asset_manifest_path = ROOT / ASSET_MANIFEST_RELATIVE
    if sha256_file(asset_manifest_path) != ASSET_MANIFEST_SHA256:
        raise ValueError("authorized private fold0 assets manifest changed")
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    if asset_manifest.get("dataset") != ASSET_DATASET or asset_manifest.get("private") is not True:
        raise ValueError("private asset manifest identity or privacy flag changed")
    asset_entries = {entry["path"]: entry for entry in asset_manifest["files"]}
    for name in ("cache/train_targets.npz", "cache/valid_targets.npz"):
        expected = next(item for item in external_descriptors if item["path"] == name)
        entry = asset_entries.get(name)
        if (
            entry is None
            or entry.get("sha256") != expected["sha256"]
            or entry.get("size") != expected["size"]
        ):
            raise ValueError(f"private asset descriptor changed: {name}")

    for name in ("cache/chains.npz", "cache/chain_folds.npz"):
        path = ROOT / name
        if path.stat().st_size > 2_000_000:
            raise ValueError(f"refusing unexpected large feature-only graph cache: {name}")
        prior = next(item for item in v5_packet["local_input_files"] if item["path"] == name)
        if path.stat().st_size != prior["size"] or sha256_file(path) != prior["sha256"]:
            raise ValueError(f"feature-only graph cache changed from V5: {name}")

    source_paths = _source_paths()
    source_hashes = {path.as_posix(): sha256_file(ROOT / path) for path in source_paths}
    code_descriptors = [descriptor(ROOT / path) for path in source_paths]
    code_manifest_sha256 = canonical_sha256(sorted(code_descriptors, key=lambda item: item["path"]))
    recipe_path = LINEAGE / "RECIPE_V6_AMP4096.json"
    recipe_sha256 = sha256_file(recipe_path)
    authorization = _ensure_pending_authorization(
        ROOT / AUTH_RELATIVE,
        code_manifest_sha256=code_manifest_sha256,
        recipe_sha256=recipe_sha256,
        subset_sha256=subset_sha,
    )
    local_descriptors = [descriptor(ROOT / path) for path in _local_input_paths()]
    if authorization.get("status") == "approved":
        from e540_kaggle_worker import validate_conditional_authorization

        contract = {
            "packet_id": PACKET_ID,
            "code_manifest_sha256": code_manifest_sha256,
            "recipe_sha256": recipe_sha256,
            "subset_manifest_sha256": subset_sha,
            "kernel_process_wall_seconds_max": 6000,
            "worker_wall_seconds_max": 5200,
        }
        validate_conditional_authorization(ROOT / AUTH_RELATIVE, contract)

    asset_mounts = {
        name: {
            "dataset": ASSET_DATASET,
            "filename": asset_entries[name]["filename"],
        }
        for name in ("cache/train_targets.npz", "cache/valid_targets.npz")
    }
    external_args = [
        "datasets/train.parquet=train.parquet",
        "datasets/valid.parquet=valid.parquet",
        f"cache/train_targets.npz={ASSET_DATASET}:{asset_entries['cache/train_targets.npz']['filename']}",
        f"cache/valid_targets.npz={ASSET_DATASET}:{asset_entries['cache/valid_targets.npz']['filename']}",
    ]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    packet = {
        "schema": "wnn.e540-kaggle-cohort.v4",
        "packet_id": PACKET_ID,
        "recipe": "RECIPE_V6_AMP4096.json",
        "experiment_card": "EXPERIMENT_CARD_AMP4096_V6.json",
        "status": "prepared_pending_independent_root_review_and_conditional_authorization",
        "fit_admitted": False,
        "provider_launch_authorized": False,
        "grader_owner": "root independent grader",
        "generated_from": {
            "source_checkout": str(ROOT),
            "git_base_commit": _git_base_commit(),
            "source_files_are_hash_bound": True,
            "source_sha256": source_hashes,
            "code_manifest_sha256": code_manifest_sha256,
        },
        "scientific_factors": {
            "unchanged_from": "E540_AMP4096_V5",
            "selected_components": 512,
            "evaluation_groups": 468,
            "amp_initial_scale_both_arms": 4096.0,
            "training_loss": "clip2-target WMSE with absolute clipped-target weights and chunk denominator",
            "epochs": 1,
            "checkpoint_epoch": 0,
            "scientific_factor_change": False,
        },
        "v5_execution_diagnostic": {
            "classification": "execution failure before profile; no scientific outcome",
            "worker_error": "train_rnn.py: error: unrecognized arguments: scripts/train_rnn.py",
            "mounted_input_identity": "all four expected and observed sizes and SHA-256 values matched",
            "fit_started": False,
        },
        "data": {
            "split_manifest": SPLIT_RELATIVE.as_posix(),
            "split_manifest_sha256": sha256_file(ROOT / SPLIT_RELATIVE),
            "portable_registry": PORTABLE_RELATIVE.as_posix(),
            "portable_registry_sha256": sha256_file(portable_path),
            "private_assets_manifest_sha256": ASSET_MANIFEST_SHA256,
            "external_expected_identities": external_descriptors,
            "asset_mounts": asset_mounts,
            "subset_identity_source": "exact immutable V5 subset JSON; no NPZ arrays or raw data re-read locally",
            "no_local_parquet_footer_or_full_target_npz_read": True,
        },
        "membership": v5_packet["membership"],
        "conditional_authorization": {
            "receipt": AUTH_RELATIVE.as_posix(),
            "receipt_sha256": sha256_file(ROOT / AUTH_RELATIVE),
            "owner": "root",
            "status": authorization["status"],
            "authorization_required_before_mounted_data_hashing": True,
            "conditional_fit_after_both_profile_gates": True,
            "source_manifest_sha256": code_manifest_sha256,
            "recipe_sha256": recipe_sha256,
            "subset_manifest_sha256": subset_sha,
        },
        "execution": {
            "provider": "Kaggle private Tesla T4 kernel",
            "kernel_process_wall_seconds_max": 6000,
            "worker_wall_seconds_max": 5200,
            "runner_finalization_reserve_seconds": 300,
            "worker_projection_margin_seconds": 300,
            "startup_parser_validation": "actual train_rnn.py profile/fit and predict_run.py parsers for both arms, stdlib-only, before mounted input hashing",
            "startup_failure_behavior": "persist startup_preflight.json and stop before mounted input hashing or fit",
            "fit_schedule": "profile both arms; then run the matched one-epoch pair only if the strict gate passes",
            "prediction": "FP32 feature-only predictions for exactly the 468 frozen evaluation groups",
            "grading": "not included; root owns independent grading",
            "provider_mutations_performed": 0,
        },
        "source_packet_files_sha256": source_hashes,
        "local_input_files": local_descriptors,
        "external_input_files": external_descriptors,
    }
    packet_path = output_dir / "E540_KAGGLE_PACKET_V6.json"
    packet_path.write_text(json.dumps(packet, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    spec = {
        "version": 1,
        "name": SPEC_NAME,
        "kind": "paired_profile_then_conditional_fit",
        "argv": ["{python}", "scripts/e540_kaggle_worker.py"],
        "code": code_descriptors,
        "code_sha256": code_manifest_sha256,
        "inputs": [*local_descriptors, *external_descriptors],
        "artifacts": ARTIFACTS,
        "limits": {
            "cpus": 2,
            "ram_bytes": 8 * 1024**3,
            "timeout_seconds": 5200,
            "gpu_memory_bytes": 4_300_000_000,
            "gpu_stop_bytes": 5_000_000_000,
        },
        "environment": {
            "PYTHONHASHSEED": "0",
            "WNN_THREADS": "2",
            "WNN_LOADER_RAM_GB": "1",
            "WNN_LOADER_WORKERS": "0",
            "WNN_PORTABLE_DATA_MANIFEST": PORTABLE_RELATIVE.as_posix(),
            "WNN_DEFERRED_IDENTITY_PATHS": ",".join(EXTERNAL_NAMES),
            "WNN_E540_PACKET_ID": PACKET_ID,
            "WNN_E540_AUTHORIZATION_PATH": AUTH_RELATIVE.as_posix(),
            "WNN_E540_CODE_MANIFEST_SHA256": code_manifest_sha256,
            "WNN_E540_RECIPE_SHA256": recipe_sha256,
            "WNN_E540_SUBSET_SHA256": subset_sha,
            "WNN_E540_KERNEL_WALL_SECONDS": "6000",
            "WNN_E540_WORKER_WALL_SECONDS": "5200",
        },
    }
    spec_path = output_dir / f"{SPEC_NAME}.spec.json"
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    build_bundle(
        ROOT,
        [spec_path],
        output_dir / "bundle",
        dataset,
        external_args,
        [ASSET_DATASET],
    )
    kernel = output_dir / "bundle/kernel/kernel.py"
    packet["generated_artifacts_sha256"] = {
        "bundle/kernel/kernel.py": {
            "size": kernel.stat().st_size,
            "sha256": sha256_file(kernel),
        }
    }
    packet["bundle_archive_sha256"] = sha256_file(output_dir / "bundle/bundle.wnn")
    packet["bundle_manifest_sha256"] = sha256_file(output_dir / "bundle/bundle-manifest.json")
    packet["kernel_metadata_sha256"] = sha256_file(
        output_dir / "bundle/kernel/kernel-metadata.json"
    )
    packet_path.write_text(json.dumps(packet, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="vladkochetov07/e540-v6-argv-20261008")
    args = parser.parse_args()
    result = build_packet(args.output_dir, dataset=args.dataset)
    print(
        json.dumps(
            {
                "packet": str(args.output_dir / "E540_KAGGLE_PACKET_V6.json"),
                "bundle_archive_sha256": result["bundle_archive_sha256"],
                "conditional_authorization": result["conditional_authorization"]["status"],
                "provider_mutations_performed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
