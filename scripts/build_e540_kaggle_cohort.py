#!/usr/bin/env python3
"""Assemble the private E540 V4 AMP4096 Kaggle packet without provider calls."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_bundle import build_bundle  # noqa: E402
from kaggle_common import canonical_sha256, sha256_file  # noqa: E402
from prepare_e540_fit_subset import build_manifest  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402


LINEAGE = ROOT / "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540"
SPLIT_RELATIVE = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
PORTABLE_RELATIVE = Path("research/campaign_2026-10-04/portable_inputs.json")
ASSET_MANIFEST_RELATIVE = Path("research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json")
SUBSET_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/E540_FIT_SUBSET_V4.json"
)
PACKET_ID = "E540_AMP4096_V4"
ASSET_DATASET = "vladkochetov07/wnn-c20261004-fold0-assets"
ASSET_MANIFEST_SHA256 = "19fe0eb6445553c2adff00e821a4fa16ca5969473ca8be420c32cb8e4b4b72c7"
EXTERNAL_DATA_NAMES = (
    "datasets/train.parquet",
    "datasets/valid.parquet",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
)
ARTIFACTS = [
    "e540_cohort",
    "artifacts/runs/e540v4-linear-s540-amp4096",
    "artifacts/runs/e540v4-ple-s540-amp4096",
    "cache/pred_e540v4-linear-s540-amp4096-eval_valid.npy",
    "cache/pred_e540v4-linear-s540-amp4096-eval_valid.json",
    "cache/pred_e540v4-ple-s540-amp4096-eval_valid.npy",
    "cache/pred_e540v4-ple-s540-amp4096-eval_valid.json",
]


def descriptor(path: Path, logical_name: str | None = None) -> dict[str, object]:
    """Bind a small source/config file by its exact bytes."""
    return {
        "path": logical_name or path.relative_to(ROOT).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def expected_external_descriptors(portable: dict[str, object]) -> list[dict[str, object]]:
    """Use the sealed portable registry for large mounted sources without reading them locally."""
    sources = portable["sources"]
    result = []
    for name in EXTERNAL_DATA_NAMES:
        entry = sources[name]
        identity = entry["original_identity"]
        result.append(
            {
                "path": name,
                "size": int(identity["bytes"]),
                "sha256": str(entry["sha256"]),
            }
        )
    return result


def _subset_without_full_data(split_path: Path, data_root: Path) -> dict[str, object]:
    """Build membership from split/graph metadata and only the two NPZ index arrays."""
    previous = os.environ.pop("WNN_PORTABLE_DATA_MANIFEST", None)
    try:
        campaign = load_campaign_split(split_path, data_root)
        subset = build_manifest(
            split_path,
            fraction=None,
            seed=540,
            root=data_root,
            component_count=512,
        )
    finally:
        if previous is not None:
            os.environ["WNN_PORTABLE_DATA_MANIFEST"] = previous
    if subset.get("split_manifest_sha256") != campaign.sha256:
        raise ValueError("E540 subset manifest and campaign split identities disagree")
    if subset.get("selected_component_count") != 512:
        raise ValueError(
            "E540 V4 requires exactly 512 selected fitting components; "
            f"observed {subset.get('selected_component_count')} from "
            f"{subset.get('eligible_component_count')} eligible components"
        )
    if subset.get("evaluation_group_count") != 468:
        raise ValueError("E540 V4 requires exactly 468 frozen evaluation groups")
    return subset


def _stage_small_graph_caches(data_root: Path, portable: dict[str, object]) -> None:
    """Copy only the two small feature-only component indexes into the isolated worktree."""
    cache_root = ROOT / "cache"
    cache_root.mkdir(exist_ok=True)
    for name in ("cache/chains.npz", "cache/chain_folds.npz"):
        source = data_root / name
        target = ROOT / name
        expected = portable["sources"][name]
        if source.stat().st_size > 2_000_000:
            raise ValueError(f"refusing to stage unexpectedly large graph cache: {name}")
        digest = sha256_file(source)
        if digest != expected["sha256"]:
            raise ValueError(f"feature-only graph cache differs from portable identity: {name}")
        if target.exists():
            if target.stat().st_size != source.stat().st_size or sha256_file(target) != digest:
                raise FileExistsError(f"isolated worktree has conflicting graph cache bytes: {target}")
        else:
            shutil.copyfile(source, target)


def _git_base_commit() -> str:
    """Read the worktree base commit without treating it as the dirty source identity."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_packet(data_root: Path, output_dir: Path, dataset: str) -> dict[str, object]:
    """Freeze subset, inputs, source hashes, controller spec, and private bundle."""
    output_dir = output_dir.resolve()
    data_root = data_root.resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite packet output: {output_dir}")
    if not dataset.startswith("vladkochetov07/"):
        raise ValueError("E540 private packet dataset must belong to the authorized account")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    split_path = ROOT / SPLIT_RELATIVE
    subset = _subset_without_full_data(split_path, data_root)
    subset_path = ROOT / SUBSET_RELATIVE
    subset_bytes = (json.dumps(subset, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if subset_path.exists():
        if subset_path.read_bytes() != subset_bytes:
            raise FileExistsError(f"refusing to replace an existing frozen E540 subset: {subset_path}")
    else:
        subset_path.write_bytes(subset_bytes)

    portable_path = ROOT / PORTABLE_RELATIVE
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    _stage_small_graph_caches(data_root, portable)
    asset_manifest_path = ROOT / ASSET_MANIFEST_RELATIVE
    if sha256_file(asset_manifest_path) != ASSET_MANIFEST_SHA256:
        raise ValueError("fold0 private asset manifest changed from the reviewed identity")
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    if asset_manifest.get("dataset") != ASSET_DATASET or asset_manifest.get("private") is not True:
        raise ValueError("fold0 asset manifest does not name the authorized private asset dataset")
    asset_entries = {entry["path"]: entry for entry in asset_manifest["files"]}
    for name in ("cache/train_targets.npz", "cache/valid_targets.npz"):
        entry = asset_entries.get(name)
        expected = portable["sources"][name]
        if (
            entry is None
            or entry.get("sha256") != expected["sha256"]
            or entry.get("size") != expected["original_identity"]["bytes"]
        ):
            raise ValueError(f"private asset descriptor disagrees with portable source identity: {name}")

    source_paths = [
        Path("scripts/build_e540_kaggle_cohort.py"),
        Path("scripts/e540_kaggle_worker.py"),
        Path("scripts/kaggle_common.py"),
        Path("scripts/kaggle_bundle.py"),
        Path("scripts/kaggle_gpu_runner.py"),
        Path("scripts/kaggle_kernel.py"),
        Path("scripts/predict_run.py"),
        Path("scripts/prepare_e540_fit_subset.py"),
        Path("scripts/train_rnn.py"),
        Path("src/wnn/__init__.py"),
        *sorted(path.relative_to(ROOT) for path in (ROOT / "src/wnn").rglob("*.py") if path.name != "__init__.py"),
    ]
    source_hashes = {path.as_posix(): sha256_file(ROOT / path) for path in source_paths}
    local_inputs = [
        ROOT / SPLIT_RELATIVE,
        portable_path,
        asset_manifest_path,
        ROOT / "cache/chains.npz",
        ROOT / "cache/chain_folds.npz",
        subset_path,
        LINEAGE / "RECIPE_V2.json",
        LINEAGE / "RECIPE_V3_AMP4096.json",
        LINEAGE / "RECIPE_V4_AMP4096.json",
        LINEAGE / "EXPERIMENT_CARD_V2.json",
        LINEAGE / "EXPERIMENT_CARD_AMP4096_V3.json",
        LINEAGE / "EXPERIMENT_CARD_AMP4096_V4.json",
        LINEAGE / "README_V4.md",
    ]
    local_descriptors = [descriptor(path) for path in local_inputs]
    external_descriptors = expected_external_descriptors(portable)
    asset_mounts = {
        name: {
            "dataset": ASSET_DATASET,
            "filename": asset_entries[name]["filename"],
        }
        for name in ("cache/train_targets.npz", "cache/valid_targets.npz")
    }
    packet = {
        "schema": "wnn.e540-kaggle-cohort.v2",
        "packet_id": PACKET_ID,
        "recipe": "RECIPE_V4_AMP4096.json",
        "experiment_card": "EXPERIMENT_CARD_AMP4096_V4.json",
        "status": "prepared_pending_independent_root_review_live_quota_and_actual_profile",
        "fit_admitted": False,
        "provider_launch_authorized": False,
        "grader_owner": "root independent grader",
        "generated_from": {
            "source_checkout": str(ROOT),
            "git_base_commit": _git_base_commit(),
            "source_files_are_hash_bound": True,
            "source_sha256": source_hashes,
        },
        "data": {
            "split_manifest": SPLIT_RELATIVE.as_posix(),
            "split_manifest_sha256": sha256_file(split_path),
            "portable_registry": PORTABLE_RELATIVE.as_posix(),
            "portable_registry_sha256": sha256_file(portable_path),
            "private_assets_manifest_sha256": ASSET_MANIFEST_SHA256,
            "external_expected_identities": external_descriptors,
            "asset_mounts": asset_mounts,
            "no_full_dataset_or_parquet_footer_hash_performed_locally": True,
            "no_full_target_npz_hash_or_label_array_load_performed_locally": True,
            "subset_identity_method": "split metadata and graph caches plus only ordered row_group/seq_ix index arrays from target NPZs",
            "membership_lineage": "V4 carries forward the frozen V3 exact 512-component selection; the preserved V2 ceil(10%) rule selected 315 of 3,148 eligible components, not its stated 512.",
        },
        "membership": {
            "subset_manifest": SUBSET_RELATIVE.as_posix(),
            "subset_manifest_sha256": sha256_file(subset_path),
            "selected_components": subset["selected_component_count"],
            "selected_component_sha256": subset["selected_component_sha256"],
            "eligible_components": subset["eligible_component_count"],
            "realized_component_fraction": subset["target_component_fraction"],
            "selected_groups": subset["selected_group_counts"],
            "selected_group_sha256": subset["selected_group_sha256"],
            "selected_members_sha256": subset["selected_members_sha256"],
            "fit_group_sha256": subset["selected_group_sha256"],
            "fit_members_sha256": subset["selected_members_sha256"],
            "evaluation_groups": subset["evaluation_group_count"],
            "evaluation_membership_sha256": subset["evaluation_membership_sha256"],
            "component_complete": True,
            "evaluation_disjoint_from_fit": True,
            "epoch0_loader": {
                "selected_sequence_count": sum(subset["selected_group_counts"].values()),
                "batch_size": 2,
                "drop_last": True,
                "expected_consumed_sequence_count": sum(subset["selected_group_counts"].values())
                - sum(subset["selected_group_counts"].values()) % 2,
                "expected_dropped_sequence_count": sum(subset["selected_group_counts"].values()) % 2,
                "dropped_sequence_identity": "unknown; actual sampler batch indices are not captured by this packet",
                "paired_order_status": "same seed and sampler configuration; realized order equivalence is not independently verified",
            },
            "exposure": "Previously inspected valid members remain exposed; no public forecast is implied.",
        },
        "execution": {
            "provider": "Kaggle private T4 kernel",
            "kernel_process_wall_seconds_max": 6000,
            "worker_wall_seconds_max": 5200,
            "worker_projection_margin_seconds": 300,
            "runner_finalization_reserve_seconds": 300,
            "timing_scope": "The 6,000-second watchdog begins at kernel.py process entry and covers bootstrap through runner completion. Kaggle queue/cold-start before process entry and provider output transport after process exit are outside packet control; no hard provider-from-push 6,000-second bound is claimed.",
            "gpu_worker_count": 1,
            "fit_schedule": "Profile both arms, then gate the paired one-epoch fits together; one process reuses each input digest.",
            "export": "FP32 feature-only predictions for exactly the 468 frozen evaluation groups; no ONNX or full-valid export.",
            "grading": "Not included; root supplies the grader and verifies membership before scoring.",
            "provider_mutations_performed": 0,
        },
        "comparison": {
            "arms": ["linear", "ple"],
            "run_names": ["e540v4-linear-s540-amp4096", "e540v4-ple-s540-amp4096"],
            "same_seed": 540,
            "same_subset_and_seeded_sampler_configuration": True,
            "realized_pair_order_verified": False,
            "same_parameter_shapes": True,
            "loss": "clip2-target WMSE with the existing absolute clipped-target weights and chunk denominator in both arms",
            "auxiliary_objectives": "disabled",
            "distillation": "disabled",
            "propensity": "disabled",
            "amp_initial_scale": 4096.0,
            "profile_warmup_batches": 1,
            "profile_timed_batches": 3,
            "expected_profile_attempts_per_arm": 316,
            "zero_amp_skip_gate": True,
            "max_allocator_reserved_bytes": 4300000000,
        },
        "source_packet_files_sha256": source_hashes,
        "local_input_files": local_descriptors,
        "external_input_files": external_descriptors,
    }
    packet_path = output_dir / "E540_KAGGLE_PACKET_V4.json"
    output_dir.mkdir(parents=True)
    packet_path.write_text(
        json.dumps(packet, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    code_descriptors = [descriptor(ROOT / path) for path in source_paths]
    inputs = [*local_descriptors, *external_descriptors]
    spec = {
        "version": 1,
        "name": "e540-v4-warm-profile-fit-20261008",
        "kind": "paired_profile_then_conditional_fit",
        "argv": ["{python}", "scripts/e540_kaggle_worker.py"],
        "code": code_descriptors,
        "code_sha256": canonical_sha256(sorted(code_descriptors, key=lambda item: item["path"])),
        "inputs": inputs,
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
            "WNN_DEFERRED_IDENTITY_PATHS": ",".join(EXTERNAL_DATA_NAMES),
        },
    }
    spec_path = output_dir / "e540-v4-warm-profile-fit.spec.json"
    spec_path.write_text(
        json.dumps(spec, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    external_args = [
        "datasets/train.parquet=train.parquet",
        "datasets/valid.parquet=valid.parquet",
        f"cache/train_targets.npz={ASSET_DATASET}:{asset_entries['cache/train_targets.npz']['filename']}",
        f"cache/valid_targets.npz={ASSET_DATASET}:{asset_entries['cache/valid_targets.npz']['filename']}",
    ]
    build_bundle(
        ROOT,
        [spec_path],
        output_dir / "bundle",
        dataset,
        external_args,
        [ASSET_DATASET],
    )
    generated_kernel = output_dir / "bundle/kernel/kernel.py"
    packet["generated_artifacts_sha256"] = {
        "bundle/kernel/kernel.py": {
            "size": generated_kernel.stat().st_size,
            "sha256": sha256_file(generated_kernel),
        }
    }
    packet["bundle_manifest_sha256"] = sha256_file(output_dir / "bundle/bundle-manifest.json")
    packet["bundle_archive_sha256"] = sha256_file(output_dir / "bundle/bundle.wnn")
    packet["kernel_metadata_sha256"] = sha256_file(output_dir / "bundle/kernel/kernel-metadata.json")
    packet_path.write_text(
        json.dumps(packet, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "packet": str(packet_path),
                "bundle": str(output_dir / "bundle"),
                "subset_sha256": packet["membership"]["subset_manifest_sha256"],
                "bundle_archive_sha256": packet["bundle_archive_sha256"],
                "provider_mutations_performed": 0,
            },
            sort_keys=True,
        )
    )
    return packet


def main() -> None:
    """Build the isolated packet locally; do not upload or push it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="vladkochetov07/e540-v4-warm-20261008")
    args = parser.parse_args()
    build_packet(args.data_root, args.output_dir, args.dataset)


if __name__ == "__main__":
    main()
