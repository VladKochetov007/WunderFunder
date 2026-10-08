#!/usr/bin/env python3
"""Build a frozen, root-reviewable E540 profile-to-grade packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

SOURCE_FILES = (
    "scripts/build_e540_packet.py",
    "scripts/e540_data_shape_smoke.py",
    "scripts/export_onnx.py",
    "scripts/predict_run.py",
    "scripts/prepare_e540_fit_subset.py",
    "scripts/train_rnn.py",
    "tests/test_numeric_ple.py",
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/EXPERIMENT_CARD.json",
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/EXPERIMENT_CARD_V2.json",
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/HYPOTHESIS_CARD.json",
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/RECIPE_V1.json",
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/RECIPE_V2.json",
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/README.md",
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/E540_INDEPENDENT_RECIPE_DELIVERY_REVIEW_V2_20261008.md",
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/DATA_SHAPE_SMOKE_ACTUAL_TRAIN_RG0_V1.jsonl",
    "research/campaign_2026-10-06/bandit_continuation/numerical_embedding_E540/DATA_SHAPE_SMOKE_ATTEMPTS_V1.json",
    "src/wnn/data.py",
    "src/wnn/features.py",
    "src/wnn/gmetric.py",
    "src/wnn/nn.py",
    "src/wnn/numeric_ple.py",
    "src/wnn/optim.py",
    "src/wnn/resources.py",
    "src/wnn/run_identity.py",
    "src/wnn/sentinel.py",
    "src/wnn/splits.py",
    "src/wnn/streaming.py",
)


def _sha256_ints(values: tuple[int, ...]) -> str:
    """Hash a stable little-endian encoding of row-group identities."""
    array = np.asarray(sorted(values), dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _git_commit(root: Path) -> str | None:
    """Return the exact source revision when the checkout is a Git worktree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=root,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _base_args(run_name: str, numeric_map: str, split_path: Path, subset_path: Path) -> list[str]:
    """Return the common, explicit pair recipe arguments."""
    return [
        "python",
        "scripts/train_rnn.py",
        "--device",
        "cuda",
        "--name",
        run_name,
        "--numeric-map",
        numeric_map,
        "--ple-intervals",
        "8",
        "--seed",
        "540",
        "--train-source",
        "train+valid",
        "--split-manifest",
        str(split_path),
        "--fit-subset-manifest",
        str(subset_path),
        "--hidden",
        "128",
        "--layers",
        "1",
        "--dropout",
        "0",
        "--batch",
        "2",
        "--chunk",
        "256",
        "--epochs",
        "1",
        "--loss",
        "wmse",
        "--loss-norm",
        "chunk",
        "--optim",
        "adamw",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--sched",
        "onecycle",
        "--precision",
        "fp16",
        "--batch-order",
        "seeded",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "0",
        "--no-eval-during-fit",
        "--deterministic",
    ]


def _command(argv: list[str], cwd: Path) -> dict[str, object]:
    """Serialize an exact command without invoking it."""
    return {"cwd": str(cwd), "argv": argv, "shell": shlex.join(argv)}


def build_packet(split_manifest: Path, output_dir: Path) -> dict[str, object]:
    """Bind exact source/data/split identities and provisional commands."""
    from prepare_e540_fit_subset import build_manifest

    split_path = split_manifest if split_manifest.is_absolute() else ROOT / split_manifest
    split_path = split_path.resolve(strict=True)
    output = output_dir if output_dir.is_absolute() else ROOT / output_dir
    output = output.resolve()
    for name in ("datasets/train.parquet", "datasets/valid.parquet"):
        if not (ROOT / name).is_file():
            raise FileNotFoundError(
                f"run the packet builder in the integrated project checkout with {name} present"
            )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty packet directory: {output}")

    campaign = load_campaign_split(split_path, ROOT)
    subset = build_manifest(split_path, fraction=0.1, seed=540, root=ROOT)
    if subset["selected_component_count"] != 512:
        raise ValueError("E540 V2 requires exactly 512 complete fitting components")
    if len(campaign.evaluation_groups) != 468:
        raise ValueError("E540 V2 requires the frozen 468-group evaluation membership")
    subset_path = output / "E540_FIT_SUBSET_V2.json"
    split_source = json.loads(split_path.read_text(encoding="utf-8"))
    data_identities = {
        name: split_source["sources"][name]
        for name in ("datasets/train.parquet", "datasets/valid.parquet")
    }

    names = {
        "linear": "e540-linear-s540",
        "ple": "e540-ple-s540",
    }
    profile_commands: dict[str, dict[str, object]] = {}
    fit_commands: dict[str, dict[str, object]] = {}
    prediction_exports: dict[str, dict[str, object]] = {}
    streaming_exports: dict[str, dict[str, object]] = {}
    for mode, run_name in names.items():
        base = _base_args(run_name, mode, split_path, subset_path)
        profile = [*base, "--profile-warmup", "1", "--profile-steps", "2"]
        profile_commands[mode] = _command(profile, ROOT)
        fit_commands[mode] = _command(base, ROOT)
        prediction_exports[mode] = _command(
            [
                "python",
                "scripts/predict_run.py",
                run_name,
                "valid",
                "--epoch",
                "0",
                "--inference-precision",
                "fp32",
                "--output-name",
                f"{run_name}-eval",
                "--fit-subset-manifest",
                str(subset_path),
            ],
            ROOT,
        )
        streaming_exports[mode] = _command(
            [
                "python",
                "scripts/export_onnx.py",
                run_name,
                "--out",
                str(output / f"{mode}.onnx"),
                "--rows",
                "20000",
                "--parity-split",
                "train",
                "--parity-group",
                str(subset["selected_groups"]["train"][0]),
            ],
            ROOT,
        )

    source_hashes = {name: file_sha256(ROOT / name) for name in SOURCE_FILES}
    packet = {
        "version": 2,
        "experiment_id": "E540",
        "recipe": "RECIPE_V2.json",
        "predecessor_recipe": "RECIPE_V1.json",
        "status": "prepared_pending_root_admission",
        "fit_admitted": False,
        "provider_launch_authorized": False,
        "grader_owner": "root independent grader",
        "generated_from": {
            "checkout_root": str(ROOT),
            "git_commit": _git_commit(ROOT),
            "source_files_sha256": source_hashes,
        },
        "inputs": {
            "split_manifest": str(split_path),
            "split_manifest_sha256": campaign.sha256,
            "data_identities_from_split_manifest": data_identities,
            "no_full_dataset_hash_performed": True,
        },
        "membership": {
            "fitting_subset_manifest": str(subset_path),
            "fitting_subset": subset,
            "evaluation_group_count": len(campaign.evaluation_groups),
            "evaluation_group_sha256": _sha256_ints(campaign.evaluation_groups),
            "evaluation_components_are_excluded_from_fit": True,
            "exposure": "Previously inspected validation remains exposed; no holdout reset is implied.",
        },
        "comparison": {
            "same_seed": 540,
            "same_subset_and_loader_order": True,
            "same_initialized_parameter_shapes_and_values": True,
            "arms": names,
            "allocated_parameters_per_arm": 274050,
            "capacity_caveat": "The 896 linear-control ramp input coordinates are zero; their connected input-projection weights do not add data-dependent expressivity.",
            "profile_commands": profile_commands,
            "fit_commands": fit_commands,
            "prediction_export_commands": prediction_exports,
            "causal_stream_export_commands": streaming_exports,
            "grade_command": None,
            "grade_specification": {
                "owner": "root supplies and independently runs the grader",
                "metric_id": "global_wp_clip2_v1",
                "scored_members": "the exact evaluation members in the split manifest",
                "expected_scored_group_count": 468,
                "prediction_row_order": "the exact sorted evaluation_groups in the E540 subset manifest",
                "prediction_output_scope": "only those frozen evaluation groups; feature-only parquet reads",
                "paired_unit": "evaluation component",
                "report": [
                    "candidate and control scores",
                    "paired delta",
                    "component bootstrap",
                    "validity",
                    "runtime and memory",
                ],
            },
        },
        "profile_budget": {
            "provisional_batch": 2,
            "provisional_chunk_rows": 256,
            "provisional_hidden": 128,
            "profile_warmup_batches": 1,
            "profile_steps": 2,
            "pair_fit_target_gpu_minutes": 20,
            "target_is_unmeasured": True,
            "root_must_review_live_quota_headroom_and_both_arm_profiles": True,
        },
        "stages": [
            "profile",
            "fit",
            "prediction_export",
            "causal_stream_export",
            "independent_grade",
        ],
        "scope": "No profile, fit, provider mutation, grading, or public score is performed by this builder.",
    }
    output.mkdir(parents=True, exist_ok=True)
    subset_path.write_text(
        json.dumps(subset, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    packet_path = output / "E540_PACKET_V1.json"
    packet_path.write_text(
        json.dumps(packet, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    packet = build_packet(args.split_manifest, args.output_dir)
    print(
        json.dumps(
            {
                "status": packet["status"],
                "fit_admitted": packet["fit_admitted"],
                "split_sha256": packet["inputs"]["split_manifest_sha256"],
                "source_file_count": len(packet["generated_from"]["source_files_sha256"]),
                "packet": str(args.output_dir / "E540_PACKET_V1.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
