from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from wnn.data import N_FEATURES
from wnn.features import N_EVENT_FEATURES
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.environment import TrainingEnvironment
from wnn.training.masked_group_pretraining import (
    BASE_INPUT_WIDTH,
    pretraining_source_paths,
    transferred_backbone_state,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the conversion command for one paired supervised cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def task_model_arguments(hidden: int) -> dict[str, Any]:
    """Return the matched plain-GRU task model configuration."""
    return {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": hidden,
        "encoder_width": hidden,
        "recurrent_width": hidden,
        "readout_width": hidden,
        "head_layers": 1,
        "layers": 1,
        "dropout": 0.0,
        "readout_activation": "gelu",
        "numeric_map": "none",
        "research_features": "none",
        "research_feature_control": "none",
        "raw_only": False,
        "online_norm": False,
        "heuristics": False,
        "train_source": "train+valid",
        "no_eval_during_fit": True,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 0,
    }


def make_checkpoint(
    model_state: dict[str, torch.Tensor],
    args: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Wrap a model state in the canonical fixed-checkpoint layout."""
    return {
        "model": {name: value.detach().cpu() for name, value in model_state.items()},
        "args": args,
        "n_in": BASE_INPUT_WIDTH,
        "epoch": 0,
        "readout": None,
        "readout_tilted": False,
        "provenance": provenance,
    }


def prepare_pair(args: argparse.Namespace) -> None:
    """Write cold and denoising-initialized checkpoints with shared normalization/head."""
    root = args.root.resolve()
    environment = TrainingEnvironment(root=root)
    split = load_campaign_split(args.split_manifest, root)
    pretrained_path = args.pretrained.resolve()
    pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    recipe = pretrained.get("recipe", {})
    expected_model = {
        "family": "plain_gru",
        "hidden": args.hidden,
        "layers": 1,
        "input_width": BASE_INPUT_WIDTH + N_FEATURES,
        "raw_group_count": 13,
        "mask_block_rows": 32,
    }
    expected_optimization = {
        "precision": "fp32",
        "epochs": 2,
        "batch_size": 4,
        "chunk_rows": 2000,
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "optimizer_betas": [0.9, 0.999],
        "max_grad_norm": 1.0,
        "scheduler": "OneCycleLR_cosine_pct_start_0.05",
        "scheduler_div_factor": 10.0,
        "scheduler_final_div_factor": 100.0,
    }
    if recipe.get("schema") != "causal_group_mask_pretraining_v1":
        raise ValueError("pretraining checkpoint recipe schema is unsupported")
    if recipe.get("split_manifest_sha256") != split.sha256:
        raise ValueError("pretraining checkpoint belongs to a different source-qualified fold")
    if recipe.get("fit_members") != {
        "train": list(split.train_groups),
        "valid": list(split.valid_groups),
    } or recipe.get("evaluation_members") != list(split.evaluation_groups):
        raise ValueError(
            "pretraining checkpoint membership differs from this source-qualified fold"
        )
    current_sources = {
        name: file_sha256(path)
        for name, path in pretraining_source_paths(environment.source_root).items()
    }
    if recipe.get("source_sha256") != current_sources:
        raise ValueError("pretraining checkpoint was made with different source code")
    stats_path = args.stats.resolve()
    if not stats_path.is_file() or recipe.get("stats_sha256") != file_sha256(stats_path):
        raise ValueError("pretraining checkpoint uses different or missing statistics")
    stats_metadata_path = stats_path.with_suffix(stats_path.suffix + ".json")
    if not stats_metadata_path.is_file() or recipe.get("stats_metadata_sha256") != file_sha256(
        stats_metadata_path
    ):
        raise ValueError("pretraining checkpoint uses different or missing statistics metadata")
    if recipe.get("seed") != args.seed or recipe.get("model", {}).get("hidden") != args.hidden:
        raise ValueError("pretraining checkpoint seed or hidden width differs from this cell")
    if recipe.get("model") != expected_model or recipe.get("optimization") != expected_optimization:
        raise ValueError(
            "pretraining checkpoint does not match the frozen model and optimizer recipe"
        )
    if not recipe.get("feature_only"):
        raise ValueError("pretraining checkpoint is not declared feature-only")
    pretrained_state = pretrained["model"]
    if pretrained_state["inp.weight"].shape[1] != BASE_INPUT_WIDTH + N_FEATURES:
        raise ValueError("pretraining checkpoint has an unexpected mask-indicator width")
    if BASE_INPUT_WIDTH != N_FEATURES + N_EVENT_FEATURES:
        raise ValueError("canonical base input width changed")
    torch.manual_seed(args.seed)
    model_args = task_model_arguments(args.hidden)
    cold_model = environment.make_model(model_args, BASE_INPUT_WIDTH, 2)
    cold_state = cold_model.state_dict()
    cold_state["mean"].copy_(pretrained_state["mean"][:BASE_INPUT_WIDTH])
    cold_state["std"].copy_(pretrained_state["std"][:BASE_INPUT_WIDTH])
    candidate_state = transferred_backbone_state(pretrained_state, cold_state)
    run_root = root / "artifacts" / "runs"
    control_name = f"{args.output_prefix}_control_init"
    candidate_name = f"{args.output_prefix}_masked_init"
    manifest_path = (
        root / "artifacts" / "masked_group_pretraining" / f"{args.output_prefix}_pair.json"
    )
    pretrained_sha256 = file_sha256(pretrained_path)
    converter_sha256 = file_sha256(Path(__file__).resolve())
    if args.verify_only:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"initializer pair manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "fold": split.fold,
            "seed": args.seed,
            "split_manifest_sha256": split.sha256,
            "pretraining_checkpoint_sha256": pretrained_sha256,
            "statistics_sha256": file_sha256(stats_path),
            "statistics_metadata_sha256": file_sha256(stats_metadata_path),
            "initializer_source_sha256": converter_sha256,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("initializer pair manifest differs from the current recipe")
        for run_name, key in (
            (manifest.get("control_run"), "control_sha256"),
            (manifest.get("candidate_run"), "candidate_sha256"),
        ):
            checkpoint_path = run_root / str(run_name) / "model.pt"
            if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != manifest.get(key):
                raise ValueError(f"initializer checkpoint hash mismatch: {checkpoint_path}")
        print(json.dumps(manifest, indent=2))
        return
    fit_members = {
        "train": list(split.train_groups),
        "valid": list(split.valid_groups),
    }
    provenance_base = {
        "split_manifest_sha256": split.sha256,
        "fit_row_groups": fit_members,
        "evaluation_row_groups": list(split.evaluation_groups),
        "evaluation_excluded": True,
        "feature_only_pretraining": True,
        "pretraining_checkpoint_sha256": pretrained_sha256,
    }
    checkpoint_args = {
        **model_args,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": split.sha256,
        "seed": args.seed,
        "initializer_schema": "causal_group_mask_v1",
    }
    control_provenance = {
        **provenance_base,
        "initializer": "cold model with the same fitting-only normalization and task-head seed",
    }
    candidate_provenance = {
        **provenance_base,
        "initializer": "pretrained raw/event input projection and GRU; cold task readout retained",
        "pretraining_source_sha256": recipe["source_sha256"],
    }
    for run_name, state, provenance in (
        (control_name, cold_state, control_provenance),
        (candidate_name, candidate_state, candidate_provenance),
    ):
        run_dir = run_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = make_checkpoint(state, checkpoint_args, provenance)
        path = run_dir / "model.pt"
        temporary = path.with_suffix(f".tmp-{args.seed}")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    manifest = {
        "schema": "causal_group_mask_initializer_pair_v1",
        "fold": split.fold,
        "seed": args.seed,
        "split_manifest_sha256": split.sha256,
        "pretraining_checkpoint": str(pretrained_path),
        "pretraining_checkpoint_sha256": pretrained_sha256,
        "statistics_sha256": file_sha256(stats_path),
        "statistics_metadata_sha256": file_sha256(stats_metadata_path),
        "initializer_source_sha256": converter_sha256,
        "control_run": control_name,
        "control_sha256": file_sha256(run_root / control_name / "model.pt"),
        "candidate_run": candidate_name,
        "candidate_sha256": file_sha256(run_root / candidate_name / "model.pt"),
        "matched_normalization": True,
        "matched_task_head": True,
        "transferred_components": ["raw/event input projection", "GRU"],
        "discarded_components": ["mask-indicator input columns", "reconstruction readout"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    """Prepare one source-qualified control/candidate initializer pair."""
    args = build_parser().parse_args()
    if args.hidden <= 0:
        raise SystemExit("--hidden must be positive")
    prepare_pair(args)


if __name__ == "__main__":
    main()
