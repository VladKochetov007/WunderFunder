"""Write frozen GRU/LSTM batch-ladder anchors for the fitting-only search."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "E-MULTISTAGE-OPTUNA-B300-20261031-R3"
RECIPE_SCHEMA = "wnn.multistage-recipe-nodes.v1"
ANCHOR_BATCH_SIZES = (16, 32, 64)
_MODEL_DEFAULTS = {
    "heads": 12,
    "ff_multiplier": 4,
    "context_rows": 512,
    "dropout": 0.0,
    "checkpoint_blocks": True,
    "hint_dim": 0,
    "hint_projection": "learned",
    "attention_backend": "math",
    "compile_blocks": False,
    "input_feature_mode": "raw112",
}


def _sequence_config(architecture: str, batch_size: int, seed: int) -> dict[str, object]:
    """Build one explicit strict-FP32, one-GPU sequence recipe."""
    return {
        "schema_version": 1,
        "epochs": 4,
        "batch_size": batch_size,
        "accumulate_chunks": 4,
        "chunk_rows": 256,
        "seed": seed,
        "lr": 0.001,
        "weight_decay": 0.01,
        "optimizer": "adamw",
        "muon_momentum": 0.95,
        "muon_ns_steps": 5,
        "muon_ns_eps": 1e-7,
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.999,
        "adamw_eps": 1e-8,
        "grad_clip": 1.0,
        "warmup_fraction": 0.05,
        "schedule": "legacy_warmup_cosine",
        "min_lr_ratio": 0.0,
        "kd_weight": 0.0,
        "hint_weight": 0.0,
        "hint_mode": "aligned",
        "normalization_sequences": 32,
        "training_mask": "all_after_warmup",
        "normalization_scope": "prefix",
        "required_world_size": 1,
        "required_device": "B300",
        "expected_fold": 0,
        "precision_mode": "fp32",
        "checkpoint_retention": "all_epochs",
        "monitoring": {
            "train_eval_sequences": 16,
            "diagnostics_every_steps": 50,
            "train_eval_mask": "all_after_warmup",
            "eval_batch_size": 16,
            "training_telemetry_every_steps": 1,
        },
        "model": {
            **_MODEL_DEFAULTS,
            "architecture": architecture,
            "hidden": 256,
            "layers": 2,
        },
    }


def _node(architecture: str, batch_size: int, seed: int = 1301) -> dict[str, Any]:
    """Create one frozen batch-ladder anchor with source-qualified scoring."""
    family = "lstm" if architecture == "lstm" else "gru"
    label = "LSTM256x2" if family == "lstm" else "GRU256x2"
    config_name = f"{family}-b{batch_size}-r3.json"
    cli = [
        "--arch",
        "gru",
        "--cell",
        "lstm" if family == "lstm" else "gru",
        "--hidden",
        "256",
        "--layers",
        "2",
        "--dropout",
        "0",
        "--epochs",
        "4",
        "--lr",
        "0.001",
        "--wd",
        "0.01",
        "--batch-size",
        str(batch_size),
        "--chunk",
        "256",
        "--precision",
        "fp32",
    ]
    return {
        "id": f"S1-{label}-B{batch_size}-AUX-CONTROL-R3",
        "family": family,
        "stage": 1,
        "args": {
            "trainer": "scripts/train_sequence.py",
            "config_path": f"recipes/configs/{config_name}",
            "config_sha256": _config_reference(family, batch_size)["sha256"],
            "legacy_cli_compatibility": cli,
            "sequence_config": _sequence_config(architecture, batch_size, seed),
        },
        "parents": [],
        "fit_split": "multistage_auxiliary_fit",
        "eval_metric": {
            "name": "wp",
            "scope": "inner_train_proxy",
            "source": "train",
            "role": "control",
            "row_mask": "all_after_warmup AND row_index>=99",
            "is_scored_used": False,
            "reduction": "mean_of_two_pooled_clipped_weighted_Pearson_targets",
        },
        "checkpoint_roles": {
            "fixed_final": "model.pt at the configured final epoch",
            "eligible_control_selected": (
                "epoch snapshot selected by control WP only when the recipe freezes this rule"
            ),
            "best_observed": "maximum control WP; diagnostic and exposed development evidence",
            "epoch_index": "epoch_index.json binds each retained epoch checkpoint SHA256",
        },
        "recipe_status": "runnable after one-GPU exact-identity B300 profile admission",
        "model_identity_note": (
            "Uses the canonical RNNModel GRU/LSTM cell through the sequence trainer; it is a "
            "new sequence-runtime recipe, not an exact replay of legacy batch training."
        ),
    }


def config_documents() -> dict[str, dict[str, object]]:
    """Return exact per-anchor configs keyed by their immutable relative paths."""
    return {
        f"recipes/configs/{family}-b{batch_size}-r3.json": _sequence_config(
            family, batch_size, 1301
        )
        for family in ("gru", "lstm")
        for batch_size in ANCHOR_BATCH_SIZES
    }


def _config_reference(family: str, batch_size: int) -> dict[str, str]:
    """Return one exact relative config path with its frozen byte hash."""
    relative = f"recipes/configs/{family}-b{batch_size}-r3.json"
    encoded = (
        json.dumps(config_documents()[relative], indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    return {"path": relative, "sha256": hashlib.sha256(encoded).hexdigest()}


def recipe_document() -> dict[str, object]:
    """Return the immutable GRU/LSTM batch ladder and its bounded search spaces."""
    return {
        "schema": RECIPE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "recipe_revision": "R3",
        "budget": {
            "max_fit_attempts_including_auxiliary_ancestors_and_final_refits": 48,
            "max_node_wall_seconds_from_first_gpu_profile": 21600,
            "max_individual_fit_wall_seconds": 7200,
            "profile_wall_seconds_per_attempt": 180,
        },
        "data_protocol": {
            "primary_fit": "10325 official-train sequences; zero official-valid fit sequences",
            "primary_evaluation": "all 1873 official-valid sequences after recipe freeze",
            "inner_fit": "primary official-train fit members minus whole heldout components",
            "inner_evaluation": "whole known components held out from official train",
            "chronology_claim_for_inner": False,
            "normalizers_and_teachers": "fit-members only; never include auxiliary heldout members",
        },
        "profile_ladder": {
            "initial_profile_attempts": 6,
            "profile_updates": 50,
            "profile_wall_seconds_per_attempt": 180,
            "worker_world_size": 1,
            "batch_sizes_per_family": list(ANCHOR_BATCH_SIZES),
        },
        "stage1": {
            "base_config": _config_reference("gru", 16),
            "families": {
                family: {
                    "base_config": _config_reference(family, 16),
                    "config_path": "batch_size",
                    "fixed_overrides": {},
                    "search_space": {
                        "batch_size": {
                            "type": "categorical",
                            "choices": list(ANCHOR_BATCH_SIZES),
                        }
                    },
                }
                for family in ("gru", "lstm")
            },
        },
        "nodes": [
            _node(family, batch_size)
            for family in ("gru", "lstm")
            for batch_size in ANCHOR_BATCH_SIZES
        ],
    }


def write_recipe_document(path: Path) -> None:
    """Create a new frozen JSON file, accepting an identical existing file only."""
    encoded = (
        json.dumps(recipe_document(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise FileExistsError(
                "recipe output exists and differs; refusing to replace it"
            ) from None


def write_config_documents(directory: Path) -> dict[str, str]:
    """Create each exact training config once and return relative paths and hashes."""
    root = directory.resolve()
    hashes: dict[str, str] = {}
    for relative, config in config_documents().items():
        destination = root / Path(relative).name
        encoded = (json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("xb") as stream:
                stream.write(encoded)
        except FileExistsError:
            if destination.read_bytes() != encoded:
                raise FileExistsError(
                    f"config exists and differs; refusing to replace it: {destination}"
                ) from None
        hashes[relative] = hashlib.sha256(encoded).hexdigest()
    return hashes


def main() -> None:
    """Write the frozen control recipes to the requested JSON path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    args = parser.parse_args()
    write_recipe_document(args.output)
    hashes = write_config_documents(args.config_dir)
    sys.stdout.write(
        json.dumps(
            {
                "recipe": str(args.output.resolve()),
                "config_dir": str(args.config_dir.resolve()),
                "config_sha256": hashes,
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
