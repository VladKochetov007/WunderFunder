"""Write R4 family-search recipes extending the frozen R3 controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from wnn.multistage_recipe_config_r4 import (
    AUXILIARY_CONTROL_SEQUENCES,
    AUXILIARY_FIT_SEQUENCES,
    AUXILIARY_MANIFEST_SHA256,
    EXPERIMENT_ID,
    L4_GALAM_PRESET,
    R3_RECIPE_SHA256,
    RECIPE_SCHEMA,
    _config_ref,
    _encoded_json,
    _galam_config,
    _search_spaces,
    config_documents,
)


def _metric() -> dict[str, object]:
    """Describe the whole-cohort fitting-only objective and its nonofficial mask."""
    return {
        "name": "wp",
        "scope": "inner_train_proxy",
        "source": "train",
        "role": "control",
        "row_mask": "all_after_warmup AND row_index>=99",
        "is_scored_used": False,
        "reduction": "mean_of_two_pooled_clipped_weighted_Pearson_targets",
    }


def _checkpoint_roles() -> dict[str, str]:
    """Keep fixed, control-selected, and best-observed moments distinct."""
    return {
        "fixed_final": "model.pt at the configured final epoch",
        "eligible_control_selected": (
            "epoch snapshot selected by control WP only when the recipe freezes this rule"
        ),
        "best_observed": "maximum control WP; diagnostic and exposed development evidence",
        "epoch_index": "epoch_index.json binds each retained epoch checkpoint SHA256",
    }


def _node(
    family: str,
    config_ref: dict[str, str],
    config: dict[str, object],
    identity_note: str,
) -> dict[str, object]:
    """Create one runnable architecture anchor with exact config and proxy scope."""
    relative = config_ref["path"]
    return {
        "id": f"S1-{family.upper()}-R4-AUX-CONTROL",
        "family": family,
        "stage": 1,
        "args": {
            "trainer": "scripts/train_sequence.py",
            "config_path": relative,
            "config_sha256": config_ref["sha256"],
            "sequence_config": config,
        },
        "parents": [],
        "fit_split": "multistage_auxiliary_fit",
        "eval_metric": _metric(),
        "checkpoint_roles": _checkpoint_roles(),
        "recipe_status": "runnable after exact-identity one-GPU B300 profile admission",
        "model_identity_note": identity_note,
    }


def recipe_document(preset_path: Path = L4_GALAM_PRESET) -> dict[str, object]:
    """Return R4 family searches without modifying or superseding frozen R3 controls."""
    documents = config_documents(preset_path)
    galam, source_sha = _galam_config(preset_path)
    gru_ref = _config_ref("gru", documents)
    galam_ref = _config_ref("galam", documents)
    transformer_ref = _config_ref("causal_transformer", documents)
    spaces = _search_spaces()
    base_refs = {
        family: _config_ref(family, documents)
        for family in ("gru", "lstm", "galam", "causal_transformer")
    }
    families: dict[str, object] = {
        family: {
            "base_config": base_refs[family],
            "search_space": spaces[family],
            "fixed_overrides": {},
        }
        for family in spaces
    }
    return {
        "schema": RECIPE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "recipe_revision": "R4",
        "extends_recipe": {
            "revision": "R3",
            "sha256": R3_RECIPE_SHA256,
            "policy": "preserve R3 controls and anchor files unchanged",
        },
        "data_binding": {
            "auxiliary_manifest_sha256": AUXILIARY_MANIFEST_SHA256,
            "primary_fit_sequences": 10325,
            "auxiliary_fit_sequences": AUXILIARY_FIT_SEQUENCES,
            "auxiliary_control_sequences": AUXILIARY_CONTROL_SEQUENCES,
            "chronology_claim": False,
            "galam_fit_batch_remainders": {"16": 4, "32": 4, "64": 4},
        },
        "model_sources": {
            "galam_preset": {
                "path": str(L4_GALAM_PRESET).replace("\\", "/"),
                "sha256": source_sha,
                "adaptation": (
                    "L4 schematic_lstm_full shape/fidelity fields reused through the canonical "
                    "GALAM sequence adapter; auxiliary fit-only normalization, ordinal sequence "
                    "timestamps, batch32/one-GPU training, four-epoch anchor. Not an exact replay."
                ),
            },
            "transformer": {
                "adapter": "canonical causal Transformer sequence model",
                "attention_backend": "math",
                "precision": "strict fp32",
            },
        },
        "profile_ladder": {
            "additional_initial_profile_attempts": 2,
            "profile_updates": 50,
            "profile_wall_seconds": 180,
            "worker_world_size": 1,
            "profile_families": ["galam", "causal_transformer"],
        },
        "stage1": {
            "base_config": gru_ref,
            "families": families,
            "anchors": [
                {
                    "id": "S1-GALAM-G1-LSTM64-R4-AUX-CONTROL",
                    "family": "galam",
                    "params": {
                        "global_blocks": 1,
                        "lstm_layers": 1,
                        "local_attention_blocks": 1,
                        "attention_history": 64,
                        "lr": 0.0001,
                        "epochs": 4,
                        "batch_size": 32,
                    },
                    "config": galam_ref,
                },
                {
                    "id": "S1-CAUSAL-TRANSFORMER-H384-L4-C256-R4-AUX-CONTROL",
                    "family": "causal_transformer",
                    "params": {
                        "hidden": 384,
                        "layers": 4,
                        "context_rows": 256,
                        "lr": 0.0003,
                        "dropout": 0.0,
                        "epochs": 4,
                        "batch_size": 16,
                        "chunk_rows": 256,
                    },
                    "config": transformer_ref,
                },
            ],
        },
        "nodes": [
            _node(
                "galam",
                galam_ref,
                galam,
                "Uses the existing L4 schematic_lstm_full preset through the canonical GALAM "
                "sequence adapter, with strict FP32 and fit-only normalization; assumed details "
                "remain explicit and this is not an exact historical replay.",
            ),
            _node(
                "causal_transformer",
                transformer_ref,
                documents[transformer_ref["path"]],
                "New causal Transformer candidate through the canonical sequence trainer, "
                "strict-FP32 math attention and bounded rolling context.",
            ),
        ],
    }


def write_artifacts(output_root: Path, preset_path: Path = L4_GALAM_PRESET) -> dict[str, object]:
    """Write immutable R4 configs and family extension, refusing byte changes."""
    output_root = output_root.resolve()
    documents = config_documents(preset_path)
    hashes: dict[str, str] = {}
    for relative, config in documents.items():
        target = output_root / Path(relative)
        encoded = _encoded_json(config)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as stream:
                stream.write(encoded)
        except FileExistsError:
            if target.read_bytes() != encoded:
                raise FileExistsError(f"refusing to replace changed R4 config: {target}") from None
        hashes[relative] = hashlib.sha256(encoded).hexdigest()
    recipe_path = output_root / "recipes" / "stage1-family-extension-r4.json"
    encoded_recipe = _encoded_json(recipe_document(preset_path))
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with recipe_path.open("xb") as stream:
            stream.write(encoded_recipe)
    except FileExistsError:
        if recipe_path.read_bytes() != encoded_recipe:
            raise FileExistsError(f"refusing to replace changed R4 recipe: {recipe_path}") from None
    return {
        "recipe": str(recipe_path),
        "recipe_sha256": hashlib.sha256(encoded_recipe).hexdigest(),
        "config_sha256": hashes,
    }


def main() -> None:
    """Write R4 artifacts under an experiment packet directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--galam-preset", type=Path, default=L4_GALAM_PRESET)
    args = parser.parse_args()
    receipt = write_artifacts(args.packet_root, args.galam_preset)
    sys.stdout.write(json.dumps(receipt, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
