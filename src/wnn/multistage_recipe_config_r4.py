"""Write R4 family-search configs extending the frozen R3 controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPERIMENT_ID = "E-MULTISTAGE-OPTUNA-B300-20261031-R1"
RECIPE_SCHEMA = "wnn.multistage-recipe-family-extension.v1"
AUXILIARY_MANIFEST_SHA256 = "26b448bad7f60612cea482f4abaf5f184e5a724228eef227d7e78a8155190075"
AUXILIARY_FIT_SEQUENCES = 8260
AUXILIARY_CONTROL_SEQUENCES = 2065
L4_GALAM_PRESET = Path(
    "research/experiment_drafts/galam-l4-quality8-20261028/launch/configs/g1-galam-lstm64.json"
)
L4_GALAM_PRESET_SHA256 = "698a654256177000b0e575a45da6ffb94066753cfbffc812ed429eeb1a9f9b5b"
R3_RECIPE_SHA256 = "d6c5c98b84df0735a56179d4709a2fbd27fef42d8b6706e87937822ad0686d40"
R4_CONFIG_DIRECTORY = "recipes/configs/r4"
_COMMON_MODEL = {
    "heads": 8,
    "ff_multiplier": 4,
    "context_rows": 256,
    "dropout": 0.0,
    "checkpoint_blocks": True,
    "hint_dim": 0,
    "hint_projection": "learned",
    "attention_backend": "math",
    "compile_blocks": False,
    "input_feature_mode": "raw112",
}


def _base_config(architecture: str) -> dict[str, object]:
    """Create a source-bound strict-FP32 control for a canonical family."""
    return {
        "schema_version": 1,
        "epochs": 4,
        "batch_size": 16,
        "accumulate_chunks": 4,
        "chunk_rows": 256,
        "seed": 1301,
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
        "dropout_policy": "explicit_regularization_v1",
        "monitoring": {
            "train_eval_sequences": 16,
            "diagnostics_every_steps": 50,
            "train_eval_mask": "all_after_warmup",
            "eval_batch_size": 16,
            "training_telemetry_every_steps": 1,
        },
        "model": _COMMON_MODEL | {"architecture": architecture, "hidden": 256, "layers": 2},
    }


def _galam_config(preset_path: Path) -> tuple[dict[str, object], str]:
    """Adapt the known L4 GALAM preset to the fitting-only sequence runtime."""
    source = preset_path.resolve(strict=True)
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != L4_GALAM_PRESET_SHA256:
        raise ValueError("known L4 GALAM preset bytes changed")
    preset = json.loads(source_bytes)
    if not isinstance(preset, dict) or not isinstance(preset.get("model"), dict):
        raise ValueError("known L4 GALAM preset has an unexpected shape")
    model = preset["model"]
    if model.get("architecture") != "galam" or not isinstance(model.get("galam"), dict):
        raise ValueError("known L4 preset is not the schematic GALAM LSTM family")
    config = _base_config("galam")
    config.update(
        {
            "epochs": 4,
            "batch_size": 32,
            "accumulate_chunks": 1,
            "chunk_rows": 256,
            "lr": 0.0001,
            "schedule": "warmup_cosine_v2",
            "monitoring": {
                "train_eval_sequences": 16,
                "diagnostics_every_steps": 50,
                "train_eval_mask": "all_after_warmup",
                "eval_batch_size": 32,
                "training_telemetry_every_steps": 1,
            },
            "model": _galam_runtime_model(model["galam"]),
        }
    )
    config.pop("dropout_policy")
    return config, source_sha256


def _galam_runtime_model(nested: object) -> dict[str, object]:
    """Keep raw JSON aliases present for exact search-path updates and adapter checks."""
    if not isinstance(nested, dict):
        raise TypeError("GALAM preset must contain a nested configuration object")
    width, history = nested["width"], nested["attention_history"]
    global_blocks, ff_width = nested["global_blocks"], nested["feed_forward_width"]
    heads = nested["attention_heads"]
    if any(type(value) is not int for value in (width, history, global_blocks, ff_width, heads)):
        raise TypeError("GALAM preset aliases require integer shape fields")
    return {
        "architecture": "galam",
        "galam": nested,
        "hidden": width,
        "layers": global_blocks,
        "heads": heads,
        "ff_multiplier": ff_width / width,
        "context_rows": history,
        "dropout": 0.0,
        "checkpoint_blocks": False,
        "hint_dim": 0,
        "hint_projection": "learned",
        "attention_backend": "math",
        "compile_blocks": False,
        "input_feature_mode": "raw112",
        "precision_mode": "fp32",
    }


def _transformer_config() -> dict[str, object]:
    """Define a bounded causal Transformer engineering anchor and search base."""
    config = _base_config("transformer")
    model = _COMMON_MODEL | {
        "architecture": "transformer",
        "hidden": 384,
        "layers": 4,
        "heads": 8,
        "ff_multiplier": 4,
        "context_rows": 256,
        "dropout": 0.0,
        "checkpoint_blocks": True,
        "hint_dim": 0,
        "hint_projection": "learned",
        "attention_backend": "math",
        "compile_blocks": False,
        "input_feature_mode": "raw112",
    }
    config["model"] = model
    config["batch_size"] = 16
    config["lr"] = 0.0003
    return config


def config_documents(preset_path: Path = L4_GALAM_PRESET) -> dict[str, dict[str, object]]:
    """Return the exact R4 family configs as packet-relative JSON documents."""
    galam, _ = _galam_config(preset_path)
    return {
        f"{R4_CONFIG_DIRECTORY}/gru-base-r4.json": _base_config("gru"),
        f"{R4_CONFIG_DIRECTORY}/lstm-base-r4.json": _base_config("lstm"),
        f"{R4_CONFIG_DIRECTORY}/galam-g1-adapt-r4.json": galam,
        f"{R4_CONFIG_DIRECTORY}/causal-transformer-base-r4.json": _transformer_config(),
    }


def _encoded_json(value: object) -> bytes:
    """Serialize an immutable JSON document with the packet's standard formatting."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _config_ref(family: str, documents: dict[str, dict[str, object]]) -> dict[str, str]:
    """Bind one exact family config path and byte hash."""
    relative = f"{R4_CONFIG_DIRECTORY}/{family}-base-r4.json"
    if family == "galam":
        relative = f"{R4_CONFIG_DIRECTORY}/galam-g1-adapt-r4.json"
    if family == "causal_transformer":
        relative = f"{R4_CONFIG_DIRECTORY}/causal-transformer-base-r4.json"
    return {
        "path": relative,
        "sha256": hashlib.sha256(_encoded_json(documents[relative])).hexdigest(),
    }


def _categorical(choices: list[object], path: str) -> dict[str, object]:
    """Declare a bounded categorical choice at one existing config leaf."""
    return {"kind": "categorical", "choices": choices, "config_path": path}


def _fanout(choices: list[object], paths: list[str]) -> dict[str, object]:
    """Declare one bounded value applied consistently to derived aliases."""
    return {"kind": "categorical", "choices": choices, "config_paths": paths}


def _search_spaces() -> dict[str, dict[str, object]]:
    """Return inspected categorical spaces for the four canonical architectures."""
    recurrent: dict[str, object] = {
        "hidden": _categorical([128, 256, 384], "model.hidden"),
        "layers": _categorical([2, 3], "model.layers"),
        "lr": _categorical([0.0003, 0.001, 0.002], "lr"),
        "dropout": _categorical([0.0, 0.1], "model.dropout"),
        "epochs": _categorical([2, 4, 6], "epochs"),
        "batch_size": _categorical([16, 32, 64], "batch_size"),
    }
    galam: dict[str, object] = {
        "global_blocks": _fanout([1, 2], ["model.galam.global_blocks", "model.layers"]),
        "lstm_layers": _categorical([1, 2], "model.galam.lstm_layers"),
        "local_attention_blocks": _categorical([1, 2], "model.galam.local_attention_blocks"),
        "attention_history": _fanout(
            [64, 128, 256], ["model.galam.attention_history", "model.context_rows"]
        ),
        "lr": _categorical([0.00005, 0.0001, 0.0003], "lr"),
        "epochs": _categorical([2, 4, 6], "epochs"),
        "batch_size": _categorical([16, 32, 64], "batch_size"),
    }
    transformer: dict[str, object] = {
        "hidden": _categorical([384, 512], "model.hidden"),
        "layers": _categorical([3, 4, 6], "model.layers"),
        "context_rows": _categorical([128, 256, 512], "model.context_rows"),
        "lr": _categorical([0.0001, 0.0003, 0.0005], "lr"),
        "dropout": _categorical([0.0, 0.1], "model.dropout"),
        "epochs": _categorical([2, 4, 6], "epochs"),
        "batch_size": _categorical([8, 16, 32], "batch_size"),
        "chunk_rows": _categorical([128, 256], "chunk_rows"),
    }
    return {
        "gru": recurrent,
        "lstm": json.loads(json.dumps(recurrent)),
        "galam": galam,
        "causal_transformer": transformer,
    }
