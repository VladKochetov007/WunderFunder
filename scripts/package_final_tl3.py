"""Package the fixed TL3 + Source4 LSTM callback with source-bound receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from wnn.final_submit_manifest import (
    FINAL_FIT_POLICY,
    FINAL_FIT_SCHEMA,
    TOTAL_SEQUENCES,
    TRAIN_SEQUENCES,
    VALID_SEQUENCES,
    FinalFitMembership,
)
from wnn.training.final_submit_bridge import (
    SOURCE_VALIDATION_SHA256,
    validate_final_fit_artifact,
    validate_final_fit_provenance,
)
from wnn.training.sequence_final_fit import validate_final_lstm_recipe

ROOT = Path(__file__).resolve().parents[1]
MAX_ZIP_BYTES = 20_000_000
ROLES = ("w18", "tiedlead_s5", "tiedlead_s6", "source4_lstm")
WIDTHS = (211, 263, 263, 112)
RUNS = (
    "tl3c_w18final_distilled_s5",
    "tl3c_fttall_tlead_s5",
    "tl3c_fttall_tlead_s6",
    "primary-refit-03-s1t000000-lstm",
)
FINAL_RUNS = (
    "tl3final_w18final_distilled_s5",
    "tl3final_fttall_tlead_s5",
    "tl3final_fttall_tlead_s6",
    "final-lstm192x1",
)
FINAL_RECEIPT_SCHEMA = "wnn.final-model-set-receipt.v1"
FINAL_FIXED_EPOCHS = {
    "w18": {"configured_epochs": 2, "checkpoint_epoch": 1},
    "tiedlead_s5": {"configured_epochs": 4, "checkpoint_epoch": 3},
    "tiedlead_s6": {"configured_epochs": 4, "checkpoint_epoch": 3},
    "source4_lstm": {
        "configured_epochs": 8,
        "completed_epochs": 8,
        "checkpoint_epoch": 7,
    },
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_FINAL_MODEL_RECEIPT_BYTES = 1_000_000
EXPECTED_CHECKPOINT_SHA256 = (
    "a66488e99ed49373722c206df0fa8fd97c179246a72369b69732bb32029baae6",
    "f447271a524fde75779f171c1e796cf9827fda6bad8e3525f05def466393a13c",
    "ef654318599b835c4b69426b56680335d11605f0f0a82557d52bccfcc3a5ed41",
    "842c6d5602c32052045d7567fe09365aa358eab7cc5251d7cfb8744b25c61c90",
)
BLEND = {
    "schema": "wnn.nested-fp64-fp32.v1",
    "tl3_weights": [0.34, 0.33, 0.33],
    "outer_weights": [0.9, 0.1],
    "tl3_cast": "float32",
    "outer_cast": "float32",
    "clip": "official-global-clip-after-callback",
}


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash metadata as sorted compact UTF-8 JSON with NaN and Infinity forbidden."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _final_membership(receipt: Mapping[str, object], receipt_path: Path) -> FinalFitMembership:
    """Rebuild the canonical all-labeled fit identity from a trusted final receipt."""
    fit = receipt.get("final_fit")
    if not isinstance(fit, Mapping):
        raise ValueError("final-model receipt must contain final_fit identities")
    sources = fit.get("source_sha256")
    if not isinstance(sources, Mapping):
        raise ValueError("final-model receipt final_fit source hashes are malformed")
    train_sha, valid_sha = sources.get("train"), sources.get("valid")
    manifest_sha = fit.get("manifest_sha256")
    members_sha = fit.get("fit_members_sha256")
    source_validation_sha = fit.get("source_validation_sha256")
    hashes = (train_sha, valid_sha, manifest_sha, members_sha, source_validation_sha)
    if any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in hashes):
        raise ValueError("final-model receipt contains a malformed final-fit SHA256")
    if source_validation_sha != SOURCE_VALIDATION_SHA256:
        raise ValueError("final-model receipt does not use the pinned source-validation receipt")
    groups = fit.get("fit_row_groups")
    if (
        not isinstance(groups, Mapping)
        or set(groups) != {"train", "valid"}
        or any(
            not isinstance(groups.get(source), list)
            or any(type(group) is not int for group in groups[source])
            for source in ("train", "valid")
        )
    ):
        raise ValueError("final-model receipt fit row groups must be integer source ordinals")
    expected_final_fit = {
        "schema": FINAL_FIT_SCHEMA,
        "policy": FINAL_FIT_POLICY,
        "manifest_sha256": manifest_sha,
        "fit_members_sha256": members_sha,
        "source_sha256": {"train": train_sha, "valid": valid_sha},
        "source_validation_sha256": source_validation_sha,
        "fit_row_groups": {
            "train": list(range(TRAIN_SEQUENCES)),
            "valid": list(range(VALID_SEQUENCES)),
        },
        "independent_evaluation_members": [],
        "independent_score_claim": "forbidden",
        "post_fit_primary_valid_scoring": "forbidden",
    }
    if dict(fit) != expected_final_fit:
        raise ValueError("final-model receipt does not bind the exact complete-fit identity")
    encoded_members = json.dumps(
        [
            [source, group]
            for source, count in (("train", TRAIN_SEQUENCES), ("valid", VALID_SEQUENCES))
            for group in range(count)
        ],
        separators=(",", ":"),
    ).encode("ascii")
    computed_members_sha = hashlib.sha256(encoded_members).hexdigest()
    if computed_members_sha != members_sha:
        raise ValueError("final-model receipt fit_members_sha256 differs from all 12,480 members")
    return FinalFitMembership(
        path=receipt_path,
        manifest_sha256=manifest_sha,
        fit_members_sha256=members_sha,
        train_groups=tuple(range(TRAIN_SEQUENCES)),
        valid_groups=tuple(range(VALID_SEQUENCES)),
        train_sha256=train_sha,
        valid_sha256=valid_sha,
        source_validation_sha256=source_validation_sha,
    )


def validate_final_model_receipt(
    receipt: object, receipt_sha256: str, path: Path
) -> dict[str, object]:
    """Validate receipt policy, full membership, fixed endpoints, and all role identities."""
    if not isinstance(receipt, Mapping) or receipt.get("schema") != FINAL_RECEIPT_SCHEMA:
        raise ValueError("unsupported final-model receipt schema")
    expected_fields = {
        "schema",
        "final_fit",
        "fit_sequences",
        "independent_evaluation_members",
        "independent_evaluation_sequences",
        "independent_score_claim",
        "fixed_epochs",
        "members",
    }
    if set(receipt) != expected_fields:
        raise ValueError("final-model receipt contains missing or unsupported fields")
    if not isinstance(receipt_sha256, str) or _SHA256.fullmatch(receipt_sha256) is None:
        raise ValueError("final-model receipt SHA256 must be 64 lowercase hexadecimal characters")
    if (
        type(receipt.get("fit_sequences")) is not int
        or receipt.get("fit_sequences") != TOTAL_SEQUENCES
    ):
        raise ValueError("final-model receipt must bind exactly 12,480 fitting sequences")
    if (
        receipt.get("independent_evaluation_members") != []
        or type(receipt.get("independent_evaluation_sequences")) is not int
        or receipt.get("independent_evaluation_sequences") != 0
    ):
        raise ValueError("final-model receipt cannot contain independent evaluation members")
    if receipt.get("independent_score_claim") != "forbidden":
        raise ValueError("final-model receipt must forbid independent score claims")
    fixed_epochs = receipt.get("fixed_epochs")
    if (
        not isinstance(fixed_epochs, Mapping)
        or set(fixed_epochs) != set(FINAL_FIXED_EPOCHS)
        or any(
            not isinstance(fixed_epochs.get(role), Mapping)
            or set(fixed_epochs[role]) != set(endpoint)
            or any(type(fixed_epochs[role][name]) is not int for name in endpoint)
            or dict(fixed_epochs[role]) != endpoint
            for role, endpoint in FINAL_FIXED_EPOCHS.items()
        )
    ):
        raise ValueError(
            "final-model receipt fixed endpoint epochs differ from the approved recipe"
        )
    membership = _final_membership(receipt, path)
    members = receipt.get("members")
    if not isinstance(members, Mapping) or set(members) != set(ROLES):
        raise ValueError("final-model receipt must name exactly the four fixed model roles")
    for role, expected_run in zip(ROLES, FINAL_RUNS, strict=True):
        member = members[role]
        if not isinstance(member, Mapping) or member.get("run") != expected_run:
            raise ValueError(f"final-model receipt has an unsupported {role} run label")
        required = {"checkpoint_sha256", "configuration_sha256", "provenance_sha256"}
        if role == "source4_lstm":
            required.add("data_sha256")
        if set(member) != required | {"run"}:
            raise ValueError(
                f"final-model receipt {role} identity fields are incomplete or unknown"
            )
        if any(
            not isinstance(member.get(name), str) or _SHA256.fullmatch(member[name]) is None
            for name in required
        ):
            raise ValueError(f"final-model receipt {role} contains a malformed SHA256")
    return {"receipt": dict(receipt), "membership": membership, "sha256": receipt_sha256}


def load_final_model_receipt(path: Path, expected_sha256: str) -> dict[str, object]:
    """Load bounded non-symlink receipt bytes and verify their caller-pinned digest."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("final-model receipt must be a regular file, not a symlink")
    size = path.stat().st_size
    if size <= 0 or size > MAX_FINAL_MODEL_RECEIPT_BYTES:
        raise ValueError("final-model receipt exceeds its bounded read size")
    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if len(raw) != size or actual_sha256 != expected_sha256:
        raise ValueError("final-model receipt bytes differ from the supplied SHA256")
    value = json.loads(raw)
    return validate_final_model_receipt(value, actual_sha256, path.resolve())


def _validate_final_checkpoint(
    role: str,
    checkpoint: Mapping[str, object],
    member: Mapping[str, object],
    membership: FinalFitMembership,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Validate checkpoint-native fit lineage and receipt-bound config/provenance hashes."""
    config, model_config = _source_config(dict(checkpoint), role)
    expected_run = FINAL_RUNS[ROLES.index(role)]
    if role == "source4_lstm":
        data = _mapping(checkpoint.get("data"), "source4 data provenance")
        provenance = _mapping(checkpoint.get("provenance"), "source4 provenance")
        epoch = checkpoint.get("epoch")
        completed = checkpoint.get("completed_epochs")
        if type(epoch) is not int or epoch != 7 or type(completed) is not int or completed != 8:
            raise ValueError("Source4 final checkpoint must contain completed epoch 8 (index 7)")
        if type(config.get("epochs")) is not int or config.get("epochs") != 8:
            raise ValueError("Source4 final checkpoint must configure exactly 8 epochs")
        if provenance.get("completed_epochs") != 8 or provenance.get("requested_epochs") != 8:
            raise ValueError("Source4 provenance does not prove all 8 configured epochs completed")
        validate_final_lstm_recipe(
            config, split_manifest=None, teacher_cache=None, initializer=None, resume=None
        )
        validate_final_fit_provenance(data, membership)
        expected_fit_members = [
            [source, group]
            for source, groups in (
                ("train", membership.train_groups),
                ("valid", membership.valid_groups),
            )
            for group in groups
        ]
        fit_members = data.get("fit_members")
        if (
            not isinstance(fit_members, list)
            or any(
                not isinstance(fit_member, list)
                or len(fit_member) != 2
                or fit_member[0] not in ("train", "valid")
                or type(fit_member[1]) is not int
                for fit_member in fit_members
            )
            or fit_members != expected_fit_members
        ):
            raise ValueError(
                "Source4 checkpoint has incomplete or reordered source-qualified fit members"
            )
        if (
            data.get("evaluation_members") != []
            or data.get("evaluation_role") != "none"
            or data.get("evaluation_source") != "none"
            or data.get("evaluation_mask") != "none"
            or data.get("monitoring_plan") is not None
            or "evaluation_annotation" in provenance
        ):
            raise ValueError("Source4 final checkpoint contains an evaluation or monitoring path")
        history = provenance.get("history")
        if (
            not isinstance(history, list)
            or len(history) != 8
            or any(
                not isinstance(epoch_record, Mapping)
                or any(
                    name.startswith(("valid_", "train_eval_"))
                    for name in epoch_record
                    if isinstance(name, str)
                )
                for epoch_record in history
            )
        ):
            raise ValueError(
                "Source4 final checkpoint history contains evaluation metrics or is incomplete"
            )
        sources = data.get("sources")
        if (
            not isinstance(sources, Mapping)
            or sources.get("datasets/train.parquet") != {"sha256": membership.train_sha256}
            or sources.get("datasets/valid.parquet") != {"sha256": membership.valid_sha256}
        ):
            raise ValueError(
                "Source4 checkpoint data provenance differs from final-fit source hashes"
            )
        normalization = data.get("normalization")
        if (
            not isinstance(normalization, Mapping)
            or normalization.get("scope") != "prefix"
            or normalization.get("sequence_count") != 32
            or normalization.get("fit_prefix_sequences") != 32
        ):
            raise ValueError(
                "Source4 checkpoint must preserve fitting-only static prefix32 normalization"
            )
        data_sha = canonical_sha256(data)
        if data_sha != member.get("data_sha256"):
            raise ValueError(
                "Source4 checkpoint data provenance SHA differs from final-model receipt"
            )
    else:
        args = config
        provenance = _mapping(checkpoint.get("provenance"), "legacy provenance")
        epoch = checkpoint.get("epoch")
        endpoint = FINAL_FIXED_EPOCHS[role]
        if args.get("name") != expected_run:
            raise ValueError(f"{role} checkpoint argument run label differs from final receipt")
        if (
            type(args.get("epochs")) is not int
            or args.get("epochs") != endpoint["configured_epochs"]
            or type(args.get("checkpoint_epoch")) is not int
            or args.get("checkpoint_epoch") != endpoint["checkpoint_epoch"]
            or type(epoch) is not int
            or epoch != endpoint["checkpoint_epoch"]
        ):
            raise ValueError(f"{role} checkpoint is not the fixed configured final endpoint")
        if args.get("checkpoint_policy") != "fixed":
            raise ValueError(f"{role} checkpoint does not use the fixed endpoint policy")
        if provenance.get("args") != args:
            raise ValueError(f"{role} checkpoint arguments differ from embedded provenance args")
        validate_final_fit_artifact(args, provenance, membership, checkpoint_epoch=epoch)
        data_sha = None
    if member.get("configuration_sha256") != canonical_sha256(config):
        raise ValueError(f"{role} checkpoint configuration SHA differs from final-model receipt")
    if member.get("provenance_sha256") != canonical_sha256(provenance):
        raise ValueError(f"{role} checkpoint provenance SHA differs from final-model receipt")
    if role != "source4_lstm" and set(member) != {
        "run",
        "checkpoint_sha256",
        "configuration_sha256",
        "provenance_sha256",
    }:
        raise ValueError(f"{role} final receipt must not include a Source4-only data SHA")
    return config, model_config, {"provenance": dict(provenance), "data_sha256": data_sha}


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return dict(value)


def _source_config(
    checkpoint: dict[str, object], role: str
) -> tuple[dict[str, object], dict[str, object]]:
    """Read the legacy run arguments or sequence-fit model config."""
    if role == "source4_lstm":
        config = _mapping(checkpoint.get("config"), "config")
        model = _mapping(config.get("model"), "config.model")
        if model.get("architecture") != "lstm":
            raise ValueError("Source4 checkpoint is not a sequence LSTM")
        if model.get("input_feature_mode", "raw112") != "raw112":
            raise ValueError("Source4 LSTM must use raw112 input")
        if model.get("hidden") != 192 or model.get("layers") != 1:
            raise ValueError("Source4 LSTM must be the fixed 192x1 candidate")
        if model.get("dropout", 0.0) != 0.0 or model.get("hint_dim", 0) != 0:
            raise ValueError("Source4 LSTM must not require dropout or representation hints")
        if config.get("precision_mode", "fp32") != "fp32":
            raise ValueError("Source4 LSTM must preserve its strict-FP32 recipe")
        if (
            config.get("normalization_scope") != "prefix"
            or config.get("normalization_sequences") != 32
        ):
            raise ValueError("Source4 LSTM must retain fitting-only prefix32 normalization")
        return config, model

    args = _mapping(checkpoint.get("args"), "args")
    if args.get("arch", "gru") not in ("gru", None):
        raise ValueError(f"{role} checkpoint is not a plain RNNModel")
    if args.get("cell", "gru") != "gru":
        raise ValueError(f"{role} checkpoint must use a GRU cell")
    if args.get("hidden") != 256 or args.get("layers") != 2:
        raise ValueError(f"{role} checkpoint must use the fixed GRU 256x2 candidate")
    independent_widths = ("encoder_width", "recurrent_width", "readout_width")
    if any(args.get(name) is not None for name in independent_widths):
        raise ValueError(f"{role} checkpoint uses unsupported independent model widths")
    if args.get("head_layers", 1) != 1 or args.get("dropout", 0.0) != 0.0:
        raise ValueError(f"{role} checkpoint uses an unsupported head or recurrent dropout")
    if args.get("enc", "linear") != ("tiedlead" if role.startswith("tiedlead") else "linear"):
        raise ValueError(f"{role} checkpoint uses an unsupported input encoder")
    if args.get("readout_activation", "gelu") != "gelu":
        raise ValueError(f"{role} checkpoint uses a non-GELU readout")
    if args.get("numeric_map", "none") != "none":
        raise ValueError(f"{role} checkpoint uses an unsupported numeric feature map")
    if args.get("sentinel_indicators", "none") != "none" or checkpoint.get(
        "sentinel_preprocessing"
    ):
        raise ValueError(f"{role} checkpoint uses unsupported sentinel preprocessing")
    unsupported_flags = {
        "online_norm": False,
        "heuristics": False,
        "raw_only": False,
        "research_features": "none",
        "prop_cache": False,
        "propensity": "",
        "prop_input": "",
        "origin_view": "off",
        "sentinel_values": "unchanged",
    }
    for name, expected in unsupported_flags.items():
        if args.get(name, expected) != expected:
            raise ValueError(f"{role} checkpoint uses unsupported {name} preprocessing")
    if (
        args.get("input_lead")
        or args.get("snap_codebook")
        or args.get("privileged_mode", "none") != "none"
    ):
        raise ValueError(f"{role} checkpoint contains teacher-only lookahead inputs")
    if role.startswith("tiedlead") and args.get("numeric_map", "none") != "none":
        raise ValueError("tiedlead requires the original un-mapped 263-column schema")
    return args, args


def validate_tensor_layout(role: str, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Fail closed unless named PyTorch tensors match the native RNN schema."""
    index = ROLES.index(role)
    input_width = WIDTHS[index]
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{role}: tensor {name} contains non-finite values")
    mean = arrays.get("mean")
    std = arrays.get("std")
    if mean is None or std is None or mean.shape != (input_width,) or std.shape != (input_width,):
        raise ValueError(f"{role}: normalization tensors must have width {input_width}")
    if np.any(std <= 0):
        raise ValueError(f"{role}: normalization scales must be positive")
    inp = arrays.get("inp.weight")
    inp_bias = arrays.get("inp.bias")
    if inp is None or inp.ndim != 2 or inp.shape[1] != input_width or inp_bias is None:
        raise ValueError(f"{role}: inp tensors do not match width {input_width}")
    encoder_width = int(inp.shape[0])
    if inp_bias.shape != (encoder_width,):
        raise ValueError(f"{role}: inp.bias shape does not match encoder width")

    weight_ih = arrays.get("gru.weight_ih_l0")
    weight_hh = arrays.get("gru.weight_hh_l0")
    if weight_ih is None or weight_hh is None or weight_hh.ndim != 2 or weight_hh.shape[1] == 0:
        raise ValueError(f"{role}: recurrent tensors are missing or malformed")
    hidden_width = int(weight_hh.shape[1])
    gate_count = int(weight_hh.shape[0] // hidden_width)
    if weight_hh.shape[0] % hidden_width or gate_count not in (3, 4):
        raise ValueError(f"{role}: recurrent tensors are neither GRU nor LSTM")
    if role == "source4_lstm" and gate_count != 4:
        raise ValueError("Source4 tensor layout is not a four-gate LSTM")

    allowed = {
        "mean",
        "std",
        "inp.weight",
        "inp.bias",
        "head.0.weight",
        "head.0.bias",
        "head.2.weight",
        "head.2.bias",
    }
    layer = 0
    while f"gru.weight_ih_l{layer}" in arrays:
        input_size = encoder_width if layer == 0 else hidden_width
        shapes = {
            f"gru.weight_ih_l{layer}": (gate_count * hidden_width, input_size),
            f"gru.weight_hh_l{layer}": (gate_count * hidden_width, hidden_width),
            f"gru.bias_ih_l{layer}": (gate_count * hidden_width,),
            f"gru.bias_hh_l{layer}": (gate_count * hidden_width,),
        }
        for name, shape in shapes.items():
            if name not in arrays or arrays[name].shape != shape:
                raise ValueError(f"{role}: {name} must have shape {shape}")
            allowed.add(name)
        layer += 1
    if layer == 0:
        raise ValueError(f"{role}: recurrent layer 0 is missing")

    head = arrays.get("head.0.weight")
    head_bias = arrays.get("head.0.bias")
    output = arrays.get("head.2.weight")
    output_bias = arrays.get("head.2.bias")
    if head is None or head.ndim != 2 or head.shape[1] != hidden_width + encoder_width:
        raise ValueError(f"{role}: head.0.weight has an unsupported shape")
    readout_width = int(head.shape[0])
    if head_bias is None or head_bias.shape != (readout_width,):
        raise ValueError(f"{role}: head.0.bias has an unsupported shape")
    if output is None or output.shape[0] < 2 or output.shape[1] != readout_width:
        raise ValueError(f"{role}: head.2.weight must provide at least two outputs")
    if output_bias is None or output_bias.shape != (output.shape[0],):
        raise ValueError(f"{role}: head.2.bias does not match head.2.weight")
    tied_keys = {
        "tied.0.weight": (128, 104),
        "tied.0.bias": (128,),
        "tied.2.weight": (64, 128),
        "tied.2.bias": (64,),
        "tied_out.weight": (encoder_width, 128),
        "tied_out.bias": (encoder_width,),
        "lead_out.weight": (encoder_width, 64),
        "lead_out.bias": (encoder_width,),
    }
    if role.startswith("tiedlead"):
        for name, shape in tied_keys.items():
            if name not in arrays or arrays[name].shape != shape:
                raise ValueError(f"{role}: {name} must have shape {shape}")
            allowed.add(name)
    elif any(name in arrays for name in tied_keys):
        raise ValueError(f"{role}: tiedlead tensors are not allowed for this member")
    expected_layout = {
        "encoder_width": 192 if role == "source4_lstm" else 256,
        "hidden_width": 192 if role == "source4_lstm" else 256,
        "recurrent_layers": 1 if role == "source4_lstm" else 2,
        "readout_width": 192 if role == "source4_lstm" else 256,
    }
    actual_layout = {
        "encoder_width": encoder_width,
        "hidden_width": hidden_width,
        "recurrent_layers": layer,
        "readout_width": readout_width,
    }
    if actual_layout != expected_layout:
        raise ValueError(
            f"{role}: tensor layout {actual_layout} differs from fixed {expected_layout}"
        )
    unknown = sorted(set(arrays) - allowed)
    if unknown:
        raise ValueError(f"{role}: unsupported model tensors: {', '.join(unknown)}")
    return {
        "input_width": input_width,
        "encoder_width": encoder_width,
        "hidden_width": hidden_width,
        "recurrent_layers": layer,
        "gate_count": gate_count,
        "readout_width": readout_width,
        "output_count_before_trim": int(output.shape[0]),
    }


def _load_member(
    path: Path,
    role: str,
    destination: Path,
    final_receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    """Load and export one pinned engineering or receipt-bound final checkpoint."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{role} checkpoint must be a regular non-symlink file: {path}")
    actual_sha = sha256_file(path)
    if final_receipt is None:
        expected_sha = EXPECTED_CHECKPOINT_SHA256[ROLES.index(role)]
        run_reference = RUNS[ROLES.index(role)]
        if actual_sha != expected_sha:
            raise ValueError(
                f"{role} checkpoint SHA-256 differs from the fixed engineering reference"
            )
        final_fit_membership = None
        receipt_member = None
    else:
        receipt_value = _mapping(final_receipt.get("receipt"), "final-model receipt")
        final_fit_membership = final_receipt.get("membership")
        members = _mapping(receipt_value.get("members"), "final-model receipt members")
        receipt_member = _mapping(members.get(role), f"final-model receipt {role}")
        expected_sha = receipt_member["checkpoint_sha256"]
        run_reference = receipt_member["run"]
        if actual_sha != expected_sha:
            raise ValueError(f"{role} checkpoint SHA-256 differs from final-model receipt")
        if not isinstance(final_fit_membership, FinalFitMembership):
            raise ValueError("final-model receipt lacks its validated all-labeled fit membership")
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = _mapping(checkpoint, f"{role} envelope")
    if final_receipt is None:
        config, model_config = _source_config(checkpoint, role)
        provenance_record = _mapping(checkpoint.get("provenance"), f"{role} provenance")
        final_data_sha = None
    else:
        config, model_config, final_metadata = _validate_final_checkpoint(
            role, checkpoint, receipt_member, final_fit_membership
        )
        provenance_record = final_metadata["provenance"]
        final_data_sha = final_metadata["data_sha256"]
    state = _mapping(checkpoint.get("model"), f"{role} model state")
    arrays: dict[str, np.ndarray] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{role} model state must contain named tensors")
        arrays[name] = np.ascontiguousarray(tensor.detach().cpu().float().numpy(), dtype=np.float32)
    output_weight = arrays.get("head.2.weight")
    output_bias = arrays.get("head.2.bias")
    if output_weight is None or output_bias is None or output_weight.shape[0] < 2:
        raise ValueError(f"{role} checkpoint lacks a two-target output head")
    original_output_count = int(output_weight.shape[0])
    arrays["head.2.weight"] = np.ascontiguousarray(output_weight[:2])
    arrays["head.2.bias"] = np.ascontiguousarray(output_bias[:2])
    layout = validate_tensor_layout(role, arrays)
    layout["output_count_before_trim"] = original_output_count
    weight_name = f"weights_{role}.npz"
    np.savez(destination / weight_name, **arrays)

    config_sha = canonical_sha256(config)
    provenance_sha = canonical_sha256(provenance_record)
    if final_receipt is not None and (
        config_sha != receipt_member["configuration_sha256"]
        or provenance_sha != receipt_member["provenance_sha256"]
    ):
        raise ValueError(f"{role} config or provenance SHA differs from final-model receipt")
    tensor_manifest = {
        name: {
            "shape": list(value.shape),
            "dtype": "<f4",
            "sha256": hashlib.sha256(
                np.ascontiguousarray(value, dtype="<f4").tobytes()
            ).hexdigest(),
        }
        for name, value in sorted(arrays.items())
    }
    return {
        "role": role,
        "run_reference": run_reference,
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": actual_sha,
        "configuration_sha256": config_sha,
        "configuration": config,
        "model_configuration": model_config,
        "provenance_sha256": provenance_sha,
        "provenance": provenance_record,
        "data_sha256": final_data_sha,
        "weights": weight_name,
        "weights_sha256": sha256_file(destination / weight_name),
        "tensor_manifest": tensor_manifest,
        "layout": layout,
    }


def _write_package(args: argparse.Namespace) -> dict[str, object]:
    output = args.output_zip.resolve()
    receipt_path = (args.receipt or output.with_name(f"{output.stem}_build_receipt.json")).resolve()
    if receipt_path == output:
        raise ValueError("artifact and build receipt paths must differ")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("final package and receipt are create-only outputs")
    if (args.final_model_receipt is None) != (args.final_model_receipt_sha256 is None):
        raise ValueError("--final-model-receipt requires --final-model-receipt-sha256")
    final_receipt = (
        load_final_model_receipt(args.final_model_receipt, args.final_model_receipt_sha256)
        if args.final_model_receipt is not None
        else None
    )
    native_root = args.native_from.resolve(strict=True)
    for level in ("v2", "avx2", "v3"):
        library = native_root / level / "wnn_native.abi3.so"
        if library.is_symlink() or not library.is_file():
            raise FileNotFoundError(f"missing native {level} library: {library}")

    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".final-tl3-", dir=output.parent) as temporary:
        stage = Path(temporary)
        package = stage / "pkg"
        package.mkdir()
        shutil.copyfile(ROOT / "submission" / "solution_final_tl3.py", package / "solution.py")
        (package / "precision.txt").write_text("f32\n", encoding="utf-8")
        member_records = [
            _load_member(path, role, package, final_receipt)
            for role, path in zip(
                ROLES, (args.w18, args.tiedlead_s5, args.tiedlead_s6, args.lstm), strict=True
            )
        ]
        final_mode = final_receipt is not None
        members_manifest = {
            "schema": "wnn.final-tl3-lstm-members.v1",
            "precision": "f32",
            "blend": BLEND,
            "candidate_scope": (
                "approved_full_corpus_final_model_set"
                if final_mode
                else "engineering_train_only_reference"
            ),
            "training_and_selection_scope": (
                "all official train and valid labels fit; no independent evaluation"
                if final_mode
                else "engineering preflight with train-only checkpoint lineage; validation is development-exposed"
            ),
            "independent_evaluation_sequences": 0,
            "independent_score_claim": "forbidden",
            "members": [],
        }
        if final_mode:
            receipt_value = final_receipt["receipt"]
            members_manifest["final_fit"] = receipt_value["final_fit"]
            members_manifest["final_model_receipt"] = {
                "schema": FINAL_RECEIPT_SCHEMA,
                "sha256": final_receipt["sha256"],
            }
        for record in member_records:
            members_manifest["members"].append(
                {
                    key: record[key]
                    for key in (
                        "role",
                        "run_reference",
                        "checkpoint_sha256",
                        "configuration_sha256",
                        "model_configuration",
                        "provenance_sha256",
                        "data_sha256",
                        "weights",
                        "weights_sha256",
                    )
                }
                | {"input_width": record["layout"]["input_width"]}
            )
        (package / "members.json").write_text(
            json.dumps(members_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        native_hashes: dict[str, str] = {}
        for level in ("v2", "avx2", "v3"):
            relative = f"{level}/wnn_native.abi3.so"
            target = package / relative
            target.parent.mkdir()
            shutil.copyfile(native_root / relative, target)
            native_hashes[relative] = sha256_file(target)

        zip_members: dict[str, str] = {}
        with zipfile.ZipFile(
            stage / "submission.zip", "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(item for item in package.rglob("*") if item.is_file()):
                relative = path.relative_to(package).as_posix()
                archive.write(path, relative)
                zip_members[relative] = sha256_file(path)
        staged_zip = stage / "submission.zip"
        if staged_zip.stat().st_size > args.max_zip_bytes:
            raise ValueError(
                f"final submission is {staged_zip.stat().st_size} bytes; limit is {args.max_zip_bytes}"
            )
        with zipfile.ZipFile(staged_zip) as archive:
            if archive.testzip() is not None:
                raise ValueError("final submission ZIP failed CRC verification")
        receipt = {
            "schema": "wnn.final-tl3-lstm-build-receipt.v1",
            "artifact": {
                "path": str(output),
                "bytes": staged_zip.stat().st_size,
                "sha256": sha256_file(staged_zip),
                "members": zip_members,
            },
            "native_libraries": native_hashes,
            "solution_sha256": sha256_file(ROOT / "submission" / "solution_final_tl3.py"),
            "packager_sha256": sha256_file(Path(__file__).resolve()),
            "fixed_blend": BLEND,
            "candidate_scope": members_manifest["candidate_scope"],
            "training_and_selection_scope": members_manifest["training_and_selection_scope"],
            "final_model_receipt": members_manifest.get("final_model_receipt"),
            "checkpoint_sources": member_records,
        }
        with output.open("xb") as destination:
            destination.write(staged_zip.read_bytes())
        with receipt_path.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    """Create a hash-bound final candidate ZIP; refuse overwrites and oversized output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w18", type=Path, required=True)
    parser.add_argument("--tiedlead-s5", dest="tiedlead_s5", type=Path, required=True)
    parser.add_argument("--tiedlead-s6", dest="tiedlead_s6", type=Path, required=True)
    parser.add_argument("--lstm", type=Path, required=True)
    parser.add_argument("--native-from", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--final-model-receipt", type=Path)
    parser.add_argument("--final-model-receipt-sha256")
    parser.add_argument("--max-zip-bytes", type=int, default=MAX_ZIP_BYTES)
    args = parser.parse_args()
    if not 0 < args.max_zip_bytes <= MAX_ZIP_BYTES:
        parser.error(f"--max-zip-bytes must be in 1..{MAX_ZIP_BYTES}")
    print(json.dumps(_write_package(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
