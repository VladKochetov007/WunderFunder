from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from candidate_contract import (
    MEMBERS,
    verify_ancestry,
    verify_candidate_args,
    verify_membership,
    verify_metadata,
    verify_source,
)
from candidate_io import JsonObject, object_value, pinned_path, sha256

if TYPE_CHECKING:
    import numpy as np

TIED_FEATURE_SOURCES = {
    "src/wnn/tiedlead.py": "71aca6a39e9317120890ab4894998dc4e9f4ffa5b35b0337309585f5c1acbb99",
    "src/wnn/nn.py": "3c7a6f9cecebd47cfe52134cd0689d5548a1ac22d0be3b415cfb9ed9f6f1919f",
    "src/wnn/training/inputs.py": "3e0ba28e997675627db62f2064353c599d89445acf487a327c6fccc3491432e5",
    "src/wnn/training/prediction_inputs.py": "a8c67145b4e78e139c473d37b26deb00799d49f939c759c5a73bc0c0f6c3e72e",
    "scripts/predict_run.py": "981d7557082f8e33e025741dad2b2ab2e86fb20eb4347ab58eca1d53f00c9520",
    "src/wnn/features.py": "c2b317d1f148d0cad9632726a1436b94f4ae76016458deaa70aa36fdca21ecb2",
    "src/wnn/data.py": "a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00",
}


def tensor_schema(tiedlead: bool) -> dict[str, tuple[int, ...]]:
    """Return the exact legacy H256x2 tensor contract before auxiliary-head removal."""
    width = 263 if tiedlead else 211
    shapes = {
        "mean": (width,),
        "std": (width,),
        "inp.weight": (256, width),
        "inp.bias": (256,),
        "head.0.weight": (256, 512),
        "head.0.bias": (256,),
        "head.2.weight": (23, 256),
        "head.2.bias": (23,),
    }
    for layer in (0, 1):
        for part in ("ih", "hh"):
            shapes[f"gru.weight_{part}_l{layer}"] = (768, 256)
            shapes[f"gru.bias_{part}_l{layer}"] = (768,)
    if tiedlead:
        shapes.update(
            {
                "tied.0.weight": (128, 104),
                "tied.0.bias": (128,),
                "tied.2.weight": (64, 128),
                "tied.2.bias": (64,),
                "tied_out.weight": (256, 128),
                "tied_out.bias": (256,),
                "lead_out.weight": (256, 64),
                "lead_out.bias": (256,),
            }
        )
    return shapes


def _extract_arrays(checkpoint: JsonObject, tiedlead: bool) -> dict[str, np.ndarray]:
    import numpy as np
    import torch

    state = object_value(checkpoint.get("model"), "checkpoint.model")
    expected = tensor_schema(tiedlead)
    if state.keys() != expected.keys():
        raise ValueError(
            f"unhandled or missing tensors: extra={sorted(state.keys() - expected.keys())}; "
            f"missing={sorted(expected.keys() - state.keys())}"
        )
    arrays = {}
    for name, shape in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"unexpected tensor shape/type: {name}; expected {shape}")
        if value.dtype != torch.float32 or value.layout != torch.strided:
            raise ValueError(f"{name}: checkpoint must contain dense FP32 evaluation weights")
        array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite checkpoint tensor: {name}")
        arrays[name] = array
    if (arrays["std"] <= 0).any():
        raise ValueError("normalization standard deviations must be strictly positive")
    return arrays


def _validate_preprocessing(
    provenance: JsonObject, arrays: dict[str, np.ndarray], tied: bool
) -> None:
    normalization = object_value(provenance.get("normalization_identity"), "normalization_identity")
    for name in ("mean", "std"):
        expected = {
            "shape": list(arrays[name].shape),
            "dtype": "torch.float32",
            "bytes_sha256": hashlib.sha256(arrays[name].tobytes()).hexdigest(),
        }
        if normalization.get(name) != expected:
            raise ValueError(f"{name}: checkpoint preprocessing differs from fitted provenance")
    if tied:
        schema = object_value(provenance.get("input_feature_schema"), "input_feature_schema")
        if (
            schema.get("name") != "raw112-event99-lead52-v1"
            or schema.get("width") != 263
            or schema.get("source_sha256") != TIED_FEATURE_SOURCES
            or schema.get("normalization_membership") != "training loader fitting members only"
        ):
            raise ValueError("tiedlead preprocessing source/schema is not the audited PR49 version")


def export_member(record: JsonObject, split: JsonObject, output: Path, coef: float) -> JsonObject:
    """Export one pinned scored EMA state without constructing or executing a model.

    Only a receiver invokes this function. torch.load's restricted weights-only loader
    reads tensors and primitive checkpoint metadata; no fallback to arbitrary pickle runs.
    """
    provenance = verify_metadata(record, split)
    ancestry = verify_ancestry(record, split)
    run = str(record["run"])
    verify_candidate_args(object_value(provenance.get("args"), "provenance.args"), run)
    checkpoint_path = pinned_path(record["checkpoint"], "checkpoint")
    import numpy as np
    import torch
    from torch.torch_version import TorchVersion

    # PR49 records torch.__version__ in diagnostics as this str subclass. Keep the
    # unpickler restricted; no arbitrary-pickle fallback if another object is found.
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = object_value(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True), "checkpoint"
        )
    if checkpoint.get("epoch") != 3 or checkpoint.get("args") != provenance["args"]:
        raise ValueError(f"{run}: checkpoint epoch/args disagree with its sidecar")
    for name in (
        "sentinel_preprocessing",
        "research_feature_prior",
        "quantization",
        "factorization",
    ):
        if name in checkpoint:
            raise ValueError(f"{run}: unsupported checkpoint representation {name}")
    if checkpoint.get("readout") is not None or checkpoint.get("readout_tilted"):
        raise ValueError(f"{run}: outcome-code readout is unsupported")
    tied = run != MEMBERS[0]
    if checkpoint.get("n_in") != (263 if tied else 211):
        raise ValueError(f"{run}: input width differs from its family")
    embedded = object_value(checkpoint.get("provenance"), "checkpoint.provenance")
    verify_source(embedded, run)
    verify_membership(embedded, split)
    arrays = _extract_arrays(checkpoint, tied)
    _validate_preprocessing(provenance, arrays, tied)
    _validate_preprocessing(embedded, arrays, tied)
    # model_ep3 is already the scored EMA snapshot; never average it again.
    arrays["head.2.weight"] = np.ascontiguousarray(arrays["head.2.weight"][:2])
    arrays["head.2.bias"] = np.ascontiguousarray(arrays["head.2.bias"][:2])
    weight_file = f"{run}.npz"
    weight_path = output / weight_file
    with weight_path.open("xb") as stream:
        np.savez(stream, **arrays)
    if sha256(checkpoint_path) != record["checkpoint"]["sha256"]:
        raise ValueError("checkpoint changed during export")
    return {
        "run": run,
        "coef": coef,
        "family": "tiedlead263" if tied else "event211",
        "checkpoint": record["checkpoint"],
        "provenance": record["provenance"],
        "ancestors": ancestry,
        "training_source_sha256": record["training_source_sha256"],
        "weight_file": weight_file,
        "weight_sha256": sha256(weight_path),
        "snapshot": "model_ep3 already contains scored EMA weights; first 2 of 23 head rows",
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": value.nbytes,
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
            for name, value in arrays.items()
        },
    }
