from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

import torch

from wnn.training.sequence_epoch_prediction import epoch_admission, validate_epoch_payload
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_provenance import canonical_sha256, file_sha256, source_identity

SWA_SCHEMA = "wnn.sequence-swa.v1"
SWA_EPOCHS = (3, 4, 5, 6)


def equal_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare preserved buffers by shape, dtype and exact serialized value bytes."""
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.contiguous().reshape(-1).view(torch.uint8),
            right.contiguous().reshape(-1).view(torch.uint8),
        )
    )


def average_states(
    states: list[dict[str, torch.Tensor]],
    parameter_names: set[str],
) -> dict[str, torch.Tensor]:
    """Average four FP32 parameter states equally, preserving every buffer exactly."""
    if len(states) != 4 or not parameter_names or not parameter_names.issubset(states[0]):
        raise ValueError("fixed SWA requires four states and the complete parameter names")
    if any(state.keys() != states[0].keys() for state in states):
        raise ValueError("SWA parent state keys differ")
    result = copy.deepcopy(states[0])
    for name in result:
        tensors = [state[name] for state in states]
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("SWA state entries must be tensors")
        first = tensors[0]
        if any(value.shape != first.shape or value.dtype != first.dtype for value in tensors):
            raise ValueError(f"SWA parent tensor shapes/dtypes differ: {name}")
        if any(
            value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in tensors
        ):
            raise ValueError(f"nonfinite SWA parent tensor: {name}")
        if name in parameter_names:
            if first.dtype != torch.float32:
                raise ValueError("SWA parameters must use FP32")
            averaged = first.detach().clone().mul_(0.25)
            for value in tensors[1:]:
                averaged.add_(value, alpha=0.25)
            if not bool(torch.isfinite(averaged).all()):
                raise ValueError("nonfinite FP32 SWA result")
            result[name] = averaged
        elif not all(equal_tensor(first, value) for value in tensors[1:]):
            raise ValueError(f"SWA buffer must be identical in all parents: {name}")
    return result


def validate_swa_payload(payload: dict[str, object]) -> None:
    """Admit only the fixed inference-only E3/4/5/6 GRU readout, never a resume state."""
    config, provenance = payload.get("config"), payload.get("provenance")
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise ValueError("SWA payload lacks config/provenance")
    swa, identity = provenance.get("swa"), provenance.get("identity")
    if (
        payload.get("checkpoint_schema") != SWA_SCHEMA
        or not isinstance(swa, dict)
        or not isinstance(identity, dict)
        or swa.get("inference_only") is not True
        or swa.get("epochs") != list(SWA_EPOCHS)
        or swa.get("weights") != [0.25] * 4
        or swa.get("algorithm") != "FP32 quarter-scaled parameter sum in epoch order"
        or swa.get("buffer_policy") != "preserve exactly; all four parents must agree"
    ):
        raise ValueError("unsupported or malformed fixed SWA policy")
    if (
        config.get("epochs") != 6
        or config.get("checkpoint_retention") != "all_epochs"
        or config.get("model", {}).get("architecture") != "gru"
        or payload.get("epoch") != 5
        or payload.get("completed_epochs") != 6
        or identity.get("config_sha256") != canonical_sha256(config)
        or identity.get("data") != payload.get("data")
        or identity.get("data_sha256") != canonical_sha256(payload.get("data"))
        or identity.get("source_sha256") != canonical_sha256(identity.get("source_files"))
    ):
        raise ValueError("SWA identity, data or original six-epoch schedule differs")
    if any(key in payload for key in ("optimizer", "scheduler", "rank_rng_states", "next_epoch")):
        raise ValueError("SWA is inference-only and cannot carry a warm-resume claim")
    parents = swa.get("parents")
    if not isinstance(parents, list) or len(parents) != 4:
        raise ValueError("SWA requires exactly four hash-qualified parents")
    for epoch, parent in zip(SWA_EPOCHS, parents, strict=True):
        if (
            not isinstance(parent, dict)
            or parent.get("completed_epochs") != epoch
            or parent.get("identity_sha256") != canonical_sha256(identity)
            or any(
                not isinstance(parent.get(field), str)
                or len(parent[field]) != 64
                or any(c not in "0123456789abcdef" for c in parent[field])
                for field in ("checkpoint_sha256", "index_sha256")
            )
        ):
            raise ValueError("SWA parent order/hash/producer identity differs")
    if len({parent["checkpoint_sha256"] for parent in parents}) != 4:
        raise ValueError("SWA parents must name four distinct epoch checkpoints")


def build_swa(index_path: Path, index_sha256: str, output: Path) -> dict[str, object]:
    """Build a new immutable SWA envelope from a pinned six-epoch retained index."""
    index = json.loads(index_path.read_bytes())
    if index.get("completed_epochs") != 6:
        raise ValueError("fixed SWA requires the complete six-epoch index")
    root = index_path.parent if index_path.name == "epoch_index.json" else index_path.parent.parent
    payloads, parents = [], []
    for epoch in SWA_EPOCHS:
        entry = index["entries"][epoch - 1]
        checkpoint = root / entry["checkpoint_path"]
        admission = epoch_admission(checkpoint, index_path, index_sha256)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if file_sha256(checkpoint) != entry["checkpoint_sha256"]:
            raise ValueError("SWA parent bytes changed while being loaded")
        validate_epoch_payload(payload, admission)
        payloads.append(payload)
        parents.append(
            {
                "completed_epochs": epoch,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "index_path": str(index_path),
                "index_sha256": index_sha256,
                "identity_sha256": canonical_sha256(index["identity"]),
            }
        )
    final = payloads[-1]
    for payload in payloads:
        if (
            payload["identity"] != final["identity"]
            or payload["config"] != final["config"]
            or payload["data"] != final["data"]
            or payload["provenance"]["normalization"] != final["provenance"]["normalization"]
        ):
            raise ValueError("SWA parents differ in seed/config/source/data/normalization ancestry")
    model = build_sequence_model(final["config"]["model"])
    parameters = set(dict(model.named_parameters()))
    buffers = set(dict(model.named_buffers()))
    if parameters | buffers != set(final["model"]):
        raise ValueError("SWA parent state differs from the canonical model")
    payload = copy.deepcopy(final)
    payload["checkpoint_schema"] = SWA_SCHEMA
    payload["model"] = average_states([parent["model"] for parent in payloads], parameters)
    payload["provenance"]["swa"] = {
        "inference_only": True,
        "epochs": list(SWA_EPOCHS),
        "weights": [0.25] * 4,
        "algorithm": "FP32 quarter-scaled parameter sum in epoch order",
        "buffer_policy": "preserve exactly; all four parents must agree",
        "parameter_names": sorted(parameters),
        "buffer_names": sorted(buffers),
        "parents": parents,
        "implementation_source": source_identity(
            Path(__file__).parents[3], Path(__file__).parents[3] / "scripts/train_sequence.py"
        ),
    }
    payload["provenance"]["checkpoint_policy"] = "fixed SWA E3/4/5/6; no validation selection"
    validate_swa_payload(payload)
    model.load_state_dict(payload["model"], strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".swa-", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"checkpoint": str(output), "sha256": file_sha256(output), "parents": parents}


def main() -> None:
    """Materialize the prescribed post-fit inference readout without fitting or grading."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_swa(args.index, args.index_sha256, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
