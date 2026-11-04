from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import canonical_json_sha256, file_sha256, read_json
from wnn.training.sequence_config import load_config


def qualified_parent(
    path: Path, *, allowed: set[int]
) -> tuple[dict[str, object], Path, dict[str, object]]:
    """Require an immutable fixed-final clean parent for fresh initialization."""
    parent_path = path.resolve(strict=True)
    parent = read_json(parent_path)
    evaluation = (
        parent.get("objective", {}).get("evaluation")
        if isinstance(parent.get("objective"), Mapping)
        else None
    )
    if (
        parent.get("status") != "COMPLETE"
        or parent.get("stage") not in allowed
        or not isinstance(parent.get("objective"), Mapping)
        or not isinstance(evaluation, Mapping)
        or evaluation.get("scope") != "inner_train_proxy"
    ):
        raise ValueError("initializer parent must be a complete qualified inner-proxy fit")
    fit_dir = Path(str(parent["fit_output"])).resolve(strict=True)
    checkpoint = fit_dir / "model.pt"
    if file_sha256(checkpoint) != parent.get("fit_model_sha256"):
        raise ValueError("parent fixed-final checkpoint changed after scoring")
    config_path = parent_path.parent / "config.json"
    if file_sha256(config_path) != parent.get("config_sha256"):
        raise ValueError("parent config changed after its fit result")
    config = load_config(config_path)
    if (
        config.get("kd_weight") != 0
        or config.get("hint_weight") != 0
        or not isinstance(config.get("model"), Mapping)
        or config["model"].get("hint_dim") != 0
        or config.get("precision_mode") != "fp32"
    ):
        raise ValueError("stage2 initializer parent must be label-only strict-FP32 with no hints")
    import torch

    checkpoint_sha = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    data = payload.get("data") if isinstance(payload, Mapping) else None
    if (
        not isinstance(data, Mapping)
        or data.get("teacher") is not None
        or data.get("initializer") is not None
    ):
        raise ValueError("stage2 initializer parent must have direct label-only ancestry")
    payload_config = payload.get("config") if isinstance(payload, Mapping) else None
    epochs = payload_config.get("epochs") if isinstance(payload_config, Mapping) else None
    provenance = payload.get("provenance") if isinstance(payload, Mapping) else None
    identity = provenance.get("identity") if isinstance(provenance, Mapping) else None
    source_files = identity.get("source_files") if isinstance(identity, Mapping) else None
    if (
        payload.get("checkpoint_schema") != "wnn.sequence-fit.v1"
        or data.get("training_mask") != "all_after_warmup"
        or data.get("evaluation_role") != "control"
        or data.get("evaluation_source") != "train"
        or payload_config != config
        or type(epochs) is not int
        or payload.get("completed_epochs") != epochs
        or payload.get("epoch") != epochs - 1
        or not isinstance(identity, Mapping)
        or identity.get("config_sha256") != canonical_json_sha256(config)
        or identity.get("data_sha256") != canonical_json_sha256(data)
        or identity.get("data") != data
        or not isinstance(source_files, Mapping)
        or identity.get("source_sha256") != canonical_json_sha256(source_files)
    ):
        raise ValueError("stage2 initializer parent has an unqualified fit/config endpoint")
    if file_sha256(checkpoint) != checkpoint_sha or not isinstance(
        payload.get("provenance"), Mapping
    ):
        raise ValueError("parent checkpoint changed or omitted its provenance")
    return parent, checkpoint, config


def parent_identity_sha(path: Path) -> str:
    """Read the exact producer identity embedded in a fixed-final checkpoint."""
    import torch

    payload = torch.load(
        Path(str(read_json(path)["fit_output"])) / "model.pt", map_location="cpu", weights_only=True
    )
    provenance = payload.get("provenance") if isinstance(payload, Mapping) else None
    identity = provenance.get("identity") if isinstance(provenance, Mapping) else None
    if not isinstance(identity, Mapping):
        raise ValueError("parent checkpoint omits its run identity")
    return canonical_json_sha256(identity)
