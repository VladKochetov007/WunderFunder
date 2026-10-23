"""Check recovered SWA tensors and prediction-export encodings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from checkpoint_io import canonical_mean_state, load_source_checkpoints
from recovery_contract import (
    DERIVED_SHA,
    INITIALIZATION_STATUS,
    PACKET,
    SWA_NAME,
    read_json,
    require,
    sha256,
)


def verify_mean(project_root: Path, source_fit_dir: Path, recovery_root: Path) -> dict[str, Any]:
    """Recompute canonical CP3/4/5 mean and compare it to the recovered SWA checkpoint."""
    import torch

    source = load_source_checkpoints(source_fit_dir)
    mean_state = canonical_mean_state(project_root)
    expected = mean_state([checkpoint["model"] for checkpoint in source])
    path = recovery_root / "artifacts/runs" / SWA_NAME / "model_ep5.pt"
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        actual = torch.load(path, map_location="cpu", weights_only=True)
    require(
        actual.get("epoch") == 5 and actual.get("n_in") == 211,
        "Recovered SWA checkpoint architecture differs",
    )
    actual_state = actual.get("model", {})
    require(
        actual_state.keys() == expected.keys(),
        "Recovered SWA tensor names differ from the canonical mean",
    )
    require(
        all(torch.equal(actual_state[name], value) for name, value in expected.items()),
        "Recovered SWA tensors differ from the canonical equal mean",
    )
    derivation = actual.get("weight_derivation", {})
    require(
        derivation.get("schema") == "wnn.kaggle-h512-b16-offline-swa345.v1",
        "Recovered SWA lacks the recovery derivation receipt",
    )
    require(
        derivation.get("initial_parameter_sha256") is None,
        "Recovered SWA invented an initialization identity",
    )
    require(
        derivation.get("initialization_identity_status") == INITIALIZATION_STATUS,
        "Recovered SWA initialization limitation changed",
    )
    require(
        actual.get("provenance", {}).get("weight_derivation") == derivation,
        "Recovered SWA checkpoint provenance differs from its derivation record",
    )
    require(
        actual.get("args") == source[-1].get("args"),
        "Recovered SWA trainer arguments differ from CP5",
    )
    return {"path": path.relative_to(recovery_root).as_posix(), "sha256": sha256(path)}


def validate_export(
    recovery_root: Path, project_root: Path, run_name: str, checkpoint_path: Path
) -> dict[str, Any]:
    """Check an FP32 evaluation-only export and bind its NPY header and payload."""
    prediction = recovery_root / "cache" / f"pred_{run_name}_valid.npy"
    sidecar = prediction.with_suffix(".json")
    metadata = read_json(sidecar)
    checkpoint_sha = sha256(checkpoint_path)
    require(
        metadata.get("run") == run_name
        and metadata.get("split") == "valid"
        and metadata.get("epoch") == 5,
        f"Export identity differs for {run_name}",
    )
    require(
        metadata.get("checkpoint_sha256") == checkpoint_sha,
        f"Export checkpoint differs for {run_name}",
    )
    require(
        metadata.get("campaign_split_manifest_sha256") == DERIVED_SHA,
        f"Export manifest differs for {run_name}",
    )
    require(
        metadata.get("campaign_export_scope") == "evaluation"
        and metadata.get("feature_only") is True,
        f"Export is not evaluation-only feature data for {run_name}",
    )
    require(
        metadata.get("inference_precision") == "fp32",
        f"Export inference precision differs for {run_name}",
    )
    groups = metadata.get("row_groups")
    require(
        isinstance(groups, list) and len(groups) == 457 and len(set(groups)) == 457,
        f"Export membership is malformed for {run_name}",
    )
    expected_groups = read_json(project_root / PACKET / "fold1_profile_view.json")["evaluation"][
        "valid"
    ]
    require(groups == expected_groups, f"Export group ordering differs for {run_name}")
    values = np.load(prediction, mmap_mode="r", allow_pickle=False)
    require(values.shape == (457, 20_000, 2), f"Export shape differs for {run_name}")
    require(
        values.dtype == np.dtype("float32") and values.flags.c_contiguous,
        f"Export dtype/layout differs for {run_name}",
    )
    require(values.nbytes == 457 * 20_000 * 2 * 4, f"Export payload size differs for {run_name}")
    require(
        prediction.stat().st_size == values.offset + values.nbytes,
        f"Export NPY header/payload size differs for {run_name}",
    )
    require(bool(np.isfinite(values).all()), f"Export has nonfinite predictions for {run_name}")
    prediction_hash = sha256(prediction)
    require(
        metadata.get("prediction_sha256") == prediction_hash,
        f"Export sidecar prediction hash differs for {run_name}",
    )
    return {
        "run_name": run_name,
        "checkpoint_path": checkpoint_path.relative_to(recovery_root).as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "prediction_path": prediction.relative_to(recovery_root).as_posix(),
        "prediction_sha256": prediction_hash,
        "prediction_size_bytes": prediction.stat().st_size,
        "sidecar_path": sidecar.relative_to(recovery_root).as_posix(),
        "sidecar_sha256": sha256(sidecar),
        "campaign_split_manifest_sha256": DERIVED_SHA,
        "campaign_export_scope": "evaluation",
        "feature_only": True,
        "inference_precision": "fp32",
        "row_groups": groups,
    }
