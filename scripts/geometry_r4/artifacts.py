"""Artifact identity, pair matching and bounded prediction validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
    ADMISSION,
    PREDICTION_CHUNK_GROUPS,
    PREDICTIONS,
    ROOT,
    SCORES,
    SEQUENCE_LENGTH,
    TARGET_COUNT,
    read_json,
    run_name,
    sha256_file,
)


# JSON provenance and score receipts contain intentionally heterogeneous fields.
def prediction_paths(fold: int, seed: int, arm: str) -> tuple[Path, Path]:
    """Return the stable research archive paths for an endpoint's predictions."""
    stem = f"fold{fold}_seed{seed}_{arm}"
    return PREDICTIONS / f"{stem}.npy", PREDICTIONS / f"{stem}.json"


def score_path(fold: int, seed: int, arm: str) -> Path:
    """Return the stable official-score receipt path for one endpoint."""
    return SCORES / f"fold{fold}_seed{seed}_{arm}.json"


def checkpoint_complete(name: str) -> bool:
    """Check whether a fit reached the fixed epoch-2 checkpoint and final history."""
    output = ROOT / "artifacts/runs" / name
    checkpoint = output / "model_ep2.pt"
    selected = output / "model.pt"
    provenance_path = output / "provenance.json"
    history_path = output / "history.json"
    if not all(path.is_file() for path in (checkpoint, selected, provenance_path, history_path)):
        return False
    provenance = read_json(provenance_path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return (
        len(history) == 3
        and provenance.get("selected_epoch") == 2
        and provenance.get("checkpoint_sha256", {}).get("model_ep2.pt") == sha256_file(checkpoint)
        and provenance.get("checkpoint_sha256", {}).get("model.pt") == sha256_file(selected)
    )


def prediction_complete(
    path: Path,
    sidecar_path: Path,
    groups: Sequence[int],
    split_sha256: str,
    run: str,
    epoch: int,
    checkpoint_sha256: str,
) -> bool:
    """Check exact rows, checkpoint ancestry, array shape, dtype and finite values."""
    if not path.is_file() or not sidecar_path.is_file():
        return False
    metadata = read_json(sidecar_path)
    if not (
        metadata.get("run") == run
        and metadata.get("epoch") == epoch
        and metadata.get("checkpoint_sha256") == checkpoint_sha256
        and metadata.get("row_groups") == list(groups)
        and metadata.get("split_manifest_sha256") == split_sha256
        and metadata.get("campaign_split_manifest_sha256") == split_sha256
        and metadata.get("campaign_export_scope") == "evaluation"
        and metadata.get("feature_only") is True
        and metadata.get("prediction_sha256") == sha256_file(path)
    ):
        return False
    try:
        predictions = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return False
    if predictions.shape != (len(groups), SEQUENCE_LENGTH, TARGET_COUNT):
        return False
    if predictions.dtype != np.dtype("float32"):
        return False
    return all(
        np.isfinite(predictions[start : start + PREDICTION_CHUNK_GROUPS]).all()
        for start in range(0, len(groups), PREDICTION_CHUNK_GROUPS)
    )


def score_complete(path: Path, run: str, prediction_sha256: str, split_sha256: str) -> bool:
    """Check that an archived score belongs to the fixed official endpoint."""
    if not path.is_file():
        return False
    try:
        score = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    identity_matches = (
        score.get("metric_id") == "global_wp_clip2_v1"
        and score.get("run") == run
        and score.get("epoch") == 2
        and score.get("split_manifest_sha256") == split_sha256
        and score.get("prediction_sha256") == prediction_sha256
        and score.get("finite_fraction") == 1.0
    )
    try:
        finite_metrics = all(math.isfinite(float(score[metric])) for metric in ("wp", "t0", "t1"))
    except (KeyError, TypeError, ValueError):
        finite_metrics = False
    return identity_matches and finite_metrics


def verify_pair(fold: int, seed: int, expected_updates: int) -> dict[str, Any]:
    """Require matched initialization, fit membership, prior and evaluation rows."""
    names = {arm: run_name(fold, seed, arm) for arm in ("control", "features")}
    provenance = {
        arm: read_json(ROOT / "artifacts/runs" / name / "provenance.json")
        for arm, name in names.items()
    }
    left, right = provenance["control"], provenance["features"]
    for field in (
        "initial_parameter_sha256",
        "fit_membership_identity",
        "evaluation_row_groups",
        "split_manifest_sha256",
        "causal_feature_prior",
        "causal_feature_prior_source_sha256",
    ):
        if left.get(field) != right.get(field):
            raise ValueError(f"paired endpoints differ in {field}: fold={fold}, seed={seed}")
    if left.get("seed") != seed or right.get("seed") != seed:
        raise ValueError(f"paired endpoint seed mismatch: fold={fold}, seed={seed}")
    control_features = left.get("causal_feature_control", {})
    treatment_features = right.get("causal_feature_control", {})
    if control_features.get("kind") != "all" or control_features.get("zeroed_suffix_width") != 60:
        raise ValueError(
            f"control did not zero exactly 60 feature channels: fold={fold}, seed={seed}"
        )
    if (
        treatment_features.get("kind") != "none"
        or treatment_features.get("zeroed_suffix_width") != 0
    ):
        raise ValueError(f"treatment unexpectedly masks feature channels: fold={fold}, seed={seed}")
    if (
        left.get("split_manifest_sha256")
        != read_json(ADMISSION)["split_manifest_sha256"][
            f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
        ]
    ):
        raise ValueError(f"paired endpoint split hash mismatch: fold={fold}, seed={seed}")
    training_signatures = {
        arm: _training_signature(provenance[arm], expected_updates, fold, seed)
        for arm in ("control", "features")
    }
    if training_signatures["control"] != training_signatures["features"]:
        raise ValueError(f"paired sampler order or update counts differ: fold={fold}, seed={seed}")
    return {
        "initial_parameter_sha256": left["initial_parameter_sha256"],
        "fit_membership_identity": left["fit_membership_identity"],
        "split_manifest_sha256": left["split_manifest_sha256"],
        "prior_sha256": hashlib.sha256(
            json.dumps(left["causal_feature_prior"], sort_keys=True).encode()
        ).hexdigest(),
        "control": names["control"],
        "features": names["features"],
        "training_signature": training_signatures["control"],
    }


def _training_signature(
    provenance: dict[str, Any], expected_updates: int, fold: int, seed: int
) -> list[dict[str, Any]]:
    """Require matched per-epoch data order and the registered update count."""
    rows = provenance.get("update_diagnostics")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError(f"training diagnostics are incomplete: fold={fold}, seed={seed}")
    signature: list[dict[str, Any]] = []
    for row in rows:
        epoch = row.get("epoch", {})
        if not row.get("sample_order_identity"):
            raise ValueError(f"sample-order identity is missing: fold={fold}, seed={seed}")
        signature.append(
            {
                "epoch_index": row.get("epoch_index"),
                "sample_order_identity": row["sample_order_identity"],
                "optimizer_attempts": epoch.get("optimizer_attempts"),
                "optimizer_completed_updates": epoch.get("optimizer_completed_updates"),
                "amp_skipped_updates": epoch.get("amp_skipped_updates"),
                "completed_batches": epoch.get("completed_batches"),
            }
        )
    cumulative = provenance.get("cumulative_updates", {})
    if cumulative.get("optimizer_attempts") != expected_updates:
        raise ValueError(
            f"optimizer update count differs from fold {fold} plan: "
            f"seed={seed}, observed={cumulative.get('optimizer_attempts')}, "
            f"expected={expected_updates}"
        )
    return signature
