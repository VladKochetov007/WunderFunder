from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from wnn.metric import WARMUP, score  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Create a score receipt without replacing prior evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite score receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def grade(root: Path, run: str, fold: int, recipe: Path, output: Path) -> dict[str, Any]:
    """Score one fixed checkpoint on exactly its frozen evaluation groups."""
    split_path = root / "research/campaign_2026-09-29/artifacts/splits" / f"fold{fold}.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("fold") != fold or split.get("n_folds") != 4:
        raise ValueError("invalid source-qualified fold manifest")
    groups = [int(value) for value in split["evaluation"]["valid"]]
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("evaluation groups must be nonempty and unique")
    prediction_path = root / "cache" / f"pred_{run}_valid.npy"
    metadata_path = prediction_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    os.environ["WNN_PROJECT_ROOT"] = str(root)
    verified_split = load_campaign_split(split_path, root)
    split_sha = verified_split.sha256
    if list(verified_split.evaluation_groups) != groups:
        raise ValueError("JSON and revalidated component membership disagree")
    if (
        metadata.get("campaign_split_manifest_sha256") != split_sha
        or metadata.get("campaign_export_scope") != "evaluation"
        or metadata.get("feature_only") is not True
        or metadata.get("row_groups") != groups
    ):
        raise ValueError("prediction is not bound to the exact frozen evaluation membership")
    recipe_value = json.loads(recipe.read_text(encoding="utf-8"))
    epoch = recipe_value["training"]["checkpoint_epoch"]
    if metadata.get("epoch") != epoch:
        raise ValueError("prediction did not use the fixed recipe checkpoint")
    with np.load(root / "cache/valid_targets.npz", allow_pickle=False) as cache:
        available = np.asarray(cache["row_group"], dtype=np.int64)
        position = {int(group): index for index, group in enumerate(available)}
        if len(position) != len(available) or not set(groups).issubset(position):
            raise ValueError("target cache row-group index is invalid or incomplete")
        indexes = np.asarray([position[group] for group in groups], dtype=np.int64)
        targets = np.asarray(cache["y"][indexes], dtype=np.float32)
        scored = np.asarray(cache["mask"][indexes], dtype=bool)
    predictions = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if predictions.shape != targets.shape or predictions.shape != (len(groups), 20_000, 2):
        raise ValueError("prediction shape does not match evaluation targets")
    checkpoint = root / "artifacts/runs" / run / f"model_ep{epoch}.pt"
    if (
        metadata.get("prediction_sha256") != sha256(prediction_path)
        or metadata.get("checkpoint_sha256") != sha256(checkpoint)
        or metadata.get("inference_precision") != "fp32"
        or metadata.get("quantization_preset") is not None
    ):
        raise ValueError("prediction or fixed checkpoint bytes/precision differ from sidecar")
    mask = scored & (np.arange(targets.shape[1])[None, :] >= WARMUP)
    result = score(targets, np.asarray(predictions, dtype=np.float32), mask)
    group_bytes = np.asarray(groups, dtype="<i8").tobytes()
    mask_bytes = np.asarray(mask, dtype=np.uint8).tobytes()
    receipt = {
        "schema": "wnn.causal-smoothing-cell-score.v1",
        "run": run,
        "fold": fold,
        "metric": result,
        "evaluation_groups": groups,
        "evaluation_group_count": len(groups),
        "evaluation_membership_sha256": hashlib.sha256(group_bytes).hexdigest(),
        "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "mask_selection": "need_prediction (step >= 99) AND is_scored",
        "warmup_rows": WARMUP,
        "scored_rows": int(mask.sum()),
        "scored_rows_by_group": mask.sum(axis=1).astype(int).tolist(),
        "target_cache": str(root / "cache/valid_targets.npz"),
        "target_cache_bytes": (root / "cache/valid_targets.npz").stat().st_size,
        "target_cache_mtime_ns": (root / "cache/valid_targets.npz").stat().st_mtime_ns,
        "verified_source_identities": {
            "sha256": split_sha,
            "sources": split.get("sources", {}),
            "fit_train_groups": len(verified_split.train_groups),
            "fit_valid_groups": len(verified_split.valid_groups),
            "evaluation_groups": len(verified_split.evaluation_groups),
        },
        "prediction": str(prediction_path),
        "prediction_sha256": sha256(prediction_path),
        "prediction_metadata_sha256": sha256(metadata_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": split_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "recipe_sha256": sha256(recipe),
    }
    write_json(output, receipt)
    return receipt


def main() -> None:
    """Grade one previously predicted campaign cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = grade(
        args.root.resolve(), args.run, args.fold, args.recipe.resolve(), args.output.resolve()
    )
    print(
        json.dumps(
            {"score": receipt["metric"], "scored_rows": receipt["scored_rows"]}, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
