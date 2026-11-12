from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from reference_assets import audit_ancestors, bank_path, write_new_json

from wnn.splits import CampaignSplit, file_sha256, load_campaign_split
from wnn.training.sequence_reference_contract import canonical_sha256, read_object
from wnn.training.sequence_reference_pins import (
    COMPOSITION,
    PR49_PREDICTOR_SHA256,
    PR49_SOURCE_COMMIT,
    PUBLISHED_RUNS,
)

# Legacy sidecars mix hashes, arguments, row lists and booleans at this import boundary.


def load_bank(
    data_root: Path, split: CampaignSplit, run: str, source: str, producer: dict[str, Any]
) -> tuple[np.ndarray, dict[int, int], dict[str, Any]]:
    """Verify the actual legacy bank and its pinned, complete export receipt."""
    path = bank_path(data_root, run, source)
    if path.is_symlink() or not path.is_file():
        raise ValueError("reference bank must be a regular non-symlink file")
    meta, metadata_hash = read_object(path.with_suffix(".json"))
    receipt, receipt_hash = read_object(path.with_suffix(".export.json"))
    prediction_hash = file_sha256(path)
    if (
        receipt.get("schema") != "pr49-reference-export-v1"
        or receipt.get("complete") is not True
        or receipt.get("run") != run
        or receipt.get("source") != source
        or receipt.get("source_commit") != PR49_SOURCE_COMMIT
        or receipt.get("predictor_sha256") != PR49_PREDICTOR_SHA256
        or receipt.get("producer") != producer
        or receipt.get("reference_sha256") != canonical_sha256(producer)
        or receipt.get("metadata_sha256") != metadata_hash
        or receipt.get("prediction_sha256") != prediction_hash
    ):
        raise ValueError("legacy reference export receipt/source/array binding mismatch")
    pinned = PUBLISHED_RUNS[run]
    if (
        meta.get("run") != run
        or meta.get("split") != source
        or meta.get("epoch") != pinned["epoch"]
        or meta.get("checkpoint_sha256") != pinned["checkpoint_sha256"]
        or meta.get("checkpoint_file") != f"model_ep{pinned['epoch']}.pt"
        or meta.get("run_provenance_sha256") != pinned["provenance_sha256"]
        or meta.get("prediction_sha256") != prediction_hash
        or meta.get("split_manifest_sha256") != split.sha256
        or meta.get("campaign_split_manifest_sha256") != split.sha256
        or meta.get("inference_precision") != "fp32"
        or meta.get("feature_only") is not True
        or meta.get("campaign_export_scope")
        != ("chronological-full-train-index-feature-only" if source == "train" else "evaluation")
        or meta.get("quote_cache") is not None
        or meta.get("quantization_preset") is not None
        or meta.get("recurrent_rank") is not None
        or meta.get("gate_mode") != "soft"
    ):
        raise ValueError("legacy prediction sidecar is not the frozen FP32 full-row endpoint")
    groups = meta.get("row_groups")
    expected = (
        {g for s, g, _ in split.components if s == "train"}
        if source == "train"
        else set(split.evaluation_groups)
    )
    if (
        not isinstance(groups, list)
        or any(type(g) is not int or g < 0 for g in groups)
        or len(set(groups)) != len(groups)
        or set(groups) != expected
    ):
        raise ValueError("legacy reference bank must cover its full source-qualified export set")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (len(groups), 20000, 2):
        raise ValueError("legacy reference bank shape or dtype differs from full FP32 output")
    evidence = {
        "run": run,
        "source": source,
        "prediction_sha256": prediction_hash,
        "metadata_sha256": metadata_hash,
        "export_receipt_sha256": receipt_hash,
        "row_groups_sha256": canonical_sha256(groups),
    }
    return array, {g: i for i, g in enumerate(groups)}, evidence


def materialize(data_root: Path, split: CampaignSplit, output: Path) -> None:
    """Write both reference cohorts by explicit row-group joins, never array truncation."""
    producer = audit_ancestors(data_root, split)
    banks = {
        source: [load_bank(data_root, split, r["run"], source, producer) for r in COMPOSITION]
        for source in ("train", "valid")
    }
    output.mkdir(parents=True, exist_ok=False)
    for membership, source, groups in (
        ("fit", "train", split.train_groups),
        ("evaluation", "valid", split.evaluation_groups),
    ):
        folder = output / membership
        folder.mkdir()
        entries = []
        for group in groups:
            arrays = [np.asarray(array[index[group]]) for array, index, _ in banks[source]]
            if any(not np.isfinite(array).all() for array in arrays):
                raise ValueError("reference bank contains a nonfinite full-row prediction")
            mixed = np.float32(0.34) * arrays[0]
            mixed += np.float32(0.33) * arrays[1]
            mixed += np.float32(0.33) * arrays[2]
            if not np.isfinite(mixed).all():
                raise ValueError("reference blend overflowed")
            name = f"{source}_{group}.npy"
            with (folder / name).open("xb") as stream:
                np.save(stream, mixed, allow_pickle=False)
            entries.append(
                {
                    "split": source,
                    "row_group": group,
                    "file": name,
                    "sha256": file_sha256(folder / name),
                }
            )
        # A failed post-pass check leaves arrays without a complete manifest.
        for array, index, evidence in banks[source]:
            del array, index
            path = bank_path(data_root, evidence["run"], source)
            if (
                file_sha256(path) != evidence["prediction_sha256"]
                or file_sha256(path.with_suffix(".json")) != evidence["metadata_sha256"]
                or file_sha256(path.with_suffix(".export.json"))
                != evidence["export_receipt_sha256"]
            ):
                raise ValueError("reference producer files changed during materialization")
        if audit_ancestors(data_root, split) != producer:
            raise ValueError("reference ancestors changed during materialization")
        write_new_json(
            folder / "manifest.json",
            {
                "schema": "sequence-reference-cache-v1",
                "complete": True,
                "membership": membership,
                "sequence_length": 20000,
                "prediction_width": 2,
                "hint_dim": 0,
                "dtype": "float32",
                "split_sha256": split.sha256,
                "reference_sha256": canonical_sha256(producer),
                "producer": producer,
                "legacy_banks": [evidence for _, _, evidence in banks[source]],
                "sequences": entries,
            },
        )


def main() -> None:
    """Import six checked PR49 full prediction banks into two native reference caches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    split = load_campaign_split(args.split_manifest, data_root)
    materialize(data_root, split, args.output_root)


if __name__ == "__main__":
    main()
