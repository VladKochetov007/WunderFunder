"""Score the three pinned TL3 role caches without loading or running a model.

The script reads only existing role prediction arrays, checkpoint/provenance
metadata, the official split and validation targets. It never writes to caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from wnn.data import SEQUENCE_LENGTH, WARMUP
from wnn.gmetric import GlobalAccumulator
from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    CampaignSplit,
    file_sha256,
    load_campaign_split,
)

PRIMARY_SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
EVALUATION_MEMBERSHIP_SHA256 = "008f12e284e0f72ff423fee68f34c37054759bc4a345e9581620c1fe9b2acdf7"
EXPECTED_VALID_ROWS = 3_528_273
EXPECTED_ARRAY_BYTES = 299_680_128
REFERENCE_ANCESTRY_LIMIT = "published-legacy-consumed-cache-byte-ancestry-not-reconstructed"


@dataclass(frozen=True)
class RolePin:
    """Native receipt identities for one fixed TL3 component."""

    run: str
    epoch: int
    checkpoint_sha256: str
    prediction_sha256: str
    mixture_manifest_provenance_sha256: str


ROLE_PINS = (
    RolePin(
        "tl3c_w18final_distilled_s5",
        1,
        "a1934af24bd1520b78be874c294532b1e2d62acdd8bd1dc3d6b5c220b2dad107",
        "e9a664be18feec38563be730c8a1f64dbc2a9907ed0d4d4873309d674f78eeb6",
        "3a67347d5c1133860c5cd3f910e296a0a5ee956aca54ec68290a763cf2946457",
    ),
    RolePin(
        "tl3c_fttall_tlead_s5",
        3,
        "cf944dffac66d674894e4c589d839c407b3d28cb01d35a26f58ff08c7743f65f",
        "3c124a7e89110472b6b8fb878e09087fddf00a2b7f4983b8115c8acbd53c7352",
        "fae7831bb2549162fe0c5d90e0f97f94619e7f9fd391e8b9efb71ecb821f6340",
    ),
    RolePin(
        "tl3c_fttall_tlead_s6",
        3,
        "24a565b03e35dc36679bf809f63f605f24f693404bd0cfda58bfc82177a33598",
        "67d3c7804d571b1accf096d14b10b7c768d27cb87849654a36d04ed72b0588e6",
        "ad95ccc860b89c2bdb8f508a3fe74a05176a9e3226e8b9126eb65bc518161a52",
    ),
)


def _canonical_sha256(value: object) -> str:
    """Hash JSON metadata canonically without conflating it with file-byte hashes."""
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, maximum_bytes: int = 4 << 20) -> tuple[dict[str, Any], str]:
    """Read a small regular metadata file and return its raw byte digest."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink metadata file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"metadata file is empty or too large: {path}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"metadata file changed while being read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"metadata file must contain a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_split(split: CampaignSplit) -> list[int]:
    """Validate native membership and valid-source components without invented fields."""
    groups = list(split.evaluation_groups)
    component_valid_groups = [
        group for source, group, _component in split.components if source == "valid"
    ]
    if (
        split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
        or split.sha256 != PRIMARY_SPLIT_SHA256
        or len(split.train_groups) != 10_325
        or split.valid_groups
        or len(groups) != 1_873
        or list(split.train_groups) != list(range(10_325))
        or groups != list(range(1_873))
        or len(component_valid_groups) != len(groups)
        or set(component_valid_groups) != set(groups)
    ):
        raise ValueError("split is not the frozen 10,325/1,873 official chronological split")
    split.assert_fit_groups(split.train_groups, split.valid_groups)
    return groups


def _validate_run_provenance(
    value: dict[str, Any], split: CampaignSplit, pin: RolePin, checkpoint_file: str
) -> None:
    """Check fixed endpoint and source-qualified memberships from run provenance JSON."""
    fitting = value.get("fit_row_groups")
    args = value.get("args")
    hashes = value.get("checkpoint_sha256")
    if (
        value.get("split_manifest_sha256") != split.sha256
        or not isinstance(fitting, dict)
        or fitting.get("train") != list(split.train_groups)
        or fitting.get("valid") != []
        or value.get("evaluation_row_groups") != list(split.evaluation_groups)
        or not isinstance(args, dict)
        or args.get("name") != pin.run
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != pin.epoch
        or args.get("no_eval_during_fit") is not False
        or args.get("train_source") != "train"
        or args.get("split_manifest_sha256") != split.sha256
        or not isinstance(hashes, dict)
        or hashes.get(checkpoint_file) != pin.checkpoint_sha256
    ):
        raise ValueError(f"run provenance does not bind the fixed train-only role: {pin.run}")


def _load_role(
    project_root: Path, split: CampaignSplit, groups: list[int], pin: RolePin
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bind one existing role array to its sidecar, checkpoint and run provenance."""
    cache_root = project_root / "cache"
    run_root = project_root / "artifacts" / "runs" / pin.run
    array_path = cache_root / f"pred_tl3ref_{pin.run}_valid.npy"
    metadata_path = cache_root / f"pred_tl3ref_{pin.run}_valid.json"
    checkpoint_file = f"model_ep{pin.epoch}.pt"
    checkpoint_path = run_root / checkpoint_file
    provenance_path = run_root / "provenance.json"
    metadata, metadata_sha256 = _read_json(metadata_path)
    provenance, provenance_file_sha256 = _read_json(provenance_path)
    if (
        metadata.get("run") != pin.run
        or metadata.get("epoch") != pin.epoch
        or metadata.get("checkpoint_file") != checkpoint_file
        or metadata.get("checkpoint_sha256") != pin.checkpoint_sha256
        or metadata.get("prediction_sha256") != pin.prediction_sha256
        or metadata.get("split_manifest_sha256") != split.sha256
        or metadata.get("evaluation_membership_sha256") != EVALUATION_MEMBERSHIP_SHA256
        or metadata.get("inference_precision") != "fp32"
        or metadata.get("campaign_export_scope") != "evaluation"
        or metadata.get("feature_only") is not True
        or metadata.get("row_groups") != groups
        or metadata.get("run_provenance_sha256") != provenance_file_sha256
    ):
        raise ValueError(f"role sidecar differs from its frozen native identity: {pin.run}")
    _validate_run_provenance(provenance, split, pin, checkpoint_file)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError(f"role checkpoint is missing or unsafe: {checkpoint_path}")
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if checkpoint_sha256 != pin.checkpoint_sha256:
        raise ValueError(f"role checkpoint bytes differ from the frozen pin: {pin.run}")
    if array_path.is_symlink() or not array_path.is_file():
        raise ValueError(f"role prediction array is missing or unsafe: {array_path}")
    if array_path.stat().st_size != EXPECTED_ARRAY_BYTES:
        raise ValueError(f"role prediction array has an unexpected byte size: {pin.run}")
    prediction_sha256 = file_sha256(array_path)
    if prediction_sha256 != pin.prediction_sha256:
        raise ValueError(f"role prediction bytes differ from the frozen pin: {pin.run}")
    predictions = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if predictions.dtype != np.float32 or predictions.shape != (len(groups), SEQUENCE_LENGTH, 2):
        raise ValueError(f"role prediction array has an unexpected shape or dtype: {pin.run}")
    return predictions, {
        "run": pin.run,
        "epoch": pin.epoch,
        "checkpoint_file": checkpoint_file,
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_path": str(array_path),
        "prediction_sha256": prediction_sha256,
        "sidecar_path": str(metadata_path),
        "sidecar_file_sha256": metadata_sha256,
        "run_provenance_path": str(provenance_path),
        "run_provenance_file_sha256": provenance_file_sha256,
        "run_provenance_canonical_sha256": _canonical_sha256(provenance),
        "mixture_manifest_provenance_sha256": pin.mixture_manifest_provenance_sha256,
        "evaluation_membership_sha256": EVALUATION_MEMBERSHIP_SHA256,
    }


def _score(
    project_root: Path,
    split: CampaignSplit,
    groups: list[int],
    predictions: list[np.ndarray],
) -> tuple[dict[str, dict[str, Any]], int]:
    """Stream full-valid targets and accumulate standalone role scores plus fixed blend."""
    roles = [pin.run for pin in ROLE_PINS]
    accumulators = {name: GlobalAccumulator() for name in (*roles, "fixed_tl3_blend")}
    parquet_path = project_root / "datasets" / "valid.parquet"
    if parquet_path.is_symlink() or not parquet_path.is_file():
        raise ValueError("official valid.parquet must be a regular non-symlink file")
    parquet = pq.ParquetFile(parquet_path)
    if not {"t0", "t1", "is_scored"}.issubset(parquet.schema_arrow.names):
        raise ValueError("official valid.parquet must expose targets and the official is_scored mask")
    selected_rows = 0
    for position, group in enumerate(groups):
        table = parquet.read_row_group(group, columns=["t0", "t1", "is_scored"], use_threads=False)
        if table.num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"validation row group {group} is not one complete sequence")
        targets = np.column_stack(
            [table[name].to_numpy() for name in ("t0", "t1")]
        ).astype(np.float32, copy=False)
        mask = table["is_scored"].to_numpy().astype(bool, copy=True)
        mask[:WARMUP] = False
        selected_rows += int(mask.sum())
        for name, values in zip(roles, predictions, strict=True):
            prediction = values[position]
            if not np.isfinite(prediction[WARMUP:]).all():
                raise ValueError(f"nonfinite post-warmup prediction in {name}, group {group}")
            accumulators[name].add(targets, prediction, mask)
        blend = (
            np.float32(0.34) * np.asarray(predictions[0][position], dtype=np.float32)
        )
        blend += np.float32(0.33) * np.asarray(predictions[1][position], dtype=np.float32)
        blend += np.float32(0.33) * np.asarray(predictions[2][position], dtype=np.float32)
        accumulators["fixed_tl3_blend"].add(targets, blend, mask)
    if selected_rows != EXPECTED_VALID_ROWS:
        raise ValueError("selected official-valid rows differ from the frozen 3,528,273 count")
    return {name: acc.result() for name, acc in accumulators.items()}, selected_rows


def grade(project_root: Path, split_manifest: Path) -> dict[str, Any]:
    """Validate cached role inputs and write no files while computing the scores."""
    project_root = project_root.resolve(strict=True)
    split_manifest = split_manifest.resolve(strict=True)
    split = load_campaign_split(split_manifest, project_root)
    groups = _validate_split(split)
    loaded = [_load_role(project_root, split, groups, pin) for pin in ROLE_PINS]
    arrays = [item[0] for item in loaded]
    role_records = [item[1] for item in loaded]
    scores, selected_rows = _score(project_root, split, groups, arrays)
    for pin, record in zip(ROLE_PINS, role_records, strict=True):
        if file_sha256(Path(record["sidecar_path"])) != record["sidecar_file_sha256"]:
            raise ValueError(f"role sidecar changed while being scored: {pin.run}")
        if file_sha256(Path(record["run_provenance_path"])) != record["run_provenance_file_sha256"]:
            raise ValueError(f"role provenance changed while being scored: {pin.run}")
        if file_sha256(Path(record["prediction_path"])) != pin.prediction_sha256:
            raise ValueError(f"role prediction changed while being scored: {pin.run}")
        if file_sha256(
            project_root / "artifacts" / "runs" / pin.run / record["checkpoint_file"]
        ) != pin.checkpoint_sha256:
            raise ValueError(f"role checkpoint changed while being scored: {pin.run}")
    return {
        "schema": "wnn.tl3-role-cache-score.v1",
        "split_manifest_sha256": split.sha256,
        "valid_sequences": len(groups),
        "selected_rows": selected_rows,
        "members": role_records,
        "scores": scores,
        "fixed_blend": {
            "weights": [0.34, 0.33, 0.33],
            "arithmetic": "fp32-left-to-right-0.34-0.33-0.33-no-clip",
        },
        "cache_files_written": False,
        "inputs_unchanged_during_score": True,
        "model_loaded_or_run": False,
        "validation_exposure": "exposed official valid; development evidence only",
        "provenance_hash_interpretation": (
            "raw run-provenance file SHA is matched to its cache sidecar; the separate "
            "canonical provenance pin recorded by the mixture manifest is reported, "
            "not assumed equal to the run file or its canonical JSON hash"
        ),
        "training_prediction_ancestry_limit": REFERENCE_ANCESTRY_LIMIT,
        "independent_score_claim": "not_claimed",
    }


def main() -> None:
    """Run the bounded read-only score pass and create one create-only receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = grade(args.project_root, args.split_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

