from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
PACKET = Path(__file__).resolve().parent
RECIPE_PATH = PACKET / "recipe.json"
FOLD_REL = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
SPLIT_SHA = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
INPUT_REL = "datasets/valid.parquet"
FEATURE_COLUMNS = [
    f"i{i}_{group}{column}"
    for i in range(2)
    for group, width in (("p", 22), ("v", 22), ("dp", 4), ("dv", 4))
    for column in range(width)
] + [f"a{i}" for i in range(8)]
FEATURE_GROUPS = (
    ("i0_bid_price_like", 0, 11),
    ("i0_ask_price_like", 11, 22),
    ("i0_bid_volume_like", 22, 33),
    ("i0_ask_volume_like", 33, 44),
    ("i0_trade_price", 44, 48),
    ("i0_trade_volume", 48, 52),
    ("i1_bid_price_like", 52, 63),
    ("i1_ask_price_like", 63, 74),
    ("i1_bid_volume_like", 74, 85),
    ("i1_ask_volume_like", 85, 96),
    ("i1_trade_price", 96, 100),
    ("i1_trade_volume", 100, 104),
    ("additional", 104, 112),
)
MAX_FOOTER_BYTES = 16 * 1024 * 1024
EXPECTED_GROUPS = 468
SEQUENCE_LENGTH = 20_000
WARMUP = 99


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading its payload into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_footer_size(path: Path) -> int:
    """Read the fixed-size Parquet trailer before asking Arrow to parse metadata."""
    with path.open("rb") as stream:
        stream.seek(-8, 2)
        trailer = stream.read(8)
    if len(trailer) != 8 or trailer[4:] != b"PAR1":
        raise ValueError(f"{path} does not end with a valid Parquet trailer")
    return struct.unpack("<I", trailer[:4])[0]


def feature_vector(values: np.ndarray) -> np.ndarray:
    """Summarize each declared input group using only its first 99 rows."""
    output = np.empty((len(FEATURE_GROUPS), 3), dtype=np.float32)
    for index, (_, start, end) in enumerate(FEATURE_GROUPS):
        part = np.asarray(values[:, start:end], dtype=np.float64)
        scale = float(part.std())
        early = float(part[:20].mean())
        late = float(part[-20:].mean())
        output[index] = (float(part.mean()), np.log1p(scale), (late - early) / max(scale, 1e-6))
    return output.reshape(-1)


def extract(root: Path, output_dir: Path) -> dict[str, object]:
    """Extract label-blind group summaries for the fixed F0 evaluation sequences."""
    root = root.resolve(strict=True)
    output_dir = output_dir.resolve()
    approved_scratch = Path("/home/vlad/development/wunder/research_tmp").resolve(strict=True)
    if not output_dir.is_relative_to(approved_scratch):
        raise ValueError(f"output directory must be under {approved_scratch}")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "warmup_features.npy"
    receipt_path = output_dir / "warmup_features_receipt.json"
    if feature_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite an existing warmup feature artifact")

    split_path = root / FOLD_REL
    if sha256_file(split_path) != SPLIT_SHA:
        raise ValueError("F0 split manifest hash differs from the frozen recipe")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    groups = [int(group) for group in split["evaluation"]["valid"]]
    if len(groups) != EXPECTED_GROUPS or len(set(groups)) != EXPECTED_GROUPS:
        raise ValueError("unexpected F0 group membership")
    sequences = {int(row["row_group"]): row for row in split["sequences"]["valid"]}

    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    expected_input = recipe["valid_input_source"]
    parquet_path = root / INPUT_REL
    if parquet_path.stat().st_size != expected_input["size_bytes"]:
        raise ValueError("validation Parquet size differs from the frozen recipe")
    footer_bytes = parquet_footer_size(parquet_path)
    if footer_bytes != expected_input["footer_metadata_bytes"]:
        raise ValueError("validation Parquet footer size differs from the frozen recipe")
    if footer_bytes > MAX_FOOTER_BYTES:
        raise MemoryError(f"Parquet footer is {footer_bytes} bytes; limit is {MAX_FOOTER_BYTES}")
    parquet_file = pq.ParquetFile(parquet_path, memory_map=True)
    if parquet_file.metadata.num_row_groups != expected_input["row_groups"]:
        raise ValueError("validation Parquet row-group count differs from the bounded profile")
    if parquet_file.schema.names[:2] != ["seq_ix", "step_in_seq"]:
        raise ValueError("validation Parquet columns differ from the expected input schema")
    if any(
        parquet_file.metadata.row_group(group).num_rows != expected_input["rows_per_group"]
        for group in groups
    ):
        raise ValueError("an evaluation row group is not one complete 20,000-row sequence")

    features = np.empty((EXPECTED_GROUPS, len(FEATURE_GROUPS) * 3), dtype=np.float32)
    sequence_ids: list[int] = []
    input_digest = hashlib.sha256()
    for row, group in enumerate(groups):
        expected_sequence = sequences[group]
        batches = parquet_file.iter_batches(
            batch_size=WARMUP,
            row_groups=[group],
            columns=["seq_ix", "step_in_seq", *FEATURE_COLUMNS],
            use_threads=False,
        )
        try:
            batch = next(batches)
        except StopIteration as error:
            raise ValueError(f"empty warmup batch for row group {group}") from error
        sequence_column = batch.column(batch.schema.get_field_index("seq_ix")).to_numpy()
        position_column = batch.column(batch.schema.get_field_index("step_in_seq")).to_numpy()
        if len(batch) != WARMUP or set(map(int, sequence_column)) != {
            int(expected_sequence["seq_ix"])
        }:
            raise ValueError(f"row group {group} does not match its frozen sequence identity")
        if not np.array_equal(position_column, np.arange(WARMUP)):
            raise ValueError(f"row group {group} does not begin at warmup position zero")
        matrix = np.column_stack(
            [
                batch.column(batch.schema.get_field_index(name)).to_numpy()
                for name in FEATURE_COLUMNS
            ]
        ).astype(np.float32, copy=False)
        if matrix.shape != (WARMUP, 112) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid warmup feature matrix for row group {group}")
        input_digest.update(struct.pack("<qq", group, int(expected_sequence["seq_ix"])))
        input_digest.update(np.asarray(sequence_column, dtype="<i8").tobytes())
        input_digest.update(np.asarray(position_column, dtype="<i8").tobytes())
        input_digest.update(np.asarray(matrix, dtype="<f4", order="C").tobytes())
        features[row] = feature_vector(matrix)
        sequence_ids.append(int(sequence_column[0]))
        del batches

    input_slice_sha256 = input_digest.hexdigest()
    expected_slice_sha256 = expected_input.get("warmup_slice_sha256")
    if expected_slice_sha256 is not None and input_slice_sha256 != expected_slice_sha256:
        raise ValueError("selected warmup input bytes differ from the frozen recipe")

    np.save(feature_path, features, allow_pickle=False)
    receipt: dict[str, object] = {
        "schema": "wnn.warmup_expert_gate.features.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "recipe_sha256": sha256_file(RECIPE_PATH),
        "split_manifest": FOLD_REL,
        "split_manifest_sha256": SPLIT_SHA,
        "fold": 0,
        "exposure": "repeatedly exposed F0; label-blind extraction only",
        "input": {
            "path": INPUT_REL,
            "size_bytes": parquet_path.stat().st_size,
            "footer_metadata_bytes": footer_bytes,
            "row_groups": parquet_file.metadata.num_row_groups,
            "rows_per_group": SEQUENCE_LENGTH,
            "warmup_slice_sha256": input_slice_sha256,
            "columns_read": ["seq_ix", "step_in_seq", *FEATURE_COLUMNS],
            "feature_rows_read": len(groups) * WARMUP,
            "targets_or_masks_read": False,
        },
        "membership": {
            "row_groups": groups,
            "sequence_ids": sequence_ids,
            "row_group_order_sha256": hashlib.sha256(
                json.dumps(groups, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
        },
        "features": {
            "path": str(feature_path),
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "groups": [name for name, _, _ in FEATURE_GROUPS],
            "statistics_per_group": ["mean", "log1p_std", "normalized_first20_to_last20_drift"],
            "sha256": sha256_file(feature_path),
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    """Run the bounded, target-free warmup feature extraction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = extract(args.root, args.output_dir)
    print(
        json.dumps(
            {"status": "features_extracted", "feature_sha256": receipt["features"]["sha256"]}
        )
    )


if __name__ == "__main__":
    main()
