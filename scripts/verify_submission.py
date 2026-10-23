"""Run a submission callback, stream official WP, and preserve all predictions.

    python scripts/verify_submission.py SUBMISSION.zip DATA.parquet OUT.npz

NPZ output contains every sequence's ``pred``, ``y`` and ``mask`` arrays. NPY
output stores predictions plus adjacent ``.targets.npy`` and ``.mask.npy`` files.
Arrays are backed by temporary disk mappings while scoring, so memory does not
grow with the number of sequences. ``--campaign-split-manifest`` selects the
manifest's source-qualified evaluation groups in their frozen order. Legacy
block WP is an optional diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import (  # noqa: E402
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    BlockAccumulator,
    DataPoint,
    GlobalAccumulator,
    validate_sequence,
)
from wnn.splits import load_campaign_split  # noqa: E402

_LOG = logging.getLogger(__name__)


def global_wp(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> list[float]:
    """Return both official target scores for compatibility with older callers."""
    accumulator = GlobalAccumulator()
    accumulator.add(y, p, m)
    result = accumulator.result()
    return [float(result["t0"]), float(result["t1"])]


def _campaign_selection(
    manifest_path: Path, data: Path, row_group_count: int
) -> tuple[tuple[int, ...], dict[int, int], dict[str, object], str, tuple[int, ...]]:
    """Load the frozen evaluation order and bind it to validation sequence IDs."""
    expected_data = (ROOT / "datasets/valid.parquet").resolve(strict=True)
    if data.resolve(strict=True) != expected_data:
        raise ValueError("campaign replay requires the exact datasets/valid.parquet source")
    split = load_campaign_split(manifest_path, ROOT)
    raw_manifest = split.path.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != split.sha256:
        raise ValueError("campaign split manifest changed after source verification")
    manifest = json.loads(raw_manifest)
    data_identity = manifest["sources"].get("datasets/valid.parquet")
    if data_identity is None:
        raise ValueError("campaign manifest does not bind datasets/valid.parquet")
    valid_sequences = manifest["sequences"]["valid"]
    sequence_by_group = {int(item["row_group"]): int(item["seq_ix"]) for item in valid_sequences}
    if len(sequence_by_group) != len(valid_sequences):
        raise ValueError("duplicate valid row-group identity in the campaign manifest")
    if len(set(sequence_by_group.values())) != len(valid_sequences):
        raise ValueError("duplicate valid sequence ID in the campaign manifest")
    groups = tuple(map(int, split.evaluation_groups))
    if not groups or any(group < 0 or group >= row_group_count for group in groups):
        raise ValueError("campaign evaluation group is outside the validation Parquet")
    try:
        sequence_ids = tuple(sequence_by_group[group] for group in groups)
    except KeyError as error:
        raise ValueError("campaign evaluation group lacks a sequence ID") from error
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("duplicate evaluation sequence ID in the campaign manifest")
    source: dict[str, object] = {
        "data": str(expected_data),
        "data_identity": data_identity,
        "campaign_split": str(split.path),
        "fold": split.fold,
        "n_folds": split.n_folds,
    }
    return groups, sequence_by_group, source, split.sha256, sequence_ids


def _output_paths(out: Path, receipt: Path | None) -> tuple[Path, ...]:
    """Return every create-only output path, rejecting collisions and old files."""
    paths = (
        (out,)
        if out.suffix == ".npz"
        else (
            out,
            out.with_suffix(".targets.npy"),
            out.with_suffix(".mask.npy"),
        )
    )
    if receipt is not None:
        paths += (receipt,)
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("prediction artifacts and receipt must use distinct paths")
    existing = next((path for path in paths if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"refusing to replace artifact: {existing}")
    return paths


def verify_submission(
    zpath: Path,
    data: Path,
    out: Path,
    campaign_split_manifest: Path | None = None,
    receipt: Path | None = None,
) -> dict[str, object]:
    """Replay all rows or one frozen campaign evaluation split to disk artifacts."""
    if out.suffix not in {".npz", ".npy"}:
        raise ValueError("prediction output must end in .npz or .npy")
    output_paths = _output_paths(out, receipt)
    if campaign_split_manifest is not None:
        expected_data = (ROOT / "datasets/valid.parquet").resolve(strict=True)
        if data.resolve(strict=True) != expected_data:
            raise ValueError("campaign replay requires the exact datasets/valid.parquet source")
    pf = pq.ParquetFile(data)
    if campaign_split_manifest is None:
        row_groups = tuple(range(pf.num_row_groups))
        sequence_by_group: dict[int, int] | None = None
        source: dict[str, object] = {"data": str(data.resolve())}
        split_sha256 = None
        sequence_ids: tuple[int, ...] | None = None
    else:
        row_groups, sequence_by_group, source, split_sha256, sequence_ids = _campaign_selection(
            campaign_split_manifest, data, pf.num_row_groups
        )
    if not row_groups:
        raise ValueError("no selected rows")
    out.parent.mkdir(parents=True, exist_ok=True)
    if receipt is not None:
        receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        with zipfile.ZipFile(zpath) as archive:
            archive.extractall(tmp)
        sys.path.insert(0, str(tmp))
        try:
            sys.modules.pop("solution", None)
            t0 = time.perf_counter()
            model = importlib.import_module("solution").PredictionModel()
            load_s = time.perf_counter() - t0
            shape = (len(row_groups), SEQUENCE_LENGTH, 2)
            predictions = np.lib.format.open_memmap(
                tmp / "pred.npy", mode="w+", dtype=np.float32, shape=shape
            )
            targets = np.lib.format.open_memmap(
                tmp / "y.npy", mode="w+", dtype=np.float32, shape=shape
            )
            masks = np.lib.format.open_memmap(
                tmp / "mask.npy", mode="w+", dtype=bool, shape=shape[:2]
            )
            accumulator = GlobalAccumulator()
            block_accumulator = BlockAccumulator()
            diagnostic_error = None
            rows, busy = 0, 0.0
            seen = set()
            for output_index, group in enumerate(row_groups):
                table = pf.read_row_group(group, use_threads=False)
                seq, need = validate_sequence(table)
                if sequence_by_group is not None and sequence_by_group[group] != seq:
                    raise ValueError(
                        f"row group {group} has sequence ID {seq}, expected "
                        f"{sequence_by_group[group]} from the campaign manifest"
                    )
                if seq in seen:
                    raise ValueError("duplicate sequence ID")
                seen.add(seq)
                features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(
                    np.float32
                )
                y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(
                    np.float32
                )
                p = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
                started = time.perf_counter()
                for step in range(SEQUENCE_LENGTH):
                    value = model.predict(DataPoint(seq, step, bool(need[step]), features[step]))
                    if need[step]:
                        value = np.asarray(value, dtype=np.float32)
                        if value.shape != (2,) or not np.isfinite(value).all():
                            raise ValueError("return two finite scores on every required row")
                        p[step] = value
                    elif value is not None:
                        raise ValueError("warm-up must return None after updating state")
                busy += time.perf_counter() - started
                rows += SEQUENCE_LENGTH
                mask = table["is_scored"].to_numpy().astype(bool) & need
                accumulator.add(y, p, mask)
                if diagnostic_error is None:
                    try:
                        block_accumulator.add(y, p, mask)
                    except ValueError as error:
                        diagnostic_error = str(error)
                predictions[output_index], targets[output_index], masks[output_index] = p, y, mask
            official = accumulator.result()
            block_wp = None
            if diagnostic_error is None:
                if block_accumulator.scores:
                    block_wp = float(block_accumulator.result()["weighted_pearson"])
                else:
                    diagnostic_error = "no eligible blocks"
            for array in (predictions, targets, masks):
                array.flush()
            if out.suffix == ".npz":
                # NumPy writes contiguous memmaps in chunks without stacking arrays.
                with out.open("xb") as stream:
                    np.savez(stream, pred=predictions, y=targets, mask=masks)
            else:
                for source_path, target_path in zip(
                    (tmp / "pred.npy", tmp / "y.npy", tmp / "mask.npy"),
                    output_paths[:3],
                    strict=True,
                ):
                    with (
                        target_path.open("xb") as destination,
                        source_path.open("rb") as source_stream,
                    ):
                        shutil.copyfileobj(source_stream, destination)
            result: dict[str, object] = {
                "metric_id": str(official["metric_id"]),
                "load_s": round(load_s, 2),
                "us_per_row": round(1e6 * busy / rows, 1),
                "rows": rows,
                "sequences": len(seen),
                "selected_rows": int(official["selected_rows"]),
                "block_wp": None if block_wp is None else round(block_wp, 6),
                "block_diagnostic_error": diagnostic_error,
                "global_wp": round(float(official["weighted_pearson"]), 6),
                "global_t0": round(float(official["t0"]), 6),
                "global_t1": round(float(official["t1"]), 6),
                "eta_min_39M_rows": round(39.4e6 * busy / rows / 60, 1),
                "row_groups": list(row_groups),
                "source": source,
                "split_sha256": split_sha256,
                "sequence_ids": None if sequence_ids is None else list(sequence_ids),
                "official_result": official,
            }
            if receipt is not None:
                with receipt.open("x") as stream:
                    json.dump(result, stream, allow_nan=False, indent=2)
                    stream.write("\n")
            return result
        finally:
            sys.path.remove(str(tmp))
            sys.modules.pop("solution", None)


def main() -> None:
    """Run the submission verification CLI and log its JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--campaign-split-manifest", type=Path)
    parser.add_argument("--receipt", type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    _LOG.info(
        json.dumps(
            verify_submission(
                arguments.submission,
                arguments.data,
                arguments.output,
                arguments.campaign_split_manifest,
                arguments.receipt,
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
