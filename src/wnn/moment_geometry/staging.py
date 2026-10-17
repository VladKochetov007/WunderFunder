"""Freeze component-disjoint membership and stage bounded disk arrays for a pilot."""

from __future__ import annotations

import argparse
import json
import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.data import FEATURE_COLUMNS, TARGET_COLUMNS
from wnn.features import event_features
from wnn.moment_geometry.features import EXTRA_WIDTH, geometry_features, infer_geometry
from wnn.moment_geometry.membership import freeze_membership
from wnn.moment_geometry.provenance import bind_sources, validate_storage
from wnn.moment_geometry.staging_resources import (
    advise_file_range,
    estimate_staging_bytes,
    flush_and_discard_sequence,
)

LOGGER = logging.getLogger(__name__)
MAX_MEMBERSHIP_JSON_BYTES = 2_000_000
STAGING_RESERVE_BYTES = 5_000_000_000


def _load_membership(path: Path) -> dict[str, object]:
    """Read a bounded, previously frozen membership JSON file."""
    size = path.stat().st_size
    if size <= 0 or size > MAX_MEMBERSHIP_JSON_BYTES:
        raise ValueError("membership JSON exceeds its bounded read size")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("fit"), list):
        raise ValueError("membership JSON is missing fitting rows")
    if not isinstance(value.get("eval"), list):
        raise ValueError("membership JSON is missing evaluation rows")
    return value


def _advise_parquet_row_group(
    reader: pq.ParquetFile, descriptor: int, group: int, columns: list[str]
) -> None:
    """Discard only compressed column chunks read from one source row group."""
    metadata = reader.metadata.row_group(group)
    requested = set(columns)
    ranges = []
    found = set()
    for index in range(metadata.num_columns):
        column = metadata.column(index)
        if column.path_in_schema not in requested:
            continue
        start = column.data_page_offset
        if column.dictionary_page_offset >= 0:
            start = min(start, column.dictionary_page_offset)
        ranges.append((start, start + column.total_compressed_size))
        found.add(column.path_in_schema)
    if found != requested:
        raise ValueError("requested Parquet column chunk was absent from row-group metadata")
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    for start, end in merged:
        advise_file_range(descriptor, start, end - start)


def stage_raw(root: Path, output: Path, membership: dict[str, object]) -> None:
    """Read one Parquet sequence at a time and stage source-qualified arrays on disk."""
    fit_count, eval_count = len(membership["fit"]), len(membership["eval"])
    estimate = estimate_staging_bytes(fit_count, eval_count)
    required = int(estimate["persistent_total_bytes"]) + STAGING_RESERVE_BYTES
    validate_storage(output, required_bytes=required)
    binding = bind_sources(root, membership)
    (output / "source_binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    arrays: dict[tuple[str, str], np.memmap] = {}
    array_paths: dict[tuple[str, str], Path] = {}
    for role in ("fit", "eval"):
        count = len(membership[role])
        for name, width in (("raw", 112), ("y", 2)):
            path = output / f"{role}_{name}.npy"
            arrays[role, name] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(count, 20_000, width)
            )
            array_paths[role, name] = path
        mask_path = output / f"{role}_mask.npy"
        arrays[role, "mask"] = np.lib.format.open_memmap(
            mask_path, mode="w+", dtype=bool, shape=(count, 20_000)
        )
        array_paths[role, "mask"] = mask_path
        np.save(output / f"{role}_component.npy", [row[2] for row in membership[role]])
        np.save(output / f"{role}_fold.npy", [row[3] for row in membership[role]])

    for split in ("train", "valid"):
        source_path = root / "datasets" / f"{split}.parquet"
        reader = pq.ParquetFile(source_path)
        descriptor = os.open(source_path, os.O_RDONLY)
        columns = ["seq_ix", *FEATURE_COLUMNS, *TARGET_COLUMNS, "need_prediction"]
        if split == "valid":
            columns.append("is_scored")
        try:
            for role in ("fit", "eval"):
                for index, (source, group, _, _) in enumerate(membership[role]):
                    if source != split:
                        continue
                    table = reader.read_row_group(group, columns=columns, use_threads=False)
                    expected_seq_ix = binding["selected_sequences"][role][index]["seq_ix"]
                    actual_ids = table["seq_ix"].to_numpy()
                    if not np.all(actual_ids == expected_seq_ix):
                        raise ValueError("source-qualified sequence identity changed")
                    if table.num_rows != 20_000:
                        raise ValueError("incomplete source sequence")
                    raw = np.column_stack([table[column].to_numpy() for column in FEATURE_COLUMNS])
                    targets = np.column_stack(
                        [table[column].to_numpy() for column in TARGET_COLUMNS]
                    )
                    mask = table["need_prediction"].to_numpy().copy()
                    if role == "eval":
                        mask &= table["is_scored"].to_numpy()
                    arrays[role, "raw"][index] = raw
                    arrays[role, "y"][index] = targets
                    arrays[role, "mask"][index] = mask
                    for name in ("raw", "y", "mask"):
                        flush_and_discard_sequence(
                            array_paths[role, name], arrays[role, name], index, durable=True
                        )
                    del actual_ids, mask, raw, targets, table
                    _advise_parquet_row_group(reader, descriptor, group, columns)
        finally:
            reader.close()
            os.close(descriptor)
    LOGGER.info(
        "Staged %d fit and %d eval sequences; projected persistent output %.2f GB",
        fit_count,
        eval_count,
        int(estimate["persistent_total_bytes"]) / 1e9,
    )


def stage_features(output: Path) -> dict[str, object]:
    """Fit constants on fitting sequences, then transform each causal history independently."""
    fit_path = output / "fit_raw.npy"
    fit = np.load(fit_path, mmap_mode="r")
    eval_raw = np.load(output / "eval_raw.npy", mmap_mode="r")
    estimate = estimate_staging_bytes(len(fit), len(eval_raw))
    required = (
        int(estimate["base_features_incremental_bytes"])
        + int(estimate["discrete_features_incremental_bytes"])
        + STAGING_RESERVE_BYTES
    )
    validate_storage(output, required_bytes=required)

    beta, intercept, atoms = infer_geometry(
        fit, after_sequence=partial(flush_and_discard_sequence, fit_path, fit, durable=False)
    )
    total, sums, squares = 0, np.zeros(231), np.zeros(231)
    for index, raw in enumerate(fit):
        values = np.concatenate(
            [raw, event_features(raw), geometry_features(raw, beta, intercept, atoms)], axis=-1
        )
        total += len(values)
        sums += values.sum(axis=0, dtype=np.float64)
        squares += np.square(values.astype(np.float64)).sum(axis=0)
        flush_and_discard_sequence(fit_path, fit, index, durable=False)
    mean = sums / total
    scale = np.maximum(np.sqrt(np.maximum(squares / total - mean**2, 0)), 0.05)
    for role in ("fit", "eval"):
        raw_path = output / f"{role}_raw.npy"
        raw_data = np.load(raw_path, mmap_mode="r")
        features_path = output / f"{role}_x.npy"
        features = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(raw_data), 20_000, 231),
        )
        for index, raw in enumerate(raw_data):
            values = np.concatenate(
                [raw, event_features(raw), geometry_features(raw, beta, intercept, atoms)], axis=-1
            )
            features[index] = (values - mean) / scale
            flush_and_discard_sequence(features_path, features, index, durable=True)
            flush_and_discard_sequence(raw_path, raw_data, index, durable=False)
        del features, raw_data

    prefix = np.array(fit[0, :512], copy=True)
    changed = np.array(fit[0], copy=True)
    changed[512:] += np.float32(0.731)
    reference = geometry_features(fit[0], beta, intercept, atoms)[:512]
    future_changed = geometry_features(changed, beta, intercept, atoms)[:512]
    sliced = geometry_features(prefix, beta, intercept, atoms)
    flush_and_discard_sequence(fit_path, fit, 0, durable=False)
    if not np.array_equal(reference, future_changed) or not np.array_equal(reference, sliced):
        raise ValueError("causal suffix invariance failed")
    result = {
        "selected": ["i0_p0", "i0_p3", "i1_p0", "i1_p3"],
        "beta": beta.tolist(),
        "intercept": intercept.tolist(),
        "atoms": atoms.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "extra_width": EXTRA_WIDTH,
        "causal_suffix_invariance": "exact",
        "fit_sequences": len(fit),
        "disk_estimate": estimate,
        "cache_policy": "flush each written sequence and advise DONTNEED after every raw/X memmap pass",
    }
    (output / "preprocessing.json").write_text(json.dumps(result, indent=2) + "\n")
    del fit, eval_raw
    return result


def _parse_fit_components(value: str) -> int | None:
    """Parse a positive component count or the explicit all-eligible mode."""
    if value == "all":
        return None
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use a positive count or 'all'") from error
    if count < 1:
        raise argparse.ArgumentTypeError("component count must be positive")
    return count


def main() -> None:
    """Run membership or staging; only the membership phase reads small cached metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("membership", "raw", "features"), required=True)
    parser.add_argument("--fit-components", type=_parse_fit_components, default=12)
    parser.add_argument("--membership-seed", type=int, default=20261017)
    parser.add_argument("--reuse-evaluation-membership", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    validate_storage(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "membership":
        membership = freeze_membership(
            args.root,
            args.output,
            fit_components=args.fit_components,
            membership_seed=args.membership_seed,
            reuse_evaluation_membership=args.reuse_evaluation_membership,
        )
        LOGGER.info("Fit %d sequences; eval %d", len(membership["fit"]), len(membership["eval"]))
    elif args.phase == "raw":
        stage_raw(args.root, args.output, _load_membership(args.output / "membership.json"))
    else:
        LOGGER.info("%s", stage_features(args.output))


if __name__ == "__main__":
    main()
