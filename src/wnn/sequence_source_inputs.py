"""Build full-coverage, feature-only inputs without changing historical caches."""

from __future__ import annotations

import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.data import FEATURE_COLUMNS, SEQUENCE_LENGTH
from wnn.sequence_inner_splits import _publish_exclusive
from wnn.splits import file_sha256

GRAPH = "cache/source_holdout_chains.npz"
INDEX = {source: f"cache/source_holdout_{source}_index.npz" for source in ("train", "valid")}
RECEIPT = "cache/source_holdout_inputs.json"
METHOD = "feature-only-mutual-L1-nearest-endpoints-within-sequence-q99-v1"


def source_identity(path: Path) -> dict[str, int | str]:
    """Bind source files with full SHA256; parquets also retain footer identity."""
    before = path.stat()
    identity: dict[str, int | str] = {"bytes": before.st_size, "mtime_ns": before.st_mtime_ns}
    if path.suffix == ".parquet":
        with path.open("rb") as stream:
            stream.seek(-8, 2)
            trailer = stream.read(8)
            size = int.from_bytes(trailer[:4], "little")
            if trailer[4:] != b"PAR1" or size > before.st_size - 8:
                raise ValueError(f"invalid parquet footer: {path}")
            stream.seek(-size - 8, 2)
            digest = hashlib.sha256()
            remaining = size + 8
            while remaining:
                block = stream.read(min(remaining, 1 << 20))
                if not block:
                    raise ValueError(f"truncated parquet footer: {path}")
                digest.update(block)
                remaining -= len(block)
        identity.update(footer_bytes=size, footer_sha256=digest.hexdigest())
        identity["sha256"] = file_sha256(path)
    else:
        identity["sha256"] = file_sha256(path)
    after = path.stat()
    if any(
        getattr(after, field) != getattr(before, field)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError(f"source changed while fingerprinting: {path}")
    return identity


def verify_identity(root: Path, name: str, expected: dict[str, int | str]) -> None:
    """Check immutable local fingerprints or the shared portable full-byte registry."""
    observed = source_identity(root / name)
    registry = os.environ.get("WNN_PORTABLE_DATA_MANIFEST")
    if registry:
        from wnn.portability import verify_portable_source

        verify_portable_source(root / name, name, expected, Path(registry))
        observed.pop("mtime_ns")
        expected = {key: value for key, value in expected.items() if key != "mtime_ns"}
    if observed != expected:
        raise ValueError(f"source changed: {name}")


def sequence_ids(reader: pq.ParquetFile) -> np.ndarray:
    """Verify every physical group is exactly one complete integer-ID sequence."""
    if "seq_ix" not in reader.schema.names or reader.num_row_groups < 1:
        raise ValueError("source must contain nonempty seq_ix row groups")
    column = reader.schema.names.index("seq_ix")
    values = []
    for group in range(reader.num_row_groups):
        metadata = reader.metadata.row_group(group)
        if metadata.num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"row group {group} must contain {SEQUENCE_LENGTH} rows")
        statistics = metadata.column(column).statistics
        if (
            statistics is not None
            and statistics.has_min_max
            and statistics.null_count == 0
            and type(statistics.min) is int
            and statistics.min == statistics.max
        ):
            sequence = statistics.min
        else:
            array = reader.read_row_group(group, columns=["seq_ix"], use_threads=False)[
                "seq_ix"
            ].to_numpy()
            if array.dtype.kind not in "iu" or not np.all(array == array[0]):
                raise ValueError(f"row group {group} does not contain one integer seq_ix")
            sequence = int(array[0])
        if sequence < 0:
            raise ValueError(f"negative seq_ix in row group {group}")
        values.append(sequence)
    if len(set(values)) != len(values):
        raise ValueError("duplicate seq_ix within source")
    return np.asarray(values, dtype=np.int64)


def _edges(job: tuple[pq.ParquetFile, int]) -> np.ndarray:
    reader, group = job
    table = reader.read_row_group(group, columns=FEATURE_COLUMNS, use_threads=False)
    array = np.stack([table[name].to_numpy() for name in FEATURE_COLUMNS], axis=1).astype(
        np.float32
    )
    if array.shape != (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)) or not np.isfinite(array).all():
        raise ValueError(f"nonfinite or incomplete feature row group {group}")
    return array[[0, -1, SEQUENCE_LENGTH // 2, SEQUENCE_LENGTH // 2 + 1]].copy()


def _successors(edges: np.ndarray) -> np.ndarray:
    """Use the historical V1 feature-only linkage rule, including every source group."""
    from sklearn.neighbors import NearestNeighbors

    head, tail = edges[:, 0], edges[:, 1]
    reference = np.abs(edges[:, 3] - edges[:, 2]).sum(axis=1)
    distances, candidates = (
        NearestNeighbors(n_neighbors=2, metric="manhattan").fit(head).kneighbors(tail)
    )
    _, backward = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(tail).kneighbors(head)
    result = np.full(len(edges), -1, dtype=np.int64)
    threshold = np.quantile(reference, 0.99)
    for index in range(len(edges)):
        rank = int(candidates[index, 0] == index)
        successor = candidates[index, rank]
        if backward[successor, 0] == index and distances[index, rank] <= threshold:
            result[index] = successor
    return result


def _publish_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    _publish_exclusive(path, buffer.getvalue())


def validate_input_receipt(root: Path) -> dict[str, dict[str, int | str]]:
    """Verify graph/index ancestry was derived from these exact immutable sources."""
    path = root / RECEIPT
    if path.stat().st_size > 32_000:
        raise ValueError("source holdout input receipt exceeds read bound")
    receipt = json.loads(path.read_text())
    names = {GRAPH, *INDEX.values(), "datasets/train.parquet", "datasets/valid.parquet"}
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {"schema", "method", "sources"}
        or receipt["schema"] != "wnn.sequence-source-inputs.v1"
        or receipt["method"] != METHOD
        or not isinstance(receipt["sources"], dict)
        or set(receipt["sources"]) != names
    ):
        raise ValueError("invalid source holdout input receipt")
    for name, identity in receipt["sources"].items():
        verify_identity(root, name, identity)
    return receipt["sources"]


def prepare_source_inputs(root: Path, workers: int = 2) -> None:
    """Prepare immutable full indexes and graph, failing on corrupt or missing groups.

    All valid features participate only in graph-based overlap exclusion. Targets
    are never read. Existing complete inputs are validated; partial caches require
    explicit operator cleanup, and historical caches are never replaced.
    """
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("source preparation workers must be between one and sixteen")
    root = root.resolve()
    if (root / RECEIPT).exists():
        validate_input_receipt(root)
        return
    if any((root / name).exists() for name in (GRAPH, *INDEX.values())):
        raise FileExistsError("partial source holdout caches exist without a sealed receipt")
    sources = {
        f"datasets/{source}.parquet": source_identity(root / f"datasets/{source}.parquet")
        for source in INDEX
    }
    readers = {
        source: pq.ParquetFile(root / name)
        for source, name in ((source, f"datasets/{source}.parquet") for source in INDEX)
    }
    indexes = {source: sequence_ids(reader) for source, reader in readers.items()}
    members = [(source, group) for source in INDEX for group in range(len(indexes[source]))]
    jobs = [(readers[source], group) for source, group in members]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        edges = np.stack(list(executor.map(_edges, jobs)))
    successors = _successors(edges)
    for name, identity in sources.items():
        verify_identity(root, name, identity)
    (root / "cache").mkdir(exist_ok=True)
    for source, ids in indexes.items():
        _publish_npz(root / INDEX[source], {"row_group": np.arange(len(ids)), "seq_ix": ids})
    _publish_npz(
        root / GRAPH,
        {
            "split": np.asarray([source for source, _ in members]),
            "row_group": np.asarray([group for _, group in members]),
            "next": successors,
        },
    )
    sources.update({name: source_identity(root / name) for name in (GRAPH, *INDEX.values())})
    receipt = {"schema": "wnn.sequence-source-inputs.v1", "method": METHOD, "sources": sources}
    _publish_exclusive(
        root / RECEIPT, (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    )
