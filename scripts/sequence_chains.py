"""Recover time adjacency of sequences (V1): tail of one sequence -> head of another.

    scripts/limited .venv/bin/python scripts/sequence_chains.py
Reads the first/last 3 rows of every train and validation sequence, links each tail to its
nearest head (L1 over the 112 features, excluding the a-columns that restart per sequence is
not assumed: all columns used), keeps mutual best matches whose distance is within the range of
consecutive-row distances inside sequences, and builds chains. Writes cache/chains.npz with
split, row_group, next (index of the successor or -1), chain id and position in chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.chronological_source_index import (
    OFFICIAL_SOURCE_PINS,
    SourceRowGroupReadError,
    verify_source_index_receipt,
)
from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


_FILES: dict[str, pq.ParquetFile] = {}
_OFFICIAL_FILES: dict[str, pq.ParquetFile] = {}


def _edge(args: tuple[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split, g = args
    pf = _FILES.get(split)
    if pf is None:  # open once per worker: train metadata alone is ~1 GB
        pf = _FILES[split] = pq.ParquetFile(ROOT / "datasets" / f"{split}.parquet")
    t = pf.read_row_group(g, columns=FEATURE_COLUMNS + ["t0", "t1"], use_threads=False)
    a = np.column_stack([t.column(c).to_numpy() for c in FEATURE_COLUMNS + ["t0", "t1"]]).astype(
        np.float32
    )
    return a[:2].copy(), a[-2:].copy(), a[10000:10002].copy()


def _official_edge(args: tuple[str, int, str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read feature-only endpoints and midpoint for one source-qualified row group."""
    split, row_group, root_text, sequence_length = args
    source = Path(root_text) / "datasets" / f"{split}.parquet"
    cache_key = str(source)
    parquet = _OFFICIAL_FILES.get(cache_key)
    if parquet is None:
        parquet = _OFFICIAL_FILES[cache_key] = pq.ParquetFile(source)
    try:
        table = parquet.read_row_group(row_group, columns=FEATURE_COLUMNS, use_threads=False)
        values = np.column_stack(
            [table.column(name).to_numpy() for name in FEATURE_COLUMNS]
        ).astype(np.float32, copy=False)
    except Exception as exc:
        raise SourceRowGroupReadError(
            split, row_group, str(source), f"{type(exc).__name__}: {exc}"
        ) from exc
    if table.num_rows != sequence_length:
        raise SourceRowGroupReadError(
            split,
            row_group,
            str(source),
            f"feature row count is {table.num_rows}; expected {sequence_length}",
        )
    finite = np.isfinite(values)
    if not finite.all():
        first_bad = int(np.argmax(~finite))
        bad_row, bad_column = divmod(first_bad, values.shape[1])
        raise SourceRowGroupReadError(
            split,
            row_group,
            str(source),
            f"nonfinite float32 feature at row={bad_row}, column={FEATURE_COLUMNS[bad_column]}",
        )
    middle = sequence_length // 2
    return values[:2].copy(), values[-2:].copy(), values[middle : middle + 2].copy()


def _successors(
    head: np.ndarray, tail: np.ndarray, mid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Apply the historical mutual-nearest feature-edge rule to sequence samples."""
    from sklearn.neighbors import NearestNeighbors

    reference_distances = np.abs(mid[:, 1, :112] - mid[:, 0, :112]).sum(1)
    nn_head = NearestNeighbors(n_neighbors=2, metric="manhattan").fit(head)
    distances, indexes = nn_head.kneighbors(tail)
    nn_tail = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(tail)
    _, back = nn_tail.kneighbors(head)
    successors = np.full(len(head), -1, dtype=np.int64)
    threshold = float(np.quantile(reference_distances, 0.99))
    for index in range(len(head)):
        candidate = indexes[index, 0] if indexes[index, 0] != index else indexes[index, 1]
        distance = distances[index, 0] if indexes[index, 0] != index else distances[index, 1]
        if back[candidate, 0] == index and distance <= threshold:
            successors[index] = candidate
    return successors, distances, indexes, threshold, reference_distances


def _write_graph(
    path: Path,
    split: np.ndarray,
    row_group: np.ndarray,
    successors: np.ndarray,
    source_audit_sha256: str,
    source_sha256: np.ndarray,
    index_map_sha256: np.ndarray,
    index_receipt_sha256: np.ndarray,
) -> None:
    """Write the graph and embed hashes binding it to audit and sequence indexes."""
    metadata = {
        "source_audit_sha256": np.asarray(source_audit_sha256),
        "source_sha256": source_sha256,
        "index_map_sha256": index_map_sha256,
        "index_receipt_sha256": index_receipt_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            expected_fields = {"split", "row_group", "next", *metadata}
            if set(existing.files) != expected_fields:
                raise FileExistsError(
                    f"refusing to replace noncanonical chronological graph: {path}"
                )
            matches = all(
                np.array_equal(existing[field], value)
                for field, value in (
                    ("split", split),
                    ("row_group", row_group),
                    ("next", successors),
                    *metadata.items(),
                )
            )
        if not matches:
            raise FileExistsError(f"refusing to replace changed chronological graph: {path}")
        return
    with path.open("xb") as stream:
        np.savez(stream, split=split, row_group=row_group, next=successors, **metadata)


def _write_failure_receipt(root: Path, error: SourceRowGroupReadError) -> None:
    """Preserve the exact source row group and root-audit pin for a failed read."""
    identity_path = root / "cache" / f"{error.split}_index.identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    receipt = {
        "schema": "wnn.chronological-chain-failure.v1",
        "source": f"datasets/{error.split}.parquet",
        "split": error.split,
        "row_group": error.row_group,
        "source_bytes": identity["source_bytes"],
        "source_sha256": identity["source_sha256"],
        "source_audit_sha256": identity["source_audit_sha256"],
        "error": error.detail,
    }
    path = root / "cache" / "chronological_chains.failure.json"
    encoded = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to replace earlier chain failure receipt: {path}")
        return
    with path.open("xb") as stream:
        stream.write(encoded)


def _source_group_jobs(
    root: Path, indexes: dict[str, np.ndarray]
) -> list[tuple[str, int, str, int]]:
    """Enumerate every indexed source group without applying historical exclusions."""
    return [
        (split, int(row_group), str(root), 20_000)
        for split in ("train", "valid")
        for row_group in indexes[split]
    ]


def build_official_chains(root: Path, workers: int) -> Path:
    """Build a feature-only graph over every row group named by the root audit."""
    root = root.resolve()
    if workers <= 0:
        raise ValueError("worker count must be positive")
    audit_path = root / "cache" / "source_audit.json"
    source_audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    index_identities = {
        split: json.loads((root / f"cache/{split}_index.identity.json").read_text(encoding="utf-8"))
        for split in ("train", "valid")
    }
    source_sha256 = np.asarray(
        [index_identities[split]["source_sha256"] for split in ("train", "valid")]
    )
    index_map_sha256 = np.asarray(
        [index_identities[split]["index_map_sha256"] for split in ("train", "valid")]
    )
    index_receipt_sha256 = np.asarray(
        [
            hashlib.sha256((root / f"cache/{split}_index.identity.json").read_bytes()).hexdigest()
            for split in ("train", "valid")
        ]
    )
    try:
        indexes = {
            split: verify_source_index_receipt(root, split)[0] for split in ("train", "valid")
        }
        jobs = _source_group_jobs(root, indexes)
        expected_count = sum(pin.row_group_count for pin in OFFICIAL_SOURCE_PINS.values())
        if len(jobs) != expected_count:
            raise ValueError("source-qualified graph input does not cover every official row group")
        if workers == 1:
            samples = [_official_edge(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                samples = list(executor.map(_official_edge, jobs, chunksize=16))
    except SourceRowGroupReadError as exc:
        _write_failure_receipt(root, exc)
        raise

    if len(samples) != len(jobs):
        raise RuntimeError("feature sampler returned incomplete official source coverage")
    head = np.stack([sample[0][0, :112] for sample in samples])
    tail = np.stack([sample[1][-1, :112] for sample in samples])
    mid = np.stack([sample[2] for sample in samples])
    successors, distances, indexes, threshold, reference = _successors(head, tail, mid)
    source_splits = np.asarray([split for split, _, _, _ in jobs])
    row_groups = np.asarray([row_group for _, row_group, _, _ in jobs], dtype=np.int64)
    if len(set(zip(source_splits.tolist(), row_groups.tolist(), strict=True))) != len(jobs):
        raise ValueError("source-qualified graph contains duplicate group identities")
    output = root / "cache" / "chronological_chains.npz"
    _write_graph(
        output,
        source_splits,
        row_groups,
        successors,
        source_audit_sha256,
        source_sha256,
        index_map_sha256,
        index_receipt_sha256,
    )
    linked = np.flatnonzero(successors >= 0)
    print(
        {
            "source_groups": len(jobs),
            "train_groups": int(np.sum(source_splits == "train")),
            "valid_groups": int(np.sum(source_splits == "valid")),
            "links": int(len(linked)),
            "link_distance_median": float(np.median(distances[linked, 0])) if len(linked) else None,
            "within_sequence_distance_median": float(np.median(reference)),
            "threshold": threshold,
            "graph": output.as_posix(),
        }
    )
    return output


def _historical_main() -> None:
    """Replay the pre-chronological chain producer with its frozen group filter."""
    jobs = [("train", g) for g in range(10607) if g not in CORRUPT_TRAIN_ROW_GROUPS] + [
        ("valid", g) for g in range(1873)
    ]
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "6"))) as ex:
        res = list(ex.map(_edge, jobs, chunksize=16))
    head = np.stack([r[0][0, :112] for r in res])
    tail = np.stack([r[1][-1, :112] for r in res])
    mid = np.stack([r[2] for r in res])
    nxt, d, idx, thr, ref = _successors(head, tail, mid)
    n = len(res)
    prev = np.full(n, -1)
    prev[nxt[nxt >= 0]] = np.flatnonzero(nxt >= 0)
    chain = np.full(n, -1)
    pos = np.zeros(n, int)
    c = 0
    for i in np.flatnonzero(prev < 0):
        k, p = i, 0
        seen = set()
        while k >= 0 and k not in seen:
            seen.add(k)
            chain[k], pos[k] = c, p
            k, p = nxt[k], p + 1
        c += 1
    split = np.array([s for s, _ in jobs])
    rg = np.array([g for _, g in jobs])
    t_tail = np.stack([r[1][-1, 112:] for r in res])
    t_head = np.stack([r[0][0, 112:] for r in res])
    linked = np.flatnonzero(nxt >= 0)
    same_target = np.mean(np.all(np.isclose(t_tail[linked], t_head[nxt[linked]]), 1))
    rnd = np.random.default_rng(0).permutation(n)
    rand_target = np.mean(np.all(np.isclose(t_tail, t_head[rnd]), 1))
    sizes = np.bincount(chain[chain >= 0])
    print(
        {
            "sequences": n,
            "links": int(len(linked)),
            "chains": int(c),
            "chain_size_median": float(np.median(sizes)),
            "chain_size_max": int(sizes.max()),
            "link_dist_median": float(np.median(d[linked, 0])),
            "within_seq_consecutive_dist_median": float(np.median(ref)),
            "threshold": float(thr),
            "target_identical_across_link": round(float(same_target), 4),
            "target_identical_random": round(float(rand_target), 4),
            "mixed_split_links": int(np.sum(split[linked] != split[nxt[linked]])),
        }
    )
    np.savez(
        ROOT / "cache" / "chains.npz", split=split, row_group=rg, next=nxt, chain=chain, pos=pos
    )


def main() -> None:
    """Run the historical producer by default or the explicit official builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-qualified", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WNN_THREADS", "1")))
    args = parser.parse_args()
    if args.source_qualified:
        build_official_chains(args.root, args.workers)
        return
    _historical_main()


if __name__ == "__main__":
    main()
