"""Independently replay the pooled TRAIN code lookup from raw parquet rows."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
DATA = ROOT / "research/campaign_2026-09-30/data_reconstruction"
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])


def read_group(source: pq.ParquetFile, group: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Load one raw TRAIN sequence and identify its diagnostic target lattice."""
    table = source.read_row_group(
        group, columns=[*[f"i0_p{i}" for i in range(22)], "t0", "t1"], use_threads=False
    )
    code = np.column_stack([table[f"i0_p{i}"].to_numpy() for i in range(22)]).astype(np.float32)
    y = np.column_stack([table["t0"].to_numpy(), table["t1"].to_numpy()]).astype(np.float64)
    residuals = [np.max(np.abs(y - np.rint(y / q) * q)) for q in QUANTA]
    cls = int(np.argmin(residuals))
    assert residuals[cls] <= 0.002
    return code, np.rint(y / QUANTA[cls]).astype(np.int32), cls


def lookup(codes: np.ndarray, table: dict[bytes, tuple[int, int]]) -> list[tuple[int, int] | None]:
    """The entire inference interface accepts only current code and a fit table."""
    rounded = np.ascontiguousarray(np.round(codes, 4))
    return [table.get(row.tobytes()) for row in rounded]


def main() -> None:
    """Rebuild accepted votes and every held-out count without actor imports."""
    started = time.monotonic()
    selection = json.loads((DATA / "selection.json").read_text())
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    votes: dict[bytes, Counter[tuple[int, int]]] = defaultdict(Counter)
    for group in selection["fit_train_groups"]:
        x, t, cls = read_group(source, group)
        boundaries = np.concatenate(
            ([0], np.nonzero(np.any(np.diff(x, axis=0) != 0, axis=1))[0] + 1, [len(x)])
        )
        for first, last in zip(boundaries[:-1], boundaries[1:], strict=True):
            selected = np.arange(first, last, 7)
            width = -t[selected].sum(axis=1)
            inside = (t[selected] <= 0).all(axis=1) & (width >= 1)
            code = np.round(x[first], 4).tobytes()
            votes[code].update((cls, int(value)) for value in width[inside])
    table = {}
    for code, counts in votes.items():
        if counts:
            pair, count = max(counts.items(), key=lambda item: item[1])
            if count >= 3 and count >= 0.95 * counts.total():
                table[code] = pair
    actor_table = json.loads((DATA / "interior_code_table.json").read_text())
    expected = {
        bytes.fromhex(row["code_float32_round4_hex"]): (row["class"], row["spread_ticks"])
        for row in actor_table
    }
    assert table == expected
    for row in actor_table:
        code = bytes.fromhex(row["code_float32_round4_hex"])
        assert votes[code].total() == row["fit_total_votes"]
        assert votes[code][table[code]] == row["fit_vote_count"]
    by_class = [Counter() for _ in QUANTA]
    by_group = []
    for group in selection["test_train_groups"]:
        x, t, cls = read_group(source, group)
        prediction = lookup(x, table)
        local = Counter(rows=len(x), groups=1)
        for row, pair in zip(t, prediction, strict=True):
            if pair is None:
                continue
            predicted_class, width = pair
            local["mapped_rows"] += 1
            local["class_exact_rows"] += predicted_class == cls
            if (row <= 0).all():
                local["interior_mapped_rows"] += 1
                local["interior_spread_exact_rows"] += width == -int(row.sum())
        by_class[cls].update(local)
        by_group.append({"group": group, "diagnostic_class": cls, "counts": dict(local)})
        assert lookup(x[:32], table) == prediction[:32]
    actor = json.loads((DATA / "pooled_results.json").read_text())
    observed = {str(cls): dict(counts) for cls, counts in enumerate(by_class)}
    assert observed == actor["by_diagnostic_class"]
    assert by_group == actor["by_group"]
    assert len(table) == actor["map_codes"] == 1026
    assert (
        sum(len({cls for cls, _ in v}) > 1 for v in votes.values())
        == actor["fit_codes_with_class_collision"]
        == 0
    )
    assert (
        sum(len({sp for _, sp in v}) > 1 for v in votes.values())
        == actor["fit_codes_with_spread_collision"]
        == 0
    )
    output = {
        "status": "pass",
        "independent_actor_module_imports": False,
        "raw_parquet_replay": True,
        "accepted_table_and_vote_counts_match": True,
        "all_group_and_class_counts_match": True,
        "map_codes": len(table),
        "inference_signature": "lookup(current_codes, fitting_table); no targets/class/masks/time/IDs",
        "by_diagnostic_class": observed,
        "selection_sha256": hashlib.sha256((DATA / "selection.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    destination = AUDIT / "pooled_independent_replay.json"
    with destination.open("x") as stream:
        stream.write(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
