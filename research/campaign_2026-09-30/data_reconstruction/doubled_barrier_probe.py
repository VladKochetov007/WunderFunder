from __future__ import annotations

import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from probe import HERE, ROOT, accept, digest, events, invert, key, read


def main() -> None:
    """Compare doubled first strictly adverse displacement with losing payoffs."""
    start = time.monotonic()
    panel = json.loads((HERE / "selection.json").read_text())
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    spreads = [defaultdict(Counter) for _ in range(4)]
    votes = [defaultdict(Counter) for _ in range(4)]
    for group in panel["fit_train_groups"]:
        x, y, cls, _ = read(source, group)
        ev = events(x)
        keys = [key(row) for row in x[ev]]
        sp, branch = invert(y)
        for code, first, last in zip(keys, ev, np.r_[ev[1:], len(x)], strict=True):
            rows = np.arange(first, last, 7)
            rows = rows[(branch[rows] == 1) & (sp[rows] >= 1)]
            spreads[cls][code].update(sp[rows].tolist())
        for i, row in enumerate(ev[1:], 1):
            if branch[row - 1] == branch[row] == 1:
                dy = y[row] - y[row - 1]
                votes[cls][keys[i - 1], keys[i]][(-int(dy[0]), int(dy[1]))] += 1
    maps = [accept(v) for v in spreads]
    moves = [
        {
            pair: v.most_common(1)[0][0]
            for pair, v in table.items()
            if v.most_common(1)[0][1] >= 3 and v.most_common(1)[0][1] / v.total() >= 0.8
        }
        for table in votes
    ]
    totals = [Counter() for _ in range(4)]
    triples = [Counter() for _ in range(4)]
    group_reports = []
    for group in panel["test_train_groups"]:
        x, y, cls, _ = read(source, group)
        ev = events(x)
        keys = [key(row) for row in x[ev]]
        _, branch = invert(y)
        delta = np.zeros((len(ev), 2), dtype=np.int64)
        unknown = np.zeros(len(ev), dtype=np.int64)
        for i in range(1, len(ev)):
            move = moves[cls].get((keys[i - 1], keys[i]))
            if move is None:
                unknown[i] = 1
            else:
                delta[i] = move
        cumulative = delta.cumsum(axis=0)
        gaps = unknown.cumsum()
        local = Counter(
            {"events": len(ev) - 1, "known_move_events": len(ev) - 1 - int(unknown.sum())}
        )
        for i, row in enumerate(ev):
            width = maps[cls].get(keys[i])
            if width is None or branch[row] == 1:
                continue
            length = int((ev[i + 1] if i + 1 < len(ev) else len(x)) - row)
            winner = int(y[row].max())
            offset = -int(y[row].min()) - width
            local["mapped_outside_origins"] += 1
            local["mapped_outside_rows"] += length
            local["offset_gt2_origins"] += offset > 2
            local["offset_gt2_rows"] += length * (offset > 2)
            local["common_future_quote_mark_impossible_origins"] += offset - winner < 1
            local["common_future_quote_mark_impossible_rows"] += length * (offset - winner < 1)
            future = np.arange(i + 1, min(i + 65, len(ev)))
            known = gaps[future] == gaps[i]
            displacement = cumulative[future] - cumulative[i]
            directional = displacement[:, 0] if branch[row] == 2 else -displacement[:, 1]
            hit = np.flatnonzero(known & (directional >= 1))
            if not len(hit):
                local["no_known_first_barrier_within64_origins"] += 1
                continue
            first = int(hit[0])
            realized = int(directional[first])
            opposite = int(displacement[first, 1] if branch[row] == 2 else -displacement[first, 0])
            local["first_barrier_known_origins"] += 1
            local["first_barrier_known_rows"] += length
            for prefix, condition in (("all", True), ("offset_gt2", offset > 2)):
                if condition:
                    local[f"{prefix}_eligible_origins"] += 1
                    local[f"{prefix}_eligible_rows"] += length
                    local[f"{prefix}_first_barrier_exact_origins"] += realized == offset
                    local[f"{prefix}_first_barrier_exact_rows"] += length * (realized == offset)
                    local[f"{prefix}_twice_first_barrier_exact_origins"] += 2 * realized == offset
                    local[f"{prefix}_twice_opposite_first_barrier_exact_origins"] += (
                        2 * opposite == offset
                    )
                    local[f"{prefix}_offset2_exact_origins"] += offset == 2
                    local[f"{prefix}_offset2_exact_rows"] += length * (offset == 2)
            triples[cls][offset, realized, opposite] += length
        totals[cls].update(local)
        group_reports.append({"group": group, "class": cls, "counts": dict(local)})
    result = {
        "contract_sha256": digest(HERE / "doubled_barrier_contract.json"),
        "source_sha256": digest(Path(__file__)),
        "adaptive": True,
        "by_class": {
            str(cls): {
                "counts": dict(totals[cls]),
                "actual_offset_then_first_directional_then_opposite_top30": [
                    {"actual_offset": a, "first_directional": b, "first_opposite": c, "rows": n}
                    for (a, b, c), n in triples[cls].most_common(30)
                ],
            }
            for cls in range(4)
        },
        "by_group": group_reports,
        "resources": {
            "wall_seconds": time.monotonic() - start,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    (HERE / "doubled_barrier_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(HERE / "doubled_barrier_results.json"), "resources": result["resources"]}
        )
    )


if __name__ == "__main__":
    main()
