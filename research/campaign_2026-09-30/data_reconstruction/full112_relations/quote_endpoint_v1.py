from __future__ import annotations

import hashlib
import importlib.util
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from probe import HERE, ROOT
from scipy.stats import norm


def main() -> None:
    """Compare a6 lower endpoint with feature-only TRAIN-supervised quote moves."""
    start = time.monotonic()
    spec = importlib.util.spec_from_file_location("primary_probe", HERE.parent / "probe.py")
    primary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(primary)
    panel = json.loads((HERE.parent / "selection.json").read_text())
    classes = {
        r["group"]: r["diagnostic_target_class"]
        for r in json.loads((HERE.parent / "results.json").read_text())["by_group"]
    }
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    votes = defaultdict(Counter)
    for group in panel["fit_train_groups"]:
        x, y, _, _ = primary.read(source, group)
        ev = primary.events(x)
        keys = [primary.key(v) for v in x[ev]]
        _, branch = primary.invert(y)
        for i, row in enumerate(ev[1:], 1):
            if branch[row - 1] == branch[row] == 1:
                dy = y[row] - y[row - 1]
                votes[keys[i - 1], keys[i]][(-int(dy[0]), int(dy[1]))] += 1
    moves = {
        pair: v.most_common(1)[0][0]
        for pair, v in votes.items()
        if v.most_common(1)[0][1] >= 3 and v.most_common(1)[0][1] / v.total() >= 0.8
    }
    endpoint = np.float32(norm.ppf(1e-7 - np.finfo(float).eps))
    counts = defaultdict(Counter)
    groups = []
    for group in panel["test_train_groups"]:
        names = [f"i0_p{i}" for i in range(22)] + ["a6"]
        tab = source.read_row_group(group, columns=names, use_threads=False)
        x = np.column_stack([tab.column(k).to_numpy() for k in names[:22]]).astype(np.float32)
        a6 = tab.column("a6").to_numpy().astype(np.float32)
        ev = primary.events(x)
        keys = [primary.key(v) for v in x[ev]]
        event = np.zeros(len(x), dtype=bool)
        event[ev[1:]] = True
        marker = a6 == endpoint
        record = Counter(
            {
                "rows": len(x),
                "code_events": len(ev) - 1,
                "endpoint_rows": int(marker.sum()),
                "endpoint_quiet_rows_including_origin": int((marker & ~event).sum()),
            }
        )
        for i, row in enumerate(ev[1:], 1):
            move = moves.get((keys[i - 1], keys[i]))
            if move is None:
                record["unknown_move_events"] += 1
                record["endpoint_on_unknown_move_events"] += bool(marker[row])
                continue
            kind = "nonzero" if move != (0, 0) else "zero"
            record[f"known_{kind}_events"] += 1
            record[f"endpoint_on_known_{kind}_events"] += bool(marker[row])
            record[f"not_endpoint_on_known_{kind}_events"] += not bool(marker[row])
        for cls in ("all", str(classes[group])):
            counts[cls].update(record)
        groups.append(
            {"group": group, "class_diagnostic_only": classes[group], "counts": dict(record)}
        )
    result = {
        "contract_sha256": hashlib.sha256(
            (HERE / "quote_endpoint_contract.json").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pooled_fit_voted_pairs": len(votes),
        "accepted_pairs": len(moves),
        "counts": {k: dict(v) for k, v in counts.items()},
        "by_group": groups,
        "resources": {
            "wall_seconds": time.monotonic() - start,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    (HERE / "quote_endpoint_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(HERE / "quote_endpoint_results.json"), "resources": result["resources"]}
        )
    )


if __name__ == "__main__":
    main()
