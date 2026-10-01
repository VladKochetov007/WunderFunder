from __future__ import annotations

import json
import time
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
from probe import HERE, ROOT, accept, digest, events, invert, key, read


def main() -> None:
    """Falsify simple deterministic offset alternatives without changing the panel."""
    start = time.monotonic()
    panel = json.loads((HERE / "selection.json").read_text())
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    interior = [defaultdict(Counter) for _ in range(4)]
    raw_alias = [defaultdict(set) for _ in range(4)]
    for group in panel["fit_train_groups"]:
        x, y, cls, _ = read(source, group)
        ev = events(x)
        sp, branch = invert(y)
        for code, first, last in zip(
            [key(v) for v in x[ev]], ev, np.r_[ev[1:], len(x)], strict=True
        ):
            rows = np.arange(first, last, 7)
            rows = rows[(branch[rows] == 1) & (sp[rows] >= 1)]
            interior[cls][code].update(sp[rows].tolist())
            raw_alias[cls][code].add(x[first].tobytes())
    maps = [accept(v) for v in interior]
    triples = [Counter() for _ in range(4)]
    event_triples = [Counter() for _ in range(4)]
    for group in panel["test_train_groups"]:
        x, y, cls, _ = read(source, group)
        ev = events(x)
        _, branch = invert(y)
        for code, first, last in zip(
            [key(v) for v in x[ev]], ev, np.r_[ev[1:], len(x)], strict=True
        ):
            raw_alias[cls][code].add(x[first].tobytes())
            mapped = maps[cls].get(code)
            if mapped is None:
                continue
            rows = np.arange(first, last)
            rows = rows[branch[rows] != 1]
            unique, counts = np.unique(y[rows], axis=0, return_counts=True)
            for pair, count in zip(unique, counts, strict=True):
                triple = (mapped, int(pair.max()), -int(pair.min()) - mapped)
                triples[cls][triple] += int(count)
            if branch[first] != 1:
                event_triples[cls][
                    (mapped, int(y[first].max()), -int(y[first].min()) - mapped)
                ] += 1
    result = {
        "contract_sha256": digest(HERE / "offset_followup_contract.json"),
        "source_sha256": digest(__import__("pathlib").Path(__file__)),
        "adaptive": True,
        "by_class": {},
    }
    for cls in range(4):
        report = {
            "rounded_alias_codes": sum(len(v) > 1 for v in raw_alias[cls].values()),
            "raw_code_count": sum(len(v) for v in raw_alias[cls].values()),
            "round4_code_count": len(raw_alias[cls]),
            "triples": [
                {
                    "spread": s,
                    "winner": w,
                    "offset": k,
                    "rows": n,
                    "events": event_triples[cls][s, w, k],
                }
                for (s, w, k), n in triples[cls].most_common()
            ],
            "controls": {},
        }
        for weight, counter in (("rows", triples[cls]), ("events", event_triples[cls])):
            out = Counter()
            for (_, winner, offset), n in counter.items():
                for name, value in (
                    ("offset2", 2),
                    ("common_mark", winner),
                    ("twice_winner", 2 * winner),
                    ("next_even", 2 * ((winner + 1) // 2)),
                    ("twice_next", 2 * (winner + 1)),
                ):
                    out[name] += n * (value == offset)
            report["controls"][weight] = {"total": counter.total(), "exact": dict(out)}
        result["by_class"][str(cls)] = report
    result["wall_seconds"] = time.monotonic() - start
    (HERE / "offset_followup.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(HERE / "offset_followup.json"), "wall_seconds": result["wall_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
