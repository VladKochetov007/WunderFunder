from __future__ import annotations

import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from probe import HERE, ROOT, digest, events, invert, key, read


def main() -> None:
    """Evaluate a single current-code table with no oracle class at inference."""
    start = time.monotonic()
    panel = json.loads((HERE / "selection.json").read_text())
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    counts = defaultdict(Counter)
    for group in panel["fit_train_groups"]:
        x, y, cls, _ = read(source, group)
        sp, branch = invert(y)
        ev = events(x)
        for code, first, last in zip(
            [key(v) for v in x[ev]], ev, np.r_[ev[1:], len(x)], strict=True
        ):
            rows = np.arange(first, last, 7)
            rows = rows[(branch[rows] == 1) & (sp[rows] >= 1)]
            counts[code].update((cls, int(s)) for s in sp[rows])
    table = {
        code: v.most_common(1)[0][0]
        for code, v in counts.items()
        if v and v.most_common(1)[0][1] >= 3 and v.most_common(1)[0][1] / v.total() >= 0.95
    }
    summary = [Counter() for _ in range(4)]
    groups = []
    for group in panel["test_train_groups"]:
        x, y, cls, _ = read(source, group)
        sp, branch = invert(y)
        ev = events(x)
        local = Counter({"rows": len(x), "groups": 1})
        for code, first, last in zip(
            [key(v) for v in x[ev]], ev, np.r_[ev[1:], len(x)], strict=True
        ):
            result = table.get(code)
            if result is None:
                continue
            cls_hat, width = result
            local["mapped_rows"] += int(last - first)
            local["class_exact_rows"] += int(last - first) * (cls_hat == cls)
            rows = np.arange(first, last)
            rows = rows[branch[rows] == 1]
            local["interior_mapped_rows"] += len(rows)
            local["interior_spread_exact_rows"] += int((sp[rows] == width).sum())
        summary[cls].update(local)
        groups.append({"group": group, "diagnostic_class": cls, "counts": dict(local)})
    mapped_path = HERE / "interior_code_table.json"
    mapped_path.write_text(
        json.dumps(
            [
                {
                    "code_float32_round4_hex": code.hex(),
                    "class": cls,
                    "spread_ticks": width,
                    "fit_vote_count": counts[code][cls, width],
                    "fit_total_votes": counts[code].total(),
                }
                for code, (cls, width) in table.items()
            ],
            indent=2,
        )
        + "\n"
    )
    result = {
        "contract_sha256": digest(HERE / "pooled_contract.json"),
        "source_sha256": digest(Path(__file__)),
        "map_sha256": digest(mapped_path),
        "map_codes": len(table),
        "fit_codes_with_class_collision": sum(
            len({cls for cls, _ in v}) > 1 for v in counts.values()
        ),
        "fit_codes_with_spread_collision": sum(
            len({sp for _, sp in v}) > 1 for v in counts.values()
        ),
        "by_diagnostic_class": {str(cls): dict(v) for cls, v in enumerate(summary)},
        "by_group": groups,
        "resources": {
            "wall_seconds": time.monotonic() - start,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "inference_inputs": "current price code only; target-derived class/spread used as offline TRAIN supervision and retrospective held-out grading",
    }
    (HERE / "pooled_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(HERE / "pooled_results.json"),
                "resources": result["resources"],
                "by_diagnostic_class": result["by_diagnostic_class"],
            }
        )
    )


if __name__ == "__main__":
    main()
