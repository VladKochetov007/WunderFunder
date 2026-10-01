from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from probe import HERE, ROOT, context, read


def main() -> None:
    """Falsify simple monotone elapsed-age identities between observed refreshes."""
    start = time.monotonic()
    panel = json.loads((HERE.parent / "selection.json").read_text())
    classes = {
        r["group"]: r["diagnostic_target_class"]
        for r in json.loads((HERE.parent / "results.json").read_text())["by_group"]
    }
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    counts = defaultdict(Counter)
    for group in panel["test_train_groups"]:
        x = read(source, group)
        flags, _ = context(x)
        da = np.diff(x[:, 104:], axis=0)
        for lag in (0, 1):
            reset = flags[1:] if lag == 0 else flags[:-1]
            for a in range(8):
                for block in range(6):
                    for kind, mask in (("refresh", reset[:, block]), ("quiet", ~reset[:, block])):
                        values = da[:, a][mask]
                        record = {
                            "rows": len(values),
                            "positive": int((values > 0).sum()),
                            "negative": int((values < 0).sum()),
                            "zero": int((values == 0).sum()),
                        }
                        for cls in ("all", str(classes[group])):
                            counts[f"a{a}/block{block}/lag{lag}/{kind}/class{cls}"].update(record)
    result = {
        "contract_sha256": hashlib.sha256(
            (HERE / "clock_monotonicity_contract.json").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": {k: dict(v) for k, v in counts.items()},
        "resources": {
            "wall_seconds": time.monotonic() - start,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    (HERE / "clock_monotonicity_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(HERE / "clock_monotonicity_results.json"),
                "resources": result["resources"],
            }
        )
    )


if __name__ == "__main__":
    main()
