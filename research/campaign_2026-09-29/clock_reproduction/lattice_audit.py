"""Audit target-lattice failures in clean fold-0 TRAIN fit groups only."""

from __future__ import annotations

import json
import resource
import time

import numpy as np
import pyarrow.parquet as pq
from replay import HERE, QUANTA, ROOT, digest


def main() -> None:
    started = time.monotonic()
    manifest_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    manifest = json.loads(manifest_path.read_text())
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    failures = []
    maxima = []
    by_class = {str(i): [] for i in range(4)}
    for group in manifest["fit"]["train"]:
        table = source.read_row_group(group, columns=["t0", "t1"], use_threads=False)
        targets = np.stack([table[name].to_numpy() for name in ("t0", "t1")], axis=1)
        errors = [float(np.max(np.abs(np.rint(targets / quantum) * quantum - targets)))
                  for quantum in QUANTA]
        best = int(np.argmin(errors))
        maxima.append(errors[best])
        by_class[str(best)].append(errors[best])
        if errors[best] > 0.001:
            failures.append({"train_group": group, "best_class": best,
                             "best_max_residual": errors[best], "all_max_residuals": errors,
                             "target_min": [float(x) for x in targets.min(axis=0)],
                             "target_max": [float(x) for x in targets.max(axis=0)]})
    output = {
        "split_manifest_sha256": digest(manifest_path),
        "train_fit_groups_scanned": len(manifest["fit"]["train"]),
        "tolerance": 0.001,
        "failures": failures,
        "max_residual_quantiles": np.quantile(maxima, [0, .5, .9, .99, .999, 1]).tolist(),
        "by_best_class": {key: {"groups": len(values),
                                "max_residual": max(values) if values else None,
                                "failures": sum(value > 0.001 for value in values)}
                          for key, values in by_class.items()},
        "resources": {"wall_seconds": time.monotonic() - started,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "exit_code": 0},
    }
    destination = HERE / "lattice_audit_f0.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: output[key] for key in ("train_fit_groups_scanned", "failures", "max_residual_quantiles", "by_best_class", "resources")}))


if __name__ == "__main__":
    main()
