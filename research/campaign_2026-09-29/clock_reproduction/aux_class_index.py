"""Index class-3 groups in clean fold-0 fit pools using target columns only."""

from __future__ import annotations

import json
import resource
import time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from replay import HERE, QUANTA, ROOT, digest


def main() -> None:
    started = time.monotonic()
    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split = json.loads(split_path.read_text())
    audit_path = HERE / "lattice_audit_f0.json"
    audit = json.loads(audit_path.read_text())
    if audit["split_manifest_sha256"] != digest(split_path):
        raise ValueError("TRAIN lattice audit split mismatch")
    exceptions = {int(value["train_group"]) for value in audit["failures"]}
    if exceptions != {2315}:
        raise ValueError("Unexpected TRAIN lattice exception set")
    totals = Counter()
    selected = {}
    for partition in ("train", "valid"):
        source = pq.ParquetFile(ROOT / "datasets" / f"{partition}.parquet")
        class3 = []
        for group in split["fit"][partition]:
            if partition == "train" and group in exceptions:
                totals["train/lattice_exception_class2"] += 1
                continue
            table = source.read_row_group(group, columns=["t0", "t1"], use_threads=False)
            target = np.stack([table[name].to_numpy() for name in ("t0", "t1")], axis=1)
            errors = [float(np.max(np.abs(np.rint(target / q) * q - target))) for q in QUANTA]
            cls = int(np.argmin(errors))
            if errors[cls] > 0.001:
                raise ValueError(f"Unpreregistered lattice exception {partition}/{group}: {errors[cls]}")
            totals[f"{partition}/class{cls}"] += 1
            if cls == 3:
                class3.append(int(group))
        selected[partition] = class3
    output = {
        "version": "clean-fold0-auxiliary-class-index-v1",
        "split_manifest_sha256": digest(split_path),
        "lattice_audit_sha256": digest(audit_path),
        "exception_train_groups": sorted(exceptions),
        "fit_class3_groups": selected,
        "fit_class_counts": dict(totals),
        "evaluation_groups_read": False,
        "resources": {"wall_seconds": time.monotonic() - started,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "exit_code": 0},
    }
    destination = HERE / "aux_class_index_f0.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"class_counts": dict(totals), "resources": output["resources"]}))


if __name__ == "__main__":
    main()
