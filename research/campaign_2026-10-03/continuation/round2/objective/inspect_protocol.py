from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path.cwd()


def main() -> None:
    """Inspect metadata only; no label distribution or score is computed."""
    manifest = json.loads(
        (ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text()
    )
    print(json.dumps({"manifest_keys": list(manifest), "fit_keys": list(manifest.get("fit", {}))}))
    for run in ("c20261003_combo_s3_f0", "c20261003_w320_s3_f0"):
        provenance = json.loads((ROOT / "artifacts/runs" / run / "provenance.json").read_text())
        print(
            json.dumps(
                {
                    "run": run,
                    "args": provenance.get("args"),
                    "keys": list(provenance),
                    "fit_counts": {
                        k: len(v) for k, v in provenance.get("fit_row_groups", {}).items()
                    },
                }
            )
        )
    for name in (
        "propcache_c20260929_prop_f0_train.npy",
        "pred_c20260929_since_s3_f0_train.npy",
        "pred_c20260929_since_s4_f0_train.npy",
    ):
        array = np.load(ROOT / "cache" / name, mmap_mode="r")
        print(json.dumps({"cache": name, "shape": array.shape, "dtype": str(array.dtype)}))
    print((ROOT / "cache/propcache_c20260929_prop_f0_train.json").read_text())


if __name__ == "__main__":
    main()
