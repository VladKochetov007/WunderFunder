from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.continuation_recipes import input_paths
from scripts.kaggle_assets import prepare


PORTABLE = (
    "research/campaign_2026-10-04/kaggle/"
    "W384_fold1_dualT4_v1/portable_inputs_f1.json"
)
DATASET = "vladkochetov07/wnn-c20261004-fold1-assets"
DESTINATION = Path("artifacts/kaggle_assets/wnn-c20261004-fold1-assets")


def main() -> None:
    raw = {"datasets/train.parquet", "datasets/valid.parquet", "datasets/valid_mask.parquet"}
    inputs = [
        name
        for name in input_paths(1)
        if name not in raw
        and name != "research/campaign_2026-10-04/portable_inputs.json"
    ]
    inputs.append(PORTABLE)
    prepare(ROOT, inputs, ROOT / DESTINATION, DATASET)


if __name__ == "__main__":
    main()
