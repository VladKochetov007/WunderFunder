from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SCALES = [0.6569, 0.4685, 0.2413, 0.1254]


def main() -> None:
    """Freeze samples conservatively excluding possible prior feature exposure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tables = {split: np.load(ROOT / "data" / f"census_{split}.npy") for split in ["train", "valid"]}
    exposed: dict[str, set[int]] = {"train": {4178, 6698, 5522, 7727, 8610}, "valid": set()}
    for seed, size in [(0, 30), (1, 20), (2, 40), (3, 48)]:
        exposed["train"].update(
            np.random.default_rng(seed).choice(len(tables["train"]), size, False)
        )
    for seed, size in [(5, 10), (6, 3), (7, 25), (11, 24), (12, 16), (13, 30)]:
        rng = np.random.default_rng(seed)
        for scale in SCALES:
            candidates = tables["train"][np.round(tables["train"][:, 2], 4) == scale, 1].astype(int)
            exposed["train"].update(rng.choice(candidates, size, False))
    for seed, size in [(8, 60), (9, 30), (10, 40)]:
        exposed["valid"].update(
            np.random.default_rng(seed).choice(len(tables["valid"]), size, False)
        )
    rng = np.random.default_rng(14)
    exposed["train"].update(rng.choice(len(tables["train"]), 80, False))
    exposed["valid"].update(rng.choice(len(tables["valid"]), 60, False))
    rng = np.random.default_rng(610103)
    records = {}
    for split, size in [("train", 48), ("valid", 32)]:
        candidates = np.array(sorted(set(range(len(tables[split]))) - exposed[split]))
        selected = rng.choice(candidates, size, False)
        records[split] = [
            {
                "row_group": int(rg),
                "seq_ix": int(tables[split][rg, 0]),
                "prior_target_census_exposure": True,
                "reconstructed_prior_feature_exposure": False,
            }
            for rg in selected
        ]
    provenance = {
        "seed": 610103,
        "samples": records,
        "possibly_exposed_row_groups": {
            split: sorted(int(rg) for rg in groups) for split, groups in exposed.items()
        },
        "limitations": [
            "all targets previously inspected by census",
            "unknown exploratory CLI arguments cannot be reconstructed",
            "possible prior scripts counted even when execution unconfirmed",
            "disjoint from reconstructed own feature samples, not campaign",
        ],
    }
    (ROOT / "continuation_samples.json").write_text(json.dumps(provenance, indent=2))
    state = json.loads((ROOT / "STATE.json").read_text())
    state["phase"] = "samples_frozen_ready_for_probe"
    state["inspected_sequence_ids"]["continuation_frozen_samples"] = records
    state["artifacts"] += ["HYPOTHESES.json", "continuation_samples.json"]
    (ROOT / "STATE.json").write_text(json.dumps(state, indent=2))
    logging.info("Frozen 48 train / 32 valid sequences; all census-exposed.")
    logging.info("Prior possible feature groups: %s", {k: len(v) for k, v in exposed.items()})


if __name__ == "__main__":
    main()
