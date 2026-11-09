from __future__ import annotations

import argparse
import json
from pathlib import Path

from admission import SPLIT_SHA, digest, read


def validate_checkpoint_binding(terminal: dict, producer: dict, checkpoint_sha: str) -> None:
    """Require the scored producer, terminal receipt, and fixed model bytes to agree."""
    if (
        terminal.get("model_sha256") != checkpoint_sha
        or producer.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError("official score is not bound to the completed fixed endpoint bytes")


def main() -> None:
    """Validate all six complete raw-z endpoints and report paired descriptive deltas."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    runtime, source = Path(plan["runtime"]), Path(plan["source_root"])
    import sys

    sys.path.insert(0, str(source / "src"))
    from wnn.training.sequence_config import load_config

    cells: dict[str, dict] = {}
    for cell in plan["cells"]:
        fit = runtime / "runs" / cell["cell"] / "fit"
        terminal, score = read(fit / "terminal_status.json"), read(fit / "score.json")
        checkpoint_sha = digest(fit / "model.pt")
        producer = score["producer"]
        validate_checkpoint_binding(terminal, producer, checkpoint_sha)
        official = score["score"]
        if (
            terminal.get("status") != "completed"
            or terminal.get("completed_epochs") != 4
            or terminal.get("global_updates") != 3280
            or score.get("split_sha256") != SPLIT_SHA
            or score.get("sequences") != 1873
            or official.get("selected_rows") != 3528273
            or official.get("metric_id") != "global_wp_clip2_v1"
        ):
            raise ValueError(f"incomplete full-valid fixed4E endpoint: {cell['cell']}")
        identity = terminal.get("identity", {})
        if (
            producer.get("config") != load_config(Path(cell["config_path"]))
            or producer.get("teacher") is not None
            or "reference" in producer
            or identity.get("source_sha256") != plan["expected_training_source_sha256"]
            or producer.get("provenance", {}).get("identity") != identity
        ):
            raise ValueError(f"primary endpoint ancestry/config differs: {cell['cell']}")
        events = [
            json.loads(line)
            for line in (fit / "monitor_evaluations.jsonl").read_bytes().splitlines()
        ]
        last = events[-1]
        metrics = last.get("metrics", {})
        if (
            last.get("epoch") != 4
            or metrics.get("valid/sequence_count") != 1873
            or metrics.get("valid/selected_rows") != 3528273
            or metrics.get("valid/wp") != metrics.get("valid/z/wp")
            or metrics.get("valid/loss") != metrics.get("valid/z/wmse")
        ):
            raise ValueError(f"raw-z same-forward valid monitoring is incomplete: {cell['cell']}")
        cells[cell["cell"]] = {
            "arm": cell["arm"],
            "seed": cell["seed"],
            "optimizer": cell["optimizer"],
            "chunk_rows": load_config(Path(cell["config_path"]))["chunk_rows"],
            "official_raw_z": official,
            "valid_z_same_forward": {
                key: value for key, value in metrics.items() if key.startswith("valid/z/")
            },
            "checkpoint_sha256": checkpoint_sha,
            "score_sha256": digest(fit / "score.json"),
        }
    deltas = []
    for seed in (203101, 203102):
        control = cells[f"C0_muon256_a4_s{seed}"]["official_raw_z"]
        for arm, question in (
            ("C1", "AdamW recipe vs Muon reference"),
            ("C2", "512x2 horizon vs 256x4"),
        ):
            candidate = cells[f"{arm}_{'adamw256_a4' if arm == 'C1' else 'muon512_a2'}_s{seed}"][
                "official_raw_z"
            ]
            deltas.append(
                {
                    "seed": seed,
                    "contrast": question,
                    "arm": arm,
                    "delta_wp": candidate["wp"] - control["wp"],
                }
            )
    output = runtime / "comparison.json"
    output.write_text(
        json.dumps(
            {
                "schema": "wnn.transformer-optimizer-horizon-comparison.v1",
                "status": "complete_exposed_fixed4E_screen",
                "metric_id": "global_wp_clip2_v1",
                "split_sha256": SPLIT_SHA,
                "cells": cells,
                "paired_deltas": deltas,
                "interpretation": (
                    "Two paired fitting seeds on repeatedly exposed official validation; "
                    "descriptive only, with no significance or public-gain claim. C1 is an "
                    "AdamW recipe contrast including its decay coverage, not an isolated "
                    "optimizer-only effect."
                ),
                "automatic_followup": False,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
