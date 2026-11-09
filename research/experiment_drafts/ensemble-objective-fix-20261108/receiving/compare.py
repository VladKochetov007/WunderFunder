from __future__ import annotations

import argparse
import json
from pathlib import Path

from admission import SPLIT_SHA, digest, read

from wnn.training.sequence_blend_prediction import validate_blend_producer
from wnn.training.sequence_config import load_config
from wnn.training.sequence_provenance import source_identity


def main() -> None:
    """Report all four fixed4E primary scores and same-forward z diagnostics without selecting seeds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    runtime, source = Path(plan["runtime"]), Path(plan["source_root"])
    expected_source = source_identity(source, source / "scripts/train_sequence.py")
    baseline = read(Path(plan["reference_root"]).parent / "b300/baseline-score.json")
    if (
        baseline.get("split_sha256") != SPLIT_SHA
        or abs(baseline["score"]["wp"] - 0.6923774248169172) > 2e-6
    ):
        raise ValueError("qualified full-valid reference baseline differs")
    records = {}
    for cell in plan["cells"]:
        folder = runtime / "runs" / cell["cell"] / "fit"
        score, terminal = read(folder / "score.json"), read(folder / "terminal_status.json")
        if (
            terminal.get("completed_epochs") != 4
            or terminal.get("global_updates") != 3280
            or terminal.get("status") != "completed"
            or score.get("split_sha256") != SPLIT_SHA
            or score["score"]["metric_id"] != "global_wp_clip2_v1"
        ):
            raise ValueError("incomplete fixed-final endpoint or changed official metric/split")
        producer = score["producer"]
        validate_blend_producer(producer, SPLIT_SHA, [tuple(v) for v in producer["fit_members"]])
        candidate = producer["candidate"]
        if (
            candidate["config"] != load_config(Path(cell["config_path"]))
            or candidate["provenance"]["identity"]["source_sha256"] != expected_source["sha256"]
            or producer["composition"]["alpha"] != 0.1
        ):
            raise ValueError("primary endpoint differs from fixed recipe/source/reference mixture")
        events = [
            json.loads(line)
            for line in (folder / "monitor_evaluations.jsonl").read_bytes().splitlines()
        ]
        last = events[-1]
        if (
            last.get("epoch") != 4
            or last["metrics"].get("valid/sequence_count") != 1873
            or last["metrics"].get("valid/selected_rows") != 3528273
        ):
            raise ValueError("standalone diagnostic lacks complete final official valid coverage")
        records[cell["cell"]] = {
            "primary_mixture": score["score"],
            "standalone_z_same_forward": {
                k: v for k, v in last["metrics"].items() if k.startswith("valid/z/")
            },
            "score_sha256": digest(folder / "score.json"),
            "checkpoint_sha256": digest(folder / "model.pt"),
        }
    paired = []
    for seed in (101101, 101102):
        left, right = (
            records[f"labels_s{seed}"]["primary_mixture"]["wp"],
            records[f"mixture_corr_s{seed}"]["primary_mixture"]["wp"],
        )
        paired.append(
            {
                "seed": seed,
                "package_minus_control": right - left,
                "control_minus_reference": left - baseline["score"]["wp"],
                "package_minus_reference": right - baseline["score"]["wp"],
            }
        )
    result = {
        "schema": "wnn.ensemble-objective-comparison.v1",
        "status": "complete_exposed_fixed4E_screen",
        "reference": baseline["score"],
        "cells": records,
        "paired_deltas": paired,
        "interpretation": "two fitting seeds on repeatedly exposed valid; package factors include curvature removal, correlation and std floor; no significance or public gain established",
        "automatic_followup": False,
    }
    with (runtime / "comparison.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
