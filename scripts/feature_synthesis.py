"""Pool both prespecified E415 directions without selecting a cue or head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_cues import CUE_NAMES, CUE_SCOPES
from feature_screen import compare_components, digest_file, load_cache


def synthesize(root: Path, output: Path) -> dict:
    """Validate row/cache bindings and report every fixed reciprocal-head result."""
    if output.exists():
        raise FileExistsError(output)
    caches = [load_cache(root / f"cache/c20261004_feature_f0_{p}.npz") for p in ("b", "a")]
    if set(caches[0][0]["component"]) & set(caches[1][0]["component"]):
        raise ValueError("Reciprocal evaluation components overlap")
    data = {
        key: np.concatenate([cache[key] for cache, _ in caches])
        for key in ("y", "component", "sequence", "row")
    }
    families = {}
    inputs = {}
    for cue in CUE_NAMES:
        arms = {key: [] for key in ("unchanged", "candidate", "placebo", "capacity")}
        for direction, (cache, metadata) in enumerate(caches):
            folder = root / "research/campaign_2026-10-04/feature_results" / f"e415_d{direction}_{cue}_v1"
            report_path = folder / "report.json"
            report = json.loads(report_path.read_text())
            if (report["cue"] != cue or report["ridge"] != 0.01
                    or report["bootstrap"] != 2000 or report["backend"] != "numpy"
                    or report["eval_cache_sha256"] != metadata["sha256"]
                    or report["fit_cache_sha256"] != caches[1 - direction][1]["sha256"]):
                raise ValueError(f"Changed prespecified contrast: {folder}")
            prediction_path = folder / "predictions.npz"
            with np.load(prediction_path, allow_pickle=False) as predictions:
                for key in ("component", "sequence", "row"):
                    if not np.array_equal(predictions[key], cache[key]):
                        raise ValueError(f"Prediction row identity differs: {folder}/{key}")
                if not np.array_equal(predictions["unchanged"], cache["pred"]):
                    raise ValueError(f"Backbone predictions changed: {folder}")
                for key in arms:
                    value = predictions[key]
                    if value.shape != cache["y"].shape or not np.isfinite(value).all():
                        raise ValueError(f"Invalid predictions: {folder}/{key}")
                    arms[key].append(value)
            for path in (report_path, prediction_path):
                inputs[str(path.relative_to(root))] = digest_file(path)
        predictions = {key: np.concatenate(values) for key, values in arms.items()}
        scores = compare_components(data, predictions, 2000)
        # compare_components aggregates sequences before calling the bootstrap;
        # correct its descriptive count without changing any score or interval.
        for score in scores.values():
            if "n_sequences" in score:
                score["n_sequences"] = int(len(np.unique(data["sequence"])))
        families[cue] = {"mechanism_scope": CUE_SCOPES[cue], "pooled": scores}
    result = {
        "experiment": "E-415",
        "rows": len(data["row"]),
        "sequences": len(np.unique(data["sequence"])),
        "components": len(np.unique(data["component"])),
        "families": families,
        "inputs": inputs,
        "sampling": "fixed jittered rows intersected with scoring mask; empty groups contribute zero rows",
        "selection": "none; every prespecified family and matched arm reported",
        "uncertainty": "component bootstrap conditional on fitted backbones and reciprocal heads; does not cover fitting, seed or search uncertainty",
        "scope": "previously exposed fold0, selected rows, proxy mechanisms; neither full-validation nor public score",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = synthesize(args.root.resolve(), args.output)
    print(json.dumps({"rows": report["rows"], "families": len(report["families"])}))


if __name__ == "__main__":
    main()
