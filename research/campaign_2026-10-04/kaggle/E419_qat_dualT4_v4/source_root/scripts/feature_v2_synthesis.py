"""Pool every fixed E418 arm over the two reciprocal component directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_screen import digest_file, load_cache
from feature_v2_screen import FAMILIES, compare, load_histories


def synthesize(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    caches = [load_cache(root / f"cache/c20261004_feature_f0_{suffix}.npz")
              for suffix in ("b", "a")]
    if set(caches[0][0]["component"]) & set(caches[1][0]["component"]):
        raise ValueError("Reciprocal evaluation components overlap")
    base = root / "research/campaign_2026-10-04/feature_v2_results"
    histories = [load_histories(base / f"exact_histories_f0_{suffix}.npz", data, metadata)
                 for suffix, (data, metadata) in zip(("b", "a"), caches)]
    pooled = {key: np.concatenate([data[key] for data, _ in caches])
              for key in ("y", "component", "sequence", "row")}
    families, inputs = {}, {}
    for family in FAMILIES:
        arms = {}
        expected_arms = None
        for direction, (cache, metadata) in enumerate(caches):
            folder = base / f"exact_f0_d{direction}_{family}_cpu"
            report_path, predictions_path = folder / "report.json", folder / "predictions.npz"
            report = json.loads(report_path.read_text())
            if (report["family"] != family or report["ridge"] != .01
                    or report["bootstrap"] != 2000 or report["backend"] != "numpy"
                    or report["eval_cache_sha256"] != metadata["sha256"]
                    or report["fit_cache_sha256"] != caches[1-direction][1]["sha256"]
                    or report["eval_histories_sha256"] != histories[direction][1]["sha256"]
                    or report["fit_histories_sha256"] != histories[1-direction][1]["sha256"]):
                raise ValueError(f"Changed prespecified family: {folder}")
            with np.load(predictions_path, allow_pickle=False) as predictions:
                for key in ("component", "sequence", "row"):
                    if not np.array_equal(predictions[key], cache[key]):
                        raise ValueError(f"Changed row identity: {folder}/{key}")
                if not np.array_equal(predictions["unchanged"], cache["pred"]):
                    raise ValueError("Changed untouched predictor")
                names = set(predictions.files) - {"component", "sequence", "row"}
                if expected_arms is not None and names != expected_arms:
                    raise ValueError("Directions have different control inventories")
                expected_arms = names
                for name in sorted(names):
                    values = predictions[name]
                    if values.shape != cache["y"].shape or not np.isfinite(values).all():
                        raise ValueError("Invalid prediction array")
                    arms.setdefault(name, []).append(values)
            for path in (report_path, predictions_path):
                inputs[str(path.relative_to(root))] = digest_file(path)
        families[family] = compare(pooled, {name: np.concatenate(value)
                                           for name, value in arms.items()}, 2000)
    result = {"experiment": "E-418", "rows": len(pooled["row"]),
              "sequences": len(np.unique(pooled["sequence"])),
              "components": len(np.unique(pooled["component"])),
              "families": families, "inputs": inputs,
              "scope": "Previously exposed fold0, fixed sampled rows, literal frozen-state head screens",
              "selection": "None; every fixed arm retained; no correction for family search",
              "uncertainty": "Component resampling conditional on fitted backbone and reciprocal heads"}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = synthesize(args.root.resolve(), args.output)
    print(json.dumps({"rows": result["rows"], "families": len(result["families"])}))


if __name__ == "__main__":
    main()
