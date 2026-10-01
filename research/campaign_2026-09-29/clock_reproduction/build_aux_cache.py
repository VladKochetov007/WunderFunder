"""Build matched clean-fold code8/code24 and quote8/code24 auxiliary caches.

The TRAIN-label decoder sees only manifest.fit.train. Labels are written only
for manifest.fit.train and manifest.fit.valid. Evaluation groups are never read.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from replay import (
    HERE,
    LENGTH,
    Q3,
    ROOT,
    class_and_ticks,
    digest,
    fit_decoder,
    midpoint_mean,
    path,
    read_group,
)


def label_pair(prices: np.ndarray, spreads: dict, moves: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    state = path(prices, spreads, moves)
    origin = np.searchsorted(state["events"], np.arange(LENGTH), side="right") - 1
    code8, code8_ok = midpoint_mean(state, "code", 8)
    quote8, quote8_ok = midpoint_mean(state, "quote", 8)
    code24, code24_ok = midpoint_mean(state, "code", 24)
    shared8 = code8_ok & quote8_ok
    code = np.full((LENGTH, 2), np.nan, dtype=np.float16)
    candidate = np.full((LENGTH, 2), np.nan, dtype=np.float16)
    code_rows = shared8[origin]
    next_rows = code24_ok[origin]
    code[code_rows, 0] = np.clip(Q3 * code8[origin[code_rows]], -2, 2)
    candidate[code_rows, 0] = np.clip(Q3 * quote8[origin[code_rows]], -2, 2)
    value24 = np.clip(Q3 * code24[origin[next_rows]], -2, 2).astype(np.float16)
    code[next_rows, 1] = value24
    candidate[next_rows, 1] = value24
    if not np.array_equal(np.isfinite(code), np.isfinite(candidate)):
        raise ValueError("Paired auxiliary row masks differ")
    if not np.array_equal(code[:, 1], candidate[:, 1], equal_nan=True):
        raise ValueError("Horizon-24 control differs")
    return code, candidate, {
        "code_events": len(state["events"]) - 1,
        "known_nonzero_quote_events": len(state["quotes"]),
        "unknown_transitions": int(state["unknown_prefix"][-1]),
        "shared_finite_horizon8_required_rows": int(np.sum(code_rows[99:])),
        "finite_horizon24_required_rows": int(np.sum(next_rows[99:])),
        "different_horizon8_required_rows": int(np.sum(
            code_rows[99:] & (np.abs(code[99:, 0].astype(np.float32) -
                                     candidate[99:, 0].astype(np.float32)) > 1e-3)
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started_wall, started_cpu = time.monotonic(), time.process_time()
    manifest_path = args.split_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["fold"] != 0:
        raise ValueError("This preregistered first screen is fold 0")
    fit_train = [int(group) for group in manifest["fit"]["train"]]
    fit_valid = [int(group) for group in manifest["fit"]["valid"]]
    evaluation = {int(group) for group in manifest["evaluation"]["valid"]}
    if set(fit_valid) & evaluation or len(fit_train) != len(set(fit_train)):
        raise ValueError("Evaluation/fit overlap or duplicate train group")
    lattice_audit_path = HERE / "lattice_audit_f0.json"
    lattice_audit = json.loads(lattice_audit_path.read_text())
    if lattice_audit["split_manifest_sha256"] != digest(manifest_path):
        raise ValueError("TRAIN lattice audit belongs to another split")
    exceptions = lattice_audit["failures"]
    if len(exceptions) != 1 or exceptions[0]["train_group"] != 2315 or exceptions[0]["best_class"] != 2:
        raise ValueError("Unexpected TRAIN lattice exception set; review before proceeding")
    class_index_path = HERE / "aux_class_index_f0.json"
    class_index = json.loads(class_index_path.read_text())
    if (class_index["split_manifest_sha256"] != digest(manifest_path)
            or class_index["lattice_audit_sha256"] != digest(lattice_audit_path)):
        raise ValueError("Auxiliary class index provenance mismatch")
    decoder_fit_train = [int(group) for group in class_index["fit_class3_groups"]["train"]]
    fit_valid_class3 = [int(group) for group in class_index["fit_class3_groups"]["valid"]]
    if not set(decoder_fit_train).issubset(fit_train) or not set(fit_valid_class3).issubset(fit_valid):
        raise ValueError("Class index contains a group outside the fit pool")
    if (len(decoder_fit_train) != len(set(decoder_fit_train))
            or len(fit_valid_class3) != len(set(fit_valid_class3))
            or sum(value for key, value in class_index["fit_class_counts"].items()
                   if key.startswith("train/")) != len(fit_train)
            or sum(value for key, value in class_index["fit_class_counts"].items()
                   if key.startswith("valid/")) != len(fit_valid)):
        raise ValueError("Class index count or uniqueness mismatch")
    if 2315 in decoder_fit_train:
        raise ValueError("Known class-2 lattice outlier entered the class-3 decoder")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    train_source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    valid_source = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    print(f"Fitting decoder on {len(decoder_fit_train)} class-3 clean TRAIN groups; class-2 lattice outlier 2315 excluded", flush=True)
    spreads, moves, fit = fit_decoder(train_source, decoder_fit_train)
    (args.out / "code").mkdir()
    (args.out / "quote8").mkdir()
    arrays = {}
    for variant in ("code", "quote8"):
        arrays[variant] = {}
        for split, source in (("train", train_source), ("valid", valid_source)):
            destination = args.out / variant / f"quote_path_{split}.npy"
            arrays[variant][split] = np.lib.format.open_memmap(
                destination, mode="w+", dtype=np.float16,
                shape=(source.num_row_groups, LENGTH, 2),
            )
    figures = Counter()
    for split, source, groups in (("train", train_source, decoder_fit_train),
                                  ("valid", valid_source, fit_valid_class3)):
        for position, group in enumerate(groups, 1):
            prices, target, _ = read_group(source, group, labels=True)
            assert target is not None
            regime, _ = class_and_ticks(target)
            if regime != 3:
                raise ValueError(f"Class index says class 3 but {split}/{group} is class {regime}")
            code, quote8, counters = label_pair(prices, spreads, moves)
            arrays["code"][split][group] = code
            arrays["quote8"][split][group] = quote8
            figures.update({f"{split}/{key}": value for key, value in counters.items()})
            if position % 250 == 0:
                print(f"{split} {position}/{len(groups)} fit groups cached", flush=True)
    for variant in arrays.values():
        for value in variant.values():
            value.flush()
    del arrays
    files = {
        f"{variant}/{split}": {
            "path": str(args.out / variant / f"quote_path_{split}.npy"),
            "sha256": digest(args.out / variant / f"quote_path_{split}.npy"),
        }
        for variant in ("code", "quote8") for split in ("train", "valid")
    }
    output = {
        "version": "clean-fold0-quote8-control-v1",
        "command": sys.argv,
        "split_manifest_sha256": digest(manifest_path),
        "train_lattice_audit_sha256": digest(lattice_audit_path),
        "auxiliary_class_index_sha256": digest(class_index_path),
        "decoder_fit_train_groups": decoder_fit_train,
        "auxiliary_excluded_train_groups": [2315],
        "source_script_sha256": digest(Path(__file__)),
        "decoder_script_sha256": digest(HERE / "replay.py"),
        "fit_train_groups": fit_train,
        "fit_valid_groups": fit_valid,
        "evaluation_valid_groups": sorted(evaluation),
        "fit": fit,
        "class_sequence_counts": class_index["fit_class_counts"],
        "row_counts": dict(figures),
        "files": files,
        "paired_control": "Same TRAIN-fitted decoder, same rows and finite mask; both use code24, candidate changes code8 to nonzero decoded quote8.",
        "non_class3_values": "zero; trainer's sequence-class gate makes their auxiliary weight zero",
        "excluded_groups": "zero-initialized and never read by a clean-fold no-evaluation trainer",
        "resources": {"wall_seconds": time.monotonic() - started_wall,
                      "cpu_seconds": time.process_time() - started_cpu,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "exit_code": 0},
    }
    destination = args.out / "metadata.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {destination}; resources={output['resources']}", flush=True)


if __name__ == "__main__":
    main()
