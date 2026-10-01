"""Fit-only target-contrast quality check for the paired fold-0 8-event labels.

TRAIN class-3 groups fit the optional affine diagnostic. VALID input is limited
to manifest.fit.valid class-3 groups. No evaluation group or is_scored mask is
read. This is label quality, not a causal prediction or Global WP estimate.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.gmetric import _merge_moments, _wp_from_moments

from replay import HERE, ROOT, digest


def moments(y: np.ndarray, p: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Return weighted centered moments for target contrast and one label."""
    active = weight > 0
    y = y[active].astype(np.float64)
    p = p[active].astype(np.float64)
    w = weight[active].astype(np.float64)
    if not len(y):
        return np.zeros(6, dtype=np.float64)
    total = w.sum()
    dy, dp = y - y[0], p - p[0]
    my, mp = np.dot(w, dy) / total, np.dot(w, dp) / total
    dy, dp = dy - my, dp - mp
    return np.array([total, y[0] + my, p[0] + mp,
                     np.dot(w, dy * dy), np.dot(w, dp * dp), np.dot(w, dy * dp)])


def affine_fit(moment: np.ndarray) -> tuple[float, float]:
    """Predict target contrast from the label using weighted TRAIN moments."""
    slope = moment[5] / moment[4] if moment[4] > 1e-12 else 0.0
    intercept = moment[1] - slope * moment[2]
    return float(intercept), float(slope)


def affine_mse(moment: np.ndarray, intercept: float, slope: float) -> float:
    """Evaluate a frozen affine calibration from centered sufficient statistics."""
    total = moment[0]
    if total <= 0:
        return float("nan")
    bias = intercept + slope * moment[2] - moment[1]
    return float((moment[3] + slope * slope * moment[4] - 2 * slope * moment[5])
                 / total + bias * bias)


def main() -> None:
    start_wall, start_cpu = time.monotonic(), time.process_time()
    split_path = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
    split = json.loads(split_path.read_text())
    index_path = HERE / "aux_class_index_f0.json"
    class_index = json.loads(index_path.read_text())
    cache_root = HERE / "cache_f0_v3"
    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    verification = json.loads((HERE / "cache_verification_f0_v3.json").read_text())
    if (class_index["split_manifest_sha256"] != digest(split_path)
            or metadata["split_manifest_sha256"] != digest(split_path)
            or verification["cache_metadata_sha256"] != digest(metadata_path)
            or verification["resources"]["exit_code"] != 0):
        raise ValueError("class index/cache/split provenance mismatch")
    records = {}
    aggregate = {}
    for partition in ("train", "valid"):
        groups = class_index["fit_class3_groups"][partition]
        if not set(groups).issubset(split["fit"][partition]):
            raise ValueError(f"{partition}: class index includes a group outside fit pool")
        if partition == "valid" and set(groups) & set(split["evaluation"]["valid"]):
            raise ValueError("evaluation group entered fit-valid diagnostic")
        source = pq.ParquetFile(ROOT / "datasets" / f"{partition}.parquet")
        code = np.load(cache_root / "code" / f"quote_path_{partition}.npy", mmap_mode="r")
        quote = np.load(cache_root / "quote8" / f"quote_path_{partition}.npy", mmap_mode="r")
        if code.shape != quote.shape or code.shape[0] != source.num_row_groups:
            raise ValueError("cache/source shape mismatch")
        result = {"groups": len(groups), "required_rows": 0, "eligible_rows": 0,
                  "changed_label_rows": 0, "full_target_mass": 0.0,
                  "eligible_target_mass": 0.0, "changed_target_mass": 0.0,
                  "moments": {"code8": np.zeros(6), "quote8": np.zeros(6)},
                  "raw_sse": {"code8": 0.0, "quote8": 0.0}}
        for position, group in enumerate(groups, 1):
            table = source.read_row_group(group, columns=["t0", "t1", "need_prediction"],
                                          use_threads=False)
            target = np.stack([table[name].to_numpy() for name in ("t0", "t1")], axis=1)
            need = table["need_prediction"].to_numpy().astype(bool)
            if target.shape != (20_000, 2) or need[:99].any() or not need[99:].all():
                raise ValueError(f"{partition}/{group}: required-row contract mismatch")
            if not np.isfinite(target[need]).all():
                raise ValueError(f"{partition}/{group}: nonfinite required target")
            a, b = np.asarray(code[group]), np.asarray(quote[group])
            if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                raise ValueError(f"{partition}/{group}: unequal finite supervision masks")
            if not np.array_equal(a[:, 1], b[:, 1], equal_nan=True):
                raise ValueError(f"{partition}/{group}: horizon24 differs")
            selected = need & np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
            clipped = np.clip(target.astype(np.float64), -2, 2)
            weight = np.abs(clipped).sum(axis=1)
            contrast = 0.5 * (clipped[:, 0] - clipped[:, 1])
            changed = selected & (np.abs(a[:, 0].astype(np.float32) -
                                         b[:, 0].astype(np.float32)) > 0.001)
            result["required_rows"] += int(need.sum())
            result["eligible_rows"] += int(selected.sum())
            result["changed_label_rows"] += int(changed.sum())
            result["full_target_mass"] += float(weight[need].sum())
            result["eligible_target_mass"] += float(weight[selected].sum())
            result["changed_target_mass"] += float(weight[changed].sum())
            for name, label in (("code8", a[:, 0]), ("quote8", b[:, 0])):
                values = label[selected].astype(np.float64)
                w = weight[selected]
                result["moments"][name] = _merge_moments(
                    result["moments"][name], moments(contrast[selected], values, w)
                )
                result["raw_sse"][name] += float(np.dot(w, (values - contrast[selected]) ** 2))
            if position % 1000 == 0:
                print(f"{partition} {position}/{len(groups)} class-3 fit groups", flush=True)
        aggregate[partition] = result
    train, valid = aggregate["train"], aggregate["valid"]
    output_partitions = {}
    affine_parameters = {name: affine_fit(train["moments"][name])
                         for name in ("code8", "quote8")}
    for partition, value in aggregate.items():
        output_partitions[partition] = {
            key: value[key] for key in ("groups", "required_rows", "eligible_rows",
                                       "changed_label_rows", "full_target_mass",
                                       "eligible_target_mass", "changed_target_mass")
        }
        output_partitions[partition]["labels"] = {}
        for name, moment in value["moments"].items():
            intercept, slope = affine_parameters[name]
            output_partitions[partition]["labels"][name] = {
                "weighted_corr_to_clipped_target_contrast": _wp_from_moments(moment),
                "raw_weighted_mse": value["raw_sse"][name] / moment[0],
                "train_fitted_affine_intercept": intercept,
                "train_fitted_affine_slope": slope,
                "train_fitted_affine_weighted_mse": affine_mse(moment, intercept, slope),
                "target_weight": float(moment[0]),
            }
    valid_values = output_partitions["valid"]["labels"]
    correlation_gain = (valid_values["quote8"]["weighted_corr_to_clipped_target_contrast"] -
                        valid_values["code8"]["weighted_corr_to_clipped_target_contrast"])
    mse_gain = (valid_values["code8"]["train_fitted_affine_weighted_mse"] -
                valid_values["quote8"]["train_fitted_affine_weighted_mse"])
    passes = correlation_gain >= 0.005 and mse_gain > 0
    destination = HERE / "fit_label_quality_f0.json"
    if destination.exists():
        raise FileExistsError(destination)
    output = {
        "version": "fit-only-clock-label-quality-v1",
        "command": sys.argv,
        "script_sha256": digest(Path(__file__)),
        "split_manifest_sha256": digest(split_path),
        "class_index_sha256": digest(index_path),
        "cache_metadata_sha256": digest(metadata_path),
        "no_evaluation_groups_or_scoring_masks_read": True,
        "weight": "sum of abs clipped t0 and abs clipped t1, applied to matched finite required rows",
        "target_contrast": "0.5*(clip(t0,-2,2)-clip(t1,-2,2))",
        "partitions": output_partitions,
        "fit_valid_quote_minus_code_weighted_corr": correlation_gain,
        "fit_valid_code_minus_quote_train_affine_mse": mse_gain,
        "passes_frozen_cheap_label_transport_gate": passes,
        "boundary": "Label alignment and target-mass coverage do not measure causal forecast WP or public transfer.",
        "resources": {"wall_seconds": time.monotonic() - start_wall,
                      "cpu_seconds": time.process_time() - start_cpu,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "exit_code": 0},
    }
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fit_valid_quote_minus_code_weighted_corr": correlation_gain,
                      "fit_valid_code_minus_quote_train_affine_mse": mse_gain,
                      "passes_frozen_cheap_label_transport_gate": passes}))


if __name__ == "__main__":
    main()
