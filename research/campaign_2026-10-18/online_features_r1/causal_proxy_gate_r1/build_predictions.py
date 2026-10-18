from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    bounded = np.clip(logit, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-bounded))


def _gate_sequence(
    tl3: np.ndarray,
    r3: np.ndarray,
    proxy: np.ndarray,
    config: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows = proxy.shape[0]
    if tl3.shape != (20_000, 2) or r3.shape != tl3.shape or proxy.shape != (19_900, 2):
        raise ValueError("unexpected sequence shape for causal gate")

    initial = float(config["initial_r3_weight"])
    minimum = float(config["prior_min_r3_weight"])
    maximum = float(config["prior_max_r3_weight"])
    eta = float(config["eta"])
    warmup = int(config["warmup_observations"])
    prior_logit = math.log(initial / (1.0 - initial))
    output = np.empty((19_901, 2), dtype=np.float32)
    weights = np.empty((rows, 2), dtype=np.float32)
    output[0] = 0.75 * tl3[99] + 0.25 * r3[99]

    half_life = int(gate["half_life_steps"])
    decay = math.exp(math.log(0.5) / half_life)
    complement = 1.0 - decay
    ema_tl3 = np.zeros(2, dtype=np.float64)
    ema_r3 = np.zeros(2, dtype=np.float64)
    ema_scale = np.zeros(2, dtype=np.float64)
    weights.fill(initial)
    for position in range(rows):
        z = np.clip(proxy[position].astype(np.float64), -2.0, 2.0)
        prior_row = 99 + position
        current_row = 100 + position
        previous_tl3 = np.clip(tl3[prior_row].astype(np.float64), -2.0, 2.0)
        previous_r3 = np.clip(r3[prior_row].astype(np.float64), -2.0, 2.0)
        pseudo_weight = np.abs(z)
        ema_tl3 = decay * ema_tl3 + complement * pseudo_weight * np.square(z - previous_tl3)
        ema_r3 = decay * ema_r3 + complement * pseudo_weight * np.square(z - previous_r3)
        ema_scale = decay * ema_scale + complement * pseudo_weight * np.maximum(np.square(z), 0.01)

        if position + 1 >= warmup:
            normalized_tl3 = ema_tl3 / np.maximum(ema_scale, 1e-12)
            normalized_r3 = ema_r3 / np.maximum(ema_scale, 1e-12)
            logit = prior_logit - eta * (normalized_r3 - normalized_tl3)
            r3_weight = np.clip(_sigmoid(logit), minimum, maximum)
        else:
            r3_weight = np.full(2, initial, dtype=np.float64)
        weights[position] = r3_weight.astype(np.float32)
        output[position + 1] = (
            (1.0 - r3_weight) * tl3[current_row] + r3_weight * r3[current_row]
        ).astype(np.float32)

    return (
        output,
        weights,
        {
            "rows": rows,
            "targets": 2,
            "weight_min": weights.min(axis=0).astype(float).tolist(),
            "weight_median": np.median(weights, axis=0).astype(float).tolist(),
            "weight_max": weights.max(axis=0).astype(float).tolist(),
        },
    )


def build(recipe_path: Path, output_root: Path, max_groups: int | None = None) -> dict[str, Any]:
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    if sha256_file(Path(__file__).resolve()) != recipe["code"]["prediction_builder_sha256"]:
        raise ValueError("prediction builder differs from its frozen source hash")
    repo_root = Path.cwd().resolve(strict=True)
    scratch_root = Path(str(recipe["resources"]["scratch_root"])).resolve(strict=True)
    output_root = output_root.resolve()
    if not output_root.is_relative_to(scratch_root):
        raise ValueError(f"prediction output must stay under physical scratch {scratch_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite gate artifacts in {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    proxy = Path(recipe["proxy"]["predictions_path"])
    if sha256_file(proxy) != recipe["proxy"]["predictions_sha256"]:
        raise ValueError("proxy prediction hash changed")
    proxy_predictions = np.load(proxy, mmap_mode="r", allow_pickle=False)
    expected_proxy_rows = len(recipe["selection"]["groups"]) * 19_900
    if proxy_predictions.shape != (expected_proxy_rows, 2) or proxy_predictions.dtype != np.float32:
        raise ValueError(
            f"unexpected proxy array: {proxy_predictions.shape}/{proxy_predictions.dtype}"
        )

    members: dict[str, np.ndarray] = {}
    for name, source in recipe["expert_sources"].items():
        prediction_path = repo_root / source["prediction_path"]
        if sha256_file(prediction_path) != source["prediction_sha256"]:
            raise ValueError(f"{name} prediction hash changed")
        mapped = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        if (
            list(mapped.shape) != source["prediction_shape"]
            or str(mapped.dtype) != source["prediction_dtype"]
        ):
            raise ValueError(f"{name} prediction array metadata changed")
        members[name] = mapped

    endpoints = list(recipe["endpoints"])
    gate_names = [item["id"] for item in recipe["gate"]["gate_configs"]]
    if endpoints != ["tl3", "r3", "fixed_75_tl3_25_r3", *gate_names]:
        raise ValueError("endpoint ordering differs from the frozen gate implementation")
    group_specs = recipe["selection"]["groups"]
    if max_groups is not None:
        if not 1 <= max_groups <= len(group_specs):
            raise ValueError("--limit-groups must be between 1 and the frozen group count")
        group_specs = group_specs[:max_groups]
    group_count = len(group_specs)
    rows_per_group = 19_901
    prediction_shape = (len(endpoints), group_count * rows_per_group, 2)
    predictions_path = output_root / "gate_predictions.npy"
    if predictions_path.exists():
        raise FileExistsError(predictions_path)
    predictions = np.lib.format.open_memmap(
        predictions_path, mode="w+", dtype=np.float32, shape=prediction_shape
    )
    weight_summaries: dict[str, list[dict[str, Any]]] = {name: [] for name in gate_names}

    for group_index, group in enumerate(group_specs):
        start = group_index * rows_per_group
        destination = slice(start, start + rows_per_group)
        cache_index = int(group["tl3_cache_index"])
        tl3 = np.asarray(members["tl3"][cache_index], dtype=np.float32)
        r3 = np.asarray(members["r3"][int(group["r3_cache_index"])], dtype=np.float32)
        proxy_start = int(group["proxy_offset"])
        proxy_end = proxy_start + int(group["proxy_rows"])
        group_proxy = np.asarray(proxy_predictions[proxy_start:proxy_end], dtype=np.float64)
        if (
            not np.isfinite(tl3[99:]).all()
            or not np.isfinite(r3[99:]).all()
            or not np.isfinite(group_proxy).all()
        ):
            raise ValueError(f"nonfinite source prediction for row group {group['row_group']}")

        predictions[0, destination] = tl3[99:]
        predictions[1, destination] = r3[99:]
        predictions[2, destination] = 0.75 * tl3[99:] + 0.25 * r3[99:]
        for gate_index, gate in enumerate(recipe["gate"]["gate_configs"], start=3):
            gate_output, weights, summary = _gate_sequence(
                tl3, r3, group_proxy, recipe["gate"], gate
            )
            predictions[gate_index, destination] = gate_output
            weight_summaries[gate["id"]].append(summary)

    predictions.flush()
    del predictions
    digest = sha256_file(predictions_path)
    receipt = {
        "schema": "wnn.causal_proxy_gate_predictions.v1",
        "status": (
            "one_group_resource_profile_without_labels_or_masks"
            if max_groups is not None
            else "prediction_build_completed_without_labels_or_masks"
        ),
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "prediction_builder_sha256": sha256_file(Path(__file__).resolve()),
        "prediction_path": str(predictions_path),
        "prediction_sha256": digest,
        "prediction_shape": list(prediction_shape),
        "prediction_dtype": "float32",
        "selected_groups": group_count,
        "full_recipe_group_count": len(recipe["selection"]["groups"]),
        "profile_only_prefix": max_groups is not None,
        "endpoints": endpoints,
        "evaluation_targets_or_masks_read": False,
        "sequence_state_reset": "fresh EWA state for each selected source-qualified sequence",
        "gate_weight_diagnostics_by_endpoint": weight_summaries,
    }
    receipt_path = output_root / "prediction_receipt.json"
    _write_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build causal proxy-gate predictions without labels."
    )
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-groups", type=int)
    args = parser.parse_args()
    build(args.recipe.resolve(strict=True), args.output_root, args.limit_groups)


if __name__ == "__main__":
    main()
