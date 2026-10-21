"""Apply causal steady-state Kalman filters to a pinned prediction cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PACKET = Path(__file__).resolve().parent
RECIPE_PATH = PACKET / "recipe.json"
FIXED_Q_OVER_R = 0.25
PREFIX_SIZE = 99


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ints(values: list[int]) -> str:
    """Hash an ordered integer list using canonical compact JSON bytes."""
    payload = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def robust_variance(values: np.ndarray) -> float:
    """Estimate variance from a short vector using its scaled median deviation."""
    centered = values - np.median(values)
    mad = float(np.median(np.abs(centered)))
    return (1.4826 * mad) ** 2


def prefix_q_over_r(prefix: np.ndarray) -> float:
    """Estimate local-level process/observation variance ratio from 99 outputs."""
    values = np.asarray(prefix, dtype=np.float64)
    first_difference = np.diff(values)
    second_difference = np.diff(first_difference)
    a = robust_variance(first_difference)
    b = robust_variance(second_difference)
    r_hat = max((b - 2.0 * a) / 2.0, 0.01 * a, 1e-12)
    q_hat = max(3.0 * a - b, 0.0)
    return float(np.clip(q_hat / r_hat, 0.01, 4.0))


def steady_state_gain(q_over_r: float) -> float:
    """Return the positive steady-state scalar random-walk Kalman gain."""
    ratio = float(np.clip(q_over_r, 0.01, 4.0))
    return (np.sqrt(ratio * ratio + 4.0 * ratio) - ratio) / 2.0


def filter_predictions(predictions: np.ndarray, gain: float) -> np.ndarray:
    """Causally filter one or more channels, resetting at each sequence call."""
    values = np.asarray(predictions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("expected one finite [time, channels] prediction sequence")
    output = np.empty_like(values)
    output[0] = values[0]
    for position in range(1, len(values)):
        output[position] = output[position - 1] + gain * (
            values[position] - output[position - 1]
        )
    return output


def verify_inputs(
    root: Path, recipe: dict[str, object]
) -> tuple[list[int], np.ndarray, list[int], list[int]]:
    """Verify source, split, ancestry receipt and prediction order before filtering."""
    split_path = root / str(recipe["split_manifest"])
    if sha256_file(split_path) != recipe["split_manifest_sha256"]:
        raise ValueError("source-qualified split differs from the frozen recipe")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    groups = [int(group) for group in split["evaluation"]["valid"]]
    if len(groups) != 468 or len(groups) != len(set(groups)):
        raise ValueError("the frozen F0 evaluation membership changed")
    sequence_rows = {
        int(row["row_group"]): row for row in split["sequences"]["valid"]
    }
    if any(group not in sequence_rows for group in groups):
        raise ValueError("the source-qualified split lacks a selected F0 sequence")
    sequence_ids = [int(sequence_rows[group]["seq_ix"]) for group in groups]
    components = [int(sequence_rows[group]["component"]) for group in groups]
    evaluation = cast(dict[str, object], recipe["evaluation"])
    if (
        sha256_ints(groups) != evaluation["group_order_sha256"]
        or sha256_ints(sequence_ids) != evaluation["sequence_id_order_sha256"]
        or sha256_ints(components) != evaluation["component_order_sha256"]
    ):
        raise ValueError("F0 group, sequence, or component order differs from the recipe")

    endpoint = cast(dict[str, object], recipe["parent_endpoint"])
    prediction_path = root / str(endpoint["prediction"])
    sidecar_path = root / str(endpoint["sidecar"])
    if sha256_file(prediction_path) != endpoint["prediction_sha256"]:
        raise ValueError("parent prediction bytes differ from the frozen recipe")
    if sha256_file(sidecar_path) != endpoint["sidecar_sha256"]:
        raise ValueError("parent prediction sidecar differs from the frozen recipe")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("checkpoint_sha256") != endpoint["checkpoint_sha256"]:
        raise ValueError("parent prediction sidecar names a different checkpoint")
    if sidecar.get("split_manifest_sha256") != recipe["split_manifest_sha256"]:
        raise ValueError("parent prediction sidecar names a different source split")
    if sidecar.get("row_groups") != list(range(1873)):
        raise ValueError("parent prediction sidecar lacks exact full-validation row order")

    ancestry_path = root / str(endpoint["fit_membership_preflight"])
    if sha256_file(ancestry_path) != endpoint["fit_membership_preflight_sha256"]:
        raise ValueError("parent fit-membership preflight differs from the frozen recipe")
    ancestry = json.loads(ancestry_path.read_text(encoding="utf-8"))
    if (
        ancestry.get("status") != "pass"
        or ancestry.get("split_manifest_sha256") != recipe["split_manifest_sha256"]
        or ancestry.get("evaluation_group_order_sha256") != evaluation["group_order_sha256"]
        or ancestry.get("evaluation_sequence_id_order_sha256")
        != evaluation["sequence_id_order_sha256"]
        or ancestry.get("target_numeric_values_decoded_or_used") is not False
    ):
        raise ValueError("parent fit-membership receipt is not valid for this split")

    array = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(cast(list[int], endpoint["expected_shape"]))
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ValueError("parent prediction shape or precision differs from the recipe")
    row_positions = {int(group): index for index, group in enumerate(sidecar["row_groups"])}
    if any(group not in row_positions for group in groups):
        raise ValueError("parent prediction cache does not cover every F0 evaluation group")
    return (
        groups,
        np.asarray([row_positions[group] for group in groups], dtype=np.int64),
        sequence_ids,
        components,
    )


def main() -> None:
    """Generate filter-only predictions without opening target or mask data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-groups", type=int)
    args = parser.parse_args()

    root = args.root.resolve(strict=True)
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    groups, parent_positions, sequence_ids, components = verify_inputs(root, recipe)
    if args.limit_groups is not None:
        if args.limit_groups < 1 or args.limit_groups > len(groups):
            raise ValueError("--limit-groups must be between one and the frozen group count")
        groups = groups[: args.limit_groups]
        parent_positions = parent_positions[: args.limit_groups]
        sequence_ids = sequence_ids[: args.limit_groups]
        components = components[: args.limit_groups]

    output_path = args.output.resolve()
    receipt_path = output_path.with_suffix(".json")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite a prediction profile or cache")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    endpoint = cast(dict[str, object], recipe["parent_endpoint"])
    parent_path = root / str(endpoint["prediction"])
    parent = np.load(parent_path, mmap_mode="r", allow_pickle=False)
    shape = (2, len(groups), 20_000, 2)
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=shape)
    parameters: list[dict[str, object]] = []

    for local_index, parent_index in enumerate(parent_positions):
        raw = np.asarray(parent[int(parent_index)], dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError(f"nonfinite parent predictions in row group {groups[local_index]}")

        fixed_gain = steady_state_gain(FIXED_Q_OVER_R)
        output[0, local_index] = filter_predictions(raw, fixed_gain)

        channel_parameters = []
        filtered = np.empty_like(raw)
        for channel in range(raw.shape[1]):
            ratio = prefix_q_over_r(raw[:PREFIX_SIZE, channel])
            gain = steady_state_gain(ratio)
            filtered[:, channel] = filter_predictions(raw[:, channel : channel + 1], gain)[:, 0]
            channel_parameters.append({"q_over_r": ratio, "steady_state_gain": gain})
        output[1, local_index] = filtered
        parameters.append(
            {
                "row_group": groups[local_index],
                "fixed_q_over_r": FIXED_Q_OVER_R,
                "fixed_steady_state_gain": fixed_gain,
                "prefix_moment_channels": channel_parameters,
            }
        )

    output.flush()
    del output
    endpoint_for_receipt = {
        "prediction": str(endpoint["prediction"]),
        "prediction_sha256": str(endpoint["prediction_sha256"]),
        "sidecar_sha256": str(endpoint["sidecar_sha256"]),
        "checkpoint_sha256": str(endpoint["checkpoint_sha256"]),
        "fit_membership_preflight_sha256": str(endpoint["fit_membership_preflight_sha256"]),
    }
    receipt: dict[str, object] = {
        "schema": "wnn.kalman_output_filter.prediction_receipt.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "recipe_sha256": sha256_file(RECIPE_PATH),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "split_manifest_sha256": recipe["split_manifest_sha256"],
        "parent_endpoint": endpoint_for_receipt,
        "prediction_output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "shape": list(shape),
            "dtype": "float32",
            "arm_order": ["fixed_q_over_r_025", "prefix_moments_q_over_r"],
        },
        "row_groups": groups,
        "row_group_order_sha256": sha256_ints(groups),
        "sequence_id_order_sha256": sha256_ints(sequence_ids),
        "component_order_sha256": sha256_ints(components),
        "prefix_rows_used_for_parameter_estimation": PREFIX_SIZE,
        "target_or_mask_numeric_values_read": False,
        "parameters_by_group": parameters,
        "profile_only": args.limit_groups is not None,
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "generated",
                "groups": len(groups),
                "output_bytes": output_path.stat().st_size,
                "output_sha256": receipt["prediction_output"]["sha256"],
                "receipt": str(receipt_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
