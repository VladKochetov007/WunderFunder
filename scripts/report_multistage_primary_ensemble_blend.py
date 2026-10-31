from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_multistage_primary_ensemble_worker import (  # noqa: E402
    PRIMARY_ROWS,
    TL3_WP,
    _canonical_sha256,
    _load_request,
    _load_source6,
    _read_json,
    _regular,
    _sha256,
    _verify_source6,
    _verify_splits,
    _write_new,
)


def weighted_prediction_block(predictions: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Apply the fixed simplex weights to raw scored rows before competition clipping."""
    if (
        len(predictions) != len(weights)
        or not predictions
        or weights.ndim != 1
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or abs(float(weights.sum()) - 1.0) > 1e-10
    ):
        raise ValueError("prediction members and frozen weights differ")
    block = np.stack(predictions, axis=1)
    if block.ndim != 3 or block.shape[-1] != 2 or not np.isfinite(block).all():
        raise ValueError("candidate prediction block must contain finite rows for both targets")
    return np.einsum("m,rmk->rk", weights, block).astype(np.float32)


def _load_member_receipts(
    request: Mapping[str, object], blend: Mapping[str, object]
) -> tuple[list[tuple[Path, dict[str, object]]], dict[str, object]]:
    candidates, members = request.get("candidates"), blend.get("members")
    if (
        not isinstance(candidates, list)
        or not isinstance(members, list)
        or len(candidates) != 8
        or len(members) != 8
    ):
        raise ValueError("primary ensemble requires exactly eight complete member receipts")
    first_mask: dict[str, object] | None = None
    arrays: list[tuple[Path, dict[str, object]]] = []
    keys = (
        "membership_sha256",
        "mask_sha256",
        "selected_row_index_sha256",
        "sequence_order",
        "row_order",
        "selected_rows",
        "per_sequence_selected_rows",
        "mask",
        "mask_packed_bitorder",
    )
    for index, (candidate, member) in enumerate(zip(candidates, members, strict=True)):
        if not isinstance(candidate, Mapping) or not isinstance(member, Mapping):
            raise ValueError("primary ensemble member binding is malformed")
        receipt_path = _regular(Path(str(member["receipt_path"])), "member receipt")
        if _sha256(receipt_path) != member.get("receipt_sha256"):
            raise ValueError("member receipt changed after scoring")
        receipt = _read_json(receipt_path, "member receipt")
        prediction = receipt.get("predictions")
        scorer = receipt.get("scorer")
        if (
            receipt.get("status") != "complete"
            or receipt.get("member_index") != index
            or receipt.get("name") != candidate.get("name")
            or str(receipt_path.resolve()) != candidate.get("receipt_path")
            or receipt.get("checkpoint_sha256") != candidate.get("checkpoint_sha256")
            or receipt.get("trial_result_sha256") != candidate.get("trial_result_sha256")
            or receipt.get("trial_request_sha256") != candidate.get("trial_request_sha256")
            or receipt.get("config_sha256") != candidate.get("config_sha256")
            or receipt.get("config_identity_sha256") != candidate.get("config_identity_sha256")
            or receipt.get("prediction_manifest_sha256")
            != candidate.get("prediction_manifest_sha256")
            or receipt.get("inner_objective_sha256") != candidate.get("inner_objective_sha256")
            or receipt.get("producer_identity_sha256") != candidate.get("producer_identity_sha256")
            or receipt.get("source_sha256") != candidate.get("training_source_sha256")
            or not isinstance(prediction, Mapping)
            or prediction.get("sha256") != member.get("prediction_sha256")
            or prediction.get("path") != candidate.get("predictions_path")
            or prediction.get("shape") != [PRIMARY_ROWS, 2]
            or prediction.get("dtype") != "float32"
            or prediction.get("selected_rows") != PRIMARY_ROWS
            or receipt.get("primary_split_sha256") != request["primary_split"]["sha256"]
            or receipt.get("auxiliary_split_sha256") != request["auxiliary_split"]["sha256"]
            or not isinstance(scorer, Mapping)
            or scorer.get("source6_sha256") != request["source6"].get("source_sha256")
            or scorer.get("score_script_sha256") != request["source6"].get("score_script_sha256")
        ):
            raise ValueError("member receipt does not bind its exact Stage4 candidate")
        mask = {key: prediction.get(key) for key in keys}
        if first_mask is not None and mask != first_mask:
            raise ValueError("candidate primary masks or scored-row orders differ")
        first_mask = mask
        array_path = _regular(Path(str(prediction["path"])), "scored predictions")
        if _sha256(array_path) != prediction.get("sha256"):
            raise ValueError("candidate prediction array SHA differs from its receipt")
        arrays.append((array_path, receipt))
    if first_mask is None:
        raise ValueError("candidate score masks are unavailable")
    if (
        Path(str(blend.get("output_dir"))).resolve(strict=True)
        != Path(str(request["output_dir"])).resolve(strict=True)
        or Path(str(blend.get("ensemble_predictions_path"))).resolve()
        != Path(str(request["output_dir"])) / "ensemble_scored_predictions.npy"
    ):
        raise ValueError("blend output destinations differ from the frozen score request")
    return arrays, first_mask


def _blend_and_score(
    request: Mapping[str, object],
    arrays: Sequence[tuple[Path, Mapping[str, object]]],
    common_mask: Mapping[str, object],
) -> tuple[dict[str, object], Path, list[dict[str, object]]]:
    source_root = Path(str(request["source6"]["root"])).resolve(strict=True)
    _verify_source6(source_root, request["source6"])
    (
        row_count,
        streaming_metrics,
        _,
        load_campaign_split,
        _,
        sequence_records,
        collate_sequences,
        membership_sha256,
        _,
        configure_precision_mode,
    ) = _load_source6(source_root)
    primary, _ = _verify_splits(request, load_campaign_split)
    if row_count != PRIMARY_ROWS:
        raise ValueError("Source6 scored-row count differs from the frozen mask contract")
    configure_precision_mode("fp32")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    predictions = [np.load(path, mmap_mode="r", allow_pickle=False) for path, _ in arrays]
    if any(array.dtype != np.float32 or array.shape != (PRIMARY_ROWS, 2) for array in predictions):
        raise ValueError("Source6 member prediction arrays have the wrong dtype or shape")
    candidates = request["candidates"]
    weights = np.asarray([float(candidate["weight"]) for candidate in candidates], dtype=np.float64)
    if (
        not np.isfinite(weights).all()
        or np.any(weights < 0)
        or abs(float(weights.sum()) - 1.0) > 1e-10
    ):
        raise ValueError("frozen Stage4 weights are not a nonnegative unit simplex")
    if _canonical_sha256(weights.tolist()) != request.get("weight_vector_sha256"):
        raise ValueError("frozen Stage4 weight vector hash changed")
    output = Path(str(request["output_dir"])).resolve(strict=True)
    mixed_path = output / "ensemble_scored_predictions.npy"
    if mixed_path.exists() or mixed_path.is_symlink():
        raise FileExistsError("ensemble prediction output must be fresh")
    temporary = mixed_path.with_name(mixed_path.name + ".partial")
    mixed = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(PRIMARY_ROWS, 2)
    )
    for start in range(0, PRIMARY_ROWS, 65_536):
        stop = min(start + 65_536, PRIMARY_ROWS)
        mixed[start:stop] = weighted_prediction_block(
            [array[start:stop] for array in predictions], weights
        )
    mixed.flush()
    del mixed
    temporary.replace(mixed_path)
    mixed = np.load(mixed_path, mmap_mode="r", allow_pickle=False)
    dataset = sequence_records(
        Path(str(request["data_root"])),
        [("valid", group) for group in primary.evaluation_groups],
        0,
        training_mask="is_scored",
    )
    dataset.validate_source_schema()
    accumulator = streaming_metrics()
    row_offset = 0
    mask_digest, index_digest = hashlib.sha256(), hashlib.sha256()
    sequence_counts = []
    for start in range(0, len(dataset), 64):
        indices = list(range(start, min(start + 64, len(dataset))))
        batch = collate_sequences(dataset, indices)
        targets, masks = batch["y"].numpy(), batch["mask"].numpy()
        batch_prediction = np.zeros_like(targets, dtype=np.float32)
        for offset, index in enumerate(indices):
            mask = np.asarray(masks[offset], dtype=bool)
            row_indices = np.flatnonzero(mask).astype("<u4", copy=False)
            if np.any(mask[:99]):
                raise ValueError("Source6 blend mask selected a warmup row")
            count = len(row_indices)
            batch_prediction[offset, mask] = mixed[row_offset : row_offset + count]
            row_group = dataset.members[index][1]
            mask_digest.update(np.packbits(mask, bitorder="little").tobytes())
            index_digest.update(np.asarray([row_group], dtype="<i8").tobytes())
            index_digest.update(row_indices.tobytes())
            sequence_counts.append({"row_group": row_group, "selected_rows": count})
            row_offset += count
        accumulator.add(targets, batch_prediction, masks)
    metrics = accumulator.result()
    if row_offset != PRIMARY_ROWS or metrics.get("selected_rows") != PRIMARY_ROWS:
        raise ValueError("blended score did not cover the full official scored mask")
    mask_receipt = {
        "membership_sha256": membership_sha256(dataset.members),
        "mask_sha256": mask_digest.hexdigest(),
        "selected_row_index_sha256": index_digest.hexdigest(),
        "sequence_order": "official primary evaluation membership order; source then row_group",
        "row_order": "ascending row_index among is_scored rows after warmup row 99",
        "mask": "is_scored AND row_index>=99",
        "mask_packed_bitorder": "little",
        "selected_rows": row_offset,
        "per_sequence_selected_rows": sequence_counts,
    }
    if mask_receipt != common_mask:
        raise ValueError("blend ordering differs from the eight independent Source6 score masks")
    return metrics, mixed_path, [dict(receipt) for _, receipt in arrays]


def _report(
    request: Mapping[str, object],
    blend: Mapping[str, object],
    member_receipts: Sequence[Mapping[str, object]],
    metrics: Mapping[str, object],
    mask: Mapping[str, object],
    mixed_path: Path,
) -> dict[str, object]:
    candidates = request["candidates"]
    rows = []
    for candidate, receipt in zip(candidates, member_receipts, strict=True):
        score = receipt["metrics"]
        rows.append(
            {
                "name": candidate["name"],
                "stage": candidate["stage"],
                "family": candidate["family"],
                "trial_number": candidate["trial_number"],
                "measurement": "scored on exposed official primary valid",
                "weight": candidate["weight"],
                "trained_sequence_count": 8_260,
                "inner_train_proxy_wp": candidate["inner_wp"],
                "primary_valid_wp": score["wp"],
                "primary_valid_t0": score["t0"],
                "primary_valid_t1": score["t1"],
                "primary_valid_wmse": score["wmse"],
                "primary_valid_wmse_t0": score["wmse_t0"],
                "primary_valid_wmse_t1": score["wmse_t1"],
                "trial_result_path": candidate["trial_result_path"],
                "trial_result_sha256": candidate["trial_result_sha256"],
                "trial_request_path": candidate["trial_request_path"],
                "trial_request_sha256": candidate["trial_request_sha256"],
                "config_path": candidate["config_path"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "config_sha256": candidate["config_sha256"],
                "config_identity_sha256": candidate["config_identity_sha256"],
                "source_sha256": candidate["training_source_sha256"],
                "source_revision": candidate["training_source_revision"],
                "inner_objective_path": candidate["inner_objective_path"],
                "inner_objective_sha256": candidate["inner_objective_sha256"],
                "prediction_manifest_sha256": candidate["prediction_manifest_sha256"],
                "producer_identity_sha256": candidate["producer_identity_sha256"],
                "scored_predictions_sha256": receipt["predictions"]["sha256"],
            }
        )
    output = Path(str(blend["output_dir"]))
    csv_path = output / "primary_ensemble_comparison.csv"
    comparison_rows = rows + [
        {
            **dict.fromkeys(rows[0], ""),
            "name": "Fixed Stage4 ensemble",
            "stage": 4,
            "family": "frozen weighted blend",
            "measurement": "scored on exposed official primary valid",
            "trained_sequence_count": 8_260,
            "primary_valid_wp": metrics["wp"],
            "primary_valid_t0": metrics["t0"],
            "primary_valid_t1": metrics["t1"],
            "primary_valid_wmse": metrics["wmse"],
            "primary_valid_wmse_t0": metrics["wmse_t0"],
            "primary_valid_wmse_t1": metrics["wmse_t1"],
        },
        {
            **dict.fromkeys(rows[0], ""),
            "name": "Fresh TL3 reference",
            "stage": "reference",
            "family": "TL3",
            "measurement": "fixed published score; not rescored here",
            "trained_sequence_count": 10_325,
            "primary_valid_wp": TL3_WP,
        },
    ]
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    weights = [float(candidate["weight"]) for candidate in candidates]
    weight_sha = _canonical_sha256(weights)
    png, svg = _export_plots(rows, float(metrics["wp"]), output)
    report = {
        "schema": "wnn.multistage-primary-ensemble-report.v1",
        "status": "complete",
        "report_type": "primary_development_exposed_report_only",
        "objective_eligible": False,
        "primary_checkpoint_selection_allowed": False,
        "stage4_result_path": request["stage4_result"]["path"],
        "stage4_result_sha256": request["stage4_result"]["sha256"],
        "score_request_path": str(blend["score_request_path"]),
        "score_request_sha256": blend["score_request_sha256"],
        "run_manifest": dict(request["run_manifest"]),
        "admission": dict(request["admission"]),
        "contract": dict(request["contract"]),
        "terminal_qualification": request["terminal_qualification"],
        "stage4_runner_request": dict(request["stage4_runner_request"]),
        "stage4_runner_request_sha256": request["stage4_runner_request_sha256"],
        "stage4_weights": weights,
        "stage4_weight_vector_sha256": weight_sha,
        "primary_split_sha256": request["primary_split"]["sha256"],
        "auxiliary_split_sha256": request["auxiliary_split"]["sha256"],
        "primary_split_path": request["primary_split"]["path"],
        "auxiliary_split_path": request["auxiliary_split"]["path"],
        "primary_fit_members": {"train": 10_325, "valid": 0},
        "primary_valid_fit_sequences": 0,
        "auxiliary_trained_fit_sequences": 8_260,
        "evaluation": {
            "source": "valid",
            "role": "primary_valid",
            "sequences": 1_873,
            "mask": "is_scored AND row_index>=99",
            "metrics": dict(metrics),
            "mask_and_order": dict(mask),
            "selected_rows": PRIMARY_ROWS,
        },
        "members": rows,
        "comparison_rows": comparison_rows,
        "ensemble_predictions": {
            "path": str(mixed_path.resolve()),
            "sha256": _sha256(mixed_path),
            "dtype": "float32",
            "shape": [PRIMARY_ROWS, 2],
        },
        "fixed_TL3_reference": {
            "name": "canonical fresh TL3",
            "wp": TL3_WP,
            "trained_sequence_count": 10_325,
            "source": "research/experiment_drafts/heterogeneous-tl3-20261030/results/R3/REPORT.json",
            "measured_here": False,
            "delta_wp": float(metrics["wp"]) - TL3_WP,
        },
        "source6": request["source6"],
        "portable_data_registry": request["portable_data_registry"],
        "selection_policy": "Stage4 weights were frozen on the inner train proxy. Primary-valid metrics are report-only; do not retune weights or select checkpoints from this report.",
        "comparison_csv": {"path": str(csv_path.resolve()), "sha256": _sha256(csv_path)},
        "plots": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in (png, svg)],
    }
    report_path = output / "ensemble_primary_report.json"
    _write_new(report_path, report)
    return report | {
        "report_path": str(report_path.resolve()),
        "report_sha256": _sha256(report_path),
    }


def _export_plots(
    rows: list[dict[str, object]], ensemble_wp: float, output: Path
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["name"]) for row in rows] + ["Fixed Stage4 ensemble", "Fresh TL3 reference"]
    values = [float(row["primary_valid_wp"]) for row in rows] + [ensemble_wp, TL3_WP]
    colors = ["steelblue"] * len(rows) + ["darkgreen", "gray"]
    fig, axis = plt.subplots(figsize=(max(9, len(labels) * 0.8), 5))
    axis.bar(labels, values, color=colors)
    axis.set_ylabel("Official primary-valid WP")
    axis.set_title("Frozen Stage4 ensemble on exposed official primary valid")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    png, svg = (
        output / "primary_ensemble_comparison.png",
        output / "primary_ensemble_comparison.svg",
    )
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend-request", type=Path, required=True)
    parser.add_argument("--blend-request-sha256", required=True)
    args = parser.parse_args()
    try:
        blend_path = _regular(args.blend_request, "blend request", 32_000_000)
        if _sha256(blend_path) != args.blend_request_sha256:
            raise ValueError("blend request changed after preparation")
        blend = _read_json(blend_path, "blend request", 32_000_000)
        request_path = Path(str(blend["score_request_path"]))
        request = _load_request(request_path, str(blend["score_request_sha256"]))
        arrays, mask = _load_member_receipts(request, blend)
        metrics, mixed_path, receipts = _blend_and_score(request, arrays, mask)
        result = _report(request, blend, receipts, metrics, mask, mixed_path)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"Source6 ensemble blend refused: {error}") from error
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
