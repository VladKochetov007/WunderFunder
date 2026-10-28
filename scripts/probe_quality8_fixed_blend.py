"""Report-only, fixed-checkpoint GALAM/GRU blend discovery on official valid."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from wnn.gmetric import GlobalAccumulator
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.sequence_data import SequenceRecords
from wnn.training.sequence_distributed import configure_strict_fp32
from wnn.training.sequence_epoch_prediction import epoch_admission
from wnn.training.sequence_fit_utils import collate_sequences
from wnn.training.sequence_models import detach_state
from wnn.training.sequence_predict import load_endpoint


def write_json(path: Path, value: object) -> None:
    """Write a new bounded receipt; never replace earlier output."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def score(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    """Use the unchanged official global scorer over already selected rows."""
    result = GlobalAccumulator()
    result.add(target, prediction, np.ones(len(target), dtype=bool))
    return result.result()


def weighted_correlation(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> float:
    """Compute centered diagnostic correlation; this never fits blend coefficients."""
    a, b, w = left.astype(np.float64), right.astype(np.float64), weight.astype(np.float64)
    total = w.sum()
    a = a - np.sum(w * a) / total
    b = b - np.sum(w * b) / total
    denominator = np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))
    return float(np.sum(w * a * b) / denominator) if denominator > 0 else 0.0


def fixed_blend(reference: np.ndarray, candidate: np.ndarray, weight: float) -> np.ndarray:
    """Mix raw outputs before official clipping, with no fitted coefficients."""
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("fixed blend requires aligned two-target rows")
    if not 0 <= weight <= 1 or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("invalid fixed blend inputs")
    return ((1.0 - weight) * reference + weight * candidate).astype(np.float32)


def trace_chunk(
    model: torch.nn.Module, values: torch.Tensor, state: object, output: Path
) -> tuple[torch.Tensor, torch.Tensor, object]:
    """Capture one inference-only steady-history chunk; this is not a training trace."""
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with (
        torch.profiler.profile(activities=activities, record_shapes=False) as profiler,
        torch.profiler.record_function("quality8_frozen_checkpoint_inference_chunk"),
    ):
        result = model(values, state)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(output / "inference-trace.json"))
    operators = [
        {
            "name": event.key,
            "count": event.count,
            "self_cpu_us": event.self_cpu_time_total,
            "cpu_us": event.cpu_time_total,
            "self_device_us": getattr(event, "self_device_time_total", 0.0),
            "device_us": getattr(event, "device_time_total", 0.0),
        }
        for event in profiler.key_averages()
    ]
    write_json(output / "inference-operators.json", operators)
    return result


@torch.inference_mode()
def predict_endpoint(
    endpoint: dict[str, object], plan: dict[str, object], output: Path, log: Path
) -> tuple[np.ndarray, np.ndarray, str, dict[str, object]]:
    """Evaluate all complete official valid sequences, caching scored rows only."""
    destination = output / str(endpoint["cell"])
    destination.mkdir()
    checkpoint = Path(str(endpoint["checkpoint"]))
    if file_sha256(checkpoint) != endpoint["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint changed")
    admission = epoch_admission(
        checkpoint, Path(str(endpoint["epoch_index"])), str(endpoint["epoch_index_sha256"])
    )
    model, payload = load_endpoint(checkpoint, torch.device("cuda:0"), admission)
    if payload["provenance"]["identity"]["source_sha256"] != plan["training_source_sha256"]:
        raise ValueError("checkpoint was trained with a different source")
    split = load_campaign_split(Path(str(plan["split_manifest"])), Path(str(plan["data_root"])))
    if split.sha256 != plan["split_sha256"] or payload["data"]["split_sha256"] != split.sha256:
        raise ValueError("checkpoint/split identity mismatch")
    expected_fit = [["train", group] for group in split.train_groups]
    if payload["data"]["fit_members"] != expected_fit or split.valid_groups:
        raise ValueError("endpoint fitting ancestry includes valid or mismatched train members")
    members = [("valid", group) for group in split.evaluation_groups]
    if len(members) != plan["members"]:
        raise ValueError("incomplete official valid membership")
    config = payload["config"]
    mode = config["model"].get("input_feature_mode", "raw112")
    dataset = SequenceRecords(Path(str(plan["data_root"])), members, 0, input_feature_mode=mode)
    predictions, targets = [], []
    membership_hash = hashlib.sha256()
    started = time.perf_counter()
    load_seconds, forward_seconds = 0.0, 0.0
    for begin in range(0, len(dataset), 384):
        before = time.perf_counter()
        ids = list(range(begin, min(begin + 384, len(dataset))))
        batch = collate_sequences(dataset, ids)
        load_seconds += time.perf_counter() - before
        state, chunks = None, []
        chunk_rows = int(config["chunk_rows"])
        for start in range(0, batch["x"].shape[1], chunk_rows):
            values = batch["x"][:, start : start + chunk_rows].to("cuda:0")
            torch.cuda.synchronize()
            before = time.perf_counter()
            if begin == 0 and start == chunk_rows and "galam" in str(endpoint["cell"]):
                prediction, _, state = trace_chunk(model, values, state, destination)
            else:
                prediction, _, state = model(values, state)
            state = detach_state(state)
            chunks.append(prediction.float())
            torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - before
        predicted = torch.cat(chunks, dim=1).cpu().numpy()
        mask, target = batch["mask"].numpy(), batch["y"].numpy()
        if mask[:, :99].any() or not np.isfinite(predicted).all():
            raise ValueError("invalid warmup mask or nonfinite predictions")
        predictions.append(predicted[mask].copy())
        targets.append(target[mask].copy())
        for index, selected in zip(ids, mask, strict=True):
            membership_hash.update(np.asarray(members[index], dtype="U16").tobytes())
            membership_hash.update(np.flatnonzero(selected).astype("<i4").tobytes())
        with log.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "cell": endpoint["cell"],
                        "complete_sequences": begin + len(ids),
                        "total_sequences": len(dataset),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                + "\n"
            )
        del batch, chunks, predicted, target, mask, prediction, state, values
        gc.collect()
    predicted, target = np.concatenate(predictions), np.concatenate(targets)
    if len(target) != plan["expected_scored_rows"]:
        raise ValueError("incomplete official scoring coverage")
    metrics = score(target, predicted)
    if abs(float(metrics["wp"]) - float(endpoint["reported_wp"])) > 1e-6:
        raise ValueError("export does not reproduce monitored checkpoint score")
    np.save(destination / "scored-predictions.npy", predicted, allow_pickle=False)
    receipt = {
        "endpoint": endpoint,
        "metrics": metrics,
        "row_order_sha256": membership_hash.hexdigest(),
        "scored_rows": len(target),
        "input_feature_mode": mode,
        "batch_size": 384,
        "chunk_rows": chunk_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "load_seconds": load_seconds,
        "forward_seconds_including_one_profile": forward_seconds,
        "prediction_sha256": file_sha256(destination / "scored-predictions.npy"),
    }
    write_json(destination / "receipt.json", receipt)
    del model, payload, predictions, targets, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return predicted, target, membership_hash.hexdigest(), receipt


def main() -> None:
    """Run the predeclared inference-only panel on one otherwise free physical GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["primary_galam_weights"] != [0.2] or plan["sensitivity_galam_weights"] != [0.1, 0.3]:
        raise ValueError("this diagnostic is bound to the predeclared fixed blend panel")
    if torch.cuda.device_count() != 1:
        raise ValueError("expose exactly one currently free physical GPU")
    configure_strict_fp32()
    torch.set_num_threads(8)
    args.output.mkdir()
    write_json(args.output / "plan.json", plan)
    base_prediction, base_target, base_order, baseline_receipt = predict_endpoint(
        plan["endpoints"][0], plan, args.output, args.output / "progress.jsonl"
    )
    np.save(args.output / "scored-targets.npy", base_target, allow_pickle=False)
    comparisons, receipts = [], [baseline_receipt]
    for endpoint in plan["endpoints"][1:]:
        predicted, target, order, receipt = predict_endpoint(
            endpoint, plan, args.output, args.output / "progress.jsonl"
        )
        if order != base_order or not np.array_equal(target, base_target):
            raise ValueError("candidate/reference row ordering or targets disagree")
        for weight in plan["primary_galam_weights"] + plan["sensitivity_galam_weights"]:
            blend = fixed_blend(base_prediction, predicted, weight)
            result = score(target, blend)
            comparisons.append(
                {
                    "candidate": endpoint["cell"],
                    "galam_weight": weight,
                    "primary": weight == 0.2,
                    "metrics": result,
                    "delta_wp": float(result["wp"]) - float(baseline_receipt["metrics"]["wp"]),
                }
            )
        clipped_y = np.clip(target, -2, 2)
        diagnostics = []
        for index in range(2):
            weight = np.abs(clipped_y[:, index])
            left, right = (
                np.clip(base_prediction[:, index], -2, 2),
                np.clip(predicted[:, index], -2, 2),
            )
            diagnostics.append(
                {
                    "target": index,
                    "prediction_correlation": weighted_correlation(left, right, weight),
                    "residual_correlation": weighted_correlation(
                        clipped_y[:, index] - left, clipped_y[:, index] - right, weight
                    ),
                }
            )
        receipt["reference_correlations"] = diagnostics
        receipts.append(receipt)
    write_json(
        args.output / "result.json",
        {
            "plan_sha256": file_sha256(args.plan),
            "script_sha256": file_sha256(Path(__file__)),
            "row_order_sha256": base_order,
            "scored_rows": len(base_target),
            "receipts": receipts,
            "fixed_blends": comparisons,
            "exposure": plan["exposure"],
            "reference": plan["reference"],
            "scientific_fits_started": 0,
            "training_processes_changed": False,
        },
    )


if __name__ == "__main__":
    main()
