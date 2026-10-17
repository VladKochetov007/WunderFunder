from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from wnn.moment_geometry.backend import prepare_run, step_prepared
from wnn.moment_geometry.grading import grade_by_fold
from wnn.moment_geometry.recipe import ARMS, BATCH, CHUNK, CURRENT, GEO, LOG_EVERY, RAW, SEQ
from wnn.moment_geometry.run_metadata import endpoint_config
from wnn.nn import RNNModel
from wnn.training.checkpoints import inference_checkpoint_fields
from wnn.training.forward import detach_state, forward_chunk
from wnn.training.source_identity import training_source_identity
from wnn.training.state import TrainingRun


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the fold, output, epoch, seed, and arm settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-label", default="combined-folds-0-1")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11])
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS[:3]))
    args = parser.parse_args(argv)
    if (
        args.epochs <= 0
        or len(set(args.seeds)) != len(args.seeds)
        or len(set(args.arms)) != len(args.arms)
    ):
        parser.error("epochs must be positive and seeds/arms must be unique")
    return args


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    """Memory-map and validate the fixed staged fold schema."""
    names = (
        "fit_x",
        "fit_y",
        "eval_x",
        "eval_y",
        "eval_mask",
        "eval_component",
        "fit_component",
        "eval_fold",
    )
    arrays = {key: np.load(path / f"{key}.npy", mmap_mode="r", allow_pickle=False) for key in names}
    nf, ne = arrays["fit_x"].shape[0], arrays["eval_x"].shape[0]
    width = arrays["fit_x"].shape[-1]
    if width not in (231, 271):
        raise ValueError("prepared width must be231 or271")
    schema = {
        "fit_x": ((nf, SEQ, width), np.float32),
        "fit_y": ((nf, SEQ, 2), np.float32),
        "eval_x": ((ne, SEQ, width), np.float32),
        "eval_y": ((ne, SEQ, 2), np.float32),
        "eval_mask": ((ne, SEQ), np.bool_),
        "fit_component": ((nf,), np.int64),
        "eval_component": ((ne,), np.int64),
        "eval_fold": ((ne,), np.int64),
    }
    for name, (shape, dtype) in schema.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"{name} must have shape {shape} and dtype {np.dtype(dtype)}")
    overlap = np.intersect1d(arrays["fit_component"], arrays["eval_component"])
    if not nf or not ne or overlap.size:
        raise ValueError("empty split or fit/evaluation component overlap")
    return arrays


def _inputs(source: np.ndarray, start: int, stop: int, arm: str) -> np.ndarray:
    batch = np.array(source[start:stop], dtype=np.float32, copy=True)
    if arm == "control":
        batch[..., RAW:] = 0.0
    elif arm == "moments":
        batch[..., RAW + GEO - CURRENT :] = 0.0
    elif arm == "scaled":
        batch[..., RAW + GEO :] = 0.0
    elif arm == "dictionary":
        if batch.shape[-1] != 271:
            raise ValueError("dictionary treatment requires271prepared channels")
        batch[..., RAW : RAW + GEO] = 0.0
    return batch


def train_model(
    run: TrainingRun,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    arm: str,
    epochs: int,
    log_path: Path,
    device: torch.device,
) -> dict[str, float | int]:
    """Train every fit row in fixed order with sequence-reset chronological TBPTT."""
    model = run.network.model
    total = epochs * math.ceil(fit_x.shape[0] / BATCH) * math.ceil(SEQ / CHUNK)
    step, started, losses, norms = 0, time.monotonic(), [], []
    model.train()
    with log_path.open("w", encoding="utf-8") as handle:
        for epoch in range(1, epochs + 1):
            for first in range(0, fit_x.shape[0], BATCH):
                last = min(first + BATCH, fit_x.shape[0])
                x = torch.from_numpy(_inputs(fit_x, first, last, arm)).to(device)
                y = torch.from_numpy(np.array(fit_y[first:last], dtype=np.float32, copy=True)).to(
                    device
                )
                run.batch.h = None
                for begin in range(0, SEQ, CHUNK):
                    end = min(begin + CHUNK, SEQ)
                    loss, norm, lr = step_prepared(run, x, y, begin, end)
                    step += 1
                    losses.append(loss.detach())
                    norms.append(norm.detach())
                    if step % LOG_EVERY == 0 or step == total:
                        torch.cuda.synchronize(device)
                        row = {
                            "loss": float(torch.stack(losses).mean().item()),
                            "preclip_grad_norm": float(torch.stack(norms).mean().item()),
                            "lr": lr,
                            "step": step,
                            "epoch": epoch,
                            "elapsed_seconds": time.monotonic() - started,
                            "cuda_peak_allocated_bytes": int(
                                torch.cuda.max_memory_allocated(device)
                            ),
                            "window_updates": len(losses),
                        }
                        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        handle.flush()
                        losses.clear()
                        norms.clear()
    return {
        "updates": step,
        "training_seconds": time.monotonic() - started,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def predict(
    model: RNNModel,
    eval_x: np.ndarray,
    arm: str,
    path: Path,
    device: torch.device,
) -> np.ndarray:
    """Write causal predictions to an FP32 memmap, resetting state per sequence batch."""
    result = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(eval_x.shape[0], SEQ, 2)
    )
    model.eval()
    with torch.inference_mode():
        for first in range(0, eval_x.shape[0], BATCH):
            last = min(first + BATCH, eval_x.shape[0])
            x = torch.from_numpy(_inputs(eval_x, first, last, arm)).to(device)
            hidden: Tensor | None = None
            for begin in range(0, SEQ, CHUNK):
                end = min(begin + CHUNK, SEQ)
                output, hidden = forward_chunk(model, x, slice(begin, end), hidden)
                result[first:last, begin:end] = output.float().cpu().numpy()
                hidden = detach_state(hidden)
    result.flush()
    return result


def run_one(
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
    arm: str,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    """Train, score, and save one fixed arm/seed endpoint."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    total_updates = args.epochs * math.ceil(len(arrays["fit_x"]) / BATCH) * math.ceil(SEQ / CHUNK)
    run = prepare_run(
        arrays["fit_x"].shape[-1],
        total_updates,
        args.epochs,
        device,
        seed,
        args.hidden,
        args.layers,
    )
    model = run.network.model
    initial_digest = hashlib.sha256(
        b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())
    ).hexdigest()
    stem = f"{arm}_seed{seed}"
    log = args.output / "logs" / f"{stem}.jsonl"
    prediction_path = args.output / "predictions" / f"{stem}_evalpred.npy"
    checkpoint = args.output / "checkpoints" / f"{stem}.pt"
    training = train_model(run, arrays["fit_x"], arrays["fit_y"], arm, args.epochs, log, device)
    predictions = predict(model, arrays["eval_x"], arm, prediction_path, device)
    metrics = grade_by_fold(
        arrays["eval_y"],
        predictions,
        arrays["eval_mask"],
        arrays["eval_component"],
        arrays["eval_fold"],
    )
    config = endpoint_config(arrays, args, arm, seed)
    config["initial_state_sha256"] = initial_digest
    config["canonical_trainer_source"] = training_source_identity(run.environment)
    config["canonical_loss_flag"] = run.options.args.loss
    config["optimization_backend"] = (
        "canonical TrainingRun/build_optimization/forward_standard/backward_standard/optimizer_step"
    )
    record: dict[str, object] = {
        "seed": seed,
        "arm": arm,
        "training": training,
        "evaluation": metrics["pooled"],
        "evaluation_by_fold": metrics["by_fold"],
        "checkpoint": str(checkpoint),
        "predictions": str(prediction_path),
        "config": config,
    }
    snapshot = inference_checkpoint_fields(
        model, "moment-geometry-pilot", environment=run.environment
    )
    state = {name: value.detach().cpu() for name, value in snapshot["model"].items()}
    temp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save({"state_dict": state, "config": config, "result": record}, temp)
    temp.replace(checkpoint)
    return record


def main(argv: list[str] | None = None) -> int:
    """Run the requested matched endpoints on CUDA and write each result immediately."""
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("this runner requires CUDA")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.18, device=0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    arrays = load_arrays(args.data)
    if (
        any(arm in ("dictionary", "combined") for arm in args.arms)
        and arrays["fit_x"].shape[-1] != 271
    ):
        raise ValueError("dictionary/combined arms require271channel inputs")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("output directory must be empty")
    for directory in ("checkpoints", "predictions", "logs"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for arm in args.arms:
        for seed in args.seeds:
            records.append(run_one(arrays, args, arm, seed, device))
            temp = args.output / "results.json.tmp"
            temp.write_text(json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n")
            temp.replace(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
