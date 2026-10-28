from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch

from wnn.galam_bn.evaluation import evaluate
from wnn.galam_bn.inputs import (
    feature_batches,
    load_inputs,
    runtime_identity,
    select_members,
    validate_precision_environment,
)
from wnn.galam_bn.moments import bn_order, recalibrate, state_hashes
from wnn.splits import file_sha256, load_campaign_split
from wnn.training.sequence_epoch_prediction import digest, epoch_admission
from wnn.training.sequence_precision import configure_precision_mode


def write_json(path: Path, value: object) -> None:
    """Write a new immutable JSON receipt, rejecting reuse of output names."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def device_evidence(device: torch.device) -> dict[str, object]:
    """Record the actual receiver identity and configured precision backend flags."""
    properties = torch.cuda.get_device_properties(device)
    uuid = getattr(properties, "uuid", None)
    return {
        "torch_version": str(torch.__version__),
        "cuda_build_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": properties.name,
        "gpu_uuid": None if uuid is None else str(uuid),
        "gpu_uuid_status": "unavailable_from_torch" if uuid is None else "reported",
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_compute_capability": [properties.major, properties.minor],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_allow_tf32_cublas_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
        "cublas_emulate_single_precision": os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION"),
    }


def run(args: argparse.Namespace) -> None:
    """Evaluate one frozen checkpoint before/after train-only BN recalibration."""
    started = time.monotonic()

    def check_deadline() -> None:
        """Bound new diagnostic work without managing any external jobs."""
        if time.monotonic() - started > args.max_wall_seconds:
            raise TimeoutError("BN diagnostic wall-clock allocation exhausted")

    runtime = runtime_identity(args.expected_commit)
    forbidden = (
        args.data_root.resolve(),
        Path(runtime["root"]),
        args.checkpoint.parent.parent.resolve(),
    )
    if any(args.output.resolve().is_relative_to(root) for root in forbidden):
        raise ValueError("output must be outside data, checkout and original fit directories")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("BN diagnostic uses one explicitly assigned GPU, not DDP")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible == "-1" or "," in visible:
        raise ValueError("set CUDA_VISIBLE_DEVICES to one explicitly assigned free GPU")
    validate_precision_environment(os.environ)
    configure_precision_mode("fp32")
    if not torch.cuda.is_available():
        raise RuntimeError("receiver requires CUDA; no local/CPU fallback")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("BN diagnostic requires exactly one visible CUDA device")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "request.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    try:
        index_file_hash = file_sha256(args.epoch_index)
        model, payload, fit, valid, binding = load_inputs(
            args.checkpoint,
            args.checkpoint_sha256,
            args.epoch_index,
            args.epoch_index_sha256,
            args.expected_training_source_sha256,
            args.split_manifest,
            args.data_root,
            device,
        )
        order = bn_order(model)
        members = select_members(fit.members, args.calibration_sequences, args.calibration_seed)
        prediction_bytes = len(valid) * 20_000 * 2 * 4 * 2
        disk_required = prediction_bytes + args.checkpoint.stat().st_size + 2 * 1024**3
        if shutil.disk_usage(args.output).free < disk_required:
            raise ValueError(
                "insufficient disk for two prediction sets, checkpoint and 2 GiB reserve"
            )
        plan = {
            "schema": "wnn.galam-bn-diagnostic.v1",
            "status": "prepared_on_receiver",
            "runtime": runtime,
            "device_runtime": device_evidence(device),
            "epoch_index_file_sha256": index_file_hash,
            "binding": binding,
            "calibration_members": members,
            "calibration_members_sha256": digest(members),
            "calibration_seed": args.calibration_seed,
            "normalization_order": [name for name, _ in order],
            "method": "layerwise_eval_input_pooled_unbiased_variance_v1",
            "calibration_rows": "all rows including warmup; features only; no labels or masks",
            "scientific_new_fits": 0,
            "precision": "fp32",
            "prediction_bytes": prediction_bytes,
            "disk_required_bytes": disk_required,
            "inference_only": True,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "exposure": "official valid already exposed; checkpoint selection is not undone",
        }
        write_json(args.output / "plan.json", plan)
        chunk_rows = payload["config"]["chunk_rows"]
        original_state = state_hashes(model)
        baseline = evaluate(
            model,
            valid,
            chunk_rows,
            args.eval_batch_size,
            device,
            args.output / "baseline",
            check_deadline,
        )
        if state_hashes(model) != original_state:
            raise RuntimeError("baseline evaluation changed model state")
        write_json(args.output / "baseline.json", baseline)
        reports = recalibrate(
            model,
            lambda: feature_batches(fit, members, args.batch_size),
            chunk_rows,
            device,
            check_deadline,
        )
        expected_rows = len(members) * 20_000
        if any(report["rows"] != expected_rows for report in reports):
            raise ValueError(
                "calibration did not cover each selected full sequence exactly once per BN"
            )
        changed_state = state_hashes(model)
        weights = args.output / "recalibrated.pt"
        with weights.open("xb") as stream:
            torch.save(
                {
                    "checkpoint_schema": "wnn.galam-bn-inference.v1",
                    "config": payload["config"],
                    "model": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "parent": binding,
                    "diagnostic_plan_sha256": digest(plan),
                    "inference_only": True,
                    "resume_training_allowed": False,
                },
                stream,
            )
        candidate = evaluate(
            model,
            valid,
            chunk_rows,
            args.eval_batch_size,
            device,
            args.output / "recalibrated",
            check_deadline,
        )
        if state_hashes(model) != changed_state:
            raise RuntimeError("candidate evaluation changed model state")
        if baseline["selected_rows"] != candidate["selected_rows"]:
            raise ValueError("paired evaluation masks differ")
        if file_sha256(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError("original checkpoint changed during diagnostic")
        if (
            file_sha256(args.epoch_index) != index_file_hash
            or epoch_admission(args.checkpoint, args.epoch_index, args.epoch_index_sha256)
            != binding["epoch_admission"]
        ):
            raise ValueError("pinned epoch index changed during diagnostic")
        if (
            load_campaign_split(args.split_manifest, args.data_root).sha256
            != binding["split_sha256"]
        ):
            raise ValueError("source split changed during diagnostic")
        check_deadline()
        write_json(args.output / "recalibrated.json", candidate)
        write_json(
            args.output / "result.json",
            {
                "schema": "wnn.galam-bn-diagnostic-result.v1",
                "status": "complete",
                "plan_sha256": digest(plan),
                "baseline": baseline["scores"],
                "candidate": candidate["scores"],
                "deltas": {
                    key: candidate["scores"][key] - baseline["scores"][key]
                    for key in ("wp", "t0", "t1")
                },
                "selected_rows": baseline["selected_rows"],
                "sequence_count": len(valid),
                "bn": reports,
                "original_state": original_state,
                "recalibrated_state": changed_state,
                "checkpoint_sha256": file_sha256(weights),
                "original_checkpoint_unchanged": True,
                "original_epoch_index_unchanged": True,
                "elapsed_seconds": time.monotonic() - started,
                "scores_are_exploratory": True,
                "fit_count": 0,
            },
        )
    except Exception as error:
        write_json(
            args.output / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def main() -> None:
    """Parse a pinned one-GPU diagnostic request; no training or live-job control."""
    parser = argparse.ArgumentParser(description="Frozen GALAM train-only BatchNorm diagnostic")
    for name in ("checkpoint", "epoch-index", "data-root", "split-manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "checkpoint-sha256",
        "epoch-index-sha256",
        "expected-training-source-sha256",
        "expected-commit",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--calibration-sequences", type=int, default=256)
    parser.add_argument("--calibration-seed", type=int, default=1001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=384)
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if (
        min(
            args.calibration_sequences,
            args.batch_size,
            args.eval_batch_size,
            args.max_wall_seconds,
            args.cpu_threads,
        )
        < 1
    ):
        parser.error("counts, sizes and wall-clock bound must be positive")
    run(args)


if __name__ == "__main__":
    main()
