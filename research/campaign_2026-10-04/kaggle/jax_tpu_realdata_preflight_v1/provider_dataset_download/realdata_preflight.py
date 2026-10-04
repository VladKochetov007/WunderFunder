"""Bounded four-stream JAX precision preflight on fold-0 real data.

This stages one 1000-row AdamW update per precision and evaluates the same
four held-out streams after NPZ export/reload. It is not a campaign fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
import threading
from pathlib import Path
from typing import Any



def _emit_milestone(phase: str, **details: Any) -> None:
    """Persist a flushed process/RSS checkpoint for fail-closed CPU preflight diagnostics."""
    milestone_path = os.environ.get("WNN_PREFLIGHT_MILESTONE")
    if not milestone_path:
        return
    current_rss = 0
    try:
        current_rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    record = {
        "phase": phase,
        "wall_time_epoch_seconds": time.time(),
        "elapsed_seconds": time.monotonic() - _MILESTONE_STARTED,
        "current_process_rss_bytes": current_rss,
        "ru_maxrss_platform_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "ru_maxrss_unit": "KiB on Linux",
        **details,
    }
    with open(milestone_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


_MILESTONE_STARTED = time.monotonic()
_emit_milestone("realdata_import_start")

import numpy as np

_emit_milestone("realdata_import_complete")

EXPECTED_INITIAL_CHECKPOINT_SHA256 = "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e"


def _model_dimensions(npz_path: Path) -> tuple[int, int, int, int]:
    """Infer model dimensions from exact Torch-named NPZ tensors."""
    with np.load(npz_path, allow_pickle=False) as state:
        input_width = int(state["mean"].shape[0])
        hidden = int(state["inp.weight"].shape[0])
        output_width = int(state["head.2.weight"].shape[0])
        layers = sum(name.startswith("gru.weight_ih_l") for name in state.files)
        indices = sorted(
            int(name.rsplit("l", 1)[1])
            for name in state.files
            if name.startswith("gru.weight_ih_l")
        )
    if indices != list(range(layers)):
        raise ValueError(f"checkpoint GRU layer keys are not contiguous: {indices}")
    if (input_width, hidden, layers, output_width) != (211, 256, 2, 23):
        raise ValueError(
            "preflight requires the fold-0 combo architecture (211,256,2,23), "
            f"got {(input_width, hidden, layers, output_width)}"
        )
    return input_width, hidden, layers, output_width


def _sha256_file(path: Path) -> str:
    """Hash a checkpoint/NPZ without reading the full file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_initial_export(npz_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    """Require an NPZ sidecar tying exact model tensors to the frozen combo checkpoint."""
    sidecar_path = npz_path.with_suffix(npz_path.suffix + ".json")
    sidecar = json.loads(sidecar_path.read_text())
    checkpoint_hash = _sha256_file(checkpoint_path)
    npz_hash = _sha256_file(npz_path)
    if checkpoint_hash != EXPECTED_INITIAL_CHECKPOINT_SHA256:
        raise ValueError("source checkpoint is not the frozen fold-0 combo epoch-3 artifact")
    if sidecar.get("source_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("NPZ export sidecar does not bind to the requested source checkpoint")
    if sidecar.get("npz_sha256") != npz_hash:
        raise ValueError("initial NPZ bytes differ from the export sidecar")
    if sidecar.get("dimensions") != {"n_in": 211, "hidden": 256, "layers": 2, "n_out": 23}:
        raise ValueError("initial NPZ export sidecar dimensions differ from the combo checkpoint")
    return sidecar


def _mask_sha256(mask: np.ndarray) -> str:
    """Hash a compact row-major official evaluation mask."""
    return hashlib.sha256(np.packbits(mask.reshape(-1), bitorder="little").tobytes()).hexdigest()


def _device_record(device: Any) -> dict[str, Any]:
    """Serialize a JAX device without assuming backend-specific attributes."""
    return {
        "platform": str(getattr(device, "platform", "unknown")),
        "device_kind": str(getattr(device, "device_kind", "unknown")),
        "id": int(getattr(device, "id", -1)),
        "repr": str(device),
    }


def _max_rss() -> int:
    """Return the platform's raw ru_maxrss value for bounded-run reporting."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one short real-data update per precision and export both states."""
    os.environ["JAX_PLATFORMS"] = args.backend
    _emit_milestone("jax_import_start")
    import jax
    import jax.numpy as jnp

    from wnn.metric import score
    from wnn_jax.bridge import (
        ComboTeacherMean,
        PropensityCache,
        RawSequenceStore,
        collate_evaluation_sequences,
        collate_sequences,
        evaluation_row_mask,
        group_batch_sha256,
        load_evaluation_sequence,
        load_fold_groups,
        load_sequence_supervision,
        seeded_group_batches,
        slice_time_batch,
        train_combo_chunk,
    )
    from wnn_jax.rnn import adamw_init, export_npz_state, forward_chunk, load_npz_state
    _emit_milestone("jax_and_bridge_import_complete")

    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    devices = jax.devices(args.backend)
    expected_device_count = 8 if args.backend == "tpu" else 1
    if len(devices) != expected_device_count:
        raise RuntimeError(f"preflight requires {expected_device_count} {args.backend} devices, got {devices}")
    if any(str(getattr(device, "platform", "unknown")) != args.backend for device in devices):
        raise RuntimeError(f"requested backend {args.backend!r}, got devices {devices}")
    if args.backend == "tpu" and any("TPU v5 lite" not in str(getattr(device, "device_kind", "")) for device in devices):
        raise RuntimeError(f"preflight requires eight TPU v5 lite devices, got {devices}")

    memory_path = Path(os.environ["WNN_TPU_MEMORY_SAMPLES"])
    memory_started = time.monotonic()
    stop_memory_sampler = threading.Event()

    def sample_tpu_memory(phase: str) -> None:
        records = []
        for device in devices:
            raw_stats = device.memory_stats()
            if not isinstance(raw_stats, dict):
                raise RuntimeError(f"TPU {device} memory_stats is unavailable")
            stats = {str(key): int(value) for key, value in raw_stats.items()}
            limit = stats.get("bytes_limit")
            if not limit or limit <= 0:
                raise RuntimeError(f"TPU {device} has no positive bytes_limit: {stats}")
            observed = max(stats.get("bytes_in_use", 0), stats.get("peak_bytes_in_use", 0))
            records.append({"device": _device_record(device), "memory_stats": stats,
                            "allocator_target_fraction": 0.80, "observed_hard_stop_fraction": 0.90,
                            "observed_hard_stop_bytes": int(limit * 0.90),
                            "observed_bytes_in_use_or_peak": observed})
        try:
            current_rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            current_rss = 0
        entry = {"phase": phase, "wall_time_epoch_seconds": time.time(),
                 "elapsed_seconds": time.monotonic() - memory_started,
                 "current_process_rss_bytes": current_rss, "devices": records}
        with memory_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if any(row["observed_bytes_in_use_or_peak"] >= row["observed_hard_stop_bytes"] for row in records):
            os._exit(86)

    def memory_sampler() -> None:
        try:
            while not stop_memory_sampler.is_set():
                sample_tpu_memory("periodic")
                stop_memory_sampler.wait(5.0)
        except BaseException as error:
            try:
                with memory_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"phase": "memory_sampler_failed", "error": repr(error)}) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os._exit(87)

    sample_tpu_memory("jax_devices_validated")
    threading.Thread(target=memory_sampler, name="tpu-memory-guard", daemon=True).start()
    _emit_milestone("initial_checkpoint_hash_verification_start")
    export_sidecar = _verify_initial_export(args.initial_npz, args.initial_checkpoint)
    _emit_milestone("initial_checkpoint_hash_verification_complete")
    dimensions = _model_dimensions(args.initial_npz)
    _emit_milestone("fold_split_load_start")
    groups = load_fold_groups(args.split_manifest)
    _emit_milestone("fold_split_load_complete", split_manifest_sha256=groups.manifest_sha256)
    _emit_milestone("model_parameter_initialization_start")
    params = load_npz_state(args.initial_npz, *dimensions)
    _emit_milestone("model_parameter_initialization_complete")

    train_below_hole = next((group for group in groups.train if group < 528), None)
    train_above_hole = next((group for group in groups.train if group > 528), None)
    if train_below_hole is None or train_above_hole is None:
        raise ValueError("fold fit TRAIN groups must cover both sides of excluded group 528")
    selected_groups = (
        ("train", train_below_hole),
        ("train", train_above_hole),
        ("valid", groups.valid_fit[0]),
        ("valid", groups.valid_fit[1]),
    )
    train_batches = seeded_group_batches(
        selected_groups, seed=3, epoch=0, batch_size=4
    )
    if len(train_batches) != 1:
        raise AssertionError("four-stream preflight did not produce exactly one fit batch")
    ordered_fit = train_batches[0]

    started = time.perf_counter()
    _emit_milestone("raw_data_store_initialization_start")
    store = RawSequenceStore(root)
    _emit_milestone("raw_data_store_initialization_complete")
    teacher_by_split = {
        split: ComboTeacherMean(root, split, groups.manifest_sha256)
        for split in ("train", "valid")
    }
    propensity_by_split = {
        split: PropensityCache(root, split)
        for split in ("train", "valid")
    }
    _emit_milestone("fit_sequence_and_target_load_start", expected_groups=[list(group) for group in ordered_fit])
    fit_samples = [
        load_sequence_supervision(
            root,
            split,
            row_group,
            teacher_by_split[split],
            propensity_by_split[split],
            store,
            groups=groups,
        )
        for split, row_group in ordered_fit
    ]
    _emit_milestone("fit_sequence_and_target_load_complete")
    fit_host = collate_sequences(fit_samples)
    _emit_milestone("fit_batch_collation_complete")
    fit_batch = slice_time_batch(fit_host, 0, args.chunk_rows)

    selected_eval = tuple(groups.evaluation[:4])
    _emit_milestone("evaluation_sequence_load_start", groups=list(selected_eval))
    eval_samples = [
        load_evaluation_sequence(root, row_group, store) for row_group in selected_eval
    ]
    eval_host = collate_evaluation_sequences(eval_samples)
    _emit_milestone("evaluation_sequence_load_complete")
    eval_batch = {
        name: jnp.asarray(value[:, : args.chunk_rows])
        for name, value in eval_host.items()
    }
    eval_mask = np.asarray(evaluation_row_mask(eval_batch))
    _emit_milestone("evaluation_mask_build_complete", scored_rows=int(eval_mask.sum()))
    if not eval_mask.any():
        raise ValueError("selected evaluation streams have no official scored rows in preflight chunk")

    arm_metrics: dict[str, Any] = {}
    for precision in ("fp32", "bf16"):
        arm_start = time.perf_counter()
        optimizer = adamw_init(params)
        sample_tpu_memory(f"{precision}_gradient_update_start")
        _emit_milestone(f"{precision}_gradient_update_start")
        updated, optimizer, _, _, loss = train_combo_chunk(
            params,
            optimizer,
            fit_batch,
            None,
            ema_params=params,
            ema_state=None,
            learning_rate=args.learning_rate,
            precision=precision,
            weight_decay=args.weight_decay,
        )
        loss = jax.block_until_ready(loss)
        sample_tpu_memory(f"{precision}_gradient_update_complete")
        _emit_milestone(f"{precision}_gradient_update_complete")
        loss_value = float(np.asarray(loss))
        if not np.isfinite(loss_value):
            raise FloatingPointError(f"{precision} training loss is non-finite")
        step_count = int(np.asarray(jax.device_get(optimizer.count)))
        if step_count != 1:
            raise AssertionError(f"{precision} optimizer completed {step_count} updates, expected one")
        for moment_name, moment_tree in (("first", optimizer.first), ("second", optimizer.second)):
            for parameter_name, moment in moment_tree.items():
                moment_host = np.asarray(jax.device_get(moment))
                if not np.isfinite(moment_host).all():
                    raise FloatingPointError(
                        f"{precision} AdamW {moment_name} moment is non-finite: {parameter_name}"
                    )
        _emit_milestone(f"{precision}_finite_loss_and_optimizer_checks_complete", train_loss=loss_value, adamw_steps=step_count)
        npz_path = output / f"model_{precision}.npz"
        export_npz_state(npz_path, updated)
        reloaded = load_npz_state(npz_path, *dimensions)
        sample_tpu_memory(f"{precision}_evaluation_forward_start")
        _emit_milestone(f"{precision}_evaluation_forward_start")
        prediction, _ = forward_chunk(
            reloaded,
            eval_batch["inputs"],
            precision="fp32",
        )
        prediction_np = np.asarray(prediction[..., :2], dtype=np.float32)
        if not np.isfinite(prediction_np).all():
            raise FloatingPointError(f"{precision} evaluation predictions are non-finite")
        labels_np = np.asarray(eval_batch["labels"], dtype=np.float32)
        score_mask = eval_mask.astype(bool)
        evaluation_score = score(labels_np, prediction_np, score_mask)
        sample_tpu_memory(f"{precision}_evaluation_complete")
        _emit_milestone(f"{precision}_evaluation_complete", scored_rows=int(score_mask.sum()), evaluation_score=evaluation_score)
        arm_metrics[precision] = {
            "train_loss": loss_value,
            "adamw_steps": step_count,
            "model_npz": str(npz_path),
            "model_npz_sha256": _sha256_file(npz_path),
            "evaluation_score_rows": int(score_mask.sum()),
            "evaluation_mask_sha256": _mask_sha256(score_mask),
            "diagnostic_four_stream_chunk_score": evaluation_score,
            "arm_wall_seconds": time.perf_counter() - arm_start,
        }

    mask_hashes = {metrics["evaluation_mask_sha256"] for metrics in arm_metrics.values()}
    if len(mask_hashes) != 1:
        raise AssertionError("precision arms did not use identical evaluation masks")
    report = {
        "format": "wnn.jax-realdata-preflight",
        "version": 1,
        "status": "bounded-preflight-only",
        "root": str(root),
        "fold_manifest": str(args.split_manifest),
        "fold_manifest_sha256": groups.manifest_sha256,
        "model_dimensions": {
            "n_in": dimensions[0],
            "hidden": dimensions[1],
            "layers": dimensions[2],
            "n_out": dimensions[3],
        },
        "starting_npz": str(args.initial_npz),
        "starting_npz_sha256": _sha256_file(args.initial_npz),
        "starting_export_sidecar": export_sidecar,
        "source_checkpoint": str(args.initial_checkpoint),
        "source_checkpoint_sha256": _sha256_file(args.initial_checkpoint),
        "source_files_sha256": {
            "train_parquet": _sha256_file(root / "datasets" / "train.parquet"),
            "valid_parquet": _sha256_file(root / "datasets" / "valid.parquet"),
            "teachers": {
                split: [
                    {
                        "prediction_sha256": cache.array_sha256,
                        "checkpoint_sha256": cache.checkpoint_sha256,
                    }
                    for cache in teacher_by_split[split].caches
                ]
                for split in ("train", "valid")
            },
            "propensity": {
                split: {
                    "array_sha256": propensity_by_split[split].array_sha256,
                    "checkpoint_sha256": propensity_by_split[split].checkpoint_sha256,
                }
                for split in ("train", "valid")
            },
        },
        "fit_groups": [list(group) for group in ordered_fit],
        "fit_group_batch_sha256": group_batch_sha256((ordered_fit,)),
        "evaluation_groups": list(selected_eval),
        "evaluation_mask_policy": "need_prediction AND is_scored AND row >= 99",
        "diagnostic_metric_id": "pooled_global_weighted_pearson; diagnostic slice only, no scientific claim",
        "evaluation_mask_sha256": next(iter(mask_hashes)),
        "chunk_rows": args.chunk_rows,
        "device_inventory": [_device_record(device) for device in devices],
        "requested_backend": args.backend,
        "host_inventory": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "max_rss_platform_units": _max_rss(),
            "max_rss_unit_note": "Linux ru_maxrss is KiB; capture cgroup memory separately in launcher",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "arms": arm_metrics,
        "limitations": [
            "one optimizer update on four streams is a systems preflight, not a fitted model or predictive result",
            "AdamW continuation differs from the source Muon optimizer",
            "evaluation metric uses rows 0:1000 of four held-out VALID sequences only",
        ],
    }
    report_path = output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _emit_milestone("preflight_metrics_written", metrics_path=str(report_path))
    return report


def main() -> None:
    """Parse the bounded preflight configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("research/campaign_2026-09-29/artifacts/splits/fold0.json"),
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path("artifacts/runs/c20261003_combo_s3_f0/model_ep3.pt"),
    )
    parser.add_argument(
        "--initial-npz",
        type=Path,
        default=Path("artifacts/runs/c20261003_combo_s3_f0/model_ep3.jax.npz"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("cpu", "tpu"),
        required=True,
        help="select exactly one backend before importing JAX; preflight must specify this explicitly",
    )
    parser.add_argument("--chunk-rows", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()
    if args.chunk_rows != 1000:
        parser.error("the bounded preflight is fixed to its 1000-row first chunk")
    if not args.split_manifest.is_absolute():
        args.split_manifest = args.root / args.split_manifest
    if not args.initial_npz.is_absolute():
        args.initial_npz = args.root / args.initial_npz
    if not args.initial_checkpoint.is_absolute():
        args.initial_checkpoint = args.root / args.initial_checkpoint
    if args.backend == "tpu":
        os.environ["JAX_PLATFORMS"] = "tpu"
        import jax
        devices = jax.devices("tpu")
        if len(devices) != 8:
            raise RuntimeError(f"expected exactly eight TPU devices, got {devices}")
        with jax.default_device(devices[0]):
            report = run(args)
    else:
        report = run(args)
    print(json.dumps({"status": report["status"], "metrics": str(args.output / "metrics.json")}))


if __name__ == "__main__":
    main()
