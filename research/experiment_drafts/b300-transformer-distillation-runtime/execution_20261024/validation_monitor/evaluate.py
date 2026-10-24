"""Evaluate one captured epoch checkpoint without altering its live fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

GIB = 1 << 30
WARMUP = 99
ROWS = 20_000
SOURCE_COMMIT = "f1beeb4c210a9da01735e03b1ff6e27be878b873"


def _hash(path: Path) -> str:
    """Return a SHA256 digest in bounded reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Atomically replace the observer's own JSON receipt."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append(path: Path, value: object) -> None:
    """Fsync one per-batch progress record."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _safe_metrics(metrics: Any) -> dict[str, Any] | None:
    """Return current streaming values once at least one scored row exists."""
    try:
        return metrics.result()
    except ValueError as error:
        if "no selected rows" in str(error):
            return None
        raise


def _verify_data_identity(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Match fold membership, source fingerprints and normalization ancestry."""
    fields = (
        "split_sha256",
        "fold",
        "sources",
        "fit_members",
        "evaluation_members",
        "input_schema",
        "mask",
        "training_mask",
        "teacher",
        "hint_mode",
        "hint_members",
    )
    for field in fields:
        if actual.get(field) != expected.get(field):
            raise ValueError(f"rebuilt canonical data differs from profile identity: {field}")
    for field in ("scope", "sequence_count", "source_fingerprints", "fit_prefix_sequences"):
        if actual["normalization"].get(field) != expected["normalization"].get(field):
            raise ValueError(f"normalization ancestry differs from profile identity: {field}")


def _phase(
    name: str,
    dataset: Any,
    indices: list[int],
    model: Any,
    device: Any,
    config: dict[str, Any],
    out: Path,
    deadline: float,
    limit_batches: int | None,
    batch_size: int,
    cap_bytes: int,
) -> dict[str, Any]:
    """Stream causal chunks, metrics and saved prediction/target arrays."""
    import torch

    from wnn.training.sequence_fit_utils import collate_sequences, move_and_validate_batch
    from wnn.training.sequence_precision import forward_context

    from .metrics import StreamingMetrics

    phase_dir = out / name
    phase_dir.mkdir()
    total = len(indices)
    planned = min(total, limit_batches * batch_size) if limit_batches else total
    shape = (planned, ROWS, 2)
    predictions = np.lib.format.open_memmap(
        phase_dir / ".predictions.partial.npy", mode="w+", dtype=np.float32, shape=shape
    )
    targets = np.lib.format.open_memmap(
        phase_dir / ".targets.partial.npy", mode="w+", dtype=np.float32, shape=shape
    )
    masks = np.lib.format.open_memmap(
        phase_dir / ".mask.partial.npy", mode="w+", dtype=np.bool_, shape=shape[:2]
    )
    accumulator = StreamingMetrics()
    scored_accumulator = StreamingMetrics() if name == "heldout" else None
    sse, scored_elements, processed, first_batch = 0.0, 0, 0, True
    progress = out / "PROGRESS.jsonl"
    started = time.monotonic()
    for offset in range(0, planned, batch_size):
        if time.monotonic() >= deadline:
            break
        batch_indices = indices[offset : min(offset + batch_size, planned)]
        cpu_batch = collate_sequences(dataset, batch_indices)
        batch = move_and_validate_batch(cpu_batch, device, int(config["model"]["hint_dim"]))
        outputs = []
        state = None
        with torch.inference_mode():
            for start in range(0, ROWS, int(config["chunk_rows"])):
                stop = min(start + int(config["chunk_rows"]), ROWS)
                with forward_context(str(config["precision_mode"]), device.type):
                    prediction, _, state = model(batch["x"][:, start:stop], state)
                outputs.append(prediction.float().cpu())
        torch.cuda.synchronize(device)
        predicted = torch.cat(outputs, dim=1).numpy().astype(np.float32, copy=False)
        target = batch["y"].float().cpu().numpy()
        scored = batch["mask"].cpu().numpy().astype(bool, copy=False)
        all_after_warmup = np.ones(scored.shape, dtype=bool)
        all_after_warmup[:, :WARMUP] = False
        if scored[:, :WARMUP].any() or not np.isfinite(predicted).all():
            raise ValueError("bad causal predictions or a selected warmup row")
        metric_mask = all_after_warmup if name == "train_probe" else scored
        accumulator.update(target, predicted, metric_mask, wmse_mask=all_after_warmup)
        if scored_accumulator is not None:
            scored_accumulator.update(target, predicted, scored, wmse_mask=scored)
        count = len(batch_indices)
        predictions[processed : processed + count] = predicted
        targets[processed : processed + count] = target
        masks[processed : processed + count] = metric_mask
        if name == "heldout":
            selected = scored[..., None]
            difference = predicted.astype(np.float64) - target.astype(np.float64)
            sse += float(np.square(difference[selected.repeat(2, axis=-1)]).sum())
            scored_elements += int(scored.sum()) * 2
        processed += count
        memory = {
            "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        if first_batch:
            first_batch = False
            if max(memory.values()) > cap_bytes:
                raise MemoryError("first inference batch exceeded the configured GPU memory cap")
        if max(memory.values()) > cap_bytes:
            raise MemoryError("inference exceeded the configured GPU memory cap")
        _append(
            progress,
            {
                "phase": name,
                "processed_sequences": processed,
                "total_sequences": total,
                "elapsed_seconds": time.monotonic() - started,
                "cumulative_metrics": _safe_metrics(accumulator),
                "gpu_memory": memory,
            },
        )
        del cpu_batch, batch, outputs, predicted, target
    predictions.flush()
    targets.flush()
    masks.flush()
    del predictions, targets, masks
    for stem in ("predictions", "targets", "mask"):
        partial = phase_dir / f".{stem}.partial.npy"
        values = np.load(partial, mmap_mode="r", allow_pickle=False)
        final = phase_dir / f"{stem}.npy"
        with final.open("wb") as stream:
            np.save(stream, values[:processed], allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        del values
        partial.unlink()
    members = [list(dataset.members[index]) for index in indices[:processed]]
    member_path = phase_dir / "members.json"
    _write_json(member_path, members)
    metric_values = _safe_metrics(accumulator)
    scored_metrics = _safe_metrics(scored_accumulator) if scored_accumulator is not None else None
    if metric_values is None:
        raise ValueError("phase contains no scored rows")
    complete = processed == total
    result: dict[str, Any] = {
        "status": "complete" if complete else "partial",
        "processed_sequences": processed,
        "total_sequences": total,
        "membership_sha256": _hash(member_path),
        "weighted_mse_all_after_warmup": metric_values["wmse"],
        "wmse_mask_definition": "row_index>=99 for every sequence; reconstruct from array shape",
        "saved_mask_definition": "all_after_warmup"
        if name == "train_probe"
        else "is_scored AND row_index>=99",
        "weighted_mse_metric_id": metric_values["wmse_metric_id"],
        "metrics": metric_values,
        "artifacts": {
            stem: {
                "path": str(phase_dir / f"{stem}.npy"),
                "sha256": _hash(phase_dir / f"{stem}.npy"),
            }
            for stem in ("predictions", "targets", "mask")
        },
    }
    if name == "heldout":
        if scored_metrics is None:
            raise ValueError("heldout phase contains no scored rows")
        result["official_wp"] = metric_values["wp"]
        result["official_metric_id"] = metric_values["metric_id"]
        result["weighted_mse_scored"] = scored_metrics["wmse"]
        result["weighted_mse_scored_metric_id"] = scored_metrics["wmse_metric_id"]
        result["scored_mse_raw"] = sse / scored_elements if scored_elements else None
        result["scored_mse_elements"] = scored_elements
    if limit_batches:
        result["profile_only"] = True
    return result


def run(args: argparse.Namespace) -> int:
    """Validate the snapshot, then run the fixed train probe and full heldout fold."""
    job = json.loads(args.job.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint.resolve(strict=True)
    out = args.output.resolve(strict=True)
    if {path.name for path in out.iterdir()} - {"worker.log"}:
        raise FileExistsError("output must be fresh (worker.log from the watcher is allowed)")
    raw_config = json.loads(Path(job["config"]).read_text(encoding="utf-8"))
    precision = str(raw_config.get("precision_mode", "fp32"))
    if precision in ("fp32_bf16x9", "bf16_bf16x9"):
        os.environ["CUBLAS_EMULATE_SINGLE_PRECISION"] = "1"
    else:
        os.environ.pop("CUBLAS_EMULATE_SINGLE_PRECISION", None)
    source_root = Path(job["source_root"]).resolve(strict=True)
    import sys

    sys.path.insert(0, str(source_root / "src"))
    from wnn.splits import load_campaign_split
    from wnn.training.sequence_config import load_config, validate_expected_fold
    from wnn.training.sequence_data import build_data
    from wnn.training.sequence_models import build_sequence_model
    from wnn.training.sequence_provenance import (
        canonical_sha256,
        source_identity,
        validate_profile_receipt,
    )

    from .checkpoint_io import load_model_state, read_snapshot

    config_path = Path(job["config"]).resolve(strict=True)
    profile_path = Path(job["profile_receipt"]).resolve(strict=True)
    split_path = Path(job["split_manifest"]).resolve(strict=True)
    data_root = Path(job["data_root"]).resolve(strict=True)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    identity = profile.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("approved profile has no run identity")
    config = load_config(config_path)
    if job.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("job is not bound to the pinned f1beeb4c source")
    actual_source = source_identity(source_root, source_root / "scripts/train_sequence.py")
    if actual_source["sha256"] != identity.get("source_sha256"):
        raise ValueError("source tree does not match the approved profile")
    if canonical_sha256(config) != identity.get("config_sha256"):
        raise ValueError("loaded config differs from the approved profile")
    if job.get("config_sha256") not in (None, identity.get("config_sha256")):
        raise ValueError("job config identity differs from the approved profile")
    split = load_campaign_split(split_path, data_root)
    validate_expected_fold(config, split.fold)
    if job.get("split_sha256") not in (None, split.sha256):
        raise ValueError("job split hash differs from the selected fold")
    if job.get("profile_sha256") not in (None, _hash(profile_path)):
        raise ValueError("profile receipt hash differs from the original job")
    validate_profile_receipt(profile_path, identity)
    data = build_data(config, data_root, split_path)
    identity_data = identity.get("data")
    if not isinstance(identity_data, dict):
        raise ValueError("approved profile has no data identity")
    _verify_data_identity(data.provenance, identity_data)
    snapshot = read_snapshot(
        checkpoint,
        identity,
        config_sha256=canonical_sha256(config),
        source_sha256=str(actual_source["sha256"]),
        split_sha256=split.sha256,
        epochs=int(config["epochs"]),
    )
    checkpoint_sha256 = _hash(checkpoint)
    if int(config["required_world_size"] or identity["world_size"]) != snapshot["world_size"]:
        raise ValueError("checkpoint world size differs from the frozen recipe")
    output = {
        "schema": "wnn.pr32.checkpoint-validation.v1",
        "status": "running",
        "cell": job["cell"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_snapshot": snapshot["snapshot"],
        "checkpoint_epoch": snapshot["checkpoint_epoch"],
        "completed_epochs": snapshot["completed_epochs"],
        "checkpoint_is_final": snapshot["completed_epochs"] == int(config["epochs"]),
        "global_step": snapshot["global_step"],
        "selection_policy": job.get("selection_policy"),
        "profile_identity_sha256": canonical_sha256(identity),
        "source_commit": SOURCE_COMMIT,
        "config_sha256": canonical_sha256(config),
        "split_sha256": split.sha256,
        "train_probe_members": job["train_probe_members"],
        "training_epoch_online": snapshot["history"][-1],
        "phases": {},
    }
    result_path = out / "RESULT.json"
    _write_json(result_path, output)
    import torch

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint evaluation requires CUDA")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    cap_bytes = int(args.max_gpu_gib * GIB)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < cap_bytes + GIB or cap_bytes >= total_bytes:
        raise MemoryError("insufficient free VRAM for the capped validation process")
    torch.cuda.set_per_process_memory_fraction(cap_bytes / total_bytes, device)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.set_num_interop_threads(1)
    model = build_sequence_model(config["model"])
    load_model_state(model, snapshot["model"])
    model = model.to(device=device, dtype=torch.float32).eval()
    expected = model.state_dict()
    if any(
        value.dtype != torch.float32 or not torch.isfinite(value).all().item()
        for value in expected.values()
    ):
        raise ValueError("loaded inference model has non-FP32 or nonfinite state")
    del expected, snapshot["model"]
    probe = job["train_probe_members"]
    if not isinstance(probe, list) or len(probe) != 64:
        raise ValueError("training probe must contain exactly 64 frozen members")
    member_indices = {member: index for index, member in enumerate(data.fit.members)}
    normalized_probe = [(str(source), int(group)) for source, group in probe]
    if len(set(normalized_probe)) != 64 or any(
        member not in member_indices for member in normalized_probe
    ):
        raise ValueError("training probe members are duplicated or outside this fit split")
    fit_indices = [member_indices[member] for member in normalized_probe]
    valid_indices = list(range(len(data.evaluation)))
    if len(data.evaluation) != len(split.evaluation_groups):
        raise ValueError("heldout dataset is not the complete evaluation fold")
    deadline = time.monotonic() + args.max_seconds
    batch_size = args.batch_size
    output["device"] = {
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "unknown")),
        "cap_bytes": cap_bytes,
        "free_before_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }
    _write_json(result_path, output)
    try:
        output["phases"]["train_probe"] = _phase(
            "train_probe",
            data.fit,
            fit_indices,
            model,
            device,
            config,
            out,
            deadline,
            args.limit_batches,
            batch_size,
            cap_bytes,
        )
        _write_json(result_path, output)
        if output["phases"]["train_probe"]["status"] == "complete" or args.limit_batches:
            output["phases"]["heldout"] = _phase(
                "heldout",
                data.evaluation,
                valid_indices,
                model,
                device,
                config,
                out,
                deadline,
                args.limit_batches,
                batch_size,
                cap_bytes,
            )
            _write_json(result_path, output)
    except Exception as error:
        output["status"] = "failed"
        output["error"] = f"{type(error).__name__}: {error}"
        _write_json(result_path, output)
        raise
    complete = all(
        output["phases"].get(name, {}).get("status") == "complete"
        for name in ("train_probe", "heldout")
    )
    output["status"] = "complete" if complete else "partial"
    output["profile_only"] = args.limit_batches is not None
    _write_json(result_path, output)
    return 0 if complete and args.limit_batches is None else 1


def main() -> int:
    """Parse a bounded immutable-snapshot evaluation request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(2,), default=2)
    parser.add_argument("--max-gpu-gib", type=float, default=16.0)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--limit-batches", type=int)
    args = parser.parse_args()
    if not 0 < args.max_gpu_gib <= 16 or not 1 <= args.max_seconds <= 7200:
        parser.error("GPU cap must be <=16 GiB and time limit 1..7200 seconds")
    if args.limit_batches is not None and args.limit_batches < 1:
        parser.error("limit-batches must be positive")
    try:
        return run(args)
    except Exception as error:
        result_path = args.output / "RESULT.json"
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["status"] = "failed"
                result["error"] = f"{type(error).__name__}: {error}"
                _write_json(result_path, result)
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        print(f"validation failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
