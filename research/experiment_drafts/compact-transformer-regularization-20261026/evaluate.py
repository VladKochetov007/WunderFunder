from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PINNED_SOURCE_COMMIT = "0a70b4ab96c1810484544f382df4bcf0c08e8fd4"
EXPECTED_JOBS = {"ctl_f0", "reg_f0", "ctl_f2", "reg_f2"}
GIB = 1024**3


class EvaluationDeadline(TimeoutError):
    """Raised between complete model chunks when the bounded evaluation expires."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("ab") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def _resolve(value: object, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing path field {label}")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve(strict=True)


def _load_context(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    plan_path = args.plan.resolve(strict=True)
    plan = _json(plan_path)
    if plan.get("source_commit") != PINNED_SOURCE_COMMIT:
        raise ValueError("plan source_commit does not match the pinned canonical source")
    base = plan_path.parent
    paths = {
        name: _resolve(plan.get(name), base, name)
        for name in ("source_root", "data_root", "run_root")
    }
    if not (paths["source_root"] / "src" / "wnn").is_dir():
        raise ValueError("source_root must contain src/wnn")
    sys.path.insert(0, str(paths["source_root"] / "src"))
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("plan jobs must be an array")
    by_name = {item.get("name"): item for item in jobs if isinstance(item, dict)}
    if set(by_name) != EXPECTED_JOBS or args.job not in by_name:
        raise ValueError("plan must contain exactly the four approved jobs")
    raw_job = by_name[args.job]
    job = dict(raw_job)
    for field in ("config", "split_manifest", "profile_receipt"):
        job[field] = _resolve(job.get(field), base, f"{args.job}.{field}")
    job["gpu_uuids"] = raw_job.get("gpu_uuids")
    if not isinstance(job["gpu_uuids"], list) or not job["gpu_uuids"]:
        raise ValueError("job must declare its assigned GPU UUIDs")
    if args.expected_gpu_uuid != job["gpu_uuids"][0]:
        raise ValueError("evaluator GPU argument differs from the job's first assigned UUID")
    return plan, job, paths


def _check_recipe(name: str, raw: dict[str, Any]) -> None:
    arm, fold_text = name.rsplit("_", 1)
    fold = int(fold_text.removeprefix("f"))
    expected_dropout, expected_decay = (0.0, 0.01) if arm == "ctl" else (0.1, 0.1)
    fixed = {
        "epochs": 5,
        "lr": 0.0001,
        "warmup_fraction": 0.05,
        "schedule": "warmup_cosine_v2",
        "min_lr_ratio": 0.1,
        "optimizer": "muon",
        "precision_mode": "bf16_bf16x9",
        "checkpoint_retention": "all_epochs",
        "dropout_policy": "explicit_regularization_v1",
        "expected_fold": fold,
        "kd_weight": 0.0,
        "hint_weight": 0.0,
    }
    if any(raw.get(key) != value for key, value in fixed.items()):
        raise ValueError(f"{name} recipe differs from the approved five-epoch labels-only recipe")
    model = raw.get("model")
    expected_model = {
        "architecture": "transformer",
        "hidden": 768,
        "layers": 16,
        "heads": 12,
        "ff_multiplier": 4,
        "context_rows": 1024,
        "dropout": expected_dropout,
        "attention_backend": "efficient",
        "checkpoint_blocks": False,
        "compile_blocks": True,
    }
    if not isinstance(model, dict) or any(
        model.get(key) != value for key, value in expected_model.items()
    ):
        raise ValueError(f"{name} Transformer settings differ from the approved arm")
    if raw.get("weight_decay") != expected_decay:
        raise ValueError(f"{name} weight decay differs from the approved arm")


def _probe_members(members: list[tuple[str, int]], split_sha256: str) -> list[tuple[str, int]]:
    if len(members) < 64:
        raise ValueError(
            "fit membership has fewer than 64 sequences for the fixed diagnostic probe"
        )

    def rank(member: tuple[str, int]) -> str:
        token = (
            f"compact-transformer-regularization-probe-v1:{split_sha256}:{member[0]}:{member[1]}"
        )
        return hashlib.sha256(token.encode()).hexdigest()

    selected = set(sorted(members, key=rank)[:64])
    return [member for member in members if member in selected]


def _uuid_key(value: str) -> str:
    return value.lower().removeprefix("gpu-").replace("-", "")


def _device(torch: Any, expected_uuid: str, max_gpu_gib: float) -> tuple[Any, dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    actual_uuid = getattr(properties, "uuid", None)
    if actual_uuid is None or _uuid_key(str(actual_uuid)) != _uuid_key(expected_uuid):
        raise RuntimeError(f"visible CUDA device UUID mismatch: got {actual_uuid!r}")
    total = int(properties.total_memory)
    free, _ = torch.cuda.mem_get_info(device)
    cap = int(max_gpu_gib * GIB)
    if cap <= 0 or cap > total or int(free) < cap + GIB:
        raise RuntimeError("GPU lacks the requested 12 GiB inference cap plus 1 GiB headroom")
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    return device, {
        "uuid": str(actual_uuid),
        "name": str(properties.name),
        "total_memory_bytes": total,
        "free_memory_bytes_before_load": int(free),
        "allocator_cap_bytes": cap,
    }


def _run_phase(
    *,
    torch: Any,
    dataset: Any,
    members: list[tuple[str, int]],
    model: Any,
    config: dict[str, Any],
    device: Any,
    batch_size: int,
    warmup: int,
    phase: str,
    deadline: float,
    output: Path,
    progress: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from metrics import StreamingMetrics

    from wnn.data import SEQUENCE_LENGTH
    from wnn.training.sequence_fit_utils import collate_sequences, move_and_validate_batch
    from wnn.training.sequence_precision import forward_context

    if not members:
        raise ValueError(f"{phase} phase has no sequences")
    index_by_member = {member: index for index, member in enumerate(dataset.members)}
    try:
        dataset_indices = [index_by_member[member] for member in members]
    except KeyError as error:
        raise ValueError(f"{phase} member is absent from the validated dataset") from error
    all_metrics = StreamingMetrics()
    scored_metrics = StreamingMetrics() if phase == "full_inner_heldout" else None
    elapsed_start = time.monotonic()
    total_batches = math.ceil(len(members) / batch_size)
    chunk_rows = int(config["chunk_rows"])
    prefix = "heldout" if phase == "full_inner_heldout" else "train_probe"
    artifact_paths = {
        "predictions": output / f"{prefix}.predictions.npy",
        "targets": output / f"{prefix}.targets.npy",
        "selected_mask": output / f"{prefix}.selected_mask.npy",
        "post_warmup_mask": output / f"{prefix}.post_warmup_mask.npy",
    }
    if any(path.exists() or path.is_symlink() for path in artifact_paths.values()):
        raise FileExistsError(f"{phase} artifact already exists; refusing to overwrite")
    artifact_maps = {
        "predictions": np.lib.format.open_memmap(
            artifact_paths["predictions"],
            mode="w+",
            dtype=np.float32,
            shape=(len(members), SEQUENCE_LENGTH, 2),
        ),
        "targets": np.lib.format.open_memmap(
            artifact_paths["targets"],
            mode="w+",
            dtype=np.float32,
            shape=(len(members), SEQUENCE_LENGTH, 2),
        ),
        "selected_mask": np.lib.format.open_memmap(
            artifact_paths["selected_mask"],
            mode="w+",
            dtype=np.bool_,
            shape=(len(members), SEQUENCE_LENGTH),
        ),
        "post_warmup_mask": np.lib.format.open_memmap(
            artifact_paths["post_warmup_mask"],
            mode="w+",
            dtype=np.bool_,
            shape=(len(members), SEQUENCE_LENGTH),
        ),
    }
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for batch_number, offset in enumerate(range(0, len(members), batch_size), start=1):
            if time.monotonic() >= deadline:
                raise EvaluationDeadline(f"deadline reached during {phase}")
            selection = dataset_indices[offset : offset + batch_size]
            batch = move_and_validate_batch(collate_sequences(dataset, selection), device, 0)
            row_count = int(batch["x"].shape[1])
            predictions = np.empty((len(selection), row_count, 2), dtype=np.float32)
            state = None
            with torch.inference_mode():
                for row_start in range(0, row_count, chunk_rows):
                    if time.monotonic() >= deadline:
                        raise EvaluationDeadline(f"deadline reached during {phase} chunk")
                    row_end = min(row_count, row_start + chunk_rows)
                    with forward_context(str(config["precision_mode"]), device.type):
                        prediction, _, state = model(batch["x"][:, row_start:row_end], state)
                    predictions[:, row_start:row_end] = prediction.float().cpu().numpy()
            targets = batch["y"].float().cpu().numpy()
            scored_mask = batch["mask"].cpu().numpy().astype(bool, copy=False)
            post_warmup = np.ones(scored_mask.shape, dtype=bool)
            post_warmup[:, :warmup] = False
            all_metrics.update(targets, predictions, scored_mask, wmse_mask=post_warmup)
            if scored_metrics is not None:
                scored_metrics.update(targets, predictions, scored_mask, wmse_mask=scored_mask)
            selection_slice = slice(offset, offset + len(selection))
            artifact_maps["predictions"][selection_slice] = predictions
            artifact_maps["targets"][selection_slice] = targets
            artifact_maps["selected_mask"][selection_slice] = scored_mask
            artifact_maps["post_warmup_mask"][selection_slice] = post_warmup
            torch.cuda.synchronize(device)
            progress.update(
                {
                    "phase": phase,
                    "completed_sequences": offset + len(selection),
                    "total_sequences": len(members),
                    "completed_batches": batch_number,
                    "total_batches": total_batches,
                    "elapsed_seconds": time.monotonic() - elapsed_start,
                    "artifact_rows_written": offset + len(selection),
                    "last_member_batch": [
                        [source, group]
                        for source, group in members[offset : offset + len(selection)]
                    ],
                }
            )
            _atomic_json(output / "PROGRESS.json", progress)
            _append_jsonl(output / "progress.jsonl", progress)
            del batch, predictions, targets, scored_mask, post_warmup, state
    finally:
        for array in artifact_maps.values():
            array.flush()
        for path in artifact_paths.values():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        artifact_maps.clear()
    artifact_receipts = {
        name: {
            "path": path.name,
            "sha256": _sha256(path),
            "shape": (
                [len(members), SEQUENCE_LENGTH, 2]
                if name in ("predictions", "targets")
                else [len(members), SEQUENCE_LENGTH]
            ),
            "dtype": "float32" if name in ("predictions", "targets") else "bool",
        }
        for name, path in artifact_paths.items()
    }
    torch.cuda.synchronize(device)
    metric_scope = (
        "official_inner_holdout" if phase == "full_inner_heldout" else "in_sample_diagnostic"
    )
    all_result = all_metrics.result()
    metric_result: dict[str, object] = {
        "weighted_mse_all_after_warmup": all_result["wmse"],
        "weighted_mse_rows_all_after_warmup": all_result["wmse_rows"],
        "weighted_mse_metric_id": all_result["wmse_metric_id"],
    }
    if scored_metrics is None:
        metric_result["diagnostic_wp_all_after_warmup"] = all_result["wp"]
    else:
        scored_result = scored_metrics.result()
        metric_result.update(
            {
                "official_wp": all_result["wp"],
                "official_wp_t0": all_result["t0"],
                "official_wp_t1": all_result["t1"],
                "official_selected_rows": all_result["selected_rows"],
                "official_wp_metric_id": all_result["metric_id"],
                "weighted_mse_on_scored_rows": scored_result["wmse"],
                "weighted_mse_rows_on_scored_rows": scored_result["wmse_rows"],
            }
        )
    return {
        "scope": metric_scope,
        "sequence_count": len(members),
        "member_sha256": hashlib.sha256(
            json.dumps([[s, g] for s, g in members], separators=(",", ":")).encode()
        ).hexdigest(),
        "members": [[source, group] for source, group in members],
        "artifacts": artifact_receipts,
        "metrics": metric_result,
        "elapsed_seconds": time.monotonic() - elapsed_start,
        "batch_size": batch_size,
        "chunk_rows": chunk_rows,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def run_evaluation(args: argparse.Namespace) -> int:
    process_started = time.monotonic()
    process_deadline = process_started + args.max_seconds
    started_utc_unix = time.time()
    plan, job, paths = _load_context(args)
    config_file = Path(job["config"])
    raw_config = _json(config_file)
    _check_recipe(args.job, raw_config)
    if os.environ.get("CUBLAS_EMULATE_SINGLE_PRECISION") != "1":
        raise ValueError("BF16x9 evaluations require CUBLAS_EMULATE_SINGLE_PRECISION=1")

    import torch

    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    from wnn.data import WARMUP
    from wnn.splits import SEQUENCE_INNER_SPLIT_SCHEMA
    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_data import build_data
    from wnn.training.sequence_epoch_index import load_epoch_index
    from wnn.training.sequence_epoch_prediction import epoch_admission
    from wnn.training.sequence_predict import load_endpoint
    from wnn.training.sequence_provenance import (
        canonical_sha256,
        source_identity,
        validate_profile_receipt,
    )

    output = args.output.resolve(strict=True)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("evaluation output must be a real precreated directory")
    unexpected = {entry.name for entry in output.iterdir()} - {"worker.log"}
    if unexpected:
        raise FileExistsError(f"evaluation output is not fresh: {sorted(unexpected)}")
    index_path = args.epoch_index.resolve(strict=True)
    fit_root = paths["run_root"] / args.job / "fit"
    if index_path.parent != fit_root / "epoch_indexes" or not index_path.name.endswith(".json"):
        raise ValueError("epoch index must be this job's immutable epoch_indexes snapshot")
    epoch_number = int(index_path.stem.removeprefix("epoch-"))
    if epoch_number < 1 or epoch_number > 5:
        raise ValueError("retained epoch is outside the fixed five-epoch schedule")
    config = load_config(config_file)
    profile = _json(Path(job["profile_receipt"]))
    identity = profile.get("identity")
    if not isinstance(identity, dict) or canonical_sha256(config) != identity.get("config_sha256"):
        raise ValueError("profile identity does not bind the configured recipe")
    profile_sha = validate_profile_receipt(Path(job["profile_receipt"]), identity)
    if identity.get("world_size") != len(job["gpu_uuids"]):
        raise ValueError("profile world size differs from the assigned job GPUs")
    observed_source = source_identity(
        paths["source_root"], paths["source_root"] / "scripts" / "train_sequence.py"
    )
    if observed_source.get("sha256") != identity.get("source_sha256") or observed_source.get(
        "files"
    ) != identity.get("source_files"):
        raise ValueError("canonical source tree differs from the approved fit identity")
    device, resource_start = _device(torch, args.expected_gpu_uuid, args.max_gpu_gib)
    index = load_epoch_index(index_path, identity)
    entry = index["entries"][-1]
    checkpoint_relative = Path(str(entry["checkpoint_path"]))
    if checkpoint_relative.is_absolute() or ".." in checkpoint_relative.parts:
        raise ValueError("epoch index checkpoint path must remain relative to fit")
    checkpoint = fit_root / checkpoint_relative
    admission = epoch_admission(checkpoint, index_path, args.epoch_index_sha256)
    index = admission["index"]
    if index.get("identity") != identity or index.get("index_sha256") != args.epoch_index_sha256:
        raise ValueError("epoch index is not bound to the approved profile identity")
    if int(entry["completed_epochs"]) != epoch_number:
        raise ValueError("epoch snapshot name does not match its safe frontier")
    if Path(entry["checkpoint_path"]).is_absolute():
        raise ValueError("epoch checkpoint index path must be relative")
    split_raw = _json(Path(job["split_manifest"]))
    selection = split_raw.get("selection")
    if split_raw.get("schema") != SEQUENCE_INNER_SPLIT_SCHEMA or not isinstance(selection, dict):
        raise ValueError("validation requires the approved source-qualified inner split")
    fraction = selection.get("evaluation_fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or abs(float(fraction) - 0.2) > 1e-12
    ):
        raise ValueError("inner validation fraction must be exactly 20 percent")
    data = build_data(config, paths["data_root"], Path(job["split_manifest"]))
    if data.split.fold != config["expected_fold"]:
        raise ValueError("inner split fold differs from the recipe")
    model, payload = load_endpoint(checkpoint, device, admission)
    if payload["config"] != config or payload["data"].get("split_sha256") != data.split.sha256:
        raise ValueError("retained checkpoint differs from the plan config or inner split")
    if payload["data"].get("fit_members") != [list(member) for member in data.fit.members]:
        raise ValueError("checkpoint fit membership differs from the canonical inner fit set")
    if payload["data"].get("evaluation_members") != [
        list(member) for member in data.evaluation.members
    ]:
        raise ValueError("checkpoint heldout membership differs from the full inner evaluation set")
    for field in (
        "fold",
        "sources",
        "input_schema",
        "mask",
        "training_mask",
        "teacher",
        "hint_mode",
        "hint_members",
    ):
        if payload["data"].get(field) != data.provenance.get(field):
            raise ValueError(f"checkpoint data provenance differs in {field}")
    checkpoint_normalization = payload["data"].get("normalization")
    built_normalization = data.provenance.get("normalization")
    if not isinstance(checkpoint_normalization, dict) or not isinstance(built_normalization, dict):
        raise ValueError("checkpoint normalization provenance is incomplete")
    for field in ("scope", "sequence_count", "source_fingerprints"):
        if checkpoint_normalization.get(field) != built_normalization.get(field):
            raise ValueError(f"checkpoint normalization provenance differs in {field}")
    if payload["data"].get("teacher") is not None:
        raise ValueError("labels-only control and regularized fits must have no teacher ancestry")
    history = payload["provenance"].get("history")
    if not isinstance(history, list) or len(history) < epoch_number:
        raise ValueError("retained checkpoint lacks its completed epoch loss history")
    epoch_loss = history[epoch_number - 1]
    if not isinstance(epoch_loss, dict) or epoch_loss.get("complete") is not True:
        raise ValueError("epoch history does not mark this boundary complete")
    probe = _probe_members(list(data.fit.members), data.split.sha256)
    component_by_member = {
        (source, group): component for source, group, component in data.split.components
    }
    heldout_components = sorted({component_by_member[member] for member in data.evaluation.members})
    deadline = process_deadline
    common = {
        "schema": "wnn.compact-transformer-validation.v1",
        "job": args.job,
        "fold": data.split.fold,
        "completed_epochs": epoch_number,
        "epoch": entry["epoch"],
        "global_step": entry["global_step"],
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "epoch_index_path": str(index_path),
        "epoch_index_sha256": args.epoch_index_sha256,
        "profile_receipt_path": str(Path(job["profile_receipt"]).resolve()),
        "profile_receipt_sha256": profile_sha,
        "config_sha256": identity["config_sha256"],
        "split_sha256": data.split.sha256,
        "inner_split_selection": split_raw["selection"],
        "heldout_component_ids": heldout_components,
        "source_commit": plan["source_commit"],
        "source_sha256": identity["source_sha256"],
        "evaluation_operator_sha256": _sha256(Path(__file__).resolve()),
        "metrics_operator_sha256": _sha256(Path(__file__).with_name("metrics.py")),
        "online_epoch_loss": epoch_loss,
        "resource_start": resource_start,
        "max_gpu_gib": args.max_gpu_gib,
        "started_utc_unix": started_utc_unix,
        "phases": {},
    }
    state = dict(common) | {"status": "running"}
    _atomic_json(output / "RESULT.json", state)
    progress: dict[str, Any] = {"job": args.job, "completed_epochs": epoch_number}
    try:
        state["train_probe"] = _run_phase(
            torch=torch,
            dataset=data.fit,
            members=probe,
            model=model,
            config=config,
            device=device,
            batch_size=args.batch_size,
            warmup=WARMUP,
            phase="fit_diagnostic_probe",
            deadline=deadline,
            output=output,
            progress=progress,
        )
        state["phases"]["train_probe"] = "complete"
        state["heldout"] = _run_phase(
            torch=torch,
            dataset=data.evaluation,
            members=list(data.evaluation.members),
            model=model,
            config=config,
            device=device,
            batch_size=args.batch_size,
            warmup=WARMUP,
            phase="full_inner_heldout",
            deadline=deadline,
            output=output,
            progress=progress,
        )
        state["phases"]["heldout"] = "complete"
        state["status"] = "complete"
        state["elapsed_seconds"] = time.monotonic() - process_started
        state["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        state["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        if state["peak_reserved_bytes"] > int(args.max_gpu_gib * GIB):
            raise RuntimeError("CUDA allocator exceeded the configured per-process cap")
        state["finished_utc_unix"] = time.time()
        _atomic_json(output / "RESULT.json", state)
        return 0
    except EvaluationDeadline as error:
        state["status"] = "partial"
        state["deadline_reason"] = str(error)
        state["elapsed_seconds"] = time.monotonic() - process_started
        state["finished_utc_unix"] = time.time()
        _atomic_json(output / "RESULT.json", state)
        return 2
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        state["elapsed_seconds"] = time.monotonic() - process_started
        state["finished_utc_unix"] = time.time()
        _atomic_json(output / "RESULT.json", state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one immutable retained epoch without changing its fit."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job", choices=sorted(EXPECTED_JOBS), required=True)
    parser.add_argument("--epoch-index", type=Path, required=True)
    parser.add_argument("--epoch-index-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(4, 8), default=4)
    parser.add_argument("--max-gpu-gib", type=float, default=12.0)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()
    if not 0 < args.max_gpu_gib <= 12 or not 1 <= args.max_seconds <= 900:
        parser.error("GPU cap and duration must be bounded to 12 GiB and 900 seconds")
    raise SystemExit(run_evaluation(args))


if __name__ == "__main__":
    main()
