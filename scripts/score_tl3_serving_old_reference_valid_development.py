"""Compare explicit serving policies on pinned old references, using exposed valid only.

This helper computes the official pooled clipped weighted-Pearson metric for
historical train-only checkpoints as a development diagnostic. It is neither a
final-model score nor an independent evaluation. Checkpoints and validation
bytes are SHA-pinned; workers process whole causal sequences and merge moments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.package_final_tl3 as _packager  # noqa: E402
import utils as _callback  # noqa: E402
import wnn.data as _data  # noqa: E402
import wnn.final_submit_manifest as _manifest  # noqa: E402
import wnn.training.final_submit_bridge as _bridge  # noqa: E402

SOURCE_VALIDATION_PATH = (
    ROOT
    / "research"
    / "experiment_drafts"
    / "final-submit-all-labeled-20261031"
    / "receiving"
    / "SOURCE_VALIDATION_R1.json"
)
SEQUENCE_ROWS = 20_000
MAX_WORKERS = 32


@dataclass(frozen=True)
class _WorkerConfig:
    """Small spawn-safe paths and mode shared by scoring workers."""

    checkpoints_root: str
    valid_path: str
    native_dir: str
    fast_activations: bool
    precision: str
    serving_policy: str


@dataclass
class _WorkerContext:
    """Process-local model and parquet reader reused across assigned sequences."""

    predictor: Any  # PyO3 native predictor has no static Python stub.
    sequences: _data.SequenceFile
    contracts: pq.ParquetFile
    effective_precisions: tuple[str, ...]
    execution_shapes: tuple[tuple[object, ...], ...]


# Process-local native state is large and must persist across a worker's assigned groups.
_WORKER_CONTEXT: _WorkerContext | None = None


def _sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_paths(root: Path) -> tuple[Path, ...]:
    """Return the four historical reference paths in fixed blend order."""
    return (
        root / _packager.RUNS[0] / "checkpoints" / "model.pt",
        root / _packager.RUNS[1] / "checkpoints" / "model.pt",
        root / _packager.RUNS[2] / "checkpoints" / "model.pt",
        root / "models" / _packager.RUNS[3] / "model.pt",
    )


def _validate_source_identity(source_receipt_path: Path, valid_path: Path) -> tuple[str, str]:
    """Require the pinned source receipt and its exact official validation bytes."""
    if _sha256_file(source_receipt_path) != _bridge.SOURCE_VALIDATION_SHA256:
        raise ValueError("source-validation receipt SHA256 differs from the pinned receipt")
    receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != _bridge.SOURCE_VALIDATION_SCHEMA:
        raise ValueError("unsupported source-validation receipt schema")
    sources = receipt.get("sources")
    valid_source = sources.get("valid") if isinstance(sources, dict) else None
    if not isinstance(valid_source, dict):
        raise ValueError("source-validation receipt lacks the official valid identity")
    expected_sha = valid_source.get("file_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("source-validation receipt has a malformed valid SHA256")
    if valid_source.get("row_groups") != _manifest.VALID_SEQUENCES:
        raise ValueError("source-validation receipt valid membership differs from the fixed split")
    if valid_source.get("rows") != _manifest.VALID_SEQUENCES * SEQUENCE_ROWS:
        raise ValueError("source-validation receipt valid row count differs from the fixed split")
    actual_sha = _sha256_file(valid_path)
    if actual_sha != expected_sha:
        raise ValueError("valid parquet SHA256 differs from the pinned source-validation receipt")
    return _sha256_file(source_receipt_path), actual_sha


def _verified_checkpoint_paths(root: Path) -> tuple[tuple[Path, ...], dict[str, str]]:
    """Fail closed unless every checkpoint has its historical engineering SHA."""
    paths = _checkpoint_paths(root)
    hashes: dict[str, str] = {}
    for role, path, expected in zip(
        _packager.ROLES, paths, _packager.EXPECTED_CHECKPOINT_SHA256, strict=True
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{role} reference checkpoint must be a regular file: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"{role} checkpoint is not the pinned old reference model")
        hashes[role] = actual
    return paths, hashes


def _native_weights(checkpoint_paths: tuple[Path, ...]) -> list[dict[str, np.ndarray]]:
    """Export the exact pinned checkpoint tensors to the native named-array schema."""
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output: list[dict[str, np.ndarray]] = []
    for role, path in zip(_packager.ROLES, checkpoint_paths, strict=True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise ValueError(f"{role} checkpoint lacks a named model state")
        arrays: dict[str, np.ndarray] = {}
        for name, tensor in state.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{role} checkpoint state must contain named tensors")
            arrays[name] = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        weight = arrays.get("head.2.weight")
        bias = arrays.get("head.2.bias")
        if weight is None or bias is None or weight.shape[0] < 2 or bias.shape[0] < 2:
            raise ValueError(f"{role} checkpoint lacks the two target heads")
        arrays["head.2.weight"] = np.ascontiguousarray(weight[:2])
        arrays["head.2.bias"] = np.ascontiguousarray(bias[:2])
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError(f"{role} checkpoint contains nonfinite model tensors")
        output.append(arrays)
    return output


def _initialize_worker(config: _WorkerConfig) -> None:
    """Load one isolated native predictor and parquet handle per process."""
    global _WORKER_CONTEXT
    native_dir = Path(config.native_dir).resolve(strict=True)
    if not (native_dir / "wnn_native.abi3.so").is_file():
        raise FileNotFoundError(f"native directory lacks wnn_native.abi3.so: {native_dir}")
    sys.path.insert(0, str(native_dir))
    import torch
    import wnn_native

    torch.set_num_threads(1)
    checkpoint_paths = _checkpoint_paths(Path(config.checkpoints_root).resolve(strict=True))
    weights = _native_weights(checkpoint_paths)
    predictor = wnn_native.FinalTl3Predictor(
        weights,
        precision=config.precision,
        fast_activations=config.fast_activations,
        serving_policy=config.serving_policy,
    )
    if config.precision == "q8vnni" and not all(
        name.startswith("q8vnni") for name in predictor.effective_precisions
    ):
        raise RuntimeError(
            f"worker did not select requested dynamic Q8 kernels: {predictor.effective_precisions}"
        )
    if config.precision == "f32" and any(name != "f32" for name in predictor.effective_precisions):
        raise RuntimeError(
            f"worker did not preserve requested FP32 precision: {predictor.effective_precisions}"
        )
    active_members = (0,) if config.serving_policy == "w18_only" else (0, 1, 2, 3)
    if tuple(predictor.active_member_indices) != active_members:
        raise RuntimeError(
            f"worker activated unexpected members: {predictor.active_member_indices}"
        )
    execution_shapes = tuple(tuple(shape) for shape in predictor.member_execution_shapes)
    _WORKER_CONTEXT = _WorkerContext(
        predictor=predictor,
        sequences=_data.SequenceFile(config.valid_path),
        contracts=pq.ParquetFile(config.valid_path),
        effective_precisions=tuple(predictor.effective_precisions),
        execution_shapes=execution_shapes,
    )


def _score_partition(task: tuple[int, int, int]) -> dict[str, object]:
    """Score one contiguous range of complete validation row groups."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("scoring worker was not initialized")
    partition_id, start, stop = task
    context = _WORKER_CONTEXT
    accumulator = _callback.GlobalAccumulator()
    sequence_ids: list[int] = []
    started = time.perf_counter()
    steps = np.arange(SEQUENCE_ROWS)
    need = steps >= _callback.WARMUP
    for group in range(start, stop):
        sequence = context.sequences.read(group, targets=True, mask=True)
        contract_table = context.contracts.read_row_group(
            group, columns=["seq_ix", "step_in_seq", "need_prediction"], use_threads=False
        )
        contract_id, stored_need = _callback.validate_sequence(contract_table)
        if contract_id != sequence.seq_ix or not np.array_equal(stored_need, need):
            raise ValueError(f"valid row group {group} violates the canonical row/need contract")
        if sequence.targets is None or sequence.scored is None:
            raise ValueError(f"valid row group {group} lacks labels or is_scored mask")
        if (
            sequence.features.shape != (SEQUENCE_ROWS, 112)
            or not sequence.features.flags.c_contiguous
        ):
            raise ValueError(f"valid row group {group} does not match the canonical raw112 layout")
        if not np.isfinite(sequence.features).all():
            raise ValueError(f"valid row group {group} contains nonfinite raw inputs")
        if np.asarray(sequence.scored, dtype=bool)[: _callback.WARMUP].any():
            raise ValueError(f"valid row group {group} scores a warm-up row")
        predictions = np.full((SEQUENCE_ROWS, 2), np.nan, dtype=np.float32)
        context.predictor.reset()
        for step, row in enumerate(sequence.features):
            value = context.predictor.predict(
                _callback.DataPoint(sequence.seq_ix, step, bool(need[step]), row)
            )
            if not need[step]:
                if value is not None:
                    raise ValueError(
                        f"callback emitted during warm-up at group {group}, row {step}"
                    )
                continue
            if value is None:
                raise ValueError(f"callback omitted required output at group {group}, row {step}")
            output = np.asarray(value, dtype=np.float32)
            if output.shape != (2,) or not np.isfinite(output).all():
                raise ValueError(f"callback returned invalid output at group {group}, row {step}")
            predictions[step] = output
        accumulator.add(
            sequence.targets, predictions, np.asarray(sequence.scored, dtype=bool) & need
        )
        sequence_ids.append(sequence.seq_ix)
    return {
        "partition_id": partition_id,
        "groups": stop - start,
        "sequence_ids": sequence_ids,
        "sums": accumulator.sums.tolist(),
        "moments": accumulator._moments.tolist(),
        "selected_rows": accumulator.selected_rows,
        "blocks_seen": accumulator.blocks_seen,
        "elapsed_seconds": time.perf_counter() - started,
        "effective_precisions": list(context.effective_precisions),
        "execution_shapes": [list(shape) for shape in context.execution_shapes],
    }


def _partition_ranges(total: int, count: int) -> list[tuple[int, int, int]]:
    """Split row-group ordinals into deterministic contiguous worker ranges."""
    base, remainder = divmod(total, count)
    ranges = []
    start = 0
    for index in range(count):
        stop = start + base + (index < remainder)
        ranges.append((index, start, stop))
        start = stop
    return ranges


def _merge_partials(partials: list[dict[str, object]]) -> _callback.GlobalAccumulator:
    """Merge worker-centered moments in fixed partition order without raw-moment cancellation."""
    merged = _callback.GlobalAccumulator()
    for partial in sorted(partials, key=lambda item: int(item["partition_id"])):
        sums = np.asarray(partial["sums"], dtype=np.float64)
        moments = np.asarray(partial["moments"], dtype=np.float64)
        if sums.shape != (2, 6) or moments.shape != (2, 6):
            raise ValueError("worker accumulator has an unsupported moment layout")
        merged.sums += sums
        for target in range(2):
            previous = merged._moments[target]
            incoming = moments[target]
            if incoming[0] == 0.0:
                continue
            if previous[0] == 0.0:
                previous[:] = incoming
                continue
            combined = previous[0] + incoming[0]
            ratio = incoming[0] / combined
            dy = incoming[1] - previous[1]
            dp = incoming[2] - previous[2]
            cross_weight = previous[0] * ratio
            previous[:] = (
                combined,
                previous[1] + dy * ratio,
                previous[2] + dp * ratio,
                previous[3] + incoming[3] + dy * dy * cross_weight,
                previous[4] + incoming[4] + dp * dp * cross_weight,
                previous[5] + incoming[5] + dy * dp * cross_weight,
            )
        merged.selected_rows += int(partial["selected_rows"])
        merged.blocks_seen += int(partial["blocks_seen"])
    return merged


def _load_args() -> argparse.Namespace:
    """Parse bounded source, worker and output arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--source-validation", type=Path, default=SOURCE_VALIDATION_PATH)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--activation-mode", choices=("exact", "fast"), default="exact")
    parser.add_argument("--precision", choices=("f32", "q8vnni"), default="q8vnni")
    parser.add_argument("--serving-policy", choices=("full_mix", "w18_only"), default="full_mix")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the pinned historical reference Q8 development replay and write its receipt."""
    args = _load_args()
    checkpoints_root = args.checkpoints_root.resolve(strict=True)
    valid_path = args.valid.resolve(strict=True)
    source_receipt = args.source_validation.resolve(strict=True)
    native_dir = args.native_dir.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite development receipt: {output_path}")
    if _callback.SEQUENCE_LENGTH != SEQUENCE_ROWS:
        raise RuntimeError("utils callback sequence length differs from source SequenceFile")
    if (
        _callback.FEATURE_COLUMNS != _data.FEATURE_COLUMNS
        or _callback.TARGET_COLUMNS != _data.TARGET_COLUMNS
    ):
        raise RuntimeError("SequenceFile and official callback feature/target orders differ")
    valid_receipt_sha, valid_sha_before = _validate_source_identity(source_receipt, valid_path)
    checkpoint_paths, checkpoint_hashes = _verified_checkpoint_paths(checkpoints_root)
    parquet = _data.SequenceFile(valid_path)
    if len(parquet) != _manifest.VALID_SEQUENCES or not parquet.has_targets or not parquet.has_mask:
        raise ValueError("valid parquet does not have the exact labeled 1,873-group contract")
    if not (native_dir / "wnn_native.abi3.so").is_file():
        raise FileNotFoundError(f"native extension is absent from {native_dir}")
    allowed_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    available_cpus = len(allowed_cpus) if allowed_cpus else (os.cpu_count() or 1)
    requested_workers = (
        args.workers if args.workers is not None else min(MAX_WORKERS, available_cpus)
    )
    if not 1 <= requested_workers <= MAX_WORKERS:
        raise ValueError(f"workers must be in 1..{MAX_WORKERS}")
    if requested_workers > available_cpus:
        raise ValueError("worker count exceeds the process CPU affinity")

    config = _WorkerConfig(
        checkpoints_root=str(checkpoints_root),
        valid_path=str(valid_path),
        native_dir=str(native_dir),
        fast_activations=args.activation_mode == "fast",
        precision=args.precision,
        serving_policy=args.serving_policy,
    )
    ranges = _partition_ranges(_manifest.VALID_SEQUENCES, requested_workers)
    with ProcessPoolExecutor(
        max_workers=requested_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(config,),
    ) as executor:
        partials = list(executor.map(_score_partition, ranges, chunksize=1))

    if sum(int(item["groups"]) for item in partials) != _manifest.VALID_SEQUENCES:
        raise RuntimeError("parallel scoring omitted validation row groups")
    if sum(int(item["blocks_seen"]) for item in partials) != _manifest.VALID_SEQUENCES:
        raise RuntimeError("GlobalAccumulator did not receive exactly one block per sequence")
    sequence_ids = [int(seq) for item in partials for seq in item["sequence_ids"]]
    if (
        len(sequence_ids) != _manifest.VALID_SEQUENCES
        or len(set(sequence_ids)) != _manifest.VALID_SEQUENCES
    ):
        raise RuntimeError("validation sequence IDs are missing or duplicated")
    effective_precisions = {tuple(item["effective_precisions"]) for item in partials}
    execution_shapes = {json.dumps(item["execution_shapes"], sort_keys=True) for item in partials}
    if len(effective_precisions) != 1 or len(execution_shapes) != 1:
        raise RuntimeError("workers selected inconsistent native precisions or member layouts")
    accumulator = _merge_partials(partials)
    if accumulator.blocks_seen != _manifest.VALID_SEQUENCES or accumulator.selected_rows <= 0:
        raise RuntimeError("merged validation accumulator is incomplete")
    valid_sha_after = _sha256_file(valid_path)
    checkpoint_hashes_after = {
        role: _sha256_file(path)
        for role, path in zip(_packager.ROLES, checkpoint_paths, strict=True)
    }
    if valid_sha_after != valid_sha_before or checkpoint_hashes_after != checkpoint_hashes:
        raise RuntimeError("pinned validation or reference checkpoint bytes changed during scoring")

    native_path = native_dir / "wnn_native.abi3.so"
    report = {
        "schema": "wnn.tl3-serving-old-reference-valid-development.v1",
        "candidate": "historical_tl3_source4_engineering_references",
        "claim_policy": {
            "validation_status": "exposed_development_data",
            "independent_evaluation": False,
            "final_model_score_claim": "forbidden",
            "not_a_final_full_corpus_model_set": True,
        },
        "models": {
            "roles": list(_packager.ROLES),
            "run_references": list(_packager.RUNS),
            "checkpoint_sha256": checkpoint_hashes,
            "requested_precision": args.precision,
            "serving_policy": args.serving_policy,
            "active_member_indices": [0] if args.serving_policy == "w18_only" else [0, 1, 2, 3],
            "comparison_purpose": "old-reference-serving-fidelity-only; no final model selection",
            "effective_precision_by_member": list(next(iter(effective_precisions))),
            "activation_mode": args.activation_mode,
            "execution_shapes": json.loads(next(iter(execution_shapes))),
        },
        "sources": {
            "valid_path": str(valid_path),
            "valid_sha256": valid_sha_before,
            "source_validation_receipt": str(source_receipt),
            "source_validation_receipt_sha256": valid_receipt_sha,
            "sequence_length": SEQUENCE_ROWS,
            "row_groups": list(range(_manifest.VALID_SEQUENCES)),
            "sequence_ids_sha256": hashlib.sha256(
                np.asarray(sequence_ids, dtype="<i8").tobytes()
            ).hexdigest(),
            "feature_source": "wnn.data.SequenceFile; canonical FEATURE_COLUMNS raw112 order",
            "scoring_mask": "is_scored AND step_in_seq>=99",
        },
        "execution": {
            "workers": requested_workers,
            "worker_cap": MAX_WORKERS,
            "cpu_affinity": allowed_cpus,
            "whole_sequence_contiguous_partitions": True,
            "all_rows_causally_replayed": True,
            "global_accumulator_moments_merged_in_partition_order": True,
            "native_library": str(native_path),
            "native_library_sha256": _sha256_file(native_path),
            "worker_elapsed_seconds": [float(item["elapsed_seconds"]) for item in partials],
        },
        "exposed_valid_development_metric": accumulator.result(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
