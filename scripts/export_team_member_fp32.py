"""Export one imported team GRU member through the project FP32 model path.

This is a diagnostic member exporter. It does not reproduce the package's native
quantization, activation approximations, ensemble coefficients, or callback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import resource
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from team_member_export_lib import (
    N_MODEL_FEATURES,
    N_OUTPUTS,
    N_RAW_FEATURES,
    SEQUENCE_LENGTH,
    WARMUP,
    digest,
    infer_architecture,
    iter_chunks,
    load_fp32_model,
    member_source,
    native_difference,
    select_row_groups,
)

if TYPE_CHECKING:
    import numpy as np
    import pyarrow.parquet as pq


def _read_group(
    parquet: pq.ParquetFile, group: int, feature_columns: list[str]
) -> tuple[int, np.ndarray]:
    """Read and validate one complete source-qualified sequence in row order."""
    import numpy as np

    table = parquet.read_row_group(
        group, columns=["seq_ix", "step_in_seq", *feature_columns], use_threads=False
    )
    if table.num_rows != SEQUENCE_LENGTH:
        raise ValueError(f"row group {group} has {table.num_rows} rows")
    seq_ids = table.column("seq_ix").to_numpy()
    steps = table.column("step_in_seq").to_numpy()
    if not np.all(seq_ids == seq_ids[0]):
        raise ValueError(f"row group {group} contains multiple seq_ix values")
    if not np.array_equal(steps, np.arange(SEQUENCE_LENGTH, dtype=steps.dtype)):
        raise ValueError(f"row group {group} is not a complete ordered sequence")
    features = np.ascontiguousarray(
        np.stack([table.column(name).to_numpy() for name in feature_columns], axis=0)
        .astype(np.float32, copy=False)
        .T
    )
    if features.shape != (SEQUENCE_LENGTH, N_RAW_FEATURES):
        raise ValueError(f"row group {group} has unexpected feature shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"row group {group} contains non-finite inputs")
    return int(seq_ids[0]), features


def _evict_output_pages(predictions: np.memmap, path: Path) -> None:
    """Keep the flushed prediction memmap from accumulating page-cache charges."""
    predictions.flush()
    mapped = getattr(predictions, "_mmap", None)
    if mapped is not None and hasattr(mapped, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
        mapped.madvise(mmap.MADV_DONTNEED)
    if hasattr(os, "posix_fadvise"):
        with path.open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one team member as causal FP32 PyTorch predictions on CUDA"
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--member", required=True, help="declared NPZ basename, e.g. weights_s5.npz"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--first-group", type=int, default=0)
    parser.add_argument(
        "--group-count", type=int, default=None, help="defaults to all remaining groups"
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device; CPU fallback is disabled")
    parser.add_argument(
        "--batch-sequences",
        type=int,
        choices=(1, 2),
        default=1,
        help="complete sequences in each GPU batch (profile 1 or 2)",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=500,
        help="recurrent rows per stateful forward chunk; event features use full sequences",
    )
    return parser.parse_args()


def main() -> None:
    """Export complete validation sequences one at a time to a disk-backed array."""
    args = _parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("this diagnostic exporter requires CUDA; it has no CPU fallback")
    if args.out.exists():
        raise FileExistsError(args.out)
    inventory_sha256 = digest(args.inventory)
    inventory, weights_path, weight_record, member_record = member_source(
        args.inventory, args.candidate, args.member
    )
    if digest(args.inventory) != inventory_sha256:
        raise ValueError("team inventory changed during source resolution")
    package_record = inventory["packages"][args.candidate]
    archive_path = Path(package_record["archive"])
    if digest(archive_path) != package_record["sha256"]:
        raise ValueError("team source archive differs from the inventory")
    if digest(weights_path) != weight_record["sha256"]:
        raise ValueError("selected NPZ differs from its inventory hash")
    native_runtime_note = native_difference(args.candidate, package_record)
    data_identity = inventory["data"]
    if args.data.stat().st_size != data_identity["bytes"]:
        raise ValueError("valid.parquet size differs from the inventory")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU inference is intentionally disabled")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_sha256 = digest(args.data)
    if data_sha256 != data_identity["sha256"]:
        raise ValueError("valid.parquet hash differs from the inventory")

    import pyarrow.parquet as pq

    from wnn.data import FEATURE_COLUMNS
    from wnn.features import event_features_torch

    with np.load(weights_path, allow_pickle=False) as archive:
        shapes = {name: tuple(archive[name].shape) for name in archive.files}
        dtypes = {name: str(archive[name].dtype) for name in archive.files}
    if any(dtype != "float32" for dtype in dtypes.values()):
        raise ValueError("team NPZ tensors must already be FP32")
    inventory_tensors = weight_record["tensors"]
    expected_shapes = {name: tuple(record["shape"]) for name, record in inventory_tensors.items()}
    if shapes != expected_shapes:
        raise ValueError("selected NPZ tensor shapes differ from the inventory")
    if any(record["dtype"] not in ("<f4", "=f4") for record in inventory_tensors.values()):
        raise ValueError("inventory records a non-FP32 team tensor")
    architecture = infer_architecture(shapes)
    model = load_fp32_model(weights_path, architecture, args.device)
    parquet = pq.ParquetFile(args.data)
    if parquet.num_row_groups != 1873:
        raise ValueError(
            f"expected 1873 complete validation sequences, got {parquet.num_row_groups}"
        )
    groups = select_row_groups(args.first_group, args.group_count, parquet.num_row_groups)
    chunks = iter_chunks(SEQUENCE_LENGTH, args.chunk_rows)
    args.out.mkdir(parents=True)
    prediction_path = args.out / "predictions.npy"
    predictions = np.lib.format.open_memmap(
        prediction_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(groups), SEQUENCE_LENGTH, N_OUTPUTS),
    )
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.monotonic()
    group_results: list[dict[str, object]] = []
    with (args.out / "progress.jsonl").open("x") as progress:
        for batch_start in range(0, len(groups), args.batch_sequences):
            batch_groups = groups[batch_start : batch_start + args.batch_sequences]
            began = time.monotonic()
            sequence_ids: list[int] = []
            feature_hashes: list[str] = []
            host_features: list[np.ndarray] = []
            for group in batch_groups:
                seq_ix, features = _read_group(parquet, group, FEATURE_COLUMNS)
                sequence_ids.append(seq_ix)
                feature_hashes.append(hashlib.sha256(memoryview(features).cast("B")).hexdigest())
                host_features.append(features)
            raw = torch.from_numpy(np.stack(host_features)).to(args.device, dtype=torch.float32)
            with torch.inference_mode():
                model_input = torch.cat((raw, event_features_torch(raw)), dim=-1)
                expected_shape = (len(batch_groups), SEQUENCE_LENGTH, N_MODEL_FEATURES)
                if model_input.shape != expected_shape:
                    raise ValueError(f"unexpected model input shape {tuple(model_input.shape)}")
                del raw
                state = None
                batch_predictions = np.empty(
                    (len(batch_groups), SEQUENCE_LENGTH, N_OUTPUTS), dtype=np.float32
                )
                for start, stop in chunks:
                    output, state = model(model_input[:, start:stop], h=state)
                    batch_predictions[:, start:stop] = output.detach().cpu().numpy()
            if not np.isfinite(batch_predictions).all():
                raise ValueError("member produced non-finite predictions")
            for batch_index, group in enumerate(batch_groups):
                pred = batch_predictions[batch_index]
                if pred.shape != (SEQUENCE_LENGTH, N_OUTPUTS):
                    raise ValueError(f"member produced invalid output shape: {pred.shape}")
                pred[:WARMUP] = 0.0
                output_index = batch_start + batch_index
                predictions[output_index] = pred
                group_results.append(
                    {
                        "row_group": group,
                        "seq_ix": sequence_ids[batch_index],
                        "feature_sha256": feature_hashes[batch_index],
                        "prediction_sha256": hashlib.sha256(memoryview(pred).cast("B")).hexdigest(),
                        "warmup_zeroed": WARMUP,
                        "elapsed_seconds": time.monotonic() - began,
                    }
                )
            del model_input, output, state, batch_predictions, host_features, features
            completed = batch_start + len(batch_groups)
            if completed % 16 < len(batch_groups) or completed == len(groups):
                _evict_output_pages(predictions, prediction_path)
                status = {
                    "completed": completed,
                    "total": len(groups),
                    "wall_seconds": time.monotonic() - started,
                    "utc": datetime.now(UTC).isoformat(),
                }
                progress.write(json.dumps(status) + "\n")
                progress.flush()
                print(json.dumps(status), flush=True)
    predictions.flush()
    del predictions
    result = {
        "status": "complete",
        "candidate": args.candidate,
        "member": args.member,
        "member_source": member_record,
        "command": sys.argv,
        "created_utc": datetime.now(UTC).isoformat(),
        "inventory_path": str(args.inventory.resolve()),
        "inventory_sha256": inventory_sha256,
        "archive_sha256": package_record["sha256"],
        "archive_hash_verified_by_this_run": True,
        "archive_path": str(archive_path.resolve()),
        "weight_path": str(weights_path.resolve()),
        "weight_sha256": weight_record["sha256"],
        "data_path": str(args.data.resolve()),
        "data_bytes": data_identity["bytes"],
        "data_sha256": data_sha256,
        "data_hash_verified_by_this_run": True,
        "exporter_sha256": digest(Path(__file__)),
        "row_groups": list(groups),
        "group_results": group_results,
        "architecture": asdict(architecture),
        "normalization": "member NPZ mean/std over raw112 + causal event99",
        "feature_path": "wnn.features.event_features_torch; row zero resets event state per sequence",
        "model_path": "wnn.nn.RNNModel with archived linear-head width; strict FP32 state load",
        "precision": (
            "FP32 CUDA; TF32 disabled; no autocast; stateful recurrent chunks; "
            "causal event features computed on each complete sequence"
        ),
        "batch_sequences": args.batch_sequences,
        "chunk_rows": args.chunk_rows,
        "chunk_count_per_sequence": len(chunks),
        "recurrent_chunking_note": (
            "GRU hidden state is carried across adjacent chunks; CUDA kernels may still produce "
            "small FP32 rounding differences from one unchunked full-sequence forward."
        ),
        "native_difference": native_runtime_note,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_shape": [len(groups), SEQUENCE_LENGTH, N_OUTPUTS],
        "prediction_dtype": "float32",
        "prediction_sha256": digest(prediction_path),
        "cuda_device": torch.cuda.get_device_name(args.device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "wall_seconds": time.monotonic() - started,
        "scope": "diagnostic FP32 member predictions on supplied validation; no grade or selection claim",
    }
    result_path = args.out / "RESULT.json"
    temp_path = result_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp_path.replace(result_path)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("candidate", "member", "prediction_shape", "wall_seconds")
            }
        )
    )


if __name__ == "__main__":
    main()
