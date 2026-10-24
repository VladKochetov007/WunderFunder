#!/usr/bin/env python3
"""Benchmark canonical F0 CPU batch assembly and optional bounded H2D movement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
SOURCE = Path("runtime/pr32-source-f1beeb4c")
COMMIT = "f1beeb4c210a9da01735e03b1ff6e27be878b873"
MANIFEST_SHA = "b598e8bcee941479713f979e1396f700a827241573eb2bcff06aa0fe787bb214"
PACKET = Path(
    "research/experiment_drafts/b300-transformer-distillation-runtime/large_teacher_r3_muon30"
)
BATCHES = (64, 96, 128)
CPU_SET = set(range(64, 72))
GIB = 1 << 30


def sha(path: Path) -> str:
    """Hash a file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    """Atomically save benchmark progress after each trial."""
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(name, path)


def verify_source(source: Path) -> None:
    """Verify the pinned f1beeb4c transport manifest and all 213 source members once."""
    manifest_path = source / "TRANSPORT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if sha(manifest_path) != MANIFEST_SHA or manifest.get("commit") != COMMIT:
        raise ValueError("pinned source manifest mismatch")
    files = manifest.get("files", {})
    if len(files) != 213:
        raise ValueError("source file count differs from pinned archive")
    for name, expected in files.items():
        rel = Path(name)
        path = source / rel
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or path.is_symlink()
            or not path.is_file()
            or sha(path) != expected
        ):
            raise ValueError(f"frozen source member mismatch: {name}")


def rss() -> tuple[int, int]:
    """Return current and peak resident memory in bytes."""
    current = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmRSS:")
    )
    return current, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def tensor_bytes(batch: dict) -> int:
    """Count canonical collated tensor payload bytes."""
    return sum(value.numel() * value.element_size() for value in batch.values())


def main() -> int:
    """Run three distinct real-F0 batches for each requested size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h2d", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 1800:
        parser.error("max-seconds must be in 1..1800")
    root = args.root.resolve(strict=True)
    source = root / SOURCE
    verify_source(source)
    if not CPU_SET.issubset(os.sched_getaffinity(0)):
        raise RuntimeError("launch with CPU affinity 64–71")
    os.sched_setaffinity(0, CPU_SET)
    os.environ.pop("CUBLAS_EMULATE_SINGLE_PRECISION", None)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("TMPDIR", str(root / "tmp/pr32-batch-loader-r1"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source / "src"))

    import torch

    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_data import build_data
    from wnn.training.sequence_fit_utils import (
        collate_sequences,
        move_and_validate_batch,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config(source / PACKET / "configs/t1280_f0.json")
    data_root, split = root / "data", root / "splits/fold0.json"
    fit = build_data(config, data_root, split).fit
    if len(fit) < sum(BATCHES) * 3:
        raise ValueError("not enough distinct real F0 fit members")
    out = args.output.resolve()
    if out.exists() or not out.parent.is_dir():
        raise FileExistsError("output must be fresh with an existing parent")
    out.parent.mkdir(parents=True, exist_ok=True)
    cuda = bool(args.h2d and torch.cuda.is_available())
    if cuda:
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        uuid = getattr(props, "uuid", "")
        h2d_device = {
            "name": props.name,
            "uuid": uuid.decode() if isinstance(uuid, bytes) else str(uuid),
            "total_memory_bytes": int(props.total_memory),
        }
    else:
        h2d_device = {
            "status": "skipped",
            "reason": "H2D not requested or CUDA unavailable",
        }
    result = {
        "schema": "wnn.pr32.cpu-batch-loader.v1",
        "status": "running",
        "source_commit": COMMIT,
        "manifest_sha256": MANIFEST_SHA,
        "split_sha256": sha(split),
        "fit_sequences": len(fit),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "h2d_device": h2d_device,
        "h2d_temporary_memory_limit_bytes": 2 * GIB,
        "trials": [],
        "dataset_stat": {
            p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in (
                data_root / "datasets/train.parquet",
                data_root / "datasets/valid.parquet",
            )
        },
        "engineering_only": True,
    }
    rng = random.Random(101001)
    chosen = rng.sample(range(len(fit)), sum(BATCHES) * 3)
    cursor, deadline = 0, time.monotonic() + args.max_seconds
    for size in BATCHES:
        for repeat in range(1, 4):
            if time.monotonic() >= deadline:
                result["status"] = "partial"
                result["error"] = "benchmark time limit reached"
                save(out, result)
                return 1
            indices = chosen[cursor : cursor + size]
            cursor += size
            before_rss, peak_before = rss()
            start = time.perf_counter()
            batch = collate_sequences(fit, indices)
            cpu_seconds = time.perf_counter() - start
            after_rss, peak_after = rss()
            nbytes = tensor_bytes(batch)
            trial = {
                "batch_size": size,
                "repeat": repeat,
                "distinct_real_sequences": len(indices),
                "indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
                "shapes": {key: list(value.shape) for key, value in batch.items()},
                "tensor_bytes": nbytes,
                "cpu_collate_seconds": cpu_seconds,
                "cpu_collate_sequences_per_second": size / cpu_seconds,
                "rss_before_bytes": before_rss,
                "rss_after_bytes": after_rss,
                "peak_rss_before_bytes": peak_before,
                "peak_rss_after_bytes": peak_after,
            }
            if cuda:
                free, _ = torch.cuda.mem_get_info(0)
                if nbytes + 256 * (1 << 20) > 2 * GIB or free < nbytes + 256 * (1 << 20):
                    trial["h2d"] = {
                        "status": "skipped",
                        "reason": "2 GiB temporary-memory gate",
                    }
                else:
                    torch.cuda.synchronize(0)
                    torch.cuda.reset_peak_memory_stats(0)
                    start = time.perf_counter()
                    moved = move_and_validate_batch(batch, torch.device("cuda:0"), 0)
                    torch.cuda.synchronize(0)
                    trial["h2d"] = {
                        "status": "complete",
                        "seconds": time.perf_counter() - start,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                        "bytes": tensor_bytes(moved),
                    }
                    del moved
                    torch.cuda.empty_cache()
            result["trials"].append(trial)
            result["status"] = "running"
            save(out, result)
            del batch
    result["status"] = "complete"
    result["finished_utc"] = time.time()
    save(out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
