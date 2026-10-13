from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import mmap
import os
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    DataPoint,
    GlobalAccumulator,
    PredictionModel,
    validate_sequence,
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_model(name: str, mode: str, member: int | None) -> PredictionModel:
    inventory = json.loads((HERE / "INVENTORY.json").read_text())["packages"]
    source = inventory[name]
    package = Path(source["package"])
    if digest(Path(source["archive"])) != source["sha256"]:
        raise ValueError("archive changed")
    for filename, record in source["files"].items():
        if digest(package / filename) != record["sha256"]:
            raise ValueError(f"extracted source changed: {filename}")
    if mode == "original":
        if member is not None:
            raise ValueError("individual original member is not an exact archive callback")
        spec = importlib.util.spec_from_file_location("team_solution", package / "solution.py")
        if spec is None or spec.loader is None:
            raise ImportError("missing solution")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.PredictionModel()
    if (package / "classes.json").exists():
        raise ValueError("class-gated package needs its original callback")
    native = Path(inventory["E"]["package"]) / "v3/wnn_native.abi3.so"
    if digest(native) != inventory["E"]["files"]["v3/wnn_native.abi3.so"]["sha256"]:
        raise ValueError("native library changed")
    sys.path.insert(0, str(native.parent))
    import wnn_native

    selected = source["members"] if member is None else [source["members"][member]]
    weights = []
    for item in selected:
        with np.load(package / item["weights"], allow_pickle=False) as arrays:
            weights.append(
                {k: np.ascontiguousarray(arrays[k], dtype=np.float32) for k in arrays.files}
            )
    coefficients = [np.float32(item["coef"] if member is None else 1.0) for item in selected]
    return wnn_native.EnsemblePredictor(
        weights, coefficients, fast_activations=True, precision="q8vnni", budget_s=None
    )


def replay(
    model: PredictionModel, sequence: int, features: np.ndarray, need: np.ndarray
) -> np.ndarray:
    predictions = np.zeros((len(features), 2), dtype=np.float32)
    for step, row in enumerate(features):
        value = model.predict(DataPoint(sequence, step, bool(need[step]), row))
        if not need[step]:
            if value is not None:
                raise ValueError("warmup returned prediction")
        else:
            if np.shape(value) != (2,):
                raise ValueError("wrong prediction shape")
            predictions[step] = value
    if not np.isfinite(predictions).all():
        raise ValueError("nonfinite predictions")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded exact team archive replay")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--mode", choices=("original", "rust"), default="original")
    parser.add_argument("--member", type=int)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reset-check", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("original C++ artifacts require Python3.11")
    if args.out.exists():
        raise FileExistsError(args.out)
    inventory = json.loads((HERE / "INVENTORY.json").read_text())
    parquet_path = ROOT / "datasets/valid.parquet"
    if parquet_path.stat().st_size != inventory["data"]["bytes"]:
        raise ValueError("data file size changed")
    parquet = pq.ParquetFile(parquet_path)
    if args.first < 0 or args.count < 1 or args.first + args.count > parquet.num_row_groups:
        raise ValueError("invalid row-group selection")
    args.out.mkdir(parents=True)
    began = time.monotonic()
    model = load_model(args.candidate, args.mode, args.member)
    predictions_path = args.out / "predictions.npy"
    predictions = np.lib.format.open_memmap(
        predictions_path, mode="w+", dtype=np.float32, shape=(args.count, 20000, 2)
    )
    total = GlobalAccumulator()
    entries = []
    callback_seconds = 0.0
    first_fixture = None
    with (args.out / "progress.jsonl").open("x") as log:
        for index, group in enumerate(range(args.first, args.first + args.count)):
            table = parquet.read_row_group(group, use_threads=False)
            sequence, need = validate_sequence(table)
            features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(
                np.float32
            )
            targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(
                np.float32
            )
            mask = table["is_scored"].to_numpy().astype(bool) & need
            start = time.monotonic()
            pred = replay(model, sequence, features, need)
            elapsed = time.monotonic() - start
            callback_seconds += elapsed
            if index == 0 and args.reset_check:
                first_fixture = (sequence, features.copy(), need.copy(), pred.copy())
            predictions[index] = pred
            score = GlobalAccumulator()
            score.add(targets, pred, mask)
            total.add(targets, pred, mask)
            entry = {
                "row_group": group,
                "seq_ix": sequence,
                "selected_rows": int(mask.sum()),
                "score": score.result(),
                "moments": score._moments.tolist(),
                "callback_seconds": elapsed,
                "input_sha256": array_hash(features),
                "target_sha256": array_hash(targets),
                "mask_sha256": array_hash(mask),
                "prediction_sha256": array_hash(pred),
            }
            entries.append(entry)
            if (index + 1) % 16 == 0 or index + 1 == args.count:
                predictions.flush()
                predictions._mmap.madvise(mmap.MADV_DONTNEED)
                with predictions_path.open("rb") as stream:
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                status = {
                    "completed": index + 1,
                    "total": args.count,
                    "wall_seconds": time.monotonic() - began,
                    "score": total.result(),
                    "utc": datetime.now(UTC).isoformat(),
                }
                log.write(json.dumps(status) + "\n")
                log.flush()
                print(json.dumps(status), flush=True)
    reset = None
    if first_fixture is not None:
        sequence, features, need, expected = first_fixture
        # Force a different sequence before replay, including a one-group profile.
        replay(model, -987654321, features[:99], need[:99])
        repeated = replay(model, sequence, features, need)
        reset = bool(np.array_equal(repeated, expected))
        if not reset:
            raise ValueError("reset replay differs")
    predictions.flush()
    del predictions
    result = {
        "candidate": args.candidate,
        "mode": args.mode,
        "member": args.member,
        "command": sys.argv,
        "python": sys.version,
        "source_sha256": digest(Path(__file__)),
        "inventory_sha256": digest(HERE / "INVENTORY.json"),
        "archive_sha256": inventory["packages"][args.candidate]["sha256"],
        "data": inventory["data"],
        "created_utc": datetime.now(UTC).isoformat(),
        "groups": entries,
        "score": total.result(),
        "moments": total._moments.tolist(),
        "callback_seconds": callback_seconds,
        "wall_seconds": time.monotonic() - began,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "reset_replay_bitwise": reset,
        "prediction_sha256": digest(predictions_path),
        "scope": "supplied validation: final fitted or unknown-overlap diagnostics; not clean CV",
    }
    (args.out / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("candidate", "mode", "score", "wall_seconds", "max_rss_kib")}
        )
    )


if __name__ == "__main__":
    main()
