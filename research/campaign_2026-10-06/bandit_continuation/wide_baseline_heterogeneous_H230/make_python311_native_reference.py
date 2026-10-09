"""Create a small native-output reference for H230's offline ZIP callback gate.

This helper reads exactly two complete row groups and writes predictions only.
It never computes a score or uses target columns.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--level", choices=("v2", "avx2", "v3"), default="v3")
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("run this helper inside the offline Python 3.11 image")
    if os.environ.get("LD_LIBRARY_PATH") or os.environ.get("PYTHONPATH"):
        raise RuntimeError("reference generation requires clean library and Python paths")

    root = args.project_root.resolve()
    zip_path = (root / args.zip).resolve()
    data_path = (root / args.data).resolve()
    reference_path = (root / args.reference).resolve()
    report_path = (root / args.report).resolve()
    if not zip_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("candidate ZIP or two-sequence data file is missing")

    sys.path.insert(0, str(root))
    from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, DataPoint  # noqa: E402

    parquet = pq.ParquetFile(data_path)
    if parquet.num_row_groups < 2:
        raise ValueError("reference generation needs two complete row groups")
    streams: list[tuple[int, np.ndarray, np.ndarray]] = []
    for group in range(2):
        table = parquet.read_row_group(
            group, columns=["seq_ix", "need_prediction", *FEATURE_COLUMNS], use_threads=False
        )
        if table.num_rows != SEQUENCE_LENGTH:
            raise ValueError(f"row group {group} is not one complete sequence")
        features = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(
            np.float32
        )
        streams.append(
            (int(table["seq_ix"][0].as_py()), features, table["need_prediction"].to_numpy())
        )

    os.environ["WNN_NATIVE_LEVEL"] = args.level
    predictions: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="h230-native-reference-") as temporary:
        package = Path(temporary)
        with zipfile.ZipFile(zip_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("candidate ZIP failed its CRC check")
            for member in archive.namelist():
                path = Path(member)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe ZIP member: {member}")
            archive.extractall(package)
        spec = importlib.util.spec_from_file_location("h230_checked_solution", package / "solution.py")
        if spec is None or spec.loader is None:
            raise ImportError("candidate ZIP has no loadable solution.py")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(package))
        try:
            spec.loader.exec_module(module)
            model = module.PredictionModel()
            for sequence, features, need in streams:
                output = np.zeros((len(features), 2), dtype=np.float32)
                for step, row in enumerate(features):
                    value = model.predict(DataPoint(sequence, step, bool(need[step]), row))
                    if not need[step]:
                        if value is not None:
                            raise ValueError(f"unexpected warmup prediction at row {step}")
                        continue
                    value = np.asarray(value, dtype=np.float32)
                    if value.shape != (2,) or not np.isfinite(value).all():
                        raise ValueError(f"invalid required prediction at row {step}")
                    output[step] = value
                predictions.append(output)
        finally:
            sys.path.remove(str(package))
            sys.modules.pop("h230_checked_solution", None)

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(reference_path, pred_0=predictions[0], pred_1=predictions[1])
    report = {
        "status": "passed",
        "python": sys.version,
        "native_level": args.level,
        "zip": str(zip_path.relative_to(root)),
        "zip_sha256": sha256_file(zip_path),
        "data": str(data_path.relative_to(root)),
        "data_sha256": sha256_file(data_path),
        "reference": str(reference_path.relative_to(root)),
        "reference_sha256": sha256_file(reference_path),
        "sequence_ids": [stream[0] for stream in streams],
        "rows": sum(len(stream[1]) for stream in streams),
        "score_computed": False,
        "scope": "candidate ZIP v3 native callbacks on exactly two complete sequences; prediction-only reference for cross-level offline package parity",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
