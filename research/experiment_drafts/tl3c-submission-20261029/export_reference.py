"""Export the actual admitted Torch chunk4000 predictor on a separately chosen receiver.

No training, optimizer, GPU allocation, process management or portal action occurs.
This is a receiver command; never run it on the preparation Mac.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

from candidate_contract import load_manifest  # noqa: E402
from reference_source import load_reference_models, verify_training_root  # noqa: E402
from validation_io import SPLIT_SHA, read_json, selected_groups, sha256  # noqa: E402


def arguments() -> argparse.Namespace:
    """Require an explicit data scope and receiver device, never choose a free GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("training-root", "manifest", "receipt", "data", "output-dir"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--max-sequences", type=int)
    scope.add_argument("--full-valid", action="store_true")
    return parser.parse_args()


def export(args: argparse.Namespace, output: Path) -> dict:
    """Preserve complete-sequence features, 4000-row encoder resets and recurrent carry."""
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("reference export belongs on the Linux Python 3.11 receiver")
    manifest, split = load_manifest(args.manifest)
    receipt = read_json(args.receipt)
    if receipt.get("schema") != "tl3c-submission-candidate-v1" or receipt.get(
        "input_manifest", {}
    ).get("sha256") != sha256(args.manifest):
        raise ValueError("build receipt does not bind this input manifest")
    packaged = receipt["members"]
    names = [item["run"] for item in packaged]
    by_name = {record["run"]: record for record in manifest["members"]}
    if len(set(names)) != len(names) or not names or any(name not in by_name for name in names):
        raise ValueError("build receipt member inventory differs from the frozen manifest")
    records = [by_name[name] for name in names]
    training_root = args.training_root.resolve(strict=True)
    source = verify_training_root(training_root, records, split)
    data_sha = sha256(args.data)
    if data_sha != args.data_sha256:
        raise ValueError("reference parquet SHA256 mismatch")
    split_path = Path(manifest["split_manifest"]["path"])
    groups = selected_groups(split_path, args.max_sequences, args.full_valid)
    signature = [
        {
            "run": item["run"],
            "coef": item["coef"],
            "family": item["family"],
            "checkpoint_sha256": item["checkpoint"]["sha256"],
        }
        for item in packaged
    ]
    with tempfile.TemporaryDirectory(prefix="tl3c-reference-weights-") as temporary:
        models, device, runtime = load_reference_models(
            training_root, records, split, packaged, Path(temporary), args.device
        )
        import numpy as np
        import pyarrow.parquet as pq
        import torch
        from validation_runtime import load_sequence

        from utils import SEQUENCE_LENGTH, GlobalAccumulator
        from wnn.training.evaluation import predict

        parquet = pq.ParquetFile(args.data)
        if max(groups) >= parquet.num_row_groups:
            raise ValueError("split references missing validation row groups")
        path = output / "predictions.npy"
        predictions = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(len(groups), SEQUENCE_LENGTH, 2)
        )
        metric = GlobalAccumulator()
        sequence_ids = []
        for index, group in enumerate(groups):
            stream = load_sequence(parquet, group)
            if stream["seq_ix"] in sequence_ids:
                raise ValueError("duplicate validation sequence ID")
            sequence_ids.append(stream["seq_ix"])
            raw = torch.from_numpy(stream["features"])[None]
            combined = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
            for model, environment, coefficient in models:
                # Canonical predict builds event99/lead52 once over the whole sequence,
                # then calls the model in 4000-row chunks while carrying GRU state.
                value = predict(model, raw, 4000, device, environment=environment)[0]
                combined += coefficient * value
            if not np.isfinite(combined).all():
                raise ValueError("nonfinite canonical reference")
            metric.add(stream["targets"], combined, stream["score"])
            combined[~stream["need"]] = np.nan
            predictions[index] = combined
        predictions.flush()
        del predictions
    result = metric.result()
    if args.full_valid and result["selected_rows"] != 3_528_273:
        raise ValueError("full official valid requires exactly 3528273 scored rows")
    return {
        "schema": "tl3c-torch-reference-v1",
        "reference_kind": "torch_chunk4000",
        "data_sha256": data_sha,
        "split_sha256": SPLIT_SHA,
        "row_groups": groups,
        "sequence_ids": sequence_ids,
        "member_signature": signature,
        "predictions": {"path": path.name, "sha256": sha256(path)},
        "metrics": result,
        "build_receipt_sha256": sha256(args.receipt),
        "source": source,
        "runtime": runtime,
        "chunk_rows": 4000,
        "score_scope": "full_official_valid_exposed"
        if args.full_valid
        else "profile_subset_not_primary",
        "semantics": "admitted canonical predict; whole-sequence features; chunk-local tiedlead deltas; carried GRU state",
        "submission_approved": False,
    }


def main() -> None:
    """Create a reference bundle exactly once and preserve explicit failure evidence."""
    args = arguments()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = export(args, output)
    except Exception as error:
        (output / "error.json").write_text(
            json.dumps({"error": f"{type(error).__name__}: {error}"}, indent=2) + "\n"
        )
        raise
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
