"""Compare fixed E412 native predictions with common FP32 Torch exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.gmetric import reference_global_wp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path.cwd()
    base = root / "research/campaign_2026-10-04/checks"
    output = base / "e412_fp32_reference.json"
    if output.exists():
        raise FileExistsError(output)
    files: dict[str, str] = {}
    native = {}
    for name in ("q16x_v3", "f32_v3"):
        path = base / f"e412_package_{name}.npz"
        with np.load(path, allow_pickle=False) as archive:
            native[name] = {key: archive[key] for key in archive.files}
        files[str(path.relative_to(root))] = sha256(path)
    targets = root / "cache/valid_targets.npz"
    with np.load(targets, allow_pickle=False) as archive:
        sequence_ids = archive["seq_ix"]
    positions = {int(seq): i for i, seq in enumerate(sequence_ids)}
    subset = pq.ParquetFile(root / "cache/valid_subset20.parquet")
    subset_ids = [int(subset.read_row_group(i, columns=["seq_ix"])["seq_ix"][0].as_py())
                  for i in range(subset.num_row_groups)]
    floating = native["f32_v3"]
    for key in ("y", "mask"):
        if not np.array_equal(floating[key], native["q16x_v3"][key]):
            raise ValueError(f"Native {key} differs")
    required = np.broadcast_to(np.arange(20000) >= 99, floating["mask"].shape)
    torch_predictions = {}
    for precision in ("fp16", "fp32"):
        selected = []
        for run, epoch in (("c20261004_combo_s3_final_r1", 3),
                           ("c20261004_combo_head_s3_final", 0)):
            suffix = "_fp32" if precision == "fp32" else ""
            path = root / f"cache/pred_{run}{suffix}_valid.npy"
            sidecar = path.with_suffix(".json")
            metadata = json.loads(sidecar.read_text())
            prediction_hash = sha256(path)
            if metadata["prediction_sha256"] != prediction_hash:
                raise ValueError(f"Prediction hash mismatch: {path}")
            if metadata["run"] != run or metadata["epoch"] != epoch:
                raise ValueError("Wrong fixed checkpoint")
            if metadata.get("inference_precision", "fp16") != precision:
                raise ValueError("Wrong inference precision")
            if metadata["row_groups"] != list(range(len(sequence_ids))):
                raise ValueError("Unexpected prediction row order")
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            selected.append(np.stack([np.asarray(values[positions[seq]], np.float64)
                                      for seq in subset_ids]))
            files[str(path.relative_to(root))] = prediction_hash
            files[str(sidecar.relative_to(root))] = sha256(sidecar)
        torch_predictions[precision] = (selected[0] + selected[1]) / 2
    predictions = {**{key: value["pred"] for key, value in native.items()},
                   **{f"torch_{key}": value for key, value in torch_predictions.items()}}
    scores = {name: reference_global_wp(floating["y"], value, floating["mask"])
              for name, value in predictions.items()}
    comparisons = {}
    for left, right in (("f32_v3", "torch_fp32"), ("q16x_v3", "torch_fp32"),
                        ("torch_fp16", "torch_fp32"), ("q16x_v3", "f32_v3")):
        difference = np.abs(predictions[left] - predictions[right])
        comparisons[f"{left}_minus_{right}"] = {
            "max_required_abs": float(difference[required].max()),
            "p999_required_abs": float(np.quantile(difference[required], .999)),
            "max_scored_abs": float(difference[floating["mask"]].max()),
            "wp_delta": scores[left] - scores[right],
        }
    result = {"scope": "20 existing full streams; precision diagnosis only; no checkpoint selection",
              "sequence_ids": subset_ids, "files": files, "scores": scores,
              "comparisons": comparisons,
              "original_q16x_gate_unchanged": {"max_abs": 3e-4, "wp_delta": 5e-6}}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"scores": scores, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
