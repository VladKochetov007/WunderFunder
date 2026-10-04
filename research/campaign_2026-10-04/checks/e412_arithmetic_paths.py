"""Isolate fixed E412 CPU activation, feature, and CUDA-reference differences."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import engine_fp16_check as engine  # noqa: E402
from probe_q8_model import torch_reference  # noqa: E402
from utils import FEATURE_COLUMNS  # noqa: E402
from wnn.features import event_features, event_features_torch  # noqa: E402
from wnn.gmetric import reference_global_wp  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = ROOT / "research/campaign_2026-10-04/checks"
    output = base / "e412_arithmetic_paths.json"
    archive_path = base / "e412_arithmetic_paths.npz"
    if output.exists() or archive_path.exists():
        raise FileExistsError("Preserve diagnostic outputs")
    package = ROOT / "submissions/c20261004_combo_s3_q16x/pkg"
    members = json.loads((package / "members.json").read_text())
    if len(members) != 1 or members[0]["coef"] != 1:
        raise ValueError("Expected the single merged-head predictor")
    weights = engine.load_npz(package / members[0]["weights"])
    native = engine.load_native(package / "v3")
    subset = pq.ParquetFile(ROOT / "cache/valid_subset20.parquet")
    with np.load(base / "e412_package_f32_v3.npz", allow_pickle=False) as stored:
        fast, targets, mask = stored["pred"], stored["y"], stored["mask"]
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as stored:
        position = {int(seq): i for i, seq in enumerate(stored["seq_ix"])}
    cuda_members = [np.load(ROOT / f"cache/pred_{name}_fp32_valid.npy", mmap_mode="r")
                    for name in ("c20261004_combo_s3_final_r1", "c20261004_combo_head_s3_final")]
    exact_predictions, cpu_predictions, cuda_predictions = [], [], []
    sequence_ids, feature_errors = [], []
    for group in range(subset.num_row_groups):
        table = subset.read_row_group(group, use_threads=False)
        sequence = int(table["seq_ix"][0].as_py())
        features = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(np.float32)
        need = np.arange(len(features)) >= 99
        predictor = native.EnsemblePredictor([weights], [np.float32(1)],
                                             fast_activations=False, precision="f32", budget_s=None)
        _, exact = engine.harness(predictor, sequence, features, need)
        cpu = torch_reference(weights, features)
        with torch.inference_mode():
            feature_error = np.abs(event_features(features) -
                                   event_features_torch(torch.from_numpy(features)[None])[0].numpy())
        feature_errors.append(float(feature_error.max()))
        exact_predictions.append(exact)
        cpu_predictions.append(cpu)
        cuda_predictions.append(sum(np.asarray(values[position[sequence]], np.float64)
                                    for values in cuda_members) / 2)
        sequence_ids.append(sequence)
        print(json.dumps({"sequence": sequence, "completed": group + 1,
                          "exact_native_vs_cpu_max": float(np.abs(exact[need] - cpu[need]).max()),
                          "torch_numpy_event_features_max": feature_errors[-1]}), flush=True)
    predictions = {"native_fast": fast, "native_exact": np.stack(exact_predictions),
                   "torch_cpu": np.stack(cpu_predictions), "torch_cuda": np.stack(cuda_predictions)}
    required = np.broadcast_to(np.arange(20000) >= 99, mask.shape)
    scores = {name: reference_global_wp(targets, values, mask) for name, values in predictions.items()}
    comparisons = {}
    for left, right in (("native_exact", "torch_cpu"), ("native_fast", "native_exact"),
                        ("torch_cpu", "torch_cuda"), ("native_exact", "torch_cuda")):
        error = np.abs(predictions[left] - predictions[right])
        comparisons[f"{left}_minus_{right}"] = {
            "max_required_abs": float(error[required].max()),
            "max_scored_abs": float(error[mask].max()),
            "p999_required_abs": float(np.quantile(error[required], .999)),
            "wp_delta": scores[left] - scores[right],
        }
    np.savez(archive_path, **predictions, sequence_ids=sequence_ids)
    result = {"scope": "Frozen20-stream arithmetic diagnosis; no model or threshold selection",
              "sequence_ids": sequence_ids, "torch_numpy_event_features_max": max(feature_errors),
              "scores": scores, "comparisons": comparisons,
              "torch_cpu_mkldnn_enabled": torch.backends.mkldnn.enabled,
              "native_fast_activations_original": True, "native_exact_activations": False}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
