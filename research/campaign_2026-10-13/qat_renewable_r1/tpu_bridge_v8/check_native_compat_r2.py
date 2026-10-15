"""Check the actual TPU-updated EMA export in the frozen native CPU predictor."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wnn.data import SequenceFile


def main() -> None:
    """Replay the declared fitting sequence and record cross-backend errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    packet = root / "research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8"
    output = packet / "NATIVE_COMPATIBILITY_R2.json"
    if output.exists():
        raise FileExistsError(output)
    collected = root / "artifacts/kaggle_tpu_v8_terminal_20261015_r1/cloud_preflight_tpu_v8"
    receipt = json.loads((collected / "realdata_update_receipt.json").read_text())
    model_path = collected / "realdata_reload/ema_after_one_chunk_fp32.npz"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != receipt["checkpoint"]["ema_sha256"]:
        raise ValueError("updated EMA checkpoint differs from the actual-data receipt")
    with np.load(model_path, allow_pickle=False) as archive:
        weights = {key: np.ascontiguousarray(archive[key], dtype=np.float32) for key in archive.files}
    if any(not np.isfinite(value).all() for value in weights.values()):
        raise ValueError("nonfinite updated EMA tensors")
    weights["head.2.weight"] = np.ascontiguousarray(weights["head.2.weight"][:2])
    weights["head.2.bias"] = np.ascontiguousarray(weights["head.2.bias"][:2])
    prediction_path = collected / "realdata_reload/updated_ema_valid_fit_group_prediction.npz"
    if hashlib.sha256(prediction_path.read_bytes()).hexdigest() != receipt["prediction_export"]["sha256"]:
        raise ValueError("TPU prediction export changed")
    with np.load(prediction_path, allow_pickle=False) as archive:
        reference = np.array(archive["predictions"], dtype=np.float32)
        group = int(archive["row_group"])
    if group != 254 or reference.shape != (1, 20_000, 23):
        raise ValueError("unexpected actual fitting-group prediction export")
    reference = reference[0]
    sequence = SequenceFile(root / "datasets/valid.parquet").read(group, targets=False, mask=False)
    if sequence.features.shape != (20_000, 112):
        raise ValueError("source sequence is not a complete raw trajectory")
    library = root / "artifacts/native_qat_r1"
    binary = library / "wnn_native.abi3.so"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != (
        "a4390ecf7bcd54bb6826b43f5dcf6d7a40064c64ed1450bdc1780c34c2a6e108"
    ):
        raise ValueError("native library differs from the frozen implementation")
    sys.path.insert(0, str(library))
    native = importlib.import_module("wnn_native")
    if Path(native.__file__).resolve() != binary.resolve():
        raise ValueError("native module resolved outside the pinned binary")
    predictor = native.Predictor(weights, precision="f32", fast_activations=False)
    if predictor.precision != "f32":
        raise ValueError("native predictor did not select FP32")
    actual = np.stack([predictor.step(row) for row in sequence.features])
    predictor.reset()
    repeated = np.stack([predictor.step(row) for row in sequence.features])
    finite = bool(np.isfinite(actual).all())
    reset_exact = bool(np.array_equal(actual, repeated))
    difference = np.abs(actual - reference[:, :2])
    close = bool(np.allclose(actual, reference[:, :2], rtol=1e-4, atol=1e-4))
    result = {"utc": datetime.now(UTC).isoformat(), "source_qualified_group": ["valid", group],
              "seq_ix": int(sequence.seq_ix), "rows": 20_000, "compared_outputs": 2,
              "native_precision": predictor.precision, "fast_activations": False,
              "all_outputs_finite": finite, "native_reset_replay_bitwise": reset_exact,
              "native_vs_tpu_max_abs": float(difference.max()),
              "native_vs_tpu_mean_abs": float(difference.mean()),
              "cross_backend_tolerance": {"rtol": 1e-4, "atol": 1e-4},
              "cross_backend_close": close,
              "status": "passed" if finite and reset_exact and close else "needs_diagnosis",
              "interpretation": "one fitting trajectory, updated EMA; no score or full-fit claim",
              "checkpoint_sha256": receipt["checkpoint"]["ema_sha256"],
              "native_library_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
