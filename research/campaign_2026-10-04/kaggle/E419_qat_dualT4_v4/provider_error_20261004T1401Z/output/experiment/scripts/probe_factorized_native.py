"""Check full-sequence native factor inference against its effective dense control."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_rnn import make_model  # noqa: E402

from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS, GlobalAccumulator  # noqa: E402
from wnn.features import StreamingEventFeatures, event_features  # noqa: E402
from wnn.splits import file_sha256  # noqa: E402

# Torch checkpoints and Arrow tables expose heterogeneous runtime schemas.
FACTOR_NAMES = tuple(f"gru.weight_{kind}_l{layer}" for layer in range(2) for kind in ("ih", "hh"))
MAX_ABS_LIMIT = 3e-4
MAX_WP_DELTA = 5e-6
EXPECTED_SOURCE_SHA256 = "b5181f3aa948765f22a8c3f23f6e293ba715b52ffb91eb6eac02d6b7e30a8b8d"
EXPECTED_E409_MANIFEST_SHA256 = "815bee9144348fa58c6cd622e0ba2ff48c94b3722228d2b19fafd6e227552464"


def load_native(library: Path) -> Any:
    """Load the built extension package from one scorer-level directory."""
    library = library.resolve()
    extension = library / "wnn_native.abi3.so"
    spec = importlib.util.spec_from_file_location("wnn_native", extension)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native extension at {extension}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve().parent != library:
        raise RuntimeError(f"loaded {module.__file__}, expected package in {library}")
    return module


def read_streams(data_path: Path, groups: list[int]) -> list[dict[str, Any]]:
    """Read the E409 row groups and build the trainer's 211-column causal inputs."""
    parquet = pq.ParquetFile(data_path)
    streams: list[dict[str, Any]] = []
    for group in groups:
        table = parquet.read_row_group(group, use_threads=False)
        raw = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS])
        raw = np.ascontiguousarray(raw, dtype=np.float32)
        if len(raw) != SEQUENCE_LENGTH:
            raise ValueError(f"E409 row group {group} has {len(raw)} rows")
        event = event_features(raw)
        streaming = StreamingEventFeatures()
        event_rows = np.stack([streaming.update(row).copy() for row in raw])
        if not np.array_equal(event.view(np.uint32), event_rows.view(np.uint32)):
            raise ValueError(f"batched and streaming event features differ in group {group}")
        model_input = np.ascontiguousarray(np.concatenate((raw, event), axis=1))
        targets = np.column_stack([table[name].to_numpy() for name in TARGET_COLUMNS])
        targets = np.ascontiguousarray(targets, dtype=np.float32)
        need = np.asarray(table["need_prediction"].to_numpy(), dtype=bool)
        scored = np.asarray(table["is_scored"].to_numpy(), dtype=bool) & need
        seq_ix = int(table["seq_ix"][0].as_py())
        streams.append(
            {
                "group": group,
                "seq_ix": seq_ix,
                "input": model_input,
                "targets": targets,
                "mask": scored,
            }
        )
    return streams


def arrays_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, np.ndarray]:
    """Copy inference tensors to contiguous FP32 arrays for the native bindings."""
    arrays: dict[str, np.ndarray] = {}
    for name, value in checkpoint["model"].items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"expected FP32 tensor in inference state: {name}")
        arrays[name] = np.ascontiguousarray(value.detach().cpu().numpy())
    return arrays


def factor_arrays(checkpoint: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Extract saved factors without refactoring or changing their FP32 bytes."""
    metadata = checkpoint.get("factorization")
    if not isinstance(metadata, dict) or metadata.get("rank") != 128:
        raise ValueError("expected the admitted rank-128 factorized control")
    if metadata.get("training_updates") != 0:
        raise ValueError("parity source must be the zero-update SVD export")
    factors = metadata.get("factors")
    if not isinstance(factors, dict) or set(factors) != set(FACTOR_NAMES):
        raise ValueError("factor metadata must contain exactly the four GRU matrices")
    result: dict[str, dict[str, np.ndarray]] = {}
    for name in FACTOR_NAMES:
        pair = factors[name]
        left = pair["left"].detach().cpu().numpy()
        right = pair["right"].detach().cpu().numpy()
        if left.dtype != np.float32 or right.dtype != np.float32:
            raise ValueError(f"factor dtype changed for {name}")
        if left.shape != (1152, 128) or right.shape != (128, 384):
            raise ValueError(f"factor dimensions changed for {name}")
        result[name] = {
            "left": np.ascontiguousarray(left),
            "right": np.ascontiguousarray(right),
        }
    return result


def torch_outputs(checkpoint: dict[str, Any], streams: list[dict[str, Any]]) -> list[np.ndarray]:
    """Run the full sequence through the dense effective matrices in the control."""
    model = make_model(
        checkpoint["args"], checkpoint["n_in"], checkpoint["model"]["head.2.bias"].numel()
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    results = []
    with torch.inference_mode():
        for stream in streams:
            inputs = torch.from_numpy(stream["input"])[None]
            output, _state = model(inputs, None)
            results.append(np.ascontiguousarray(output[0].cpu().numpy()))
    return results


def native_outputs(
    library: Path,
    weights: dict[str, np.ndarray],
    factors: dict[str, dict[str, np.ndarray]],
    streams: list[dict[str, Any]],
    precision: str,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Run one portable or SIMD build with reset and exact factor round-trip checks."""
    native = load_native(library)
    model = native.ModelStepper(weights, precision=precision, factors=factors)
    exported = model.factorization_tensors()
    if set(exported) != set(factors):
        raise ValueError("native factor names differ from the saved export")
    for name, pair in factors.items():
        for side, expected in pair.items():
            actual = exported[name][side]
            if actual.dtype != np.float32 or actual.tobytes() != expected.tobytes():
                raise ValueError(f"native factor bytes changed for {name}.{side}")
    storage = dict(model.storage_bytes)
    expected_factor_bytes = sum(
        value.nbytes for pair in factors.values() for value in pair.values()
    )
    if storage["factor_weights"] != expected_factor_bytes:
        raise ValueError("factor byte accounting differs from the saved factors")
    if storage["factor_weights"] > storage["total_streamed_weights"]:
        raise ValueError("factor bytes must be a subset of total streamed weights")
    if storage["import_only_dense_recurrent"] != 4 * 3 * 384 * 384 * 4:
        raise ValueError("import-only dense recurrent byte count differs from W384")
    if storage["total_streamed_weights"] != model.weight_bytes:
        raise ValueError("total streamed byte field does not match native weight_bytes")
    if not {"recurrent_state", "scratch", "import_only_dense_recurrent"} <= storage.keys():
        raise ValueError("native storage report is missing a separate category")

    outputs = []
    for stream in streams:
        model.reset()
        predicted = np.ascontiguousarray(model.run(stream["input"]))
        if predicted.shape[0] != SEQUENCE_LENGTH or predicted.shape[1] < 2:
            raise ValueError("native output does not cover the full stream and both targets")
        model.reset()
        replay = np.ascontiguousarray(model.run(stream["input"]))
        if not np.array_equal(predicted.view(np.uint32), replay.view(np.uint32)):
            raise ValueError("reset/replay changed native outputs")
        outputs.append(predicted)
    return outputs, storage


def score(predicted: list[np.ndarray], streams: list[dict[str, Any]]) -> float:
    """Return pooled WP over the fixed E409 score masks."""
    accumulator = GlobalAccumulator()
    for output, stream in zip(predicted, streams, strict=True):
        accumulator.add(stream["targets"], output[:, :2], stream["mask"])
    return float(accumulator.result()["wp"])


def metrics(
    predicted: list[np.ndarray], references: list[np.ndarray], streams: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute the declared all-row error and pooled scored-row WP diagnostic."""
    maximum = 0.0
    squared = 0.0
    count = 0
    reference_score = GlobalAccumulator()
    predicted_score = GlobalAccumulator()
    for actual, reference, stream in zip(predicted, references, streams, strict=True):
        delta = actual[:, :2] - reference[:, :2]
        maximum = max(maximum, float(np.abs(delta).max()))
        squared += float(np.square(delta.astype(np.float64)).sum())
        count += delta.size
        reference_score.add(stream["targets"], reference[:, :2], stream["mask"])
        predicted_score.add(stream["targets"], actual[:, :2], stream["mask"])
    reference_wp = float(reference_score.result()["wp"])
    predicted_wp = float(predicted_score.result()["wp"])
    return {
        "max_abs_first_two_all_rows": maximum,
        "rms_first_two_all_rows": float(np.sqrt(squared / count)),
        "reference_global_wp": reference_wp,
        "native_global_wp": predicted_wp,
        "abs_global_wp_delta": abs(predicted_wp - reference_wp),
        "passed": maximum <= MAX_ABS_LIMIT and abs(predicted_wp - reference_wp) <= MAX_WP_DELTA,
    }


def run_parity(args: argparse.Namespace) -> None:
    """Validate a previously admitted control against three hash-bound E409 streams."""
    torch.set_num_threads(1)
    manifest = json.loads(args.e409_manifest.read_text())
    if manifest.get("experiment") != "E409":
        raise ValueError("expected the E409 runtime manifest")
    expected_data_hash = manifest["data"]["sha256"]
    if file_sha256(args.e409_manifest) != EXPECTED_E409_MANIFEST_SHA256:
        raise ValueError("E409 manifest hash differs from the frozen source")
    if str(args.data) != manifest["data"]["path"]:
        raise ValueError("data path differs from the E409 hash-bound path")
    if file_sha256(args.data) != expected_data_hash:
        raise ValueError("E409 data hash changed")
    groups = manifest["row_groups"]
    if len(groups) != 3:
        raise ValueError("E420 requires exactly the three E409 20k-row streams")
    source_hash = file_sha256(args.source)
    control = torch.load(args.control, map_location="cpu", weights_only=False)
    metadata = control.get("factorization", {})
    if metadata.get("source_checkpoint_sha256") != source_hash:
        raise ValueError("zero-update export source hash differs from requested parent")
    if metadata.get("training_updates") != 0:
        raise ValueError("zero-update control records a nonzero update count")
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError("W384 source checkpoint hash differs from the sealed value")

    weights = arrays_from_checkpoint(control)
    factors = factor_arrays(control)
    effective_errors = {
        name: float(np.max(np.abs(pair["left"] @ pair["right"] - weights[name])))
        for name, pair in factors.items()
    }
    if max(effective_errors.values()) > 1e-6:
        raise ValueError("saved dense recurrent matrices differ from their left@right products")
    streams = read_streams(args.data, groups)
    references = torch_outputs(control, streams)
    original = torch.load(args.source, map_location="cpu", weights_only=False)
    original_outputs = torch_outputs(original, streams)
    report: dict[str, Any] = {
        "experiment": "E420",
        "purpose": "native full-stream representation parity; no timing or model selection",
        "source_checkpoint": str(args.source),
        "source_checkpoint_sha256": source_hash,
        "zero_update_control": str(args.control),
        "zero_update_control_sha256": file_sha256(args.control),
        "e409_manifest_sha256": file_sha256(args.e409_manifest),
        "source_closure_sha256": file_sha256(
            ROOT / "research/campaign_2026-10-04/E420_factorized/source_closure.sha256"
        ),
        "e409_data_sha256": expected_data_hash,
        "row_groups": groups,
        "sequence_ids": [stream["seq_ix"] for stream in streams],
        "gate": {"max_abs": MAX_ABS_LIMIT, "abs_global_wp_delta": MAX_WP_DELTA},
        "factor_shapes": {
            name: {side: list(value.shape) for side, value in pair.items()}
            for name, pair in factors.items()
        },
        "effective_dense_max_abs_vs_saved_products": effective_errors,
        "cases": {},
    }
    outputs_by_level: dict[str, list[np.ndarray]] = {}
    for level in ("v2", "v3"):
        library = args.library_root / level / "wnn_native"
        fast, storage = native_outputs(library, weights, factors, streams, "factor_f32")
        portable, _storage_portable = native_outputs(
            library, weights, factors, streams, "factor_f32_portable"
        )
        if not all(
            np.array_equal(a.view(np.uint32), b.view(np.uint32))
            for a, b in zip(fast, portable, strict=True)
        ):
            raise ValueError(f"{level}: portable and SIMD outputs differ")
        outputs_by_level[level] = fast
        case = metrics(fast, references, streams)
        case["portable_bitwise_equal"] = True
        case["storage_bytes"] = storage
        case["native_library_sha256"] = file_sha256(library / "wnn_native.abi3.so")
        parent_error = max(
            float(np.abs(parent[:, :2] - reference[:, :2]).max())
            for parent, reference in zip(original_outputs, references, strict=True)
        )
        reference_wp = score(references, streams)
        original_wp = score(original_outputs, streams)
        case["original_w384_approximation_context_only"] = {
            "max_abs_first_two_all_rows_vs_zero_update_dense": parent_error,
            "global_wp": original_wp,
            "global_wp_delta_vs_zero_update_dense": original_wp - reference_wp,
            "per_sequence": [
                {
                    "seq_ix": stream["seq_ix"],
                    "original_w384_wp": score([parent], [stream]),
                    "zero_update_dense_wp": score([reference], [stream]),
                    "wp_delta": score([parent], [stream]) - score([reference], [stream]),
                }
                for parent, reference, stream in zip(
                    original_outputs, references, streams, strict=True
                )
            ],
        }
        report["cases"][level] = case
        if not case["passed"]:
            raise ValueError(f"{level}: represented-dense parity gate failed")
    report["v2_v3_bitwise_equal"] = all(
        np.array_equal(a.view(np.uint32), b.view(np.uint32))
        for a, b in zip(outputs_by_level["v2"], outputs_by_level["v3"], strict=True)
    )
    if not report["v2_v3_bitwise_equal"]:
        raise ValueError("v2 and v3 factorized outputs differ")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite parity output: {args.out}")
    args.out.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    """Parse a parity-only invocation; export preparation remains a separate admitted step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--e409-manifest", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run_parity(parser.parse_args())


if __name__ == "__main__":
    main()
