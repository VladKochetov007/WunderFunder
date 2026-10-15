#!/usr/bin/env python3
"""Bounded paired native callback cost profile for production GRU/minGRU shapes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import resource
import sys
import time
from importlib.machinery import ExtensionFileLoader
from pathlib import Path
from typing import Any

import numpy as np

MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
DEFAULT_WORKTREE = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_DIR = DEFAULT_WORKTREE / "cache/min_gru_cost_profile_r1"
DEFAULT_NATIVE = Path(
    "/home/vlad/development/wunder/worktrees/min-gru-native-20261014/"
    "native/target/debug/libwnn_native.so"
)
EXPECTED_NATIVE_SHA256 = "20510deab14d051ada9260417fe02b2df7c9ff75593108c11a9bcaf368d5bbc6"
CHECKPOINT_REL = Path(
    "artifacts/runs/clean_t3_f0_control_s3_a9_t3r5_r2_p3_envr1/model_ep5.pt"
)
CHECKPOINT_SHA256 = "e30577ffb39d061bc40043f68a90285965bd2cf790f6ba5947ba78c60b776eda"
TRAIN_ROW = 37
SEQUENCE_ID = 7_598_721
RAW_SHA256 = "fe24c974e6d6e3e720d36937be06d222d00f743efd65e4a0a3ec178db891f57a"
ROWS = 20_000
RAW_WIDTH = 112
MODEL_WIDTH = 211
EVENT_WIDTH = 99
WARMUP = 99
SEED = 20_261_014
LAYERS = 2
REFERENCE_ROWS = 137
SHAPES = {
    "gru": {"encoder": 256, "recurrent": 256, "readout": 256},
    "min_gru": {"encoder": 384, "recurrent": 384, "readout": 384},
}


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence receipt: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cgroup_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"available": False}
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(line.split("::", 1)[1] for line in lines if "::" in line)
        path = Path("/sys/fs/cgroup") / relative.lstrip("/")
        if not path.exists():
            return result
        result = {"available": True, "path": str(path)}
        for name in ("memory.current", "memory.peak", "memory.max", "memory.swap.max", "cpu.max"):
            item = path / name
            if item.exists():
                result[name] = item.read_text().strip()
        events = path / "memory.events"
        if events.exists():
            result["memory.events"] = {
                key: int(value)
                for key, value in (line.split() for line in events.read_text().splitlines())
            }
    except (OSError, StopIteration, ValueError):
        return result
    return result


def resident_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1024 if sys.platform.startswith("linux") else 1))


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def prepare_inputs(main_root: Path, local_dir: Path, output: Path) -> None:
    """Extract the source-qualified row and common clean-F0 normalization buffers."""
    checkpoint_path = main_root / CHECKPOINT_REL
    parquet_path = main_root / "datasets/train.parquet"
    if sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("clean F0 checkpoint hash differs from the frozen receipt")

    sys.path.insert(0, str(main_root / "src"))
    import torch

    from wnn.data import SequenceFile

    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    mean = np.ascontiguousarray(state["mean"].detach().cpu().numpy(), dtype=np.float32)
    std = np.ascontiguousarray(state["std"].detach().cpu().numpy(), dtype=np.float32)
    if mean.shape != (MODEL_WIDTH,) or std.shape != (MODEL_WIDTH,):
        raise ValueError("clean F0 normalization is not 211-wide")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("clean F0 normalization contains invalid values")

    sequence = SequenceFile(parquet_path).read(TRAIN_ROW, targets=False, mask=False)
    raw = np.ascontiguousarray(sequence.features, dtype=np.float32)
    if sequence.seq_ix != SEQUENCE_ID or raw.shape != (ROWS, RAW_WIDTH):
        raise ValueError("train row-group 37 identity or shape changed")
    if sha256_array(raw) != RAW_SHA256:
        raise ValueError("train row-group 37 feature hash differs from the source receipt")

    local_dir.mkdir(parents=True, exist_ok=True)
    raw_path = local_dir / "train37_raw112.f32"
    mean_path = local_dir / "clean_f0_mean.npy"
    std_path = local_dir / "clean_f0_std.npy"
    for path in (raw_path, mean_path, std_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite prepared local input: {path}")
    raw.astype("<f4", copy=False).tofile(raw_path)
    np.save(mean_path, mean, allow_pickle=False)
    np.save(std_path, std, allow_pickle=False)
    manifest = {
        "format": "wnn.min-gru-real-cost-profile-inputs",
        "version": 1,
        "source": {
            "dataset": "datasets/train.parquet",
            "membership": "fitting",
            "row_group": TRAIN_ROW,
            "sequence_id": SEQUENCE_ID,
            "rows": ROWS,
            "raw_features": RAW_WIDTH,
            "raw_feature_sha256": sha256_file(raw_path),
            "source_qualified_reference_sha256": RAW_SHA256,
            "targets_or_masks_loaded": False,
        },
        "normalization": {
            "source_checkpoint": str(CHECKPOINT_REL),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
            "mean_sha256": sha256_array(mean),
            "std_sha256": sha256_array(std),
            "mean_file_sha256": sha256_file(mean_path),
            "std_file_sha256": sha256_file(std_path),
            "shared_by_both_families": True,
        },
        "local_files": {
            "raw_features": str(raw_path),
            "mean": str(mean_path),
            "std": str(std_path),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pyarrow": __import__("pyarrow").__version__,
        },
    }
    write_json(output, manifest)
    del checkpoint, state, sequence, raw
    gc.collect()


def matrix(rng: np.random.Generator, shape: tuple[int, int], fan_in: int) -> np.ndarray:
    bound = 1.0 / np.sqrt(fan_in)
    return np.ascontiguousarray(rng.uniform(-bound, bound, size=shape).astype(np.float32))


def synthesize_weights(family: str, mean: np.ndarray, std: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    widths = SHAPES[family]
    enc, recurrent, readout = widths["encoder"], widths["recurrent"], widths["readout"]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0 if family == "gru" else 1]))
    weights: dict[str, np.ndarray] = {
        "mean": np.ascontiguousarray(mean, dtype=np.float32),
        "std": np.ascontiguousarray(std, dtype=np.float32),
        "inp.weight": matrix(rng, (enc, MODEL_WIDTH), MODEL_WIDTH),
        "inp.bias": np.zeros(enc, dtype=np.float32),
    }
    if family == "gru":
        for layer in range(LAYERS):
            for part in ("ih", "hh"):
                weights[f"gru.weight_{part}_l{layer}"] = matrix(
                    rng, (3 * recurrent, recurrent), recurrent
                )
            weights[f"gru.bias_ih_l{layer}"] = np.zeros(3 * recurrent, dtype=np.float32)
            weights[f"gru.bias_hh_l{layer}"] = np.zeros(3 * recurrent, dtype=np.float32)
    else:
        for layer in range(LAYERS):
            input_width = enc if layer == 0 else recurrent
            weights[f"cells.{layer}.input_projection.weight"] = matrix(
                rng, (2 * recurrent, input_width), input_width
            )
            weights[f"cells.{layer}.input_projection.bias"] = np.zeros(
                2 * recurrent, dtype=np.float32
            )
    weights["head.0.weight"] = matrix(rng, (readout, recurrent + enc), recurrent + enc)
    weights["head.0.bias"] = np.zeros(readout, dtype=np.float32)
    weights["head.2.weight"] = matrix(rng, (2, readout), readout)
    weights["head.2.bias"] = np.zeros(2, dtype=np.float32)
    weights = {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in weights.items()}

    digest = hashlib.sha256()
    tensor_hashes: dict[str, str] = {}
    matrix_parameters = 0
    trainable_parameters = 0
    for name, value in sorted(weights.items()):
        metadata = json.dumps(
            {"name": name, "dtype": value.dtype.str, "shape": value.shape},
            separators=(",", ":"),
        ).encode()
        payload = memoryview(value).cast("B")
        digest.update(len(metadata).to_bytes(4, "little"))
        digest.update(metadata)
        digest.update(payload)
        tensor_hashes[name] = hashlib.sha256(payload).hexdigest()
        if name not in {"mean", "std"}:
            trainable_parameters += value.size
            if name.endswith("weight") or name.startswith("gru.weight_"):
                matrix_parameters += value.size
    expected_matrix_count = 972_032 if family == "gru" else 966_528
    if matrix_parameters != expected_matrix_count:
        raise AssertionError(f"{family} matrix count {matrix_parameters} != {expected_matrix_count}")
    return weights, {
        "family": family,
        "widths": widths,
        "layers": LAYERS,
        "outputs": 2,
        "seed": SEED,
        "seed_sequence": [SEED, 0 if family == "gru" else 1],
        "generator": "NumPy PCG64 via default_rng(SeedSequence([20261014, family_id]))",
        "initialization": "independent float32 uniform matrices +/-1/sqrt(fan_in); zero biases",
        "matrix_parameter_count_excluding_biases_and_norm_buffers": matrix_parameters,
        "trainable_parameter_count_including_biases": trainable_parameters,
        "trainable_f32_bytes_excluding_norm_buffers": sum(
            value.nbytes for name, value in weights.items() if name not in {"mean", "std"}
        ),
        "full_state_dict_bytes_including_norm_buffers": sum(value.nbytes for value in weights.values()),
        "state_dict_sha256": digest.hexdigest(),
        "tensor_sha256": tensor_hashes,
    }


def load_native(path: Path, expected_sha256: str) -> Any:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"native binary hash mismatch: expected {expected_sha256}, got {actual}")
    loader = ExtensionFileLoader("wnn_native", str(path))
    spec = importlib.util.spec_from_loader("wnn_native", loader)
    if spec is None:
        raise ImportError(f"cannot load PyO3 extension: {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def load_inputs(local_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = local_dir / "INPUT_MANIFEST_R1.json"
    manifest = json.loads(manifest_path.read_text())
    raw_path = local_dir / "train37_raw112.f32"
    mean_path = local_dir / "clean_f0_mean.npy"
    std_path = local_dir / "clean_f0_std.npy"
    if sha256_file(raw_path) != manifest["source"]["raw_feature_sha256"]:
        raise ValueError("local raw input fixture hash changed")
    raw = np.fromfile(raw_path, dtype="<f4")
    if raw.size != ROWS * RAW_WIDTH:
        raise ValueError("local raw feature fixture byte count changed")
    raw = np.ascontiguousarray(raw.reshape(ROWS, RAW_WIDTH), dtype=np.float32)
    mean = np.load(mean_path, allow_pickle=False)
    std = np.load(std_path, allow_pickle=False)
    if sha256_array(mean) != manifest["normalization"]["mean_sha256"]:
        raise ValueError("common normalization mean changed")
    if sha256_array(std) != manifest["normalization"]["std_sha256"]:
        raise ValueError("common normalization std changed")
    return raw, mean, std, manifest


def callback_class(native: Any, family: str) -> Any:
    return native.Predictor if family == "gru" else native.MinGruPredictor


def hidden_state(predictor: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(predictor.hidden_state, dtype=np.float32))


def profile_worker(
    family: str,
    precision: str,
    local_dir: Path,
    native_path: Path,
    expected_native_sha256: str,
    build_mode: str,
    output: Path,
) -> None:
    if family not in SHAPES or precision not in {"f32", "q8vnni"}:
        raise ValueError("invalid family or precision")
    before_rss = resident_bytes()
    before_peak = max_rss_bytes()
    before_cgroup = cgroup_snapshot()
    raw, mean, std, input_manifest = load_inputs(local_dir)
    weights, model_manifest = synthesize_weights(family, mean, std)
    native_started = time.perf_counter()
    native = load_native(native_path, expected_native_sha256)
    native_import_seconds = time.perf_counter() - native_started
    cls = callback_class(native, family)
    construct_started = time.perf_counter()
    predictor = cls(weights, fast_activations=True, precision=precision)
    construct_seconds = time.perf_counter() - construct_started
    selected_precision = predictor.precision
    if selected_precision != precision:
        raise AssertionError(f"requested {precision}, native selected {selected_precision}")
    if family == "min_gru" and predictor.fast_activations is not True:
        raise AssertionError("minGRU native callback did not select fast activations")

    expected_widths = SHAPES[family]
    if model_manifest["outputs"] != 2 or model_manifest["layers"] != 2:
        raise AssertionError("callback shape does not match the production two-layer/two-output profile")
    expected_state_shape = (2, expected_widths["recurrent"])
    zero_state = np.zeros(expected_state_shape, dtype=np.float32)
    repetitions: list[dict[str, Any]] = []
    reference_output: np.ndarray | None = None
    reference_state: np.ndarray | None = None
    for repetition in range(3):
        predictor.reset()
        state_before = hidden_state(predictor)
        if state_before.shape != expected_state_shape or not np.array_equal(state_before, zero_state):
            raise AssertionError("native reset did not restore the expected zero-state shape")
        started = time.perf_counter()
        output_values = np.empty((ROWS, 2), dtype=np.float32)
        for index, row in enumerate(raw):
            output_values[index] = predictor.step(row)
        elapsed = time.perf_counter() - started
        final_state = hidden_state(predictor)
        if output_values.shape != (ROWS, 2) or not np.isfinite(output_values).all():
            raise FloatingPointError("callback returned invalid predictions")
        if final_state.shape != expected_state_shape or not np.isfinite(final_state).all():
            raise FloatingPointError("callback returned invalid final recurrent state")
        output_hash = sha256_array(output_values)
        state_hash = sha256_array(final_state)
        if reference_output is None:
            reference_output = output_values.copy()
            reference_state = final_state.copy()
        elif not np.array_equal(output_values, reference_output) or not np.array_equal(
            final_state, reference_state
        ):
            raise AssertionError("full-sequence callback changed across reset replays")
        repetitions.append(
            {
                "index": repetition,
                "seconds": elapsed,
                "microseconds_per_raw_row": elapsed * 1_000_000 / ROWS,
                "rows_processed": ROWS,
                "including_warmup_rows": WARMUP,
                "output_sha256": output_hash,
                "final_state_sha256": state_hash,
                "reset_state_shape": list(state_before.shape),
                "reset_state_all_zero": True,
                "finite_predictions_and_state": True,
            }
        )
        del output_values, final_state
        gc.collect()

    storage = {key: int(value) for key, value in predictor.storage_bytes.items()}
    after_rss = resident_bytes()
    after_peak = max_rss_bytes()
    after_cgroup = cgroup_snapshot()
    result = {
        "format": "wnn.min-gru-real-cost-profile-worker",
        "version": 1,
        "family": family,
        "precision_requested": precision,
        "precision_selected_assertion": selected_precision,
        "fast_activations": True,
        "implementation": {
            "native_class": f"wnn_native.{cls.__name__}",
            "native_build_mode": build_mode,
            "sealed_native_binary": str(native_path),
            "native_binary_sha256": sha256_file(native_path),
            "callback": "20,000 Python-to-native step calls; each call performs raw112 sanitation, causal streaming feature construction to 211 and one recurrent/readout step",
            "state_reset": "native predictor.reset() before every complete-sequence replay",
        },
        "model": model_manifest,
        "input": {
            "source": "fitting train row-group 37, sequence 7598721",
            "rows": ROWS,
            "raw_features": RAW_WIDTH,
            "event_features": EVENT_WIDTH,
            "model_features": MODEL_WIDTH,
            "raw_feature_sha256": input_manifest["source"]["raw_feature_sha256"],
            "normalization_checkpoint_sha256": input_manifest["normalization"]["checkpoint_sha256"],
            "normalization_mean_sha256": input_manifest["normalization"]["mean_sha256"],
            "normalization_std_sha256": input_manifest["normalization"]["std_sha256"],
        },
        "native_constructor": {
            "seconds_including_imported_weight_copy_and_q8_pack": construct_seconds,
            "native_module_import_seconds_excluded": native_import_seconds,
        },
        "storage": {
            "native_weight_bytes": int(predictor.weight_bytes),
            "native_storage_bytes_breakdown": storage,
            "native_scratch_bytes": storage.get("scratch", 0),
            "python_synthetic_state_dict_bytes_including_norms": model_manifest[
                "full_state_dict_bytes_including_norm_buffers"
            ],
        },
        "full_callback_repetitions": repetitions,
        "determinism": {
            "same_predictions_bitwise_after_reset": True,
            "same_final_state_bitwise_after_reset": True,
            "all_three_output_sha256_equal": len({item["output_sha256"] for item in repetitions}) == 1,
            "all_three_state_sha256_equal": len({item["final_state_sha256"] for item in repetitions}) == 1,
        },
        "resources": {
            "cpu_model": cpu_model(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "resident_bytes_before": before_rss,
            "resident_bytes_after": after_rss,
            "process_max_rss_bytes_before": before_peak,
            "process_max_rss_bytes_after": after_peak,
            "cgroup_before": before_cgroup,
            "cgroup_after": after_cgroup,
        },
        "quality_claim": False,
        "limitations": "Synthetic deterministic weights, one source-qualified fitting sequence and one CPU host. Cost evidence only; not training capacity, quality, score or competition-host performance.",
    }
    write_json(output, result)


def reference_check(
    local_dir: Path,
    native_path: Path,
    expected_native_sha256: str,
    build_mode: str,
    output: Path,
) -> None:
    """Compare bounded exact native raw callbacks with matching Python model references."""
    import torch

    sys.path.insert(0, str(MAIN_ROOT / "src"))
    from wnn.features import event_features_torch
    from wnn.min_gru import MinGRUModel
    from wnn.nn import RNNModel

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    raw, mean, std, input_manifest = load_inputs(local_dir)
    raw = raw[:REFERENCE_ROWS]
    native = load_native(native_path, expected_native_sha256)
    results: dict[str, Any] = {}
    for family in ("gru", "min_gru"):
        weights, model_manifest = synthesize_weights(family, mean, std)
        if family == "gru":
            reference = RNNModel(MODEL_WIDTH, hidden=256, layers=LAYERS, n_out=2).eval()
        else:
            reference = MinGRUModel(
                MODEL_WIDTH,
                hidden=384,
                layers=LAYERS,
                n_out=2,
                encoder_width=384,
                recurrent_width=384,
                readout_width=384,
            ).eval()
        state = {name: torch.from_numpy(value.copy()) for name, value in weights.items()}
        reference.load_state_dict(state, strict=True)
        raw_tensor = torch.from_numpy(raw[None, ...].copy())
        engineered = torch.cat((raw_tensor, event_features_torch(raw_tensor)), dim=-1)
        if engineered.shape != (1, REFERENCE_ROWS, MODEL_WIDTH):
            raise AssertionError("reference preprocessing did not produce [1,137,211]")
        with torch.inference_mode():
            expected_output, expected_state = reference(engineered)
        expected_output_np = expected_output[0].cpu().numpy()
        expected_state_np = expected_state[:, 0].cpu().numpy()
        predictor_class = callback_class(native, family)
        predictor = predictor_class(weights, fast_activations=False, precision="f32")
        if predictor.precision != "f32":
            raise AssertionError("bounded reference did not select exact FP32 native mode")
        actual_output = np.stack([predictor.step(row) for row in raw])
        actual_state = hidden_state(predictor)
        prediction_difference = actual_output.astype(np.float64) - expected_output_np.astype(np.float64)
        state_difference = actual_state.astype(np.float64) - expected_state_np.astype(np.float64)
        if not np.isfinite(actual_output).all() or not np.isfinite(actual_state).all():
            raise FloatingPointError(f"{family} exact native chunk emitted nonfinite values")
        results[family] = {
            "state_dict_sha256": model_manifest["state_dict_sha256"],
            "rows": REFERENCE_ROWS,
            "model_input_shape": list(engineered.shape),
            "python_reference": "wnn.nn.RNNModel with exact GELU/torch GRU"
            if family == "gru"
            else "wnn.min_gru.MinGRUModel affine-prefix sequence reference",
            "native_callback": f"wnn_native.{predictor_class.__name__}, fast_activations=False, f32",
            "prediction_max_abs_error": float(np.max(np.abs(prediction_difference), initial=0.0)),
            "prediction_rmse": float(np.sqrt(np.mean(prediction_difference**2))),
            "state_max_abs_error": float(np.max(np.abs(state_difference), initial=0.0)),
            "state_rmse": float(np.sqrt(np.mean(state_difference**2))),
            "predictions_and_state_finite": True,
        }
        del predictor, reference, expected_output, expected_state, expected_output_np, expected_state_np
        del engineered, raw_tensor, state, weights
        gc.collect()
    write_json(
        output,
        {
            "format": "wnn.min-gru-real-cost-profile-reference-check",
            "version": 1,
            "native_binary_sha256": sha256_file(native_path),
            "native_binary_path": str(native_path),
            "native_build_mode": build_mode,
            "input": {
                "source": "fitting train row-group 37",
                "rows_checked": REFERENCE_ROWS,
                "raw_feature_sha256": input_manifest["source"]["raw_feature_sha256"],
                "normalization_checkpoint_sha256": input_manifest["normalization"]["checkpoint_sha256"],
            },
            "checks": results,
            "resources": {
                "cpu_model": cpu_model(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "process_max_rss_bytes": max_rss_bytes(),
                "cgroup": cgroup_snapshot(),
            },
            "quality_claim": False,
            "limitations": "Only 137 rows per family were compared to the Python reference; production cost measurements remain full-sequence native callbacks. Numerical differences are reported rather than treated as exact parity.",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    prepare = subparsers.add_parser("prepare-inputs")
    prepare.add_argument("--main-root", type=Path, default=MAIN_ROOT)
    prepare.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    prepare.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LOCAL_DIR / "INPUT_MANIFEST_R1.json",
    )
    worker = subparsers.add_parser("worker")
    worker.add_argument("--family", choices=tuple(SHAPES), required=True)
    worker.add_argument("--precision", choices=("f32", "q8vnni"), required=True)
    worker.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    worker.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    worker.add_argument("--expected-native-sha256", default=EXPECTED_NATIVE_SHA256)
    worker.add_argument("--build-mode", choices=("debug", "release"), default="debug")
    worker.add_argument("--output", type=Path, required=True)
    reference = subparsers.add_parser("reference")
    reference.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    reference.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    reference.add_argument("--expected-native-sha256", default=EXPECTED_NATIVE_SHA256)
    reference.add_argument("--build-mode", choices=("debug", "release"), default="debug")
    reference.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare-inputs":
        prepare_inputs(args.main_root, args.local_dir, args.output)
    elif args.mode == "worker":
        profile_worker(
            args.family,
            args.precision,
            args.local_dir,
            args.native,
            args.expected_native_sha256,
            args.build_mode,
            args.output,
        )
    else:
        reference_check(
            args.local_dir,
            args.native,
            args.expected_native_sha256,
            args.build_mode,
            args.output,
        )


if __name__ == "__main__":
    main()
