"""Measure scorer-callback runtime for a passing E420 factorized control."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import engine_fp16_check as engine  # noqa: E402

from wnn.splits import file_sha256  # noqa: E402

FACTOR_NAMES = tuple(f"gru.weight_{kind}_l{layer}" for layer in range(2) for kind in ("ih", "hh"))
EXPECTED_SOURCE_SHA256 = "b5181f3aa948765f22a8c3f23f6e293ba715b52ffb91eb6eac02d6b7e30a8b8d"
EXPECTED_E409_MANIFEST_SHA256 = "815bee9144348fa58c6cd622e0ba2ff48c94b3722228d2b19fafd6e227552464"
EXPECTED_CONTROL_SHA256 = "1e3c440f458e5e9826911b37b5762c8becdce44b1f33965ea80918d1568900e9"
EXPECTED_DATA_SHA256 = "cd6a349b3655bb6f98314e5db391e69b1e1ea7f12ab18f1d15c61cefd28daa37"
EXPECTED_CLOSURE_SHA256 = "dfe0e1ef3b1900dab1180e08376a5f63adc2d7e7f88326cb3557940dd4c51b10"
EXPECTED_PARITY_SHA256 = "58f1691a9573b35bf1d4bae9c28368dcb2d47c9257e954ed2c9de7714fe0078e"
PRECISIONS = {
    "factor_f32": "factor_f32",
    "effective_dense_f32": "f32",
    "original_w384_q16x": "q16x",
}
TIMING_CASES = tuple(PRECISIONS)
MAX_CALLBACK_ABS = 3e-4


def arrays_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, np.ndarray]:
    """Expose only FP32 inference weights to the native importer."""
    arrays: dict[str, np.ndarray] = {}
    for name, value in checkpoint["model"].items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"expected FP32 inference tensor in {name}")
        arrays[name] = np.ascontiguousarray(value.detach().cpu().numpy())
    return arrays


def factor_arrays(checkpoint: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Read the saved FP32 factors without running another factorization."""
    metadata = checkpoint.get("factorization")
    if not isinstance(metadata, dict) or metadata.get("rank") != 128:
        raise ValueError("expected the saved rank-128 factorization")
    if metadata.get("training_updates") != 0:
        raise ValueError("factor runtime must use the zero-update SVD control")
    factors = metadata.get("factors")
    if not isinstance(factors, dict) or set(factors) != set(FACTOR_NAMES):
        raise ValueError("factor export must contain exactly the four GRU matrices")
    output: dict[str, dict[str, np.ndarray]] = {}
    for name in FACTOR_NAMES:
        output[name] = {}
        for side, shape in (("left", (1152, 128)), ("right", (128, 384))):
            tensor = factors[name][side]
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
                raise ValueError(f"expected FP32 factor {name}.{side}")
            array = tensor.detach().cpu().numpy()
            if array.shape != shape:
                raise ValueError(f"factor shape mismatch for {name}.{side}")
            output[name][side] = np.ascontiguousarray(array)
    return output


def write_once(path: Path, value: object) -> None:
    """Write one immutable JSON result."""
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def slice_deployment_head(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep the two deployment targets; remove auxiliary output rows before native import."""
    result = dict(weights)
    weight = result["head.2.weight"]
    bias = result["head.2.bias"]
    if weight.ndim != 2 or weight.shape[0] != 23 or bias.shape != (23,):
        raise ValueError("E420 timing expects the source's 23-output head")
    result["head.2.weight"] = np.ascontiguousarray(weight[:2])
    result["head.2.bias"] = np.ascontiguousarray(bias[:2])
    return result


def validate_parity(
    parity_path: Path,
    source_path: Path,
    control_path: Path,
    data_path: Path,
    e409_manifest_path: Path,
    library_root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Require the sealed-compatible E420 parity evidence before any timed work."""
    report = json.loads(parity_path.read_text())
    if file_sha256(parity_path) != EXPECTED_PARITY_SHA256:
        raise ValueError("E420 parity report hash differs from the completed passing report")
    if report.get("experiment") != "E420":
        raise ValueError("timing requires the E420 full-stream parity report")
    if (
        report.get("purpose")
        != "native full-stream representation parity; no timing or model selection"
    ):
        raise ValueError("parity report has an unexpected purpose")
    if file_sha256(source_path) != EXPECTED_SOURCE_SHA256:
        raise ValueError("source W384 checkpoint hash changed")
    if file_sha256(control_path) != EXPECTED_CONTROL_SHA256:
        raise ValueError("zero-update rank-128 control hash changed")
    if file_sha256(data_path) != EXPECTED_DATA_SHA256:
        raise ValueError("E409 sequence data hash changed")
    if file_sha256(e409_manifest_path) != EXPECTED_E409_MANIFEST_SHA256:
        raise ValueError("E409 manifest hash changed")
    if report.get("source_checkpoint_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("parity report is not bound to the expected parent checkpoint")
    if report.get("zero_update_control_sha256") != EXPECTED_CONTROL_SHA256:
        raise ValueError("parity report is not bound to the expected rank-128 control")
    if report.get("e409_manifest_sha256") != EXPECTED_E409_MANIFEST_SHA256:
        raise ValueError("parity report is not bound to the frozen E409 manifest")
    if report.get("source_closure_sha256") != EXPECTED_CLOSURE_SHA256:
        raise ValueError("parity report has a different source closure")
    if report.get("e409_data_sha256") != EXPECTED_DATA_SHA256:
        raise ValueError("parity report is not bound to the frozen E409 input")
    if report.get("row_groups") != [0, 1, 2]:
        raise ValueError("timing requires exactly E409 row groups 0, 1 and 2")
    if report.get("v2_v3_bitwise_equal") is not True:
        raise ValueError("E420 v2/v3 predictions did not pass bitwise parity")
    cases = report.get("cases", {})
    if not all(cases.get(level, {}).get("passed") is True for level in ("v2", "v3")):
        raise ValueError("E420 factorized full-20k parity must pass on both levels")
    gate = report.get("gate", {})
    if gate.get("max_abs") != MAX_CALLBACK_ABS or gate.get("abs_global_wp_delta") != 5e-6:
        raise ValueError("E420 parity tolerances differ from the frozen gate")
    library_hashes: dict[str, str] = {}
    for level in ("v2", "v3"):
        case = cases[level]
        if (
            case.get("max_abs_first_two_all_rows", float("inf")) > MAX_CALLBACK_ABS
            or case.get("abs_global_wp_delta", float("inf")) > 5e-6
            or case.get("portable_bitwise_equal") is not True
        ):
            raise ValueError(f"{level} factorized parity metrics fail the frozen gate")
        expected = case.get("native_library_sha256")
        library = library_root / level / "wnn_native" / "wnn_native.abi3.so"
        if not isinstance(expected, str) or file_sha256(library) != expected:
            raise ValueError(f"{level} native library differs from the passing parity build")
        library_hashes[level] = expected
    return report, library_hashes


def make_case_weights(
    source: dict[str, Any], control: dict[str, Any]
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
]:
    """Create target-sliced deployment inputs from source and zero-update control states."""
    factor_full = arrays_from_checkpoint(control)
    factor_weights = slice_deployment_head(factor_full)
    dense_weights = slice_deployment_head(dict(factor_full))
    source_weights = slice_deployment_head(arrays_from_checkpoint(source))
    factors = factor_arrays(control)
    if set(factors) != set(FACTOR_NAMES):
        raise ValueError("factorized control does not carry the exact four GRU matrices")
    return factor_weights, dense_weights, source_weights, factors


def load_library(path: Path):
    """Import the selected native scorer build from its frozen package directory."""
    sys.path.insert(0, str(path.resolve()))
    module = importlib.import_module("wnn_native")
    if Path(module.__file__).resolve().parent != path.resolve():
        raise RuntimeError(f"imported {module.__file__}, expected native package under {path}")
    return module


def read_runtime_sequences(data_path: Path, row_groups: list[int]):
    """Load E409's three real raw-feature sequences for the scorer callback harness."""
    sequences = []
    for seq_ix, features, _targets, need, _mask in engine.read_sequences(data_path, row_groups):
        if len(features) != 20_000 or features.shape[1] != 112:
            raise ValueError("expected three 20,000-row, 112-column raw streams")
        sequences.append(
            (int(seq_ix), np.ascontiguousarray(features), np.asarray(need, dtype=bool))
        )
    if len(sequences) != 3:
        raise ValueError("expected exactly three E409 streams")
    return sequences


def selected_stream_sizes(data_path: Path, row_groups: list[int]) -> dict[str, int]:
    """Count selected Parquet compressed/uncompressed bytes and in-memory raw arrays."""
    parquet = pq.ParquetFile(data_path)
    selected_columns = set(engine.FEATURE_COLUMNS) | {"need_prediction"}
    compressed = uncompressed = 0
    for group in row_groups:
        row_group = parquet.metadata.row_group(group)
        for column_ix in range(row_group.num_columns):
            column = row_group.column(column_ix)
            if column.path_in_schema in selected_columns:
                compressed += column.total_compressed_size
                uncompressed += column.total_uncompressed_size
    rows = 3 * 20_000
    feature_bytes = rows * 112 * np.dtype(np.float32).itemsize
    need_bytes = rows * np.dtype(np.bool_).itemsize
    return {
        "selected_parquet_compressed_bytes": compressed,
        "selected_parquet_uncompressed_bytes": uncompressed,
        "loaded_raw_feature_bytes": feature_bytes,
        "loaded_need_mask_bytes": need_bytes,
    }


def predictor_for(native, case: str, weights, factors):
    """Build the one-member predictor whose predict path matches scorer callbacks."""
    options: dict[str, Any] = {}
    if case == "factor_f32":
        options["factors"] = [factors]
    model = native.EnsemblePredictor(
        [weights],
        [np.float32(1.0)],
        fast_activations=True,
        precision=PRECISIONS[case],
        budget_s=None,
        **options,
    )
    if model.precision != PRECISIONS[case]:
        raise ValueError(f"native precision fallback: {case} -> {model.precision}")
    if case == "factor_f32":
        exported = model.factorization_tensors()[0]
        if set(exported) != set(factors):
            raise ValueError("native callback did not import the four saved factor matrices")
        for name, pair in factors.items():
            for side, expected in pair.items():
                actual = exported[name][side]
                if actual.dtype != np.float32 or actual.tobytes() != expected.tobytes():
                    raise ValueError(f"native callback factor bytes changed for {name}.{side}")
    return model


def run_case(native, case: str, weights, factors, sequences):
    """Run one case over all three sequences for one repeat."""
    model = predictor_for(native, case, weights, factors)
    prediction_rows = []
    elapsed = 0.0
    cycles = [0]
    for seq_ix, features, need in sequences:
        seconds, output = engine.harness(model, seq_ix, features, need, cycles)
        elapsed += seconds
        prediction_rows.append(output)
    rows = sum(features.shape[0] for _, features, _ in sequences)
    storage = dict(model.storage_bytes[0])
    required_storage = {
        "total_streamed_weights",
        "factor_weights",
        "recurrent_state",
        "scratch",
        "import_only_dense_recurrent",
    }
    if not required_storage <= storage.keys():
        raise ValueError("native storage report omits a required separate accounting field")
    input_weight_bytes = sum(value.nbytes for value in weights.values())
    factor_input_bytes = (
        sum(value.nbytes for pair in factors.values() for value in pair.values())
        if case == "factor_f32"
        else 0
    )
    if storage["factor_weights"] > storage["total_streamed_weights"]:
        raise ValueError("factor bytes must be a subset of total streamed model weights")
    if storage["total_streamed_weights"] != model.weight_bytes[0]:
        raise ValueError("total streamed model-weight accounting is inconsistent")
    if storage["factor_weights"] != factor_input_bytes:
        raise ValueError("native factor byte count differs from saved factor inputs")
    expected_import_dense = 4 * 3 * 384 * 384 * 4 if case == "factor_f32" else 0
    if storage["import_only_dense_recurrent"] != expected_import_dense:
        raise ValueError("W384 dense recurrent import bytes changed")
    return np.stack(prediction_rows), {
        "precision": model.precision,
        "sequences": len(sequences),
        "rows": rows,
        "seconds": elapsed,
        "us_per_row": elapsed * 1e6 / rows,
        "kcycles_per_row": cycles[0] / rows / 1000 if engine.CYCLES.fd >= 0 else None,
        "native_storage_bytes": storage,
        "python_import_dense_parameter_bytes": input_weight_bytes,
        "python_import_factor_parameter_bytes": factor_input_bytes,
        "timed_scope": "E409 ScorerStepByStep DataPoint loop + native predict callback; includes native causal feature updates",
    }


def timing_order(repeat: int) -> tuple[str, ...]:
    """Alternate forward/reverse paired order over the five preregistered repeats."""
    return TIMING_CASES if repeat % 2 == 0 else tuple(reversed(TIMING_CASES))


def finite_pair_max_abs(left: np.ndarray, right: np.ndarray) -> float:
    """Compare scored predictions while requiring matching finite-row layouts."""
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.array_equal(np.isfinite(left), np.isfinite(right)) or not finite.any():
        raise ValueError("paired callback predictions have different valid-row masks")
    return float(np.max(np.abs(left[finite] - right[finite])))


def run(args: argparse.Namespace) -> None:
    """Run only after full E420 parity; runtime output is descriptive, never a score claim."""
    if args.repeats != 5:
        raise ValueError("E420 runtime is preregistered for exactly five alternating trials")
    if args.lib_level != "v3":
        raise ValueError("this draft times the v3 scorer build after both v2/v3 parity pass")
    if args.out.exists():
        raise FileExistsError("preserve prior timing evidence and use a new output path")
    report, library_hashes = validate_parity(
        args.parity,
        args.source,
        args.control,
        args.data,
        args.e409_manifest,
        args.library_root,
    )
    os.sched_setaffinity(0, {args.cpu})
    if engine.CYCLES.fd < 0:
        raise RuntimeError("CPU cycle counter unavailable; E420 runtime requires kcycles/row")
    native = load_library(args.library_root / args.lib_level / "wnn_native")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    control = torch.load(args.control, map_location="cpu", weights_only=False)
    if control.get("factorization", {}).get("training_updates") != 0:
        raise ValueError("runtime factor case must remain the zero-update SVD control")
    factor_weights, dense_weights, original_weights, factors = make_case_weights(source, control)
    sequences = read_runtime_sequences(args.data, report["row_groups"])
    stream_bytes = selected_stream_sizes(args.data, report["row_groups"])

    hardware = {
        "platform": platform.platform(),
        "python": sys.version,
        "affinity": sorted(os.sched_getaffinity(0)),
        "cpuinfo": Path("/proc/cpuinfo").read_text(),
        "cgroup": Path("/proc/self/cgroup").read_text(),
        "loadavg_start": Path("/proc/loadavg").read_text(),
        "cycles_counter_available": engine.CYCLES.fd >= 0,
        "parity_report_sha256": file_sha256(args.parity),
        "parity_control_sha256": report["zero_update_control_sha256"],
        "native_library_sha256_v2_v3": library_hashes,
        "timed_native_level": args.lib_level,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    trials_path = args.out / "timing_trials.jsonl"
    summaries: dict[str, list[dict[str, Any]]] = {case: [] for case in TIMING_CASES}
    predictions_by_repeat: dict[int, dict[str, np.ndarray]] = {}
    inputs = {
        "factor_f32": factor_weights,
        "effective_dense_f32": dense_weights,
        "original_w384_q16x": original_weights,
    }
    with trials_path.open("x") as trial_stream:
        for repeat in range(args.repeats):
            predictions_by_repeat[repeat] = {}
            for position, case in enumerate(timing_order(repeat)):
                predictions, metrics = run_case(native, case, inputs[case], factors, sequences)
                predictions_by_repeat[repeat][case] = predictions
                trial = {
                    "repeat": repeat,
                    "position": position,
                    "case": case,
                    "loadavg": Path("/proc/loadavg").read_text(),
                    **metrics,
                }
                summaries[case].append(trial)
                trial_stream.write(json.dumps(trial) + "\n")
                trial_stream.flush()
                print(json.dumps(trial), flush=True)
            factor = predictions_by_repeat[repeat]["factor_f32"]
            dense = predictions_by_repeat[repeat]["effective_dense_f32"]
            max_error = finite_pair_max_abs(factor, dense)
            if max_error > MAX_CALLBACK_ABS:
                raise RuntimeError(
                    f"callback-path factor/dense mismatch {max_error} exceeds {MAX_CALLBACK_ABS}"
                )

    summary = {}
    for case, values in summaries.items():
        summary[case] = {}
        for metric in ("us_per_row", "kcycles_per_row"):
            observed = [item[metric] for item in values if item[metric] is not None]
            summary[case][metric] = (
                None
                if not observed
                else {
                    "min": min(observed),
                    "median": statistics.median(observed),
                    "max": max(observed),
                    "all": observed,
                }
            )
        summary[case]["native_storage_bytes"] = values[0]["native_storage_bytes"]
        summary[case]["python_import_dense_parameter_bytes"] = values[0][
            "python_import_dense_parameter_bytes"
        ]
        summary[case]["python_import_factor_parameter_bytes"] = values[0][
            "python_import_factor_parameter_bytes"
        ]
    callback_max_abs = max(
        finite_pair_max_abs(
            predictions_by_repeat[repeat]["factor_f32"],
            predictions_by_repeat[repeat]["effective_dense_f32"],
        )
        for repeat in range(args.repeats)
    )
    write_once(
        args.out / "result.json",
        {
            "experiment": "E420_RUNTIME",
            "purpose": "actual scorer-callback timing for a parity-passing zero-update factor control; descriptive only",
            "hardware": hardware,
            "inputs": {
                "source_checkpoint": str(args.source),
                "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
                "source_epoch": int(source.get("epoch", -1)),
                "source_checkpoint_epoch": int(source.get("checkpoint_epoch", -1)),
                "zero_update_control": str(args.control),
                "zero_update_control_sha256": EXPECTED_CONTROL_SHA256,
                "control_epoch": int(control.get("epoch", -1)),
                "control_checkpoint_epoch": int(control.get("checkpoint_epoch", -1)),
                "e409_manifest": str(args.e409_manifest),
                "e409_manifest_sha256": EXPECTED_E409_MANIFEST_SHA256,
                "e409_data": str(args.data),
                "e409_data_sha256": EXPECTED_DATA_SHA256,
                "row_groups": report["row_groups"],
                "sequence_ids": [seq_ix for seq_ix, _features, _need in sequences],
                "rows_per_sequence": [
                    int(features.shape[0]) for _seq_ix, features, _need in sequences
                ],
            },
            "model": {
                "parent_hidden": 384,
                "gru_layers": 2,
                "source_output_width": 23,
                "timed_head_output_width": 2,
                "factor_rank": 128,
                "factor_matrix_names": list(FACTOR_NAMES),
                "factor_parameterization": "saved FP32 left[3H,rank] and right[rank,H]; no refactor at timing",
                "cases": {
                    "factor_f32": "rank-128 zero-update control, exact saved factors",
                    "effective_dense_f32": "same control and head, four saved left@right dense matrices",
                    "original_w384_q16x": "unfactorized parent as a separate deployment-cost context",
                },
            },
            "stream_bytes": stream_bytes,
            "timing": {
                "order": "factor_f32, effective_dense_f32, original_w384_q16x; reverse order on odd repeats",
                "repeats": args.repeats,
                "cpu": args.cpu,
                "head_outputs": 2,
                "factor_dense_callback_max_abs": callback_max_abs,
                "factor_dense_callback_max_abs_gate": MAX_CALLBACK_ABS,
                "includes": "E409 DataPoint/scorer loop and native causal event-feature callback; includes state reset on sequence transitions",
                "initialization_excluded": True,
                "public_runtime_projection": None,
            },
            "summary": summary,
            "trials": summaries,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--e409-manifest", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--lib-level", choices=("v2", "v3"), default="v3")
    parser.add_argument("--cpu", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
