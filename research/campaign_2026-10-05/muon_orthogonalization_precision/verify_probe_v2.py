"""Independent arithmetic check for one E487 result and replay-delta archive.

This verifier does not load the parquet or rerun either optimizer. It recomputes
update metrics from the saved delta vectors, and checks the consistency of the
captured hash/state receipts and shared-gradient controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PRECISIONS = ("bf16_source", "fp32_diagnostic")
SOURCES = ("native", "custom")
STRICT_RELATIVE_RMS = 0.02
STRICT_MAX_ABS = 2e-5
MUON_MATRICES = (
    "gru.weight_hh_l0",
    "gru.weight_hh_l1",
    "gru.weight_ih_l0",
    "gru.weight_ih_l1",
)


def _finite_array(value: np.ndarray, label: str) -> np.ndarray:
    """Require real finite arrays while preserving checkpoint/update dtype."""
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"nonfinite or nonnumeric {label}")
    return array


def _rms(arrays: dict[str, np.ndarray], names: list[str]) -> float:
    count = 0
    square_sum = 0.0
    for name in names:
        value = _finite_array(arrays[name], name).astype(np.float64, copy=False)
        count += value.size
        square_sum += float(np.square(value).sum(dtype=np.float64))
    if count == 0:
        raise ValueError("empty RMS family")
    return math.sqrt(square_sum / count)


def _compare(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict:
    """Independently recalculate the registered four-matrix/four-family gates."""
    names = sorted(reference)
    if names != sorted(candidate):
        raise ValueError("candidate update parameter inventory differs")
    groups = {f"matrix:{name}": [name] for name in MUON_MATRICES}
    groups.update(
        {
            "family:inp": [name for name in names if name.startswith("inp.")],
            "family:gru_bias": [name for name in names if name.startswith("gru.bias_")],
            "family:head": [name for name in names if name.startswith("head.")],
        }
    )
    if {name for members in groups.values() for name in members} != set(names):
        raise ValueError("registered update groups do not partition all parameters")
    result = {}
    for label, members in groups.items():
        if not members:
            raise ValueError(f"empty registered group: {label}")
        ref_rms = _rms(reference, members)
        cand_rms = _rms(candidate, members)
        errors = {name: candidate[name] - reference[name] for name in members}
        err_rms = _rms(errors, members)
        max_abs = max(float(np.abs(errors[name]).max()) for name in members)
        relative = 0.0 if ref_rms == 0 and err_rms == 0 else (
            None if ref_rms == 0 else err_rms / ref_rms
        )
        pass_rms = bool(relative is not None and relative <= STRICT_RELATIVE_RMS)
        pass_abs = max_abs <= STRICT_MAX_ABS
        result[label] = {
            "reference_update_rms": ref_rms,
            "candidate_update_rms": cand_rms,
            "error_update_rms": err_rms,
            "relative_error_rms": relative,
            "max_abs_error": max_abs,
            "relative_rms_limit": STRICT_RELATIVE_RMS,
            "max_abs_limit": STRICT_MAX_ABS,
            "pass_relative_rms": pass_rms,
            "pass_max_abs": pass_abs,
            "passed": bool(pass_rms and pass_abs),
        }
    return {"metrics": result, "passed": all(item["passed"] for item in result.values())}


def _cell_receipt(result: dict, key: str) -> dict:
    cells = result["crossed_replay_arithmetic"]["cell_receipts"]
    if key not in cells:
        raise ValueError(f"missing crossed replay receipt {key}")
    return cells[key]


def _npz_cell(archive, label: str, names: list[str]) -> dict[str, np.ndarray]:
    prefix = f"{label}__"
    keys = [key for key in archive.files if key.startswith(prefix)]
    found = sorted(key[len(prefix) :] for key in keys)
    if found != sorted(names):
        raise ValueError(f"delta archive parameter inventory differs for {label}")
    return {name: _finite_array(archive[f"{prefix}{name}"], f"{label}/{name}") for name in names}


def verify_payload(result: dict, archive) -> dict:
    """Check fixed inventory, finite evidence, hash relationships and arithmetic."""
    if result.get("kind") != "E487 one-batch CUDA Muon orthogonalization precision diagnostic":
        raise ValueError("unexpected result kind")
    if result.get("fit_only") is not True or result.get("test_rows_loaded") is not False:
        raise ValueError("fit/test role boundary receipt failed")
    if result.get("seed") != 271836 or result.get("initialization_bytes_match_after_device_transfer") is not True:
        raise ValueError("seed or common initialization receipt mismatch")
    raw_receipt = result.get("raw_feature_validation", {})
    if raw_receipt.get("all_24_fit_groups_checked_before_causal_sanitization") is not True:
        raise ValueError("raw fit feature validation receipt missing")
    if len(raw_receipt.get("fit_row_groups_in_manifest_order", [])) != 24:
        raise ValueError("raw fit group receipt does not contain all 24 groups")
    if len(raw_receipt.get("raw_feature_sha256_in_manifest_order", [])) != 24:
        raise ValueError("raw fit feature digest receipt does not contain all 24 groups")
    if result.get("resources", {}).get("allocator_limit_bytes") != 2 * 1024**3:
        raise ValueError("GPU allocator cap receipt differs from preregistered 2 GiB")
    if result.get("resources", {}).get("observed_gpu_stop_bytes") != 3_000_000_000:
        raise ValueError("device stop receipt differs from preregistered 3 GB")

    if "parameter_names" not in archive.files:
        raise ValueError("delta archive has no parameter_names index")
    names = archive["parameter_names"].astype(str).tolist()
    if names != sorted(set(names)):
        raise ValueError("parameter_names must be sorted and unique")
    expected_muon = set(MUON_MATRICES)
    if not expected_muon.issubset(names):
        raise ValueError("registered Muon matrix tensors are missing")

    crossed_labels = [f"{source}__{precision}" for source in SOURCES for precision in PRECISIONS]
    shared_labels = [
        f"shared_{precision}_{replay}"
        for precision in PRECISIONS
        for replay in ("native", "custom")
    ] + ["shared_bf16_source_frozen_e474"]
    expected_keys = {"parameter_names"}
    expected_keys.update(f"{label}__{name}" for label in crossed_labels + shared_labels for name in names)
    if set(archive.files) != expected_keys:
        raise ValueError("delta archive array inventory differs from the fixed 9-cell plan")

    deltas = {label: _npz_cell(archive, label, names) for label in crossed_labels + shared_labels}
    if result.get("crossed_replay_arithmetic", {}).get("delta_arrays", {}).get(
        "crossed_replay_cells"
    ) != 4 or result["crossed_replay_arithmetic"]["delta_arrays"].get(
        "shared_gradient_replay_cells"
    ) != 5:
        raise ValueError("result array inventory receipt differs from 4+5 cells")

    gradient_capture = result.get("gradient_capture", {})
    if set(gradient_capture) != set(SOURCES):
        raise ValueError("gradient source receipt must contain native and custom")
    crossed_metrics = {}
    for source in SOURCES:
        source_receipts = []
        for precision in PRECISIONS:
            label = f"{source}__{precision}"
            receipt = _cell_receipt(result, label)
            if receipt.get("raw_gradient_sha256") != gradient_capture[source].get(
                "raw_gradient_sha256"
            ):
                raise ValueError(f"raw gradient digest mismatch for {label}")
            if receipt.get("optimizer_state_finite") is not True:
                raise ValueError(f"optimizer finite-state receipt missing for {label}")
            source_receipts.append(receipt)
        if source_receipts[0].get("clipped_gradient_sha256") != source_receipts[1].get(
            "clipped_gradient_sha256"
        ):
            raise ValueError(f"precision arms changed clipped gradient bytes for {source}")

    for precision in PRECISIONS:
        reference = deltas[f"native__{precision}"]
        candidate = deltas[f"custom__{precision}"]
        actual = _compare(reference, candidate)
        recorded = result["crossed_replay_arithmetic"]["strict_native_custom_update_gates"][precision]
        if not _same_metrics(actual, recorded):
            raise ValueError(f"stored strict gate arithmetic differs for {precision}")
        crossed_metrics[precision] = actual

    sensitivity = {}
    for source in SOURCES:
        bf16, fp32 = deltas[f"{source}__bf16_source"], deltas[f"{source}__fp32_diagnostic"]
        source_sensitivity = {}
        for name in names:
            difference = fp32[name] - bf16[name]
            source_sensitivity[name] = {
                "bf16_delta_rms": _rms(bf16, [name]),
                "fp32_delta_rms": _rms(fp32, [name]),
                "precision_change_rms": _rms({name: difference}, [name]),
                "precision_change_max_abs": float(np.abs(difference).max()),
            }
        recorded = result["crossed_replay_arithmetic"]["bf16_fp32_delta_sensitivity"][source]
        if not _same_metrics(source_sensitivity, recorded):
            raise ValueError(f"stored precision sensitivity arithmetic differs for {source}")
        sensitivity[source] = source_sensitivity

    shared = result.get("shared_gradient_control", {})
    if set(shared) != set(PRECISIONS):
        raise ValueError("shared control precision inventory differs")
    shared_checks = {}
    all_shared_hashes = set()
    all_clipped_hashes = set()
    for precision in PRECISIONS:
        control = shared[precision]
        replays = control.get("replays", {})
        required_replays = {"native", "custom"}
        if precision == "bf16_source":
            required_replays.add("frozen_e474")
        if set(replays) != required_replays:
            raise ValueError(f"shared control replay inventory differs for {precision}")
        hashes, clipped, state_hashes, optimizer_hashes = set(), set(), set(), set()
        for replay_name, receipt in replays.items():
            hashes.add(receipt.get("raw_gradient_sha256"))
            clipped.add(receipt.get("clipped_gradient_sha256"))
            state_hashes.add(receipt.get("state_sha256_after"))
            optimizer_hashes.add(receipt.get("optimizer_state_sha256_after"))
            if replay_name != "frozen_e474" and receipt.get("optimizer_state_finite") is not True:
                raise ValueError(f"shared control optimizer state not certified finite {precision}/{replay_name}")
        if hashes != {control.get("shared_gradient_sha256")}:
            raise ValueError(f"shared-gradient hashes differ within {precision}")
        if len(clipped) != 1 or len(state_hashes) != 1 or len(optimizer_hashes) != 1:
            raise ValueError(f"shared clone clipped/state/optimizer hashes differ for {precision}")
        if control.get("same_gradient_delta_bytes_exact") is not True:
            raise ValueError(f"shared control delta exactness receipt failed for {precision}")
        if control.get("same_gradient_model_state_bytes_exact") is not True:
            raise ValueError(f"shared control model state exactness receipt failed for {precision}")
        if control.get("same_gradient_optimizer_state_hash_equal") is not True:
            raise ValueError(f"shared control optimizer exactness receipt failed for {precision}")
        if control.get("clip_activated") is not True:
            raise ValueError(f"shared control did not exercise global clipping for {precision}")
        all_shared_hashes |= hashes
        all_clipped_hashes |= clipped
        shared_checks[precision] = {
            "gradient_hash": next(iter(hashes)),
            "clipped_gradient_hash": next(iter(clipped)),
            "state_sha256_after": next(iter(state_hashes)),
            "optimizer_state_sha256": next(iter(optimizer_hashes)),
            "delta_arrays_exact": True,
        }
        if precision == "bf16_source":
            custom_label = "shared_bf16_source_custom"
            frozen_label = "shared_bf16_source_frozen_e474"
            if not all(
                np.array_equal(deltas[custom_label][name], deltas[frozen_label][name])
                for name in names
            ):
                raise ValueError("persisted BF16 shared deltas differ from frozen E474")
            if control.get("frozen_e474_equal") is not True:
                raise ValueError("frozen E474 equality receipt failed")

    if len(all_shared_hashes) != 1 or len(all_clipped_hashes) != 1:
        raise ValueError("shared gradient controls do not share a single fixed gradient")
    all_finite = all(
        bool(np.isfinite(value).all()) for cell in deltas.values() for value in cell.values()
    ) and all(
        _cell_receipt(result, label).get("optimizer_state_finite") is True
        for label in crossed_labels
    )
    if result["crossed_replay_arithmetic"].get("all_crossed_finite") is not all_finite:
        raise ValueError("all_crossed_finite receipt differs from independently checked deltas/state flags")
    strict_pass = all(item["passed"] for item in crossed_metrics.values())
    expected_status = "diagnostic_complete" if strict_pass else "diagnostic_complete_gate_failures_recorded"
    if result.get("strict_precision_pass") is not strict_pass or result.get("status") != expected_status:
        raise ValueError("status does not preserve the strict gate outcome")
    return {
        "status": "passed",
        "strict_precision_pass": strict_pass,
        "crossed_update_gates_recomputed": crossed_metrics,
        "precision_sensitivity_recomputed": sensitivity,
        "shared_control_receipts": shared_checks,
        "all_delta_arrays_finite": all_finite,
        "note": "captured raw-gradient and updated-state tensors are not exported; their source hashes/equality receipts are checked, while all nine update-delta cells are numerically recomputed from NPZ in saved FP32 subtraction precision",
    }


def _same_metrics(actual, recorded) -> bool:
    """Compare nested metric data without hiding numeric differences."""
    if isinstance(actual, dict):
        return isinstance(recorded, dict) and actual.keys() == recorded.keys() and all(
            _same_metrics(actual[key], recorded[key]) for key in actual
        )
    if isinstance(actual, list):
        return isinstance(recorded, list) and len(actual) == len(recorded) and all(
            _same_metrics(a, b) for a, b in zip(actual, recorded, strict=True)
        )
    if isinstance(actual, bool) or actual is None or isinstance(actual, str):
        return actual == recorded
    if isinstance(actual, (int, float)):
        return isinstance(recorded, (int, float)) and math.isclose(
            float(actual), float(recorded), rel_tol=2e-13, abs_tol=2e-15
        )
    return actual == recorded


def verify_files(result_path: Path, delta_path: Path) -> dict:
    """Verify result and NPZ content without opening any dataset or model checkpoint."""
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    delta_bytes = delta_path.read_bytes()
    proof = result["crossed_replay_arithmetic"]
    if hashlib.sha256(delta_bytes).hexdigest() != proof.get("delta_artifact_sha256"):
        raise ValueError("delta archive SHA-256 differs from result receipt")
    if len(delta_bytes) != proof.get("delta_artifact_bytes"):
        raise ValueError("delta archive size differs from result receipt")
    with np.load(delta_path, allow_pickle=False) as archive:
        return verify_payload(result, archive)


def main() -> None:
    """Write a new independent verification receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--deltas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite verifier receipt: {args.output}")
    receipt = verify_files(args.result, args.deltas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
