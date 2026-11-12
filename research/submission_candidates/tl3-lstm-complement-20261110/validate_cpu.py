#!/usr/bin/env python3
"""Run hash-bound, offline CPU receiver checks for the exact submission ZIP."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import re
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any  # JSON receipts and the externally loaded callback are heterogeneous.

import numpy as np
import pyarrow.parquet as pq

PACKET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKET_DIR))
from receiver_io import (  # noqa: E402
    SEQUENCE_LENGTH,
    VALID_BYTES,
    VALID_FOOTER_SHA256,
    VALID_ROWS,
    VALID_SCORED_ROWS,
    VALID_SEQUENCES,
    SequenceData,
    load_bound_json,
    output_storage,
    parquet_footer_identity,
    read_sequence,
    replay_sequence,
    runtime_envelope,
    safe_extract,
    sha256_file,
    write_json_once,
)

INPUT_SCHEMA = "wnn.cpu-submit-inputs.v1"
RECIPE_SCHEMA = "wnn.cpu-tl3-lstm-recipe.v1"
DEFAULT_CODE_ROOT = PACKET_DIR.parents[2]


def _check_inputs(inputs: dict[str, Any]) -> str:
    """Bind the mounted valid split and required source inventories."""
    if inputs.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"INPUTS schema must be {INPUT_SCHEMA}")
    for name in ("archives", "checkpoints", "native_runtime"):
        if name not in inputs:
            raise ValueError(f"INPUTS lacks required {name!r} inventory")
    split_hash = inputs.get("splitsha", inputs.get("split_sha256"))
    if split_hash is None and isinstance(inputs.get("split"), dict):
        split_hash = inputs["split"].get("sha256")
    if not isinstance(split_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", split_hash):
        raise ValueError("INPUTS must bind the frozen split with a SHA-256")
    data = inputs.get("validdata", inputs.get("valid_data"))
    expected = {
        "bytes": VALID_BYTES,
        "footer_sha256": VALID_FOOTER_SHA256,
        "rows": VALID_ROWS,
        "scored_rows": VALID_SCORED_ROWS,
        "sequences": VALID_SEQUENCES,
    }
    if not isinstance(data, dict):
        raise ValueError("INPUTS must include the validdata identity object")
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"INPUTS validdata.{key} differs from the frozen official split")
    runtime = inputs["native_runtime"]
    if not isinstance(runtime, dict) or not isinstance(runtime.get("libraries"), dict):
        raise ValueError("INPUTS native_runtime must bind the native archive and library digests")
    for name, digest in runtime["libraries"].items():
        if name not in {"v2", "avx2", "v3"} or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValueError("INPUTS native_runtime contains an unknown library or invalid SHA-256")
    source_sha = inputs.get("base_source_report_sha256")
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError("INPUTS must bind the canonical fresh TL3 base source report SHA-256")
    return split_hash


def _check_recipe(recipe: dict[str, Any], expected_source_sha: str) -> tuple[float, float]:
    """Require the canonical fresh TL3 plus PR53 fixed LSTM mixture and source receipt."""
    if recipe.get("schema") != RECIPE_SCHEMA:
        raise ValueError(f"recipe schema must be {RECIPE_SCHEMA}")
    baseline_weight = recipe.get("baseline_weight")
    lstm_weight = recipe.get("lstm_weight")
    if not math.isclose(float(baseline_weight), 0.9, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("recipe baseline_weight must equal 0.9")
    if not math.isclose(float(lstm_weight), 0.1, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("recipe lstm_weight must equal 0.1")
    if recipe.get("mode") != "dynamic_int8":
        raise ValueError("recipe mode must be dynamic_int8")
    source_report = recipe.get("research_reference", {}).get("source_report_sha256")
    source_sha = recipe.get("base_source_report_sha256", source_report)
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError("recipe must bind the frozen base source report SHA-256")
    if recipe.get("research_reference", {}).get("scope") != (
        "Exposed validation, unquantized research predictors; not a score for this CPU ZIP"
    ):
        raise ValueError("recipe must preserve the exposed-validation evidence scope")
    members = recipe.get("baseline_members")
    expected = {"w18": 0.34, "tiedlead5": 0.33, "tiedlead6": 0.33}
    if not isinstance(members, list) or len(members) != len(expected):
        raise ValueError(
            "recipe baseline_members must contain the three frozen canonical TL3 members"
        )
    actual: dict[str, float] = {}
    for member in members:
        if not isinstance(member, dict):
            raise ValueError("recipe baseline_members entries must be objects")
        name = member.get("name", member.get("id"))
        if name is None and isinstance(member.get("weights"), str):
            name = Path(member["weights"]).stem
        weight = member.get("weight", member.get("coef"))
        if not isinstance(name, str) or name in actual:
            raise ValueError("recipe baseline member names must be unique strings")
        actual[name] = float(weight)
    if set(actual) != set(expected) or any(
        not math.isclose(actual[name], weight, rel_tol=0.0, abs_tol=1e-12)
        for name, weight in expected.items()
    ):
        raise ValueError(
            "recipe baseline members/weights differ from the frozen canonical TL3 blend"
        )
    expected_runs = {
        "w18": "tl3c_w18final_distilled_s5",
        "tiedlead5": "tl3c_fttall_tlead_s5",
        "tiedlead6": "tl3c_fttall_tlead_s6",
    }
    expected_families = {"w18": "event211", "tiedlead5": "tiedlead263", "tiedlead6": "tiedlead263"}
    if any(
        member.get("run") != expected_runs.get(Path(str(member.get("weights", ""))).stem)
        or member.get("family") != expected_families.get(Path(str(member.get("weights", ""))).stem)
        for member in members
    ):
        raise ValueError("recipe member runs/families differ from the frozen canonical fresh TL3")
    if recipe.get("research_reference", {}).get("source_report_sha256") != source_sha:
        raise ValueError("recipe evidence and source report hashes differ")
    if source_sha != expected_source_sha:
        raise ValueError("recipe source report SHA differs from the hash-bound INPUTS manifest")
    if recipe.get("submission_ready") is not False:
        raise ValueError("recipe must not claim submission readiness")
    return float(baseline_weight), float(lstm_weight)


def _load_candidate(package_root: Path, code_root: Path) -> Any:
    """Import the ZIP's root solution while resolving the official repo ABI first."""
    code_root = code_root.resolve(strict=True)
    if not (code_root / "utils.py").is_file():
        raise FileNotFoundError(f"official utils.py is absent from code root {code_root}")
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    if str(package_root) not in sys.path:
        sys.path.insert(1, str(package_root))
    sys.modules.pop("solution", None)
    module = importlib.import_module("solution")
    if "torch" in sys.modules:
        raise RuntimeError("the shipped solution imported Torch; final ZIP must be NumPy-only")
    constructor = getattr(module, "PredictionModel", None)
    if not callable(constructor):
        raise TypeError("ZIP solution.py does not expose PredictionModel")
    model = constructor()
    if getattr(model, "mode", None) != "dynamic_int8":
        raise ValueError("loaded native baseline is not dynamic_int8")
    if not callable(getattr(model, "reset", None)):
        raise TypeError("PredictionModel.reset() is required for explicit state-reset checks")
    return model


def _native_members(package_root: Path, hashes: dict[str, str]) -> list[dict[str, Any]]:
    """Check that every packaged shared object is a 64-bit little-endian x86 ELF."""
    native: list[dict[str, Any]] = []
    for name in hashes:
        if not name.endswith(".so"):
            continue
        path = package_root / name
        header = path.read_bytes()[:20]
        if (
            len(header) < 20
            or header[:4] != b"\x7fELF"
            or header[4] != 2
            or header[5] != 1
            or int.from_bytes(header[18:20], "little") != 62
        ):
            raise ValueError(f"native member is not Linux x86_64 ELF: {name}")
        native.append({"member": name, "sha256": hashes[name]})
    if not native:
        raise ValueError("submission ZIP lacks a native x86_64 runtime member")
    return native


def _check_parquet(data_path: Path, expected_footer: dict[str, int | str]) -> pq.ParquetFile:
    """Validate full split metadata and exact ordered callback schema."""
    parquet = pq.ParquetFile(data_path)
    from utils import FEATURE_COLUMNS, TARGET_COLUMNS
    from utils import SEQUENCE_LENGTH as OFFICIAL_SEQUENCE_LENGTH

    expected_schema = [
        "seq_ix",
        "step_in_seq",
        "need_prediction",
        "is_scored",
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    ]
    if parquet.schema_arrow.names != expected_schema:
        raise ValueError("valid.parquet schema/order differs from official callback columns")
    if (
        parquet.num_row_groups != VALID_SEQUENCES
        or parquet.metadata.num_rows != VALID_ROWS
        or OFFICIAL_SEQUENCE_LENGTH != SEQUENCE_LENGTH
    ):
        raise ValueError("valid.parquet must contain 1,873 complete 20,000-row groups")
    if (
        expected_footer["bytes"] != VALID_BYTES
        or expected_footer["footer_sha256"] != VALID_FOOTER_SHA256
    ):
        raise ValueError("mounted valid.parquet bytes/footer differ from the frozen source")
    return parquet


def _run_state_checks(
    package_root: Path,
    model_type: type,
    data_a: SequenceData,
    data_b: SequenceData,
    datapoint_type: type,
    blend_weights: tuple[float, float],
    parity_rows: int,
) -> dict[str, Any]:
    """Run reset, sequence-boundary, mask-independence, causality and Torch parity checks."""
    base_weight, lstm_weight = blend_weights
    model = model_type()
    model.reset()
    a_first = replay_sequence(model, data_a, datapoint_type, base_weight, lstm_weight)
    b_middle = replay_sequence(model, data_b, datapoint_type, base_weight, lstm_weight)
    a_last = replay_sequence(model, data_a, datapoint_type, base_weight, lstm_weight)
    aba_equal = all(
        np.array_equal(left, right)
        for left, right in zip(
            (a_first.base, a_first.lstm, a_first.blended),
            (a_last.base, a_last.lstm, a_last.blended),
            strict=True,
        )
    )
    if not aba_equal:
        raise ValueError("A-B-A sequence replay changed output after sequence reset")
    model.reset()
    a_reset = replay_sequence(model, data_a, datapoint_type, base_weight, lstm_weight)
    reset_equal = all(
        np.array_equal(left, right)
        for left, right in zip(
            (a_first.base, a_first.lstm, a_first.blended),
            (a_reset.base, a_reset.lstm, a_reset.blended),
            strict=True,
        )
    )
    if not reset_equal:
        raise ValueError("explicit PredictionModel.reset() did not reproduce sequence A")

    mask_model = model_type()
    mask_model.reset()
    actual_mask = replay_sequence(mask_model, data_a, datapoint_type, base_weight, lstm_weight)
    opposite = dataclasses.replace(data_a, need_prediction=~data_a.need_prediction)
    mask_model.reset()
    changed_mask = replay_sequence(mask_model, opposite, datapoint_type, base_weight, lstm_weight)
    mask_equal = all(
        np.array_equal(left, right)
        for left, right in zip(
            (actual_mask.base, actual_mask.lstm, actual_mask.blended),
            (changed_mask.base, changed_mask.lstm, changed_mask.blended),
            strict=True,
        )
    )
    if not mask_equal:
        raise ValueError(
            "need_prediction changed causal model state instead of only gating outputs"
        )

    cutoff = min(2_048, SEQUENCE_LENGTH - 1)
    suffix = data_a.features.copy()
    suffix[cutoff:, 0] += np.float32(0.25)
    if not np.isfinite(suffix).all():
        raise ValueError("causality suffix perturbation produced a non-finite input")
    suffix_model = model_type()
    suffix_model.reset()
    original = replay_sequence(suffix_model, data_a, datapoint_type, base_weight, lstm_weight)
    suffix_model.reset()
    altered_data = dataclasses.replace(data_a, features=suffix)
    altered = replay_sequence(suffix_model, altered_data, datapoint_type, base_weight, lstm_weight)
    causal_equal = all(
        np.array_equal(left[:cutoff], right[:cutoff])
        for left, right in zip(
            (original.base, original.lstm, original.blended),
            (altered.base, altered.lstm, altered.blended),
            strict=True,
        )
    )
    if not causal_equal:
        raise ValueError("future suffix perturbation changed predictions before its cutoff")

    from utils import DataPoint

    table_features = data_a.features
    parity_model = model_type()
    from receiver_parity import run_lstm_reference_parity

    parity = run_lstm_reference_parity(
        package_root,
        parity_model,
        DataPoint,
        data_a.sequence_id,
        table_features,
        data_a.need_prediction,
        rows=parity_rows,
    )
    return {
        "a_b_a_reset_bitwise_equal": aba_equal,
        "explicit_reset_bitwise_equal": reset_equal,
        "need_mask_independent_components": mask_equal,
        "suffix_causality_bitwise_equal_before_row": cutoff,
        "torch_lstm_parity": parity,
        "diagnostic_rows_per_sequence": SEQUENCE_LENGTH,
        "diagnostic_sequences": [data_a.sequence_id, data_b.sequence_id],
        "diagnostic_callback_ns": a_first.callback_ns + b_middle.callback_ns + a_last.callback_ns,
    }


def _versions() -> dict[str, str | None]:
    """Record receiver package versions without importing optional Torch."""
    names = ("numpy", "scipy", "pyarrow", "torch", "threadpoolctl")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _parser() -> argparse.ArgumentParser:
    """Create the explicit two-stage receiver CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "full"), default="profile")
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--zip-sha256", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-sequences", type=int, choices=(2, 8), required=True)
    parser.add_argument("--test-rows", type=int)
    parser.add_argument("--parity-rows", type=int, default=2_000)
    parser.add_argument("--profile-receipt", type=Path)
    parser.add_argument("--profile-receipt-sha256")
    return parser


def _check_profile_receipt(
    args: argparse.Namespace, package_hash: str, inputs_hash: str, recipe_hash: str
) -> dict[str, Any]:
    """Require a completed profile receipt for the same immutable run inputs."""
    if args.profile_receipt is None or args.profile_receipt_sha256 is None:
        raise ValueError("full stage requires --profile-receipt and its SHA-256")
    receipt = load_bound_json(args.profile_receipt, args.profile_receipt_sha256, "profile receipt")
    if receipt.get("stage") != "profile" or receipt.get("profile_passed") is not True:
        raise ValueError("full stage requires a passing profile-stage receipt")
    for key, value in (
        ("zip_sha256", package_hash),
        ("inputs_sha256", inputs_hash),
        ("recipe_sha256", recipe_hash),
    ):
        if receipt.get(key) != value:
            raise ValueError(f"profile receipt {key} differs from the full-stage inputs")
    return receipt


def _profile_stage(
    args: argparse.Namespace,
    parquet: pq.ParquetFile,
    package_root: Path,
    model_type: type,
    blend_weights: tuple[float, float],
    env: dict[str, Any],
    package_hash: str,
    inputs_hash: str,
    recipe_hash: str,
    split_hash: str,
    member_hashes: dict[str, str],
    native_members: list[dict[str, Any]],
    data_identity_before: dict[str, int | str],
    output_fs: dict[str, str],
) -> dict[str, Any]:
    """Run the bounded profile and fail-closed ABI/state checks only."""
    from utils import FEATURE_COLUMNS, TARGET_COLUMNS, DataPoint, validate_sequence

    indices = np.linspace(0, parquet.num_row_groups - 1, args.profile_sequences, dtype=int).tolist()
    sequences: list[SequenceData] = []
    read_ns = 0
    for index in indices:
        started = time.perf_counter_ns()
        sequences.append(
            read_sequence(parquet, index, FEATURE_COLUMNS, TARGET_COLUMNS, validate_sequence)
        )
        read_ns += time.perf_counter_ns() - started
    if len({sequence.sequence_id for sequence in sequences}) != len(sequences):
        raise ValueError("duplicate seq_ix among profiled Parquet groups")
    diagnostic = _run_state_checks(
        package_root,
        model_type,
        sequences[0],
        sequences[-1],
        DataPoint,
        blend_weights,
        args.parity_rows,
    )
    profile_model = model_type()
    profile_model.reset()
    timings = []
    for sequence in sequences:
        replay = replay_sequence(profile_model, sequence, DataPoint, *blend_weights)
        timings.append(
            {
                "row_group": sequence.row_group,
                "seq_ix": sequence.sequence_id,
                "rows": SEQUENCE_LENGTH,
                "callback_ns": replay.callback_ns,
                "callback_us_per_row": replay.callback_ns / SEQUENCE_LENGTH / 1_000,
            }
        )
    identity_after = parquet_footer_identity(args.data)
    if identity_after != data_identity_before:
        raise ValueError("valid.parquet changed during profile stage")
    return {
        "schema": "wnn.cpu-submit-receiver.v1",
        "stage": "profile",
        "gate_status": "PENDING",
        "profile_passed": True,
        "ready_to_submit": False,
        "scope": "bounded 2/8 sequence profile and package/state checks; no full-valid score",
        "zip_sha256": package_hash,
        "zip_bytes": args.zip.stat().st_size,
        "inputs_sha256": inputs_hash,
        "recipe_sha256": recipe_hash,
        "split_sha256": split_hash,
        "valid_data_identity": data_identity_before,
        "profile_sequences": args.profile_sequences,
        "profile_groups": timings,
        "profile_read_seconds": read_ns / 1e9,
        "diagnostics": diagnostic,
        "native_members": native_members,
        "zip_members": member_hashes,
        "environment": env,
        "versions": _versions(),
        "output_filesystem": output_fs,
        "unresolved": [
            "full-valid quality delta",
            "explicit official TEST_ROWS runtime projection",
            "independent ancestry/manual review",
        ],
    }


def _full_stage(
    args: argparse.Namespace,
    parquet: pq.ParquetFile,
    model_type: type,
    blend_weights: tuple[float, float],
    env: dict[str, Any],
    package_hash: str,
    inputs_hash: str,
    recipe_hash: str,
    split_hash: str,
    member_hashes: dict[str, str],
    native_members: list[dict[str, Any]],
    data_identity_before: dict[str, int | str],
    output_fs: dict[str, str],
    profile_receipt: dict[str, Any],
    model_load_seconds: float,
) -> dict[str, Any]:
    """Replay all 1,873 sequences, score official global WP and bind projection."""
    from utils import (
        FEATURE_COLUMNS,
        TARGET_COLUMNS,
        DataPoint,
        GlobalAccumulator,
        validate_sequence,
    )

    if profile_receipt.get("environment") != env:
        raise ValueError("profile and full stage must use the same Python/CPU/resource envelope")
    output_paths = {
        "candidate": args.output.parent / "full-valid-candidate.npy",
        "base": args.output.parent / "full-valid-base.npy",
        "targets": args.output.parent / "full-valid-targets.npy",
        "seq_ix": args.output.parent / "full-valid-seq-ix.npy",
        "step_in_seq": args.output.parent / "full-valid-step-in-seq.npy",
    }
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite full-valid predictions: {path}")
    pair_shape = (VALID_SCORED_ROWS, 2)
    candidate_predictions = np.lib.format.open_memmap(
        output_paths["candidate"], mode="w+", dtype=np.float32, shape=pair_shape
    )
    baseline_predictions = np.lib.format.open_memmap(
        output_paths["base"], mode="w+", dtype=np.float32, shape=pair_shape
    )
    selected_targets = np.lib.format.open_memmap(
        output_paths["targets"], mode="w+", dtype=np.float32, shape=pair_shape
    )
    sequence_ids = np.lib.format.open_memmap(
        output_paths["seq_ix"], mode="w+", dtype=np.int64, shape=(VALID_SCORED_ROWS,)
    )
    steps = np.lib.format.open_memmap(
        output_paths["step_in_seq"], mode="w+", dtype=np.uint16, shape=(VALID_SCORED_ROWS,)
    )
    blend_accumulator, base_accumulator = GlobalAccumulator(), GlobalAccumulator()
    model = model_type()
    model.reset()
    read_ns = callback_ns = score_ns = rows_seen = predictions_required = scored_rows = 0
    seen_sequence_ids: set[int] = set()
    group_callback_ns: dict[int, int] = {}
    profile_indices = set(
        np.linspace(0, VALID_SEQUENCES - 1, args.profile_sequences, dtype=int).tolist()
    )
    total_started = time.perf_counter()
    for group in range(VALID_SEQUENCES):
        read_start = time.perf_counter_ns()
        data = read_sequence(parquet, group, FEATURE_COLUMNS, TARGET_COLUMNS, validate_sequence)
        read_ns += time.perf_counter_ns() - read_start
        if data.sequence_id in seen_sequence_ids:
            raise ValueError(f"duplicate seq_ix {data.sequence_id} in full valid.parquet")
        seen_sequence_ids.add(data.sequence_id)
        replay = replay_sequence(
            model,
            data,
            DataPoint,
            *blend_weights,
            retain_unscored_components=False,
            verify_blend=False,
        )
        callback_ns += replay.callback_ns
        if group in profile_indices:
            group_callback_ns[group] = replay.callback_ns
        selected_count = len(replay.scored_steps)
        selected_slice = slice(scored_rows, scored_rows + selected_count)
        group_targets = data.targets[replay.scored_steps]
        all_selected = np.ones(selected_count, dtype=bool)
        score_start = time.perf_counter_ns()
        blend_accumulator.add(group_targets, replay.blended, all_selected)
        base_accumulator.add(group_targets, replay.base, all_selected)
        score_ns += time.perf_counter_ns() - score_start
        candidate_predictions[selected_slice] = replay.blended
        baseline_predictions[selected_slice] = replay.base
        selected_targets[selected_slice] = group_targets
        sequence_ids[selected_slice] = data.sequence_id
        steps[selected_slice] = replay.scored_steps.astype(np.uint16, copy=False)
        rows_seen += SEQUENCE_LENGTH
        predictions_required += int(data.need_prediction.sum())
        scored_rows += selected_count
    for array in (
        candidate_predictions,
        baseline_predictions,
        selected_targets,
        sequence_ids,
        steps,
    ):
        array.flush()
    total_seconds = time.perf_counter() - total_started
    if rows_seen != VALID_ROWS or predictions_required != VALID_SEQUENCES * (SEQUENCE_LENGTH - 99):
        raise ValueError("full-valid replay did not cover the complete required callback grid")
    if scored_rows != VALID_SCORED_ROWS:
        raise ValueError(f"scored rows {scored_rows} differ from frozen count {VALID_SCORED_ROWS}")
    if parquet_footer_identity(args.data) != data_identity_before:
        raise ValueError("valid.parquet changed during full-valid replay")
    blend_score = blend_accumulator.result()
    base_score = base_accumulator.result()
    quality_delta = float(blend_score["weighted_pearson"]) - float(base_score["weighted_pearson"])
    if not math.isfinite(quality_delta):
        raise ValueError("full-valid quality delta is not finite")
    callback_rate = max(group_callback_ns.values()) / (SEQUENCE_LENGTH * 1e9)
    read_rate = read_ns / max(rows_seen, 1) / 1e9
    score_rate = score_ns / max(rows_seen, 1) / 1e9
    projection = None
    runtime_gate = "PENDING_TEST_ROWS"
    if args.test_rows is not None:
        if args.test_rows < 1:
            raise ValueError("TEST_ROWS must be a positive explicit integer")
        projected_seconds = (
            model_load_seconds + (callback_rate + read_rate + score_rate) * args.test_rows * 1.25
        )
        projection = {
            "test_rows": args.test_rows,
            "model_load_seconds": model_load_seconds,
            "callback_seconds_per_row_upper_profile": callback_rate,
            "parquet_read_seconds_per_row": read_rate,
            "score_seconds_per_row": score_rate,
            "safety_factor": 1.25,
            "projected_seconds": projected_seconds,
            "limit_seconds": 3_600,
            "within_limit": projected_seconds <= 3_600,
            "scope": "receiver projection proxy; not a portal runtime guarantee",
        }
        runtime_gate = "PASS" if projected_seconds <= 3_600 else "FAIL"
    del candidate_predictions, baseline_predictions, selected_targets, sequence_ids, steps
    output_hashes = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in output_paths.values()
    }
    quality_gate = quality_delta > 0.0
    if not quality_gate or runtime_gate == "FAIL":
        gate_status = "REJECTED"
    elif runtime_gate == "PENDING_TEST_ROWS":
        gate_status = "PENDING"
    else:
        gate_status = "PASS_MANUAL_REVIEW_REQUIRED"
    return {
        "schema": "wnn.cpu-submit-receiver.v1",
        "stage": "full",
        "gate_status": gate_status,
        "ready_to_submit": False,
        "scope": "full validation replay, official global clipped weighted Pearson and server runtime projection proxy",
        "zip_sha256": package_hash,
        "zip_bytes": args.zip.stat().st_size,
        "inputs_sha256": inputs_hash,
        "recipe_sha256": recipe_hash,
        "profile_receipt_sha256": args.profile_receipt_sha256,
        "split_sha256": split_hash,
        "valid_data_identity": data_identity_before,
        "rows_seen": rows_seen,
        "sequences_seen": VALID_SEQUENCES,
        "predictions_required": predictions_required,
        "scored_rows": scored_rows,
        "paired_prediction_rows_stored": scored_rows,
        "paired_uncertainty_inputs": {
            "available": True,
            "arrays": [
                "full-valid-targets.npy",
                "full-valid-base.npy",
                "full-valid-candidate.npy",
                "full-valid-seq-ix.npy",
                "full-valid-step-in-seq.npy",
            ],
            "alignment": ["seq_ix", "step_in_seq"],
        },
        "metric_id": blend_score["metric_id"],
        "metric_clip": blend_score["metric_clip"],
        "candidate_global_wp": blend_score["weighted_pearson"],
        "candidate_t0": blend_score["t0"],
        "candidate_t1": blend_score["t1"],
        "served_base_global_wp": base_score["weighted_pearson"],
        "served_base_t0": base_score["t0"],
        "served_base_t1": base_score["t1"],
        "quality_delta": quality_delta,
        "quality_gate": "PASS" if quality_gate else "FAIL",
        "quality_threshold": "strictly positive point delta; no uncertainty interval computed",
        "runtime_projection": projection,
        "runtime_gate": runtime_gate,
        "profile_group_callback_ns": group_callback_ns,
        "profile_group_ids": sorted(profile_indices),
        "read_seconds": read_ns / 1e9,
        "callback_seconds": callback_ns / 1e9,
        "score_seconds": score_ns / 1e9,
        "full_stage_seconds": total_seconds,
        "prediction_outputs": output_hashes,
        "native_members": native_members,
        "zip_members": member_hashes,
        "environment": env,
        "versions": _versions(),
        "output_filesystem": output_fs,
        "unresolved": [
            "independent ancestry/manual review",
            "portal runtime and scoring acceptance",
        ],
    }


def main() -> int:
    """Validate the fixed artifact in a constrained Linux CPU receiver."""
    started = time.perf_counter()
    process_started = time.process_time()
    args = _parser().parse_args()
    if args.parity_rows < 1 or (args.test_rows is not None and args.test_rows < 1):
        raise ValueError("parity rows and TEST_ROWS must be positive when supplied")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace receiver receipt: {args.output}")
    env = runtime_envelope()
    import threadpoolctl

    threadpoolctl.threadpool_limits(limits=1)
    env["cpu_model"] = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        None,
    )
    env["threadpools"] = threadpoolctl.threadpool_info()
    env["threadpool_limit"] = 1
    output_fs = output_storage(args.output.parent)
    package_hash = sha256_file(args.zip)
    if package_hash != args.zip_sha256:
        raise ValueError("mounted submission ZIP hash differs from the supplied ZIP SHA-256")
    inputs = load_bound_json(args.inputs, args.inputs_sha256, "INPUTS")
    recipe = load_bound_json(args.recipe, args.recipe_sha256, "recipe")
    split_hash = _check_inputs(inputs)
    blend_weights = _check_recipe(recipe, inputs["base_source_report_sha256"])
    args.code_root = args.code_root.resolve(strict=True)
    if str(args.code_root) not in sys.path:
        sys.path.insert(0, str(args.code_root))
    env["receiver_source_sha256"] = {
        name: sha256_file(PACKET_DIR / name)
        for name in ("validate_cpu.py", "receiver_io.py", "receiver_parity.py", "receiver.sh")
    }
    env["scorer_source_sha256"] = sha256_file(args.code_root / "utils.py")
    if args.stage == "full":
        profile_receipt = _check_profile_receipt(
            args, package_hash, args.inputs_sha256, args.recipe_sha256
        )
    else:
        profile_receipt = {}
    data_identity_before = parquet_footer_identity(args.data)
    parquet = _check_parquet(args.data, data_identity_before)
    with tempfile.TemporaryDirectory(prefix="wnn-cpu-submit-package-") as temporary:
        package_root = Path(temporary) / "package"
        member_hashes = safe_extract(args.zip, package_root)
        member_receipt = inputs.get("submission_members")
        if member_receipt is not None and member_receipt != member_hashes:
            raise ValueError("ZIP member hashes differ from the checksum-bound INPUTS manifest")
        bundled = inputs.get("bundled_dependency", {}).get("files", [])
        for dependency in bundled:
            member = dependency.get("file")
            expected_hash = dependency.get("sha256")
            if member_hashes.get(member) != expected_hash:
                raise ValueError(f"packaged dependency hash mismatch for {member!r}")
        for expected_hash in inputs["native_runtime"]["libraries"].values():
            if expected_hash not in member_hashes.values():
                raise ValueError(
                    "native ZIP library bytes differ from the hash-bound INPUTS manifest"
                )
        native_members = _native_members(package_root, member_hashes)
        if sys.implementation.cache_tag != "cpython-311" or platform.machine().lower() != "x86_64":
            raise RuntimeError("receiver Python/native ABI must be cpython-311 on x86_64")
        solution_path = package_root / "solution.py"
        embedded_recipe = package_root / "recipe.json"
        if not embedded_recipe.is_file() or sha256_file(embedded_recipe) != args.recipe_sha256:
            raise ValueError("ZIP recipe.json SHA differs from the mounted hash-bound recipe")
        if embedded_recipe.read_bytes() != args.recipe.read_bytes():
            raise ValueError("ZIP and mounted recipe.json bytes are not identical")
        spec = importlib.util.spec_from_file_location("solution", solution_path)
        if spec is None or spec.loader is None:
            raise ImportError("cannot load root solution.py from the measured ZIP")
        if sha256_file(solution_path) != member_hashes["solution.py"]:
            raise ValueError("extracted solution.py differs from the hashed ZIP member")
        model_started = time.perf_counter()
        model = _load_candidate(package_root, args.code_root)
        model_load_seconds = time.perf_counter() - model_started
        if args.stage == "profile":
            report = _profile_stage(
                args,
                parquet,
                package_root,
                type(model),
                blend_weights,
                env,
                package_hash,
                args.inputs_sha256,
                args.recipe_sha256,
                split_hash,
                member_hashes,
                native_members,
                data_identity_before,
                output_fs,
            )
        else:
            report = _full_stage(
                args,
                parquet,
                type(model),
                blend_weights,
                env,
                package_hash,
                args.inputs_sha256,
                args.recipe_sha256,
                split_hash,
                member_hashes,
                native_members,
                data_identity_before,
                output_fs,
                profile_receipt,
                model_load_seconds,
            )
    report["receiver_total_wall_seconds"] = time.perf_counter() - started
    report["process_cpu_seconds"] = time.process_time() - process_started
    report["process_max_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    cgroup_peak = Path("/sys/fs/cgroup/memory.peak")
    report["cgroup_memory_peak_bytes"] = (
        int(cgroup_peak.read_text().strip()) if cgroup_peak.is_file() else None
    )
    report["test_rows_explicit"] = args.test_rows
    receipt_sha256 = write_json_once(args.output, report)
    report["receipt_sha256"] = receipt_sha256
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
