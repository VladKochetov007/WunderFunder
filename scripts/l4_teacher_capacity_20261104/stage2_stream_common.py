"""Shared bounded state and arithmetic for stage-two target streaming."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_distill_cache import _target_order
from wnn.distillation_cache import SEQUENCE_ROWS, row_groups_sha256, validate_distillation_cache
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, load_campaign_split
from wnn.training.chronological_teacher import validate_chronological_train_index_order

EXPECTED_RUNS = {
    "wide_gru": (
        "l4tc_gru384x2_s1001_20261104",
        "l4tc_gru384x2_s1002_20261104",
        "l4tc_gru512x1_s1001_20261104",
        "l4tc_gru512x1_s1002_20261104",
    ),
    "causal_transformer": (
        "l4tc_transformer_d384_l8_h8_s1001_20261104",
        "l4tc_transformer_d384_l8_h8_s1002_20261104",
    ),
}
GRU_SHAPES = {
    "l4tc_gru384x2_s1001_20261104": (384, 2, 1001),
    "l4tc_gru384x2_s1002_20261104": (384, 2, 1002),
    "l4tc_gru512x1_s1001_20261104": (512, 1, 1001),
    "l4tc_gru512x1_s1002_20261104": (512, 1, 1002),
}
SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
TRAIN_SEQUENCES = 10_325
SOURCE_TRAIN_INDEX_SEQUENCES = 10_607
TARGET_BYTES = TRAIN_SEQUENCES * SEQUENCE_ROWS * 2 * np.dtype(np.float32).itemsize + 128
NATIVE_BANK_MAX_BYTES = (
    SOURCE_TRAIN_INDEX_SEQUENCES * SEQUENCE_ROWS * 2 * np.dtype(np.float32).itemsize + 128
)
SEQUENCE_BANK_MAX_BYTES = max(NATIVE_BANK_MAX_BYTES, TARGET_BYTES + TRAIN_SEQUENCES * 256)
PR60_CACHE_METADATA_SHA256 = "f9e983f311589f3510263cf5f75dbfd113b799693d2e69e078989aa313469536"
PR60_MAIN_TRAIN_SHA256 = "a5e298741d0d4ac251cb5068bf8a1a8d772f1e534efa739cc717d069c6d2f31e"
PR60_FAMILIES_TRAIN_SHA256 = "ed1ebad4912908d6a476b49b53b2169b793fefb07966ab7a107559f1105bad20"


def read_json(path: Path) -> dict[str, Any]:
    """Read a bounded-size JSON receipt as a dynamic runtime object."""
    if path.stat().st_size > 16 << 20:
        raise ValueError(f"receipt exceeds 16 MiB: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"receipt root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write a compact progress/provenance receipt."""
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def partial_root(output: Path) -> Path:
    """Return the deterministic location of one not-yet-published cache."""
    return output.with_name(output.name + ".partial")


def state_path(output: Path) -> Path:
    """Return the small receipt path for one streaming target."""
    return partial_root(output) / "stage2_state.json"


def load_state(output: Path, family: str) -> dict[str, Any]:
    """Load and bind an incremental family accumulator before accepting a teacher."""
    state = read_json(state_path(output))
    if (
        state.get("schema") != "wnn.l4-teacher-stage2-stream.v2"
        or state.get("family") != family
        or state.get("expected_runs") != list(EXPECTED_RUNS[family])
    ):
        raise ValueError("stage-2 stream state does not match the frozen family recipe")
    if state.get("pending_run") is not None:
        raise RuntimeError(
            f"previous aggregation stopped during {state['pending_run']}; preserve and inspect its partial"
        )
    return state


def sequence_hash(values: np.ndarray) -> str:
    """Hash ordered row-group identities in the shared little-endian int64 encoding."""
    return row_groups_sha256(list(map(int, values)))


def fit_source_positions(source_order: list[int], fit_order: list[int]) -> np.ndarray:
    """Map each official fit group to its position in the full source-index prediction bank."""
    if len(source_order) != len(set(source_order)):
        raise ValueError("full source row-group order contains duplicates")
    positions = {group: index for index, group in enumerate(source_order)}
    try:
        selected = [positions[group] for group in fit_order]
    except KeyError as error:
        raise ValueError(
            f"fit row group is absent from the full source index: {error.args[0]}"
        ) from error
    if len(selected) != len(set(selected)):
        raise ValueError("fit-to-source mapping repeats a full-index position")
    return np.asarray(selected, dtype="<i8")


def positions_sha256(positions: np.ndarray) -> str:
    """Hash the explicit little-endian int64 mapping from target rows to source-bank rows."""
    if positions.ndim != 1 or positions.dtype != np.dtype("<i8"):
        raise ValueError("source positions must be a one-dimensional little-endian int64 array")
    return hashlib.sha256(positions.tobytes()).hexdigest()


def initialize_target(
    output: Path,
    family: str,
    project_root: Path,
    split_path: Path,
    old_cache: Path,
) -> dict[str, Any]:
    """Validate the qualified PR60 base and reserve a one-array streaming workspace."""
    if output.exists() or partial_root(output).exists():
        raise FileExistsError("stage-2 output or partial already exists; refusing to overwrite it")
    split = load_campaign_split(split_path, project_root)
    if split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA or split.sha256 != SPLIT_SHA256:
        raise ValueError("stage-2 targets require the frozen chronological primary split")
    source_order = _target_order(project_root, "train")
    validate_chronological_train_index_order(project_root, source_order, split, "train")
    fit_order = list(map(int, split.train_groups))
    if len(source_order) != SOURCE_TRAIN_INDEX_SEQUENCES:
        raise ValueError("verified full source train index differs from its 10,607 groups")
    if fit_order != list(split.train_groups) or len(fit_order) != TRAIN_SEQUENCES:
        raise ValueError("official chronological fit membership differs from its 10,325 groups")
    fit_positions = fit_source_positions(source_order, fit_order)
    validation = validate_distillation_cache(
        old_cache, project_root, split, "source", require_families=True
    )
    old_metadata = validation["metadata"]
    old_files = old_metadata["files"]
    if (
        validation["metadata_sha256"] != PR60_CACHE_METADATA_SHA256
        or old_files["main_source_train"]["sha256"] != PR60_MAIN_TRAIN_SHA256
        or old_files["families_train"]["sha256"] != PR60_FAMILIES_TRAIN_SHA256
    ):
        raise ValueError("stage-2 input is not the independently qualified PR60 mixed cache")
    for key in ("main_source_train", "families_train"):
        if old_files[key]["row_groups"] != fit_order:
            raise ValueError(f"qualified PR60 cache group order differs for {key}")
    output.parent.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "valid"):
        old_family_path = old_cache / old_files[f"families_{split_name}"]["path"]
        if old_family_path.stat().st_dev != output.parent.stat().st_dev:
            raise OSError("new cache and read-only PR60 family arrays must share a filesystem")
    usage = shutil.disk_usage(output.parent.resolve())
    transient_peak = TARGET_BYTES + SEQUENCE_BANK_MAX_BYTES
    if usage.free - transient_peak < 3_000_000_000:
        raise OSError(
            f"streaming needs {transient_peak} temporary bytes while preserving a 3 GB free floor"
        )
    partial = partial_root(output)
    partial.mkdir(parents=True)
    accumulator_path = partial / "clipped_sum.npy"
    accumulator = np.lib.format.open_memmap(
        accumulator_path,
        mode="w+",
        dtype=np.float32,
        shape=(TRAIN_SEQUENCES, SEQUENCE_ROWS, 2),
    )
    accumulator.fill(0.0)
    accumulator.flush()
    del accumulator
    state = {
        "schema": "wnn.l4-teacher-stage2-stream.v2",
        "family": family,
        "expected_runs": list(EXPECTED_RUNS[family]),
        "processed_runs": [],
        "teacher_records": [],
        "pending_run": None,
        "output_cache": str(output.resolve()),
        "partial_root": str(partial.resolve()),
        "accumulator": str(accumulator_path.resolve()),
        "project_root": str(project_root.resolve()),
        "split_manifest": str(split_path.resolve()),
        "split_manifest_sha256": split.sha256,
        "fit_row_groups": fit_order,
        "fit_row_groups_sha256": sequence_hash(np.asarray(fit_order, dtype=np.int64)),
        "source_train_index_row_groups": source_order,
        "source_train_index_sha256": sequence_hash(np.asarray(source_order, dtype=np.int64)),
        "fit_source_positions": fit_positions.tolist(),
        "fit_source_positions_sha256": positions_sha256(fit_positions),
        "old_cache": str(old_cache.resolve()),
        "old_cache_metadata_sha256": validation["metadata_sha256"],
        "old_main_sha256": old_files["main_source_train"]["sha256"],
        "old_metadata": old_metadata,
        "started_free_bytes": usage.free,
        "target_array_bytes": TARGET_BYTES,
        "incoming_bank_max_bytes": SEQUENCE_BANK_MAX_BYTES,
    }
    write_json(state_path(output), state)
    return state


def add_clipped_values(accumulator: np.ndarray, start: int, values: np.ndarray) -> None:
    """Add a bounded teacher block to the float32 accumulator in fixed source order."""
    if values.ndim != 3 or values.shape[1:] != (SEQUENCE_ROWS, 2):
        raise ValueError("teacher predictions must have shape [sequences,20000,2]")
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("teacher predictions must be finite float32")
    np.add(
        accumulator[start : start + len(values)],
        np.clip(values, np.float32(-2.0), np.float32(2.0)),
        out=accumulator[start : start + len(values)],
    )


def combine_target_chunk(old_target: np.ndarray, clipped_sum: np.ndarray, count: int) -> np.ndarray:
    """Apply the frozen old/new family equation to one small float32 block."""
    if count < 1 or old_target.shape != clipped_sum.shape or old_target.shape[-1] != 2:
        raise ValueError(
            "target blend needs matching two-column arrays and a positive teacher count"
        )
    if old_target.dtype != np.float32 or clipped_sum.dtype != np.float32:
        raise ValueError("target blend inputs must remain float32")
    if not np.isfinite(old_target).all() or not np.isfinite(clipped_sum).all():
        raise ValueError("target blend inputs must be finite")
    family_mean = clipped_sum / np.float32(count)
    scaled_new = np.float32(0.8125) * family_mean
    return np.float32(0.8) * old_target + np.float32(0.2) * scaled_new
