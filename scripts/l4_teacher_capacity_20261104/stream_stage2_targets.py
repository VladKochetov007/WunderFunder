"""Build one frozen student target from one-at-a-time train-only teacher exports."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.l4_teacher_capacity_20261104.stage2_stream_common import (
    EXPECTED_RUNS,
    add_clipped_values,
    initialize_target,
    load_state,
    partial_root,
    write_json,
)
from scripts.l4_teacher_capacity_20261104.stage2_stream_finalize import finalize_target
from scripts.l4_teacher_capacity_20261104.stage2_stream_native import verify_native_gru
from scripts.l4_teacher_capacity_20261104.stage2_stream_teachers import (
    verify_sequence_teacher,
)
from wnn.distillation_cache import SEQUENCE_ROWS, file_sha256
from wnn.splits import load_campaign_split


def _add_teacher(
    output: Path,
    family: str,
    run: str,
    sequence_export: Path | None,
    teacher_panel: Path | None,
) -> None:
    """Validate, stream, record, and discard exactly one owned fitting-only bank."""
    state = load_state(output, family)
    expected_runs = state["expected_runs"]
    if state["processed_runs"] == expected_runs:
        raise ValueError("all teachers for this family are already aggregated")
    expected_run = expected_runs[len(state["processed_runs"])]
    if run != expected_run:
        raise ValueError(f"expected next fixed-order teacher {expected_run}, got {run}")
    project_root = Path(state["project_root"])
    split = load_campaign_split(Path(state["split_manifest"]), project_root)
    if split.sha256 != state["split_manifest_sha256"]:
        raise ValueError("chronological split changed after accumulator initialization")
    groups = list(map(int, state["fit_row_groups"]))
    accumulator = np.lib.format.open_memmap(state["accumulator"], mode="r+")
    if family == "wide_gru":
        if sequence_export is not None or teacher_panel is not None:
            raise ValueError("wide-GRU inputs use only the canonical native train prediction")
        record, values = verify_native_gru(run, split, project_root, groups)
        prediction_path = Path(record["prediction_path"])
        cache_root = (project_root / "cache").resolve()
        if prediction_path.is_symlink() or not prediction_path.resolve().is_relative_to(cache_root):
            raise ValueError("native prediction bank escaped the owned project cache")
        state["pending_run"] = run
        write_json(partial_root(output) / "stage2_state.json", state)
        for start in range(0, len(groups), 8):
            stop = min(start + 8, len(groups))
            add_clipped_values(accumulator, start, np.asarray(values[start:stop], dtype=np.float32))
        del values
        bank_paths = [prediction_path]
    else:
        if sequence_export is None or teacher_panel is None:
            raise ValueError("Transformer aggregation requires its run export and frozen panel")
        expected_root = project_root / "cache" / "l4_teacher_sequence_exports" / run
        if sequence_export.is_symlink() or sequence_export.resolve() != expected_root.resolve():
            raise ValueError("sequence export must use its exact run-scoped owned directory")
        record, entries = verify_sequence_teacher(
            run, split, project_root, sequence_export, teacher_panel
        )
        state["pending_run"] = run
        write_json(partial_root(output) / "stage2_state.json", state)
        bank_paths = []
        for index, group in enumerate(groups):
            entry = entries[group]
            filename = entry.get("file")
            if not isinstance(filename, str) or filename != f"train-{group}.npy":
                raise ValueError("sequence export contains an unexpected member filename")
            prediction_path = sequence_export / filename
            if prediction_path.is_symlink() or not prediction_path.resolve().is_relative_to(
                sequence_export.resolve()
            ):
                raise ValueError("sequence export file escaped its run-owned directory")
            if file_sha256(prediction_path) != entry.get("sha256"):
                raise ValueError(f"sequence export hash differs for {run}/{group}")
            values = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
            if values.shape != (SEQUENCE_ROWS, 2) or values.dtype != np.float32:
                raise ValueError(f"sequence prediction shape/dtype differs for {run}/{group}")
            add_clipped_values(accumulator, index, np.asarray(values[None], dtype=np.float32))
            del values
            bank_paths.append(prediction_path)
        record["prediction_files"] = len(groups)
        record["prediction_files_sha256"] = hashlib.sha256(
            "".join(entries[group]["sha256"] for group in groups).encode()
        ).hexdigest()
    accumulator.flush()
    del accumulator
    record["bank_discarded_after_aggregation"] = False
    state["teacher_records"].append(record)
    state["processed_runs"].append(run)
    state["pending_run"] = None
    write_json(partial_root(output) / "stage2_state.json", state)
    for path in bank_paths:
        if path.is_file():
            path.unlink()
    record["bank_discarded_after_aggregation"] = True
    write_json(partial_root(output) / "stage2_state.json", state)
    print(
        f"aggregated {run}; fixed family order progress {len(state['processed_runs'])}/{len(expected_runs)}"
    )


def main() -> None:
    """Initialize, incrementally aggregate, or finalize one frozen student target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "add", "finalize"))
    parser.add_argument("--family", choices=tuple(EXPECTED_RUNS), required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--old-cache", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--sequence-export", type=Path)
    parser.add_argument("--teacher-panel", type=Path)
    args = parser.parse_args()
    output = args.output_cache.resolve()
    if args.action == "init":
        if args.project_root is None or args.split_manifest is None or args.old_cache is None:
            parser.error("init requires --project-root, --split-manifest, and --old-cache")
        state = initialize_target(
            output,
            args.family,
            args.project_root.resolve(),
            args.split_manifest.resolve(),
            args.old_cache.resolve(),
        )
        print(f"initialized {args.family} target stream at {state['partial_root']}")
    elif args.action == "add":
        if args.run is None:
            parser.error("add requires --run")
        _add_teacher(output, args.family, args.run, args.sequence_export, args.teacher_panel)
    else:
        finalize_target(output, args.family)


if __name__ == "__main__":
    main()
