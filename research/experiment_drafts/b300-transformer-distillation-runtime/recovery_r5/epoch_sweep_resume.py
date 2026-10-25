"""Create or safely reopen one immutable per-fold epoch sweep directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from epoch_contract import read_object, safe_file, write_json_new
from epoch_sweep_receipts import completed_sweep, load_epoch_record


def prepare_sweep_output(
    output: Path,
    plan: dict[str, Any],
    fit: dict[str, Any],
    split_binding: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    """Open a fresh sweep or validate its plan and completed epochs for re-entry."""
    if output.is_symlink():
        raise ValueError("epoch sweep output cannot be a symlink")
    if not output.exists():
        if resume:
            raise FileNotFoundError(f"resume output directory does not exist: {output}")
        output.mkdir(parents=True, exist_ok=False)
        epochs_root = output / "epochs"
        epochs_root.mkdir()
        write_json_new(output / "plan.json", plan)
        return {"epochs_root": epochs_root, "records": [], "sweep": None}
    if not resume:
        raise FileExistsError(f"refusing to replace epoch sweep output: {output}")
    if not output.is_dir() or output.is_symlink():
        raise ValueError("resume output path must be a real directory")

    if not (output / "plan.json").exists():
        existing = list(output.iterdir())
        if existing and not (
            len(existing) == 1
            and existing[0].name == "epochs"
            and existing[0].is_dir()
            and not existing[0].is_symlink()
            and not any(existing[0].iterdir())
        ):
            raise ValueError("resume output without a plan already contains artifacts")
        epochs_root = output / "epochs"
        if epochs_root.exists() and (epochs_root.is_symlink() or not epochs_root.is_dir()):
            raise ValueError("resume epoch output path must be a directory")
        epochs_root.mkdir(exist_ok=True)
        write_json_new(output / "plan.json", plan)
        return {"epochs_root": epochs_root, "records": [], "sweep": None}

    plan_path = safe_file(output, "plan.json")
    saved_plan, raw_plan = read_object(plan_path)
    expected_bytes = (
        json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if saved_plan != plan or raw_plan != expected_bytes:
        raise ValueError("resume plan differs from the original source, fit, split, or index")
    epochs_root = output / "epochs"
    if epochs_root.is_symlink() or not epochs_root.is_dir():
        raise ValueError("resume epoch output path must be a directory")
    expected_names = {f"epoch-{number:06d}" for number in range(1, len(fit["entries"]) + 1)}
    if any(path.name not in expected_names for path in epochs_root.iterdir()):
        raise ValueError("resume output contains an unindexed epoch directory")

    records = []
    encountered_gap = False
    for entry in fit["entries"]:
        completed = entry["completed_epochs"]
        relative = f"epochs/epoch-{completed:06d}"
        epoch_path = output / relative
        if not epoch_path.exists() and not epoch_path.is_symlink():
            encountered_gap = True
            continue
        if encountered_gap:
            raise ValueError("resume output has a completed epoch after a missing earlier epoch")
        if epoch_path.is_symlink() or not epoch_path.is_dir():
            raise ValueError("resume epoch output must be a real directory")
        if not (epoch_path / "epoch.json").is_file():
            raise ValueError(
                f"epoch {completed} is partial; preserve it and resume into a fresh output directory"
            )
        records.append(load_epoch_record(output, entry, int(plan["fold"]), fit, split_binding))

    if len(records) > len(fit["entries"]):
        raise ValueError("resume output contains unindexed epoch receipts")
    sweep = completed_sweep(output, plan, records) if (output / "sweep.json").exists() else None
    if sweep is not None and len(records) != len(fit["entries"]):
        raise ValueError("completed sweep is missing verified epoch receipts")
    return {"epochs_root": epochs_root, "records": records, "sweep": sweep}
