"""Validate fixed-epoch trainer provenance and checkpoint artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adapter_common import _CODE_DIR, _json, _sha256


def extract_fit_receipt(
    run_dir: Path,
    output_dir: Path,
    *,
    run_name: str,
    arm: str,
    fold: int,
    seed: int,
    input_receipt: dict[str, Any],
    adapter_bundle_identity: dict[str, Any],
) -> dict[str, Any]:
    """Verify and hash the fixed six-epoch outputs and their training ancestry."""
    provenance_path = output_dir / "provenance.json"
    history_path = output_dir / "history.json"
    provenance = _json(provenance_path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or len(history) != 6:
        raise ValueError("fit did not produce six epoch records")
    args = provenance.get("args")
    expected_shape = {"wide": (512, 1), "deep": (384, 2)}[arm]
    if not isinstance(args, dict) or any(
        args.get(key) != value
        for key, value in {
            "name": run_name,
            "valid_fold": fold,
            "seed": seed,
            "epochs": 6,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 5,
            "no_eval_during_fit": True,
            "precision": "fp32",
            "hidden": expected_shape[0],
            "layers": expected_shape[1],
            "batch": 8,
            "chunk": 2000,
            "loss": "wmse",
            "loss_norm": "chunk",
            "batch_order": "seeded",
            "sched": "onecycle",
            "lr": 0.0001,
            "beta2": 0.999,
            "wd": 0.01,
            "head_lr_mult": 1.0,
            "clip": 1.0,
            "sync_free": True,
            "fused_adam": True,
            "deterministic": True,
        }.items()
    ):
        raise ValueError("saved training provenance disagrees with the frozen recipe")
    if args.get("init") or args.get("propensity") or args.get("prop_input"):
        raise ValueError("labels-only arm unexpectedly acquired a model or propensity ancestor")
    if args.get("distill_cache_root") or args.get("distill"):
        raise ValueError("labels-only arm unexpectedly enabled distillation")
    if provenance.get("split_manifest_sha256") != input_receipt["split_manifest_sha256"]:
        raise ValueError("saved training provenance names a different split manifest")
    if provenance.get("selected_epoch") != 5:
        raise ValueError("trainer did not select the preregistered fixed epoch 5")
    trainer_identity = provenance.get("trainer_implementation")
    expected_identity = input_receipt.get("trainer_source_identity", {})
    if (
        not isinstance(trainer_identity, dict)
        or trainer_identity.get("sha256") != expected_identity.get("source_identity_sha256")
        or trainer_identity.get("files") != expected_identity.get("files")
    ):
        raise ValueError("saved trainer provenance differs from the portable a266 source manifest")
    fit = provenance.get("fit_row_groups")
    if not isinstance(fit, dict):
        raise ValueError("saved training provenance lacks fit membership")
    if set(fit.get("train", ())) != set(input_receipt.get("fit_train_groups", ())):
        raise ValueError("trainer fit train membership differs from the frozen split")
    if set(fit.get("valid", ())) != set(input_receipt.get("fit_valid_groups", ())):
        raise ValueError("trainer fit validation membership differs from the frozen split")
    checkpoints = [f"model_ep{epoch}.pt" for epoch in range(6)] + ["model.pt"]
    required = checkpoints + ["history.json", "provenance.json", "training_state.pt"]
    files: dict[str, dict[str, Any]] = {}
    for name in required:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"required trainer output missing or empty: {path}")
        files[path.relative_to(run_dir).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    epoch_updates = []
    for record in history:
        diagnostics = record.get("update_diagnostics", {})
        epoch_updates.append(diagnostics.get("epoch", {}))
        if record.get("evaluation") != "withheld":
            raise ValueError("fit history contains an evaluation result")
    return {
        "schema": "tabdata-b300-fit-producer.v1",
        "status": "completed_fixed_epoch5",
        "run_name": run_name,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "selected_epoch": provenance.get("selected_epoch"),
        "checkpoint_count": len(checkpoints),
        "checkpoint_files": checkpoints,
        "fit_group_counts": {key: len(value) for key, value in fit.items()},
        "epoch_update_counts": epoch_updates,
        "trainer_source_identity": provenance.get("trainer_implementation"),
        "adapter_sha256": _sha256(_CODE_DIR / "train_cell.py"),
        "adapter_bundle_identity": adapter_bundle_identity,
        "input_ancestry": input_receipt,
        "outputs": files,
    }
