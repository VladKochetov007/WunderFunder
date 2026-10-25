"""Compact deterministic fixtures for prepared R5 receiver tests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PACKET = (
    Path(__file__).resolve().parents[1]
    / "research/experiment_drafts/b300-transformer-distillation-runtime/recovery_r5"
)
sys.path.insert(0, str(PACKET))

from epoch_contract import (  # noqa: E402
    EXPECTED_EPOCHS,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    validate_split_pair,
    write_json_new,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _make_splits(root: Path, fold: int) -> tuple[Path, Path, dict[str, Any]]:
    rows = {
        "train": [{"row_group": i, "component": 100 + i} for i in range(4)],
        "valid": [{"row_group": 10 + i, "component": 100 + i} for i in range(4)]
        + [{"row_group": 14, "component": 999}],
    }
    sources = {
        name: {"sha256": hashlib.sha256(name.encode()).hexdigest()} for name in ("train", "valid")
    }
    outer = {
        "schema": "wnn.sequence-campaign-split.v1",
        "version": 1,
        "fold": fold,
        "n_folds": 5,
        "sources": sources,
        "sequences": rows,
        "fit": {"train": [0, 1, 2, 3], "valid": [10, 11, 12, 13]},
        "evaluation": {"valid": [14]},
    }
    outer_path = root / f"outer-f{fold}.json"
    _write_json(outer_path, outer)
    seed = 41001 + (fold == 2)
    order = sorted(
        range(100, 104),
        key=lambda component: hashlib.sha256(
            f"wnn.sequence-inner.v1:{seed}:{fold}:{component}".encode()
        ).hexdigest(),
    )
    chosen = order[0]
    held_out = chosen - 90
    inner = {
        "schema": "wnn.sequence-inner-split.v1",
        "version": 1,
        "fold": fold,
        "n_folds": 5,
        "parent": {"path": str(outer_path.resolve()), "sha256": file_sha256(outer_path)},
        "selection": {
            "algorithm": "sha256-component-order-v1",
            "seed": seed,
            "evaluation_fraction": 0.2,
        },
        "sources": sources,
        "sequences": rows,
        "fit": {
            "train": [i for i in range(4) if 100 + i != chosen],
            "valid": [10 + i for i in range(4) if 100 + i != chosen],
        },
        "evaluation": {"valid": [held_out]},
    }
    inner_path = root / f"inner-f{fold}.json"
    _write_json(inner_path, inner)
    binding = validate_split_pair(outer_path, inner_path, fold)
    return outer_path, inner_path, binding


def _make_fit(
    root: Path, fold: int, *, seed: int | None = None, lr: float = 1e-4
) -> dict[str, Any]:
    outer, inner, binding = _make_splits(root, fold)
    config = {
        "epochs": 30,
        "expected_fold": fold,
        "seed": seed if seed is not None else 201001 + (fold == 2),
        "lr": lr,
        "checkpoint_retention": "all_epochs",
        "precision_mode": "bf16_bf16x9",
        "model": {
            "architecture": "transformer",
            "attention_backend": "efficient",
            "compile_blocks": True,
        },
    }
    config_path = root / f"config-f{fold}.json"
    _write_json(config_path, config)
    normalized = dict(config) | {
        "model": dict(config["model"]) | {"precision_mode": config["precision_mode"]}
    }
    identity = {
        "schema": "wnn.sequence-run-identity.v1",
        "config_sha256": canonical_sha256(normalized),
        "source_sha256": "a" * 64,
        "data": {
            "split_sha256": binding["inner_sha256"],
            "fold": fold,
            "fit_members": binding["fit_members"],
        },
        "runtime": {
            "attention_backend": "efficient",
            "precision": {"mode": "bf16_bf16x9", "attention_backend": "efficient"},
            "compiler": {"enabled": True, "backend": "inductor"},
        },
    }
    fit_dir = root / f"fit-f{fold}"
    fit_dir.mkdir()
    (fit_dir / "epoch_checkpoints").mkdir()
    (fit_dir / "epoch_indexes").mkdir()
    entries = []
    for completed in range(1, EXPECTED_EPOCHS + 1):
        content = f"checkpoint-{fold}-{completed}".encode()
        digest = hashlib.sha256(content).hexdigest()
        checkpoint = fit_dir / "epoch_checkpoints" / f"epoch-{completed:06d}-{digest}.pt"
        checkpoint.write_bytes(content)
        entries.append(
            {
                "epoch": completed - 1,
                "completed_epochs": completed,
                "global_step": completed,
                "checkpoint_path": str(checkpoint.relative_to(fit_dir)),
                "checkpoint_sha256": file_sha256(checkpoint),
            }
        )
    previous_hash = None
    index = None
    for completed in range(1, EXPECTED_EPOCHS + 1):
        unsigned = {
            "schema": "wnn.sequence-epoch-index.v1",
            "identity": identity,
            "completed_epochs": completed,
            "global_step": completed,
            "previous_index_sha256": previous_hash,
            "entries": entries[:completed],
        }
        index = unsigned | {"index_sha256": canonical_sha256(unsigned)}
        raw = canonical_bytes(index) + b"\n"
        (fit_dir / "epoch_indexes" / f"epoch-{completed:06d}.json").write_bytes(raw)
        previous_hash = index["index_sha256"]
    assert index is not None
    latest = fit_dir / "epoch_index.json"
    latest.write_bytes(canonical_bytes(index) + b"\n")
    final_model = fit_dir / "model.pt"
    final_model.write_bytes(b"fixed-final-model")
    _write_json(
        fit_dir / "terminal_status.json",
        {
            "status": "completed",
            "completed_epochs": 30,
            "identity": identity,
            "model_sha256": file_sha256(final_model),
        },
    )
    return {
        "outer": outer,
        "inner": inner,
        "binding": binding,
        "config": config,
        "config_path": config_path,
        "fit_dir": fit_dir,
        "index_path": latest,
        "index": index,
        "entries": entries,
    }


def _make_sweep(root: Path, fit: dict[str, Any], scores: list[float]) -> Path:
    fold = fit["binding"]["fold"]
    result = root / f"sweep-{fit['fit_dir'].name}"
    epoch_root = result / "epochs"
    epoch_root.mkdir(parents=True)
    normalized = dict(fit["config"]) | {
        "model": dict(fit["config"]["model"]) | {"precision_mode": "bf16_bf16x9"}
    }
    plan = {
        "schema": "wnn.recovery-r5-epoch-sweep.v1",
        "status": "planned",
        "fold": fold,
        "expected_epochs": 30,
        "fit_dir": str(fit["fit_dir"]),
        "config_path": str(fit["config_path"]),
        "config_file_sha256": file_sha256(fit["config_path"]),
        "config_recipe_sha256": canonical_sha256(normalized),
        "config_identity_sha256": fit["index"]["identity"]["config_sha256"],
        "data_root": str(root),
        "outer_split": str(fit["outer"].resolve()),
        "outer_split_sha256": fit["binding"]["outer_sha256"],
        "inner_split": str(fit["inner"].resolve()),
        "inner_split_sha256": fit["binding"]["inner_sha256"],
        "epoch_index": str(fit["index_path"]),
        "epoch_index_sha256": fit["index"]["index_sha256"],
        "epoch_index_file_sha256": file_sha256(fit["index_path"]),
        "final_model_sha256": file_sha256(fit["fit_dir"] / "model.pt"),
        "identity": fit["index"]["identity"],
        "runner_sha256": "b" * 64,
        "nproc_per_node": 8,
    }
    write_json_new(result / "plan.json", plan)
    records = []
    for entry, wp in zip(fit["entries"], scores, strict=True):
        completed = entry["completed_epochs"]
        epoch_dir = epoch_root / f"epoch-{completed:06d}"
        predictions = epoch_dir / "predictions"
        predictions.mkdir(parents=True)
        producer = {
            "kind": "retained_epoch",
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "split_sha256": fit["binding"]["inner_sha256"],
            "epoch_admission": {
                "index_sha256": fit["index"]["index_sha256"],
                "index": fit["index"],
                "entry": entry,
            },
        }
        sequence_member = fit["binding"]["evaluation_members"][0]
        sequence_name = f"{sequence_member[0]}-{sequence_member[1]}.npy"
        sequence_path = predictions / sequence_name
        sequence_path.write_bytes(f"predictions-{fold}-{completed}".encode())
        prediction = {
            "schema": "sequence-predictions-v1",
            "complete": True,
            "membership": "evaluation",
            "split_sha256": fit["binding"]["inner_sha256"],
            "producer": producer,
            "sequences": [
                {
                    "split": sequence_member[0],
                    "row_group": sequence_member[1],
                    "file": sequence_name,
                    "sha256": file_sha256(sequence_path),
                }
            ],
        }
        _write_json(predictions / "manifest.json", prediction)
        prediction_hash = file_sha256(predictions / "manifest.json")
        metric = {"metric_id": "official-wp-v1", "selected_rows": 20, "wp": wp, "t0": wp, "t1": wp}
        score = {
            "score": metric,
            "split_sha256": fit["binding"]["inner_sha256"],
            "predictions_manifest_sha256": prediction_hash,
            "producer": producer,
        }
        _write_json(epoch_dir / "score.json", score)
        score_hash = file_sha256(epoch_dir / "score.json")
        receipt = {
            "schema": "wnn.recovery-r5-epoch-score.v1",
            "fold": fold,
            "completed_epochs": completed,
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "config_identity_sha256": fit["index"]["identity"]["config_sha256"],
            "outer_split_sha256": fit["binding"]["outer_sha256"],
            "inner_split_sha256": fit["binding"]["inner_sha256"],
            "epoch_index_sha256": fit["index"]["index_sha256"],
            "prediction_manifest_sha256": prediction_hash,
            "score_sha256": score_hash,
            "score": metric,
        }
        _write_json(epoch_dir / "epoch.json", receipt)
        records.append(
            {
                "epoch": entry["epoch"],
                "completed_epochs": completed,
                "checkpoint_path": entry["checkpoint_path"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "prediction_manifest_sha256": prediction_hash,
                "score_sha256": score_hash,
                "score": metric,
                "receipt_sha256": file_sha256(epoch_dir / "epoch.json"),
            }
        )
    unsigned = {
        "schema": "wnn.recovery-r5-epoch-sweep.v1",
        "status": "complete",
        "fold": fold,
        "expected_epochs": 30,
        "plan_sha256": file_sha256(result / "plan.json"),
        "runner_sha256": plan["runner_sha256"],
        "epoch_index_sha256": fit["index"]["index_sha256"],
        "config_identity_sha256": fit["index"]["identity"]["config_sha256"],
        "outer_split_sha256": fit["binding"]["outer_sha256"],
        "inner_split_sha256": fit["binding"]["inner_sha256"],
        "epochs": records,
    }
    _write_json(result / "sweep.json", unsigned | {"sweep_sha256": canonical_sha256(unsigned)})
    return result
