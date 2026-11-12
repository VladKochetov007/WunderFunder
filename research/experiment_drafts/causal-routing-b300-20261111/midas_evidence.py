"""Export Signal Routes fit evidence as a MIDAS receipt; never load models or copy weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from midas_evidence_io import _copy, _write
from midas_evidence_prepare import METRIC_LIMIT, PreparedFit, prepare_fit

CAMPAIGN = "signal-routes-20261111"
STAGE = "B"  # Signal Routes initial-screen adapter; not historical R4-B.
REPOSITORY = "VladKochetov007/WunderFunder"
EVENT_LIMIT = 1 << 30
EVALUATION_LIMIT = 10_000 * 32_768
PROGRESS_LIMIT = 64 << 20
RANK_LIMIT = 256 << 20
BUNDLE_LIMIT = 2 << 30


def export_bundle(fit: Path, score: Path | None, output: Path) -> Path:
    """Validate one fit, then create a new portable MIDAS evidence bundle."""
    prepared = prepare_fit(fit, score)
    return _assemble_bundle(prepared, output)


def _assemble_bundle(prepared: PreparedFit, output: Path) -> Path:
    """Write the admitted files and strict v1 receipt into a new directory."""
    fit = prepared.fit
    meta = prepared.meta
    meta_raw = prepared.meta_raw
    config = prepared.config
    expected = prepared.expected
    owner_raw = prepared.owner_raw
    cell = prepared.cell
    fold = prepared.fold
    commit = prepared.commit
    split_raw = prepared.split_raw
    split_digest = prepared.split_digest
    epochs = prepared.epochs
    model_sha = prepared.model_sha
    model_bytes = prepared.model_bytes
    terminal_raw = prepared.terminal_raw
    status = prepared.status
    index_raw = prepared.index_raw
    train_rows = prepared.train_rows
    score_obj = prepared.score_obj
    score_raw = prepared.score_raw
    external_id = prepared.external_id

    dest = Path(output).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest = dest.parent.resolve(strict=True) / dest.name
    if dest.exists() or dest.is_symlink() or dest.is_relative_to(fit):
        raise ValueError("output must be a new directory outside the fit tree")

    stage = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    files: dict[str, dict[str, Any]] = {}
    try:
        meta_ref = _write(stage, "run_metadata.json", meta_raw, files)
        config_raw = (
            json.dumps(
                config, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            ).encode()
            + b"\n"
        )
        config_ref = _write(stage, "config.json", config_raw, files)
        split_ref = _write(stage, "split.json", split_raw, files) | {"fold": fold}
        term_ref = (
            _write(stage, "terminal_status.json", terminal_raw, files)
            if status == "completed" and terminal_raw
            else None
        )
        if terminal_raw and status != "completed":
            _write(stage, "terminal_observation.json", terminal_raw, files)
        if index_raw:
            _write(stage, "epoch_index.json", index_raw, files)
        score_ref = _write(stage, "score.json", score_raw, files) if score_raw else None
        for epoch, _, line in train_rows:
            _write(stage, f"producers/train-epoch-{epoch:06d}.json", line, files)

        metrics = [
            {
                "epoch": epoch,
                "name": "loss",
                "value": loss,
                "scope": "training",
                "split_sha256": split_digest,
                "producer_path": f"producers/train-epoch-{epoch:06d}.json",
                "producer_sha256": hashlib.sha256(line).hexdigest(),
                "checkpoint_sha256": None,
            }
            for epoch, loss, line in train_rows
        ]
        checkpoints = []
        if status == "completed" and model_sha:
            checkpoints.append(
                {
                    "epoch": epochs,
                    "path": None,
                    "reference": "model.pt",
                    "sha256": model_sha,
                    "bytes": model_bytes,
                    "parent_sha256": None,
                }
            )
        if score_obj is not None and score_ref is not None:
            metrics.append(
                {
                    "epoch": epochs,
                    "name": "global_wp_clip2_v1.wp",
                    "value": score_obj["score"]["wp"],
                    "scope": "outer_validation",
                    "split_sha256": split_digest,
                    "producer_path": score_ref["path"],
                    "producer_sha256": score_ref["sha256"],
                    "checkpoint_sha256": model_sha,
                }
            )

        receipt = {
            "schema": "midas.external-training-receipt.v1",
            "provider": "b300-team",
            "external_run_id": external_id,
            "campaign": {"id": CAMPAIGN, "stage": STAGE},
            "experiment": {
                "cell_id": cell["id"],
                "variant": cell["variant"],
                "fold": fold,
                "seed": cell["seed"],
            },
            "source": {
                "repository": REPOSITORY,
                "commit": commit,
                "runtime_source_sha256": meta["source"]["sha256"],
            },
            "run_metadata": meta_ref,
            "config": config_ref,
            "split": split_ref,
            "execution": {"status": status, "started_at": None, "finished_at": None},
            "terminal_status": term_ref,
            "metrics": metrics,
            "checkpoints": checkpoints,
        }
        receipt_raw = (
            json.dumps(
                receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            ).encode()
            + b"\n"
        )
        receipt_ref = _write(stage, "receipt.json", receipt_raw, files)
        logs = [
            ("metrics.jsonl", METRIC_LIMIT),
            ("monitor_events.jsonl", EVENT_LIMIT),
            ("monitor_evaluations.jsonl", EVALUATION_LIMIT),
            ("monitor_progress.json", PROGRESS_LIMIT),
        ]
        logs.extend((f"rank{rank}.jsonl", RANK_LIMIT) for rank in range(expected["world_size"]))
        for name, limit in logs:
            source = fit / name
            if source.exists():
                _copy(stage, source, f"logs/{name}", limit, files)
        required_logs = ("monitor_events.jsonl", "monitor_evaluations.jsonl")
        if status == "completed" and not all((fit / name).is_file() for name in required_logs):
            raise ValueError("completed fit lacks PR75 monitor event/evaluation logs")
        if sum(item["bytes"] for item in files.values()) > BUNDLE_LIMIT:
            raise ValueError("evidence bundle exceeds aggregate size bound")

        manifest = {
            "schema": "signal_routes.midas-evidence-bundle.v1",
            "campaign_stage_mapping": "B is the Signal Routes initial screen; it is not historical R4-B",
            "cell": cell,
            "scope": "outer_validation" if score_obj else None,
            "receipt_sha256": receipt_ref["sha256"],
            "model_sha256": model_sha,
            "owner_receipt_sha256": hashlib.sha256(owner_raw).hexdigest(),
            "weights_copied": False,
            "dataset_bytes_copied": False,
            "files": dict(sorted(files.items())),
        }
        manifest_raw = (
            json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            ).encode()
            + b"\n"
        )
        _write(stage, "bundle-manifest.json", manifest_raw, files)
        os.rename(stage, dest)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return dest / "receipt.json"


def main() -> None:
    """Create a portable MIDAS receipt bundle from a previously produced fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--score", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_bundle(args.fit, args.score, args.output))


if __name__ == "__main__":
    main()
