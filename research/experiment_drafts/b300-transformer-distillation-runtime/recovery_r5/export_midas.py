"""Copy bounded R5 evidence into a portable MIDAS bundle without model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any  # Validated manifests contain heterogeneous JSON fields.

from epoch_selection import _load_fold
from recovery_runtime import get_recovery_cell, load_recovery, require_no_retention_intent

LIMIT = 64 << 20


def canonical(value: object) -> str:
    """Compute the normalized runtime identity digest."""
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def read(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read one bounded regular JSON object, preserving the exact source bytes."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > LIMIT:
        raise ValueError(f"unsafe, missing, or oversized evidence: {path}")
    raw = path.read_bytes()
    if len(raw) > LIMIT:
        raise ValueError("evidence grew beyond its size bound")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("expected an evidence object")
    return value, raw


def emit(root: Path, relative: str, raw: bytes) -> dict[str, str]:
    """Copy an evidence file once; never rewrite a prepared bundle."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()}


def encoded(value: object) -> bytes:
    """Serialize copied metadata without NaN or ambiguous key order."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def validate_evaluation_override(
    evaluation_output: Path,
    output: Path,
    fit: Path,
    cell: dict[str, Any],
    config: dict[str, Any],
    config_file_sha256: str,
    config_path: Path,
    split_sha256: str,
    split_path: Path,
    data: dict[str, Any],
    source_sha256: str,
) -> Path:
    """Require an explicit sweep directory to be complete and bound to this cell."""
    requested = evaluation_output.expanduser()
    if requested.is_symlink():
        raise ValueError("evaluation output must not be a symlink")
    directory = requested.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("evaluation output must be an existing directory")
    fit_directory = fit.resolve(strict=True)
    if directory == fit_directory or directory.is_relative_to(fit_directory):
        raise ValueError("evaluation output must not be inside the fit directory")
    fold, missing = _load_fold(directory, cell["fold"])
    if (
        fold is None
        or missing
        or fold.get("missing")
        or fold.get("sweep") is None
        or fold["sweep"].get("status") != "complete"
        or len(fold.get("observed", [])) != 30
    ):
        raise ValueError("evaluation override must contain one complete 30-epoch sweep")
    verified_fit = fold["fit"]
    identity = verified_fit["identity"]
    index, index_raw = read(fit / "epoch_index.json")
    binding = fold["split_binding"]
    if (
        Path(verified_fit["fit_dir"]).resolve(strict=True) != fit.resolve(strict=True)
        or Path(fold["plan"]["config_path"]).resolve(strict=True)
        != config_path.resolve(strict=True)
        or verified_fit["config_file_sha256"] != config_file_sha256
        or identity.get("config_sha256") != canonical(config)
        or identity.get("source_sha256") != source_sha256
        or identity.get("data") != data
        or Path(binding["inner_path"]).resolve(strict=True) != split_path.resolve(strict=True)
        or binding["inner_sha256"] != split_sha256
        or binding["fold"] != cell["fold"]
        or hashlib.sha256(index_raw).hexdigest() != verified_fit["index_file_sha256"]
        or index.get("index_sha256") != verified_fit["index_sha256"]
    ):
        raise ValueError("evaluation override belongs to a different fit, config, index, or split")
    destination = output.expanduser().resolve()
    if any(
        destination == protected or destination.is_relative_to(protected)
        for protected in (fit_directory, directory)
    ) or directory.is_relative_to(destination):
        raise ValueError("evaluation override and evidence bundle paths must not overlap")
    return directory


def export_bundle(
    campaign_root: Path,
    cell_id: str,
    output: Path,
    *,
    evaluation_output: Path | None = None,
) -> Path:
    """Export source-qualified progress; absence of a terminal receipt means partial."""
    campaign, _ = load_recovery(campaign_root)
    cell = get_recovery_cell(campaign_root, cell_id)
    require_no_retention_intent(campaign_root, cell, "export")
    fit = campaign_root / cell["run_dir"] / "fit"
    metadata, metadata_raw = read(fit / "run_metadata.json")
    config = metadata["config"]
    config_path = campaign_root / cell["config"]
    expected_config, expected_config_raw = read(config_path)
    expected_config["model"]["precision_mode"] = expected_config["precision_mode"]
    if config != expected_config:
        raise ValueError("fit metadata config does not match the declared R5 cell")
    split, split_raw = read(Path(cell["split_manifest"]))
    split_sha = hashlib.sha256(split_raw).hexdigest()
    data = metadata["provenance"]["data"]
    if (
        metadata["source"]["sha256"] != campaign["runtime_source_sha256"]
        or data["split_sha256"] != split_sha
        or split_sha != cell["split_sha256"]
        or data["fold"] != cell["fold"]
    ):
        raise ValueError("fit metadata source/split identity differs from the planned cell")
    evaluation = campaign_root / cell["run_dir"] / "epoch_evaluation"
    if evaluation_output is not None:
        evaluation = validate_evaluation_override(
            evaluation_output,
            output,
            fit,
            cell,
            config,
            hashlib.sha256(expected_config_raw).hexdigest(),
            config_path,
            split_sha,
            Path(cell["split_manifest"]),
            data,
            metadata["source"]["sha256"],
        )
    output.mkdir(parents=True, exist_ok=False)
    metadata_ref = emit(output, "run_metadata.json", metadata_raw)
    config_ref = emit(output, "config.json", encoded(config))
    split_ref = emit(output, "split.json", split_raw) | {"fold": split["fold"]}
    checkpoints: list[dict[str, Any]] = []
    index = None
    if (fit / "epoch_index.json").exists():
        index, index_raw = read(fit / "epoch_index.json")
        body = {key: value for key, value in index.items() if key != "index_sha256"}
        if (
            canonical(body) != index.get("index_sha256")
            or index.get("schema") != "wnn.sequence-epoch-index.v1"
            or index["identity"]["config_sha256"] != canonical(config)
            or index["identity"]["data"] != data
            or index["identity"]["source_sha256"] != metadata["source"]["sha256"]
        ):
            raise ValueError("safe epoch index identity mismatch")
        emit(output, "epoch_index.json", index_raw)
        for entry in index["entries"]:
            checkpoints.append(
                {
                    "epoch": entry["completed_epochs"],
                    "path": None,
                    "sha256": entry["checkpoint_sha256"],
                    "bytes": None,
                    "parent_sha256": None,
                    "reference": entry["checkpoint_path"],
                }
            )
    frontier = index["completed_epochs"] if index else 0
    metrics = []
    log = fit / "metrics.jsonl"
    if log.exists():
        if log.is_symlink() or log.stat().st_size > LIMIT:
            raise ValueError("training metrics exceed the bounded snapshot size")
        raw_log = log.read_bytes()
        # Ignore a writer's incomplete last line and metrics beyond committed safe weights.
        for line in raw_log.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                continue
            row = json.loads(line)
            epoch = row.get("epoch")
            if type(epoch) is not int or not 1 <= epoch <= frontier:
                continue
            ref = emit(output, f"metrics/epoch-{epoch:06d}-train.json", line)
            metrics.append(
                {
                    "epoch": epoch,
                    "name": "loss",
                    "value": row["loss"],
                    "scope": "training",
                    "split_sha256": split_sha,
                    "checkpoint_sha256": None,
                    "producer_path": ref["path"],
                    "producer_sha256": ref["sha256"],
                }
            )
    for checkpoint in checkpoints:
        epoch = checkpoint["epoch"]
        grade = evaluation / "epochs" / f"epoch-{epoch:06d}" / "score.json"
        if not grade.exists():
            continue
        score, raw = read(grade)
        producer = score["producer"]
        if (
            score["split_sha256"] != split_sha
            or score["score"].get("metric_id") != "global_wp_clip2_v1"
            or producer["checkpoint_sha256"] != checkpoint["sha256"]
            or producer.get("config") != config
            or producer.get("kind") != "retained_epoch"
        ):
            raise ValueError("epoch score does not belong to this run/checkpoint")
        ref = emit(output, f"metrics/epoch-{epoch:06d}-score.json", raw)
        metrics.append(
            {
                "epoch": epoch,
                "name": "global_wp_clip2_v1.wp",
                "value": score["score"]["wp"],
                "scope": "inner_validation",
                "split_sha256": split_sha,
                "checkpoint_sha256": checkpoint["sha256"],
                "producer_path": ref["path"],
                "producer_sha256": ref["sha256"],
            }
        )
    terminal_ref, status = None, "partial"
    if (fit / "terminal_status.json").exists():
        terminal, raw = read(fit / "terminal_status.json")
        if (
            terminal.get("status") != "completed"
            or terminal.get("completed_epochs") != config["epochs"]
            or terminal["identity"]["config_sha256"] != canonical(config)
            or terminal["identity"]["source_sha256"] != metadata["source"]["sha256"]
            or terminal["identity"]["data"] != data
        ):
            raise ValueError("terminal fit receipt does not match this run")
        terminal_ref = emit(output, "terminal_status.json", raw)
        status = "completed"
    receipt = {
        "schema": "midas.external-training-receipt.v1",
        "provider": "b300-team",
        "external_run_id": f"{campaign['campaign_sha256']}:{cell_id}",
        "campaign": {"id": campaign["campaign_sha256"], "stage": "R5"},
        "experiment": {
            "cell_id": cell_id,
            "variant": cell["variant"],
            "fold": cell["fold"],
            "seed": cell["seed"],
        },
        "source": {
            "repository": "VladKochetov007/WunderFunder",
            "commit": None,
            "runtime_source_sha256": metadata["source"]["sha256"],
        },
        "run_metadata": metadata_ref,
        "terminal_status": terminal_ref,
        "config": config_ref,
        "split": split_ref,
        "execution": {"status": status, "started_at": None, "finished_at": None},
        "metrics": metrics,
        "checkpoints": checkpoints,
    }
    return output / emit(output, "receipt.json", encoded(receipt))["path"]


def main() -> None:
    """Write one metadata bundle; no SSH, model loading, inference, or MIDAS mutation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path)
    args = parser.parse_args()
    result = export_bundle(
        args.campaign_root.resolve(),
        args.cell,
        args.output.resolve(),
        evaluation_output=args.evaluation_output,
    )
    print(result)


if __name__ == "__main__":
    main()
