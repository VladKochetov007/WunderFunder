"""Record immutable failed attempts and validate their evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from campaign_io import read_object, require, sha256_file, write_bytes_new, write_json_new
from endpoints import run_path


def failure_path(campaign_root: Path, record: dict[str, Any]) -> Path:
    """Return the unique planned receipt path for a failed attempt."""
    return run_path(campaign_root.resolve(), record, "failure.json")


def validate_failure_receipt(
    campaign_root: Path,
    campaign: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Validate a terminal failure receipt and every external log/evidence hash."""
    root = campaign_root.resolve(strict=True)
    path = failure_path(root, record)
    receipt = read_object(path)
    sidecar = path.with_suffix(".sha256")
    require(
        sidecar.read_text(encoding="utf-8").strip() == sha256_file(path),
        "failure receipt hash mismatch",
    )
    require(
        receipt.get("schema") == "wnn.high-compute-r4-cell-failure.v1",
        "unsupported failure receipt",
    )
    require(
        receipt.get("status") == "failed",
        f"failure receipt is not terminally failed: {record['id']}",
    )
    require(
        receipt.get("phase")
        in (
            "profile_one",
            "profile_eight",
            "profile_peak",
            "admission",
            "fit",
            "evaluate",
        ),
        "failure phase is invalid",
    )
    require(
        receipt.get("campaign_sha256") == campaign["campaign_sha256"],
        f"failure campaign changed: {record['id']}",
    )
    require(
        receipt.get("cell_id") == record["id"]
        and receipt.get("stage") == record.get("failure_origin_stage", record["stage"]),
        f"failure belongs to another cell: {record['id']}",
    )
    require(
        receipt.get("source_sha256") == campaign["runtime_source_sha256"],
        f"failure source changed: {record['id']}",
    )
    require(
        receipt.get("config_sha256") == record["config_sha256"],
        f"failure config changed: {record['id']}",
    )
    require(
        receipt.get("split_sha256") == record["split_sha256"],
        f"failure split changed: {record['id']}",
    )
    evidence = receipt.get("evidence")
    require(isinstance(evidence, list) and evidence, f"failure evidence is missing: {record['id']}")
    for item in evidence:
        evidence_path = Path(item["path"]).resolve(strict=True)
        require(
            evidence_path.is_file() and sha256_file(evidence_path) == item["sha256"],
            f"failure evidence changed: {item['path']}",
        )
    status_binding = receipt.get("terminal_status")
    if status_binding is not None:
        status_file = (root / status_binding["path"]).resolve(strict=True)
        require(status_file.is_relative_to(root), "terminal failure status escapes campaign")
        require(
            sha256_file(status_file) == status_binding["sha256"], "terminal failure status changed"
        )
        require(
            read_object(status_file).get("status") == "failed", "failure status file is not failed"
        )
    return {
        "state": "failed",
        "id": record["id"],
        "variant": record["variant"],
        "fold": int(record["fold"]),
        "seed": int(record["seed"]),
        "reason": receipt["reason"],
        "failure_sha256": sha256_file(path),
        "evidence": evidence,
    }


def record_failure(
    campaign_root: Path,
    campaign: dict[str, Any],
    record: dict[str, Any],
    reason: str,
    evidence_paths: list[Path],
    phase: str = "fit",
) -> dict[str, Any]:
    """Publish one immutable profile, fit, or evaluation failure receipt with its logs."""
    root = campaign_root.resolve(strict=True)
    require(not record.get("reused"), "cannot record a new failure for an exact reused endpoint")
    require(bool(reason.strip()), "failure reason must be nonempty")
    require(evidence_paths, "at least one failure log or evidence file is required")
    require(
        phase
        in (
            "profile_one",
            "profile_eight",
            "profile_peak",
            "admission",
            "fit",
            "evaluate",
        ),
        "failure phase is invalid",
    )
    destination = failure_path(root, record)
    status_path = run_path(root, record, "fit/terminal_status.json")
    status_binding = None
    if status_path.exists():
        status = read_object(status_path)
        allowed_status = status.get("status") == "failed" or (
            phase == "evaluate" and status.get("status") == "completed"
        )
        require(allowed_status, "terminal status does not match the failure phase")
        if phase == "evaluate":
            require(
                not run_path(root, record, "fit/score.json").exists(),
                "cannot record evaluation failure after a complete score exists",
            )
        status_binding = {
            "path": status_path.relative_to(root).as_posix(),
            "sha256": sha256_file(status_path),
        }
    evidence = []
    for item in evidence_paths:
        path = item.expanduser().resolve(strict=True)
        require(path.is_file(), f"failure evidence must be a file: {path}")
        evidence.append({"path": str(path), "sha256": sha256_file(path)})
    require(
        len({item["path"] for item in evidence}) == len(evidence), "duplicate failure evidence path"
    )
    receipt: dict[str, Any] = {
        "schema": "wnn.high-compute-r4-cell-failure.v1",
        "status": "failed",
        "campaign_sha256": campaign["campaign_sha256"],
        "stage": record["stage"],
        "cell_id": record["id"],
        "variant": record["variant"],
        "fold": record["fold"],
        "seed": record["seed"],
        "reason": reason.strip(),
        "phase": phase,
        "source_sha256": campaign["runtime_source_sha256"],
        "config_sha256": record["config_sha256"],
        "split_sha256": record["split_sha256"],
        "terminal_status": status_binding,
        "evidence": evidence,
        "attempt_count": 1,
        "selection_eligible": False,
    }
    raw_hash = write_json_new(destination, receipt)
    write_bytes_new(destination.with_suffix(".sha256"), (raw_hash + "\n").encode("ascii"))
    return receipt
