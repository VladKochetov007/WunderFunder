"""Check the sealed delivery packet against real completed fits without new work."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from delivery_watch_core import (
    read_object,
    sha256_file,
    verify_runtime_envelope,
)
from delivery_watch_receipts import verify_export_receipt, verify_fit_receipt
from delivery_watch_state import _base_cli
from r01_delivery_watch import _load_config


def main() -> None:
    """Recheck existing phase receipts; never run a fit, export, grade or publisher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    config = _load_config(args.config, args.config_sha256)
    verify_runtime_envelope(config["gpu_uuid"])
    root, helper = config["root_path"], config["helper_path"]
    packet_dir = Path(__file__).resolve().parent
    manifest_path = packet_dir / "MANIFEST.json"
    manifest = read_object(manifest_path)
    if manifest.get("schema") != "tabdata-r01-delivery-watch-packet.v1":
        raise ValueError("unexpected acceptance packet manifest schema")
    for relative, expected in manifest["files"].items():
        if Path(relative).name != relative or sha256_file(packet_dir / relative) != expected:
            raise ValueError(f"acceptance packet member changed: {relative}")
    sys.path.insert(0, str(helper))
    import r01_recovery_delivery as delivery
    from r01_recovery_common import write_json_new
    from r01_recovery_placement import verify_placement

    recovery = root / "state/r01-delivery-recovery-v2"
    cli_checks = []
    for phase in ("validate", "export"):
        argv = _base_cli(config, recovery, phase) + ["--job-id", "parser-check", "--help"]
        checked = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=True)
        cli_checks.append({"phase": phase, "argv": argv, "returncode": checked.returncode})
    validated, exports = [], []
    for job in config["jobs"]:
        job_id = job["id"]
        if not (recovery / "receipts" / f"{job_id}.json").exists():
            continue
        fit = verify_fit_receipt(
            delivery, recovery, config["packet_path"], root, job, verify_placement
        )
        validated.append(
            {"job_id": job_id, "checkpoint_sha256": fit["checkpoint_identity"]["fixed_file_sha256"]}
        )
        if not (recovery / "receipts" / f"{job_id}.export.json").exists():
            continue
        exported = verify_export_receipt(recovery, root, job_id, fit, config["gpu_uuid"])
        exports.append({"job_id": job_id, "prediction_sha256": exported["prediction_sha256"]})
    if len(validated) < 4 or not exports:
        raise ValueError("expected previously completed real R01 fit/export evidence is absent")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import publish_recovery

    publisher_config = publish_recovery.verify_config(
        Path(config["publisher_config_path"]), config["publisher_config_sha256"]
    )
    _, publisher_source, publisher_placement = publish_recovery._verify_runtime(
        root, publisher_config
    )
    grader = publish_recovery._load_grader(root, publisher_source)
    prediction_checks = []
    for exported in exports:
        job = next(job for job in config["jobs"] if job["id"] == exported["job_id"])
        endpoint = publish_recovery._fit_export(root, recovery, job, publisher_placement)
        split = grader.load_campaign_split(
            root / "splits" / f"fold{job['fold']}.json", endpoint["output"]
        )
        prediction, binding = grader.validate_prediction(
            endpoint["output"], job["id"], 5, list(split.evaluation_groups), split
        )
        if tuple(prediction.shape) != (len(split.evaluation_groups), 20000, 2) or str(
            prediction.dtype
        ) != "float32":
            raise ValueError("officially checked endpoint has unexpected shape or dtype")
        if any(
            binding.get(key) != endpoint["export_receipt"].get(key)
            for key in ("checkpoint_sha256", "prediction_sha256", "sidecar_sha256")
        ):
            raise ValueError("official prediction binding differs from existing endpoint")
        prediction_checks.append({"job_id": job["id"], "binding": binding})
    snapshot = publish_recovery.collect(root, publisher_config)
    if snapshot["errors"]:
        raise ValueError(
            f"read-only recovery publisher rejects real evidence: {snapshot['errors']}"
        )
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("read-only delivery acceptance unexpectedly initialized CUDA")
    receipt = {
        "schema": "tabdata-r01-delivery-watch-acceptance.v1",
        "status": "passed_read_only",
        "time_unix": time.time(),
        "config_sha256": args.config_sha256,
        "packet_manifest_sha256": sha256_file(manifest_path),
        "helper_manifest_sha256": config["helper_manifest_sha256"],
        "validated_existing_fits": validated,
        "canonical_cli_checks": cli_checks,
        "validated_existing_exports": exports,
        "official_prediction_checks": prediction_checks,
        "official_paired_grades": len(snapshot["completed"]),
        "publisher_snapshot": snapshot,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "new_fits_exports_grades_or_comments": 0,
        "old_fit_outputs_modified": False,
    }
    write_json_new(root / "receipts" / packet_dir.name / "ACCEPTANCE.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "fits": len(validated),
                "exports": len(exports),
                "pairs": len(snapshot["completed"]),
            }
        )
    )


if __name__ == "__main__":
    main()
