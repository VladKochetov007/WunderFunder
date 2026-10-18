"""Admit the frozen eight-fit cohort and validate each independent fit receipt."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from journal import Journal, write_json_fsync
from policy import sha256_file
from process_start import start_owned
from profiles import actor_environment
from runtime import run_cohort
from runtime_gates import parse_process_envelope


def _validate_fit_receipt(job: dict[str, Any]) -> dict[str, Any]:
    """Require the fixed epoch-5 producer receipt and all hashed fit outputs."""
    run_dir = Path(job["run_dir"])
    receipt = json.loads((run_dir / "producer_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "tabdata-b300-fit-producer.v1"
        or receipt.get("status") != "completed_fixed_epoch5"
        or receipt.get("run_name") != job["id"]
        or receipt.get("arm") != job["arm"]
        or receipt.get("fold") != job["fold"]
        or receipt.get("seed") != job["seed"]
        or receipt.get("selected_epoch") != 5
        or receipt.get("checkpoint_count") != 7
    ):
        raise RuntimeError(f"fit receipt does not bind completed fixed epoch 5 for {job['id']}")
    required = {
        *(f"artifacts/runs/{job['id']}/model_ep{epoch}.pt" for epoch in range(6)),
        f"artifacts/runs/{job['id']}/model.pt",
        f"artifacts/runs/{job['id']}/history.json",
        f"artifacts/runs/{job['id']}/provenance.json",
        f"artifacts/runs/{job['id']}/training_state.pt",
    }
    outputs = receipt.get("outputs", {})
    if not required.issubset(outputs):
        raise RuntimeError(f"fit receipt omits required outputs for {job['id']}")
    for relative in required:
        path = run_dir / relative
        meta = outputs[relative]
        if not path.is_file() or path.stat().st_size != meta.get("bytes"):
            raise RuntimeError(f"fit output size differs from producer receipt: {path}")
        if sha256_file(path) != meta.get("sha256"):
            raise RuntimeError(f"fit output hash differs from producer receipt: {path}")
    parse_process_envelope(
        run_dir / "train_cell_resource_receipt.json", job["gpu_uuid"], job["cpu_ids_initial"]
    )
    return receipt


def run_fit_cohort(
    *,
    jobs: list[dict[str, Any]],
    root: Path,
    state: Path,
    journal: Journal,
    fit_deadline: float,
    wave_deadline: float,
) -> list[dict[str, Any]]:
    """Launch only fresh WAVE cells; preserve each failure while healthy siblings run."""
    fit_paths = [Path(job["run_dir"]) for job in jobs]
    for run_dir in fit_paths:
        if run_dir.exists() or run_dir.is_symlink():
            raise FileExistsError(f"fit run path is not fresh: {run_dir}")

    def launch(job: dict[str, Any]):
        run_dir = Path(job["run_dir"])
        now = time.monotonic()
        if now >= fit_deadline:
            raise TimeoutError("fit deadline passed before admitting another cell")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir()
        environment = actor_environment(root, run_dir, job["gpu_uuid"])
        timeout = fit_deadline - now
        return start_owned(
            name=job["id"],
            argv=list(job["argv"]),
            environment={**environment, "WNN_CWD": str(root)},
            cpus=tuple(job["cpu_ids_initial"]),
            log_path=state / "logs" / f"{job['id']}.log",
            timeout_seconds=timeout,
        )

    process_results = run_cohort(
        jobs,
        launch,
        journal=journal,
        root=root,
        global_deadline=wave_deadline,
        owned_paths=[root / "runs", root / "exports", root / "grades", root / "scratch", state.parent],
    )
    records = []
    for job in jobs:
        result = process_results.get(job["id"], {"status": "not_started", "returncode": None})
        record: dict[str, Any] = {"job_id": job["id"], "process": result}
        if result.get("status") == "completed" and result.get("returncode") == 0:
            try:
                receipt = _validate_fit_receipt(job)
                receipt_path = Path(job["run_dir"]) / "producer_receipt.json"
                record.update(
                    status="fixed_epoch5_receipt_verified",
                    fit_receipt_path=str(receipt_path),
                    fit_receipt_sha256=sha256_file(receipt_path),
                    selected_epoch=receipt["selected_epoch"],
                )
                journal.record("fit_receipt_verified", job_id=job["id"], run_dir=job["run_dir"])
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                json.JSONDecodeError,
            ) as error:
                record.update(status="fit_receipt_invalid", receipt_error=str(error))
                journal.record("fit_receipt_invalid", job_id=job["id"], error=str(error))
        else:
            record["status"] = result.get("status", "fit_failed")
            journal.record("fit_not_verified", job_id=job["id"], process=result)
        records.append(record)
        write_json_fsync(state / "receipts" / f"fit-{job['id']}.json", record)
    return records
