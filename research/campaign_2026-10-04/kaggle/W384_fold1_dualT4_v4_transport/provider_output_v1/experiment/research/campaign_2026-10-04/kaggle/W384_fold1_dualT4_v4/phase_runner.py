from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/kaggle/working/experiment")
OUTPUT = Path("/kaggle/working/results")
CAMPAIGN = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4"
WIDTHS = (256, 384)
WORKER_NAMES = {
    (width, probe): f"c20261004_w{width}_adamw_s3_f1_t4v4_{'probe' if probe else 'fit'}"
    for width in WIDTHS
    for probe in (True, False)
}
OPERATING_GPU_BYTES = 12 * 1024**3
STOP_GPU_BYTES = 13 * 1024**3
WORKER_RAM_BYTES = 10 * 1024**3


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_two_t4s() -> list[dict]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_gpu_runner import gpu_inventory

    devices = gpu_inventory()
    if len(devices) != 2 or any("T4" not in item["name"] for item in devices):
        raise RuntimeError("The paired protocol requires two visible Tesla T4 devices")
    return devices


def launch_runner(spec_names: list[str], output: Path) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts/kaggle_gpu_runner.py"),
        "--root",
        str(ROOT),
        "--output",
        str(output),
    ]
    for name in spec_names:
        command.extend(["--spec", str(ROOT / f"kaggle_specs/{name}.json")])
    completed = subprocess.run(command, cwd=ROOT, check=False)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if completed.returncode or summary.get("incident") or summary.get("not_started"):
        raise RuntimeError(f"GPU runner failed for phase at {output}")
    if summary.get("concurrency_admitted") != 2 or len(summary.get("finished", [])) != 2:
        raise RuntimeError(f"Both physical T4 workers were not admitted at {output}")
    return summary


def validate_workers(summary: dict, *, probe: bool) -> dict[str, dict]:
    expected = {WORKER_NAMES[(width, probe)] for width in WIDTHS}
    rows = {item["name"]: item for item in summary["finished"]}
    if set(rows) != expected:
        raise RuntimeError("The finished worker set differs from the sealed pair")
    for name, row in rows.items():
        if row.get("status") != "complete" or row.get("returncode") != 0:
            raise RuntimeError(f"Worker did not complete: {name}")
        if row.get("peak_gpu_bytes", STOP_GPU_BYTES + 1) > OPERATING_GPU_BYTES:
            raise RuntimeError(f"Measured operating GPU target exceeded: {name}")
        if row.get("peak_gpu_bytes", STOP_GPU_BYTES + 1) >= STOP_GPU_BYTES:
            raise RuntimeError(f"Measured hard stop reached: {name}")
        if row.get("peak_ram_bytes", WORKER_RAM_BYTES + 1) > WORKER_RAM_BYTES:
            raise RuntimeError(f"Worker RAM cap exceeded: {name}")
        if "T4" not in row.get("device", {}).get("name", ""):
            raise RuntimeError(f"Worker device is not a Tesla T4: {name}")
    return rows


def validate_probe_updates(rows: dict[str, dict], output: Path) -> dict[str, dict]:
    """Require an observed finite production-batch update on each probe worker."""
    result = {}
    for name in sorted(rows):
        log_path = output / name / "worker.log"
        if not log_path.is_file():
            raise RuntimeError(f"Probe worker log is missing: {name}")
        fit_lines = [line[4:] for line in log_path.read_text().splitlines() if line.startswith("FIT ")]
        if len(fit_lines) != 1:
            raise RuntimeError(f"Probe must emit exactly one FIT record: {name}")
        event = json.loads(fit_lines[0])
        counts = event.get("update_diagnostics", {}).get("epoch", {})
        if event.get("epoch") != 0 or event.get("evaluation") != "withheld":
            raise RuntimeError(f"Probe did not finish its single no-evaluation epoch: {name}")
        if counts.get("completed_batches") != 1:
            raise RuntimeError(f"Probe did not complete exactly one production batch: {name}")
        if counts.get("optimizer_attempts", 0) < 1 or counts.get("optimizer_completed_updates", 0) < 1:
            raise RuntimeError(f"Probe has no completed optimizer update: {name}")
        if counts.get("amp_skipped_updates") != 0:
            raise RuntimeError(f"Probe had an AMP-skipped update: {name}")
        if counts.get("ema_updates") != counts.get("optimizer_completed_updates"):
            raise RuntimeError(f"Probe EMA and optimizer update counts differ: {name}")
        result[name] = counts
    return result


def main() -> None:
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    evidence: dict[str, object] = {
        "protocol": "two-phase width replication; probes on both T4s complete before either fit",
        "required_widths": list(WIDTHS),
        "allocator_target_bytes_per_worker": 11 * 1024**3,
        "measured_gpu_operating_limit_bytes": OPERATING_GPU_BYTES,
        "sampled_gpu_stop_bytes": STOP_GPU_BYTES,
        "worker_ram_bytes": WORKER_RAM_BYTES,
    }
    try:
        devices = require_two_t4s()
        evidence["visible_devices"] = devices
        env = os.environ.copy()
        env.update(
            {
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "WNN_INTEGRATION_DEVICE": "cuda",
                "PYTHONPATH": str(ROOT / "src"),
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "CUDA_VISIBLE_DEVICES": devices[0]["uuid"],
            }
        )
        preflight_xml = OUTPUT / "current_source_cuda_preflight.xml"
        preflight_log = OUTPUT / "current_source_cuda_preflight.log"
        with preflight_log.open("w") as log:
            preflight = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "tests/test_trainer_resume_integration.py",
                    "tests/test_training_state.py",
                    f"--junitxml={preflight_xml}",
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        evidence["preflight"] = {
            "returncode": preflight.returncode,
            "xml_sha256": sha256_file(preflight_xml) if preflight_xml.is_file() else None,
            "log_bytes": preflight_log.stat().st_size,
        }
        if preflight.returncode:
            raise RuntimeError("Current-source CUDA preflight failed; no probe or fit started")

        probe_names = [WORKER_NAMES[(width, True)] for width in WIDTHS]
        probe_summary = launch_runner(probe_names, OUTPUT / "probe_phase")
        probe_rows = validate_workers(probe_summary, probe=True)
        probe_updates = validate_probe_updates(probe_rows, OUTPUT / "probe_phase")
        probe_devices = {
            width: probe_rows[WORKER_NAMES[(width, True)]]["device"]["uuid"] for width in WIDTHS
        }
        if len(set(probe_devices.values())) != 2:
            raise RuntimeError("Both probe workers must use distinct physical T4 UUIDs")
        evidence["probe_phase"] = {
            "reused": False,
            "concurrency_admitted": probe_summary["concurrency_admitted"],
            "workers": {
                row["name"]: {
                    "device": row["device"],
                    "peak_gpu_bytes": row["peak_gpu_bytes"],
                    "peak_ram_bytes": row["peak_ram_bytes"],
                    "status": row["status"],
                    "returncode": row["returncode"],
                    "single_batch_update": probe_updates[row["name"]],
                }
                for row in probe_summary["finished"]
            },
            "gate": "both fresh one-batch probes complete on distinct T4s below memory caps before either fit starts",
        }
        fit_names = [WORKER_NAMES[(width, False)] for width in WIDTHS]
        if devices[0]["uuid"] == probe_devices[WIDTHS[1]]:
            fit_names.reverse()
        fit_summary = launch_runner(fit_names, OUTPUT / "fit_phase")
        fit_rows = validate_workers(fit_summary, probe=False)
        fit_devices = {
            width: fit_rows[WORKER_NAMES[(width, False)]]["device"]["uuid"] for width in WIDTHS
        }
        if fit_devices != probe_devices:
            raise RuntimeError("A fit worker was assigned a different T4 than its production probe")
        evidence["fit_phase"] = {
            "concurrency_admitted": fit_summary["concurrency_admitted"],
            "workers": {
                row["name"]: {
                    "device": row["device"],
                    "peak_gpu_bytes": row["peak_gpu_bytes"],
                    "peak_ram_bytes": row["peak_ram_bytes"],
                    "wall_seconds": row["wall_seconds"],
                    "status": row["status"],
                    "returncode": row["returncode"],
                    "artifact_missing": row["artifacts"]["missing"],
                }
                for row in fit_summary["finished"]
            },
        }

        score_path = ROOT / f"{CAMPAIGN}/paired_fold1_score.json"
        env.update({"CUDA_VISIBLE_DEVICES": ""})
        score = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/screen_score.py"),
                str(ROOT / f"{CAMPAIGN}/paired_fold1_screen.json"),
                "--output",
                str(score_path),
            ],
            cwd=ROOT,
            env=env,
            check=False,
        )
        if score.returncode or not score_path.is_file():
            raise RuntimeError("Paired fixed-epoch fold1 scoring failed")
        downloaded_score = OUTPUT / "paired_fold1_score.json"
        shutil.copyfile(score_path, downloaded_score)
        report = json.loads(score_path.read_text())
        comparison = report["pooled"]["paired_component_bootstrap"]["W384_minus_W256"]
        evidence["paired_score"] = {
            "path": str(downloaded_score),
            "sha256": sha256_file(downloaded_score),
            "size": downloaded_score.stat().st_size,
            "comparison": comparison,
            "exact_global_wp": report["pooled"]["exact_global_wp"],
            "interpretation": "single exposed fold1 replication; paired diagnostic, not public validation or significance claim",
        }
        evidence["status"] = "complete"
    except BaseException as exc:
        evidence["status"] = "failed_closed"
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        evidence["traceback"] = traceback.format_exc()
        raise
    finally:
        evidence["elapsed_seconds"] = time.monotonic() - started
        write_json(OUTPUT / "phase_report.json", evidence)


if __name__ == "__main__":
    main()
