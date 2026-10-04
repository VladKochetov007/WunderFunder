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
CAMPAIGN = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v3"
WIDTHS = (256, 384)
WORKER_NAMES = {
    (width, probe): f"c20261004_w{width}_adamw_s3_f1_t4v3_{'probe' if probe else 'fit'}"
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


def validate_reused_probe_evidence(probe_evidence: dict, devices: list[dict]) -> dict[int, str]:
    """Fail closed unless both archived probes are complete, safe, and on these T4 UUIDs."""
    probes = probe_evidence.get("probes")
    if set(probes or {}) != {str(width) for width in WIDTHS}:
        raise RuntimeError("Reused probe evidence does not contain both widths")
    uuids = set()
    bound = {}
    for width in WIDTHS:
        row = probes[str(width)]
        steps = row.get("step_report", {})
        event = row.get("single_batch_update", {})
        device = row.get("device", {})
        if steps.get("status") != "complete" or len(steps.get("steps", [])) != 1 or steps["steps"][0].get("returncode") != 0:
            raise RuntimeError(f"Prior width-{width} production probe did not complete")
        if event.get("completed_batches") != 1 or event.get("optimizer_attempts", 0) < 1 or event.get("optimizer_completed_updates", 0) < 1 or event.get("amp_skipped_updates") != 0:
            raise RuntimeError(f"Prior width-{width} probe lacks a clean optimizer update")
        if "T4" not in device.get("name", "") or not device.get("uuid"):
            raise RuntimeError(f"Prior width-{width} probe did not bind a Tesla T4")
        expected_missing = [f"research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v2/plans/{row.get('name')}_steps.json"]
        if row.get("result_returncode") != 0 or row.get("result_missing") != expected_missing or row.get("prior_missing_expected") != expected_missing:
            raise RuntimeError(f"Prior width-{width} probe failure was not limited to the known report-path mapping")
        expected_step_report = f"research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v2/{row.get('name')}_steps.json"
        if row.get("step_report_path") != expected_step_report:
            raise RuntimeError(f"Prior width-{width} probe report path is not the matching generated file")
        if row.get("peak_gpu_bytes", STOP_GPU_BYTES) > OPERATING_GPU_BYTES or row.get("peak_gpu_bytes", STOP_GPU_BYTES) >= STOP_GPU_BYTES:
            raise RuntimeError(f"Prior width-{width} probe exceeded the GPU memory gate")
        if row.get("peak_ram_bytes", WORKER_RAM_BYTES + 1) > WORKER_RAM_BYTES:
            raise RuntimeError(f"Prior width-{width} probe exceeded the RAM cap")
        if not row.get("artifact_hashes_verified"):
            raise RuntimeError(f"Prior width-{width} probe artifacts were not verified")
        uuids.add(device["uuid"])
        bound[width] = device["uuid"]
    current = {device["uuid"] for device in devices}
    if len(uuids) != len(WIDTHS) or current != uuids:
        raise RuntimeError("Current physical T4 UUIDs differ from successful archived probes")
    return bound


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
        probe_evidence_path = ROOT / f"{CAMPAIGN}/probe_reuse_evidence.json"
        probe_evidence = json.loads(probe_evidence_path.read_text())
        fit_specs = {width: json.loads((ROOT / f"kaggle_specs/{WORKER_NAMES[(width, False)]}.json").read_text()) for width in WIDTHS}
        expected_spec_identity = {str(width): {"code_sha256": fit_specs[width]["code_sha256"], "input_sha256": canonical_sha(fit_specs[width]["inputs"])} for width in WIDTHS}
        if probe_evidence.get("fit_spec_identity") != expected_spec_identity:
            raise RuntimeError("Reused probe source/input identity differs from this fit bundle")
        old_uuid_set = {probe_evidence["probes"][str(width)]["device"]["uuid"] for width in WIDTHS}
        if {device["uuid"] for device in devices} == old_uuid_set:
            probe_devices = validate_reused_probe_evidence(probe_evidence, devices)
            evidence["probe_phase"] = {"reused": True, "evidence_sha256": sha256_file(probe_evidence_path), "source_identity": probe_evidence["source_identity"], "workers": probe_evidence["probes"], "gate": "reused successful one-batch production probes only when source/input signatures and both physical T4 UUIDs match"}
            fit_names = [WORKER_NAMES[(width, False)] for width in WIDTHS]
            if devices[0]["uuid"] == probe_devices[WIDTHS[1]]:
                fit_names.reverse()
        else:
            probe_summary = launch_runner(probe_names, OUTPUT / "probe_phase")
            probe_rows = validate_workers(probe_summary, probe=True)
            probe_devices = {width: probe_rows[WORKER_NAMES[(width, True)]]["device"]["uuid"] for width in WIDTHS}
            evidence["probe_phase"] = {"reused": False, "concurrency_admitted": probe_summary["concurrency_admitted"], "workers": {row["name"]: {"device": row["device"], "peak_gpu_bytes": row["peak_gpu_bytes"], "peak_ram_bytes": row["peak_ram_bytes"], "status": row["status"], "returncode": row["returncode"]} for row in probe_summary["finished"]}, "gate": "fresh corrected-path probes completed on distinct T4s and each measured under 12 GiB operating / 13 GiB stop"}
            fit_names = [WORKER_NAMES[(width, False)] for width in WIDTHS]
            if devices[0]["uuid"] == probe_devices[WIDTHS[1]]:
                fit_names.reverse()
        probe_rows = None
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
