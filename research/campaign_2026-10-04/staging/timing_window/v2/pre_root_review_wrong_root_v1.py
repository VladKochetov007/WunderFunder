"""Run E462 and E450 CPU timings only in a verified quiet-window service.

This controller never seals specs and never retries a stopped attempt. A quiet
window that does not appear within 900 seconds is recorded as deferred with a
zero exit so the dependent E425 science service can proceed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
RUNNER = CAMPAIGN / "runner"
OUTPUT = CAMPAIGN / "staging/timing_window/v2/timing_window_v2.json"
PREDECESSOR_SERVICE = "wnn-e426-sequence-v1.service"
E426_SEQUENCE = CAMPAIGN / "checks/e426_sequence_v1.json"
E426_ATTEMPTS = (
    "e426-smoke-shared-f0-s3-2b-v2",
    "e426-f0-constant-s3-4ep-v2",
    "e426-f0-aligned-s3-4ep-v2",
    "e426-f0-shuffled-s3-4ep-v2",
)
TIMINGS = (
    ("e450_seed4_i16_timing_v2", "e450-seed4-i16-timing-v2"),
    ("e462_public_q8_timing_v2", "e462-public-q8-timing-v2"),
)
QUIET_WAIT_SECONDS = 900
QUIET_POLL_SECONDS = 10
MAX_LOAD1 = 0.3
WORKER_MAX_LOAD1 = 1.0
E462_TIMING_SOURCE_SHA256 = "7178b08cf108e518c9774bbe9432917ce989521ebcf06e5d8f0b15ee508661ae"
E450_TIMING_SOURCE_SHA256 = "2223ead2ac140a1959fc932c6c75f753e594dca0c39f806303480427a49ee729"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_state(status: str, **fields: object) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"status": status, **fields}, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)


def guard() -> None:
    if (RUNNER / "PAUSE.json").exists():
        raise RuntimeError("campaign resource pause requires review")
    if not (CAMPAIGN / "CONTINUATION_AUTHORIZATION_20261004.md").is_file():
        raise FileNotFoundError("campaign continuation authorization is absent")


def systemd_properties(unit: str) -> dict[str, str]:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "LoadState", "-p", "ActiveState",
         "-p", "ExecMainStatus"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def require_success(attempt_name: str) -> dict:
    record = read_json(RUNNER / f"attempts/{attempt_name}.json")
    if (record.get("id") != attempt_name or record.get("status") != "finished"
            or record.get("exit_code") != 0 or record.get("teardown_verified") is not True):
        raise RuntimeError(f"required attempt did not finish successfully: {attempt_name}")
    return record


def require_bound_artifact(attempt_name: str, path: Path) -> dict:
    record = require_success(attempt_name)
    relative = path.relative_to(ROOT).as_posix()
    artifact = next((item for item in record.get("artifacts", [])
                     if item.get("path") == relative), None)
    if artifact is None or artifact.get("sha256") != sha256_file(path):
        raise ValueError(f"successful prerequisite attempt does not bind {relative}")
    return record


def wait_for_e426() -> dict[str, object]:
    """Wait for the exact controller handle and all four successful E426 jobs."""
    while True:
        guard()
        sequence = read_json(E426_SEQUENCE)
        observed = systemd_properties(PREDECESSOR_SERVICE)
        if observed.get("LoadState") != "loaded":
            raise RuntimeError(f"required E426 controller unit is missing: {observed}")
        if sequence.get("status") == "fits_complete":
            if observed.get("ActiveState") != "inactive":
                if observed.get("ActiveState") in ("active", "activating", "deactivating"):
                    write_state("waiting_for_e426_teardown", predecessor=PREDECESSOR_SERVICE,
                                observed=observed, sequence=sequence)
                    time.sleep(5)
                    continue
                raise RuntimeError(f"E426 service disappeared before verified teardown: {observed}")
            if observed.get("ExecMainStatus") != "0":
                raise RuntimeError(f"E426 controller exited unsuccessfully: {observed}")
            records = {name: require_success(name) for name in E426_ATTEMPTS}
            if set(sequence.get("completed", [])) != set(E426_ATTEMPTS):
                raise RuntimeError("E426 sequence completion list differs from its four required attempts")
            return {
                "service": PREDECESSOR_SERVICE,
                "properties": observed,
                "sequence_sha256": sha256_file(E426_SEQUENCE),
                "attempt_ids": list(E426_ATTEMPTS),
                "attempt_spec_sha256": {name: records[name]["spec_sha256"] for name in E426_ATTEMPTS},
            }
        if sequence.get("status") not in ("waiting_for_seed4", "running"):
            raise RuntimeError(f"E426 sequence stopped before fits_complete: {sequence}")
        if observed.get("ActiveState") not in ("active", "activating", "deactivating"):
            raise RuntimeError(f"E426 controller handle disappeared before completion: {observed}")
        write_state("waiting_for_e426", predecessor=PREDECESSOR_SERVICE,
                    observed=observed, sequence_status=sequence.get("status"),
                    completed=sequence.get("completed", []))
        time.sleep(15)


def cuda_quiet_snapshot() -> tuple[bool, dict[str, object]]:
    """Fail closed unless nvidia-smi reports zero active compute and utilization."""
    if shutil.which("nvidia-smi") is None:
        return False, {"available": False, "reason": "nvidia-smi missing"}
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    util = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if apps.returncode or util.returncode:
        return False, {"available": True, "error": apps.stderr or util.stderr,
                       "apps_returncode": apps.returncode, "util_returncode": util.returncode}
    pids = [line.strip() for line in apps.stdout.splitlines() if line.strip()]
    raw_util = [line.strip().split()[0] for line in util.stdout.splitlines() if line.strip()]
    try:
        utilization = [int(value) for value in raw_util]
    except ValueError:
        return False, {"available": True, "compute_pids": pids, "utilization": raw_util}
    quiet = not pids and all(value == 0 for value in utilization) and bool(utilization)
    return quiet, {"available": True, "compute_pids": pids,
                   "gpu_utilization_percent": utilization}


def quiet_snapshot() -> tuple[bool, dict[str, object]]:
    loadavg = list(os.getloadavg())
    cuda_quiet, cuda = cuda_quiet_snapshot()
    return loadavg[0] <= MAX_LOAD1 and cuda_quiet, {
        "loadavg": loadavg, "load1_limit": MAX_LOAD1,
        "cuda_quiet": cuda_quiet, "cuda": cuda,
    }


def required_prerequisites() -> dict[str, object] | None:
    """Return None for genuinely pending artifacts so science can proceed deferred."""
    parity = CAMPAIGN / "E462_public_q8_parity_v2/parity_report.json"
    score = CAMPAIGN / "E462_public_q8_score_v4/report/score_report.json"
    build = CAMPAIGN / "E462_public_q8_build_v2/build_report.json"
    seed4 = CAMPAIGN / "checks/e450_seed4_native_parity_v1/readiness.json"
    for path in (parity, score, build, seed4):
        if not path.is_file():
            return None
    prerequisite_specs = {
        "e462-public-q8-build-v3": CAMPAIGN / "specs/e462_public_q8_build_v3.sealed.json",
        "e462-public-q8-parity-v2": CAMPAIGN / "specs/e462_public_q8_parity_v2.sealed.json",
        "e462-public-q8-score-v4": CAMPAIGN / "specs/e462_public_q8_score_v4.sealed.json",
        "e450-seed4-native-parity-v1": CAMPAIGN / "specs/e450_seed4_native_parity_v1.sealed.json",
    }
    if any(not path.is_file() for path in prerequisite_specs.values()):
        return None
    prerequisite_attempt_names = (
        "e462-public-q8-build-v3", "e462-public-q8-parity-v2",
        "e462-public-q8-score-v4", "e450-seed4-native-parity-v1",
    )
    for name in prerequisite_attempt_names:
        attempt_path = RUNNER / f"attempts/{name}.json"
        if not attempt_path.is_file():
            return None
        attempt = read_json(attempt_path)
        if (attempt.get("status") != "finished" or attempt.get("exit_code") != 0
                or attempt.get("teardown_verified") is not True):
            return None
        if attempt.get("spec") != read_json(prerequisite_specs[name]):
            raise ValueError(f"successful prerequisite attempt has changed identity: {name}")
    parity_value, score_value, build_value, seed4_value = map(
        read_json, (parity, score, build, seed4)
    )
    if (
        parity_value.get("schema") != "e462-public-combo-q8-parity-v1"
        or parity_value.get("passed") is not True
        or parity_value.get("sequences") != 20
        or parity_value.get("rows") != 400_000
        or score_value.get("schema") != "e462-public-combo-q8-holdout-score-v1"
        or score_value.get("parity_passed") is not True
        or score_value.get("n_sequences") != 368
        or seed4_value.get("status") != "parity_passed_runtime_pending"
        or seed4_value.get("callback_parity", {}).get("passed") is not True
        or seed4_value.get("reset_and_causality", {}).get("reset_bitwise") is not True
        or seed4_value.get("reset_and_causality", {}).get("past_only_bitwise") is not True
    ):
        return None
    if score_value.get("parity_report_sha256") != sha256_file(parity):
        raise ValueError("E462 score report is not bound to the actual parity report")
    prerequisite_records = {
        "e462_build": require_bound_artifact(
            "e462-public-q8-build-v3", build),
        "e462_parity": require_bound_artifact(
            "e462-public-q8-parity-v2", parity),
        "e462_score": require_bound_artifact(
            "e462-public-q8-score-v4", score),
        "e450_seed4_parity": require_bound_artifact(
            "e450-seed4-native-parity-v1", seed4),
    }
    e450_parity_spec = read_json(prerequisite_specs["e450-seed4-native-parity-v1"])
    if prerequisite_records["e450_seed4_parity"].get("spec") != e450_parity_spec:
        raise ValueError("E450 readiness came from a different sealed native-parity spec")
    if (not (CAMPAIGN / "specs/e462_public_q8_timing_v2.sealed.json").is_file()
            or not (CAMPAIGN / "specs/e450_seed4_i16_timing_v2.sealed.json").is_file()):
        return None
    return {
        "e462_build_report_sha256": sha256_file(build),
        "e462_parity_report_sha256": sha256_file(parity),
        "e462_score_report_sha256": sha256_file(score),
        "e450_seed4_readiness_sha256": sha256_file(seed4),
        "q8r_zip_sha256": build_value["variants"]["q8r"]["zip_sha256"],
        "q8x_zip_sha256": build_value["variants"]["q8x"]["zip_sha256"],
        "seed4_zip_sha256": seed4_value["package"]["sha256"],
        "successful_attempts": {
            key: value["spec_sha256"] for key, value in prerequisite_records.items()
        },
    }


def _artifact_paths(spec: dict) -> list[str]:
    return [item["path"] if isinstance(item, dict) else item for item in spec.get("artifacts", [])]


def _flag(argv: list[str], name: str) -> str | None:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def validate_timing_spec(spec: dict, spec_name: str) -> None:
    """Keep v2 timing identities, source loops, thresholds, and outputs fixed."""
    argv = spec.get("argv")
    if (spec.get("name") != spec_name or spec.get("kind") != "cpu"
            or not isinstance(argv, list) or len(argv) < 2):
        raise ValueError(f"invalid v2 timing worker identity: {spec_name}")
    expected_worker = {
        "e450_seed4_i16_timing_v2": "research/campaign_2026-10-04/staging/timing_window/time_seed4_i16.py",
        "e462_public_q8_timing_v2": "research/campaign_2026-10-04/staging/e462/time_public_q8.py",
    }.get(spec_name)
    if expected_worker is None or expected_worker not in argv:
        raise ValueError(f"unexpected scientific timing loop: {spec_name}")
    actual_source_hash = sha256_file(ROOT / expected_worker)
    expected_source_hash = (E450_TIMING_SOURCE_SHA256 if spec_name.startswith("e450")
                            else E462_TIMING_SOURCE_SHA256)
    if actual_source_hash != expected_source_hash:
        raise ValueError(f"frozen timing loop changed: {spec_name}")
    if (_flag(argv, "--cpu") != "15" or _flag(argv, "--max-load1") != str(WORKER_MAX_LOAD1)):
        raise ValueError(f"worker affinity/load threshold changed: {spec_name}")
    repeat_flag = "--repeats" if spec_name.startswith("e450") else "--paired-repeats"
    if _flag(argv, repeat_flag) != "5":
        raise ValueError(f"worker repeat count changed: {spec_name}")
    if spec_name.startswith("e450") and _flag(
        argv, "--expected-e462-timing-sha256"
    ) != E462_TIMING_SOURCE_SHA256:
        raise ValueError("seed4 worker no longer pins the unchanged E462 timing source")
    output = _flag(argv, "--output")
    if (not output or "staging/timing_window/v2/results/" not in output
            or output in ("research/campaign_2026-10-04/staging/timing_window/results/e450_seed4_i16_v1",
                          "research/campaign_2026-10-04/staging/timing_window/results/e462_q8_v1")):
        raise ValueError(f"v2 worker output is not uniquely namespaced: {spec_name}")
    limits = spec.get("limits", {})
    if (limits.get("cpus") != 1 or limits.get("ram_bytes") != 2 * 1024**3
            or limits.get("timeout_seconds") != 1800 or limits.get("gpu_memory_bytes") != 0
            or limits.get("gpu_stop_bytes") != 0):
        raise ValueError(f"worker resource envelope changed: {spec_name}")


def project_39m_rows() -> dict[str, object]:
    """Compute competition-size row projections only from completed timing reports."""
    limit_seconds = 52 * 60
    e462_path = CAMPAIGN / "staging/timing_window/v2/results/e462_q8_v2/timing_report.json"
    seed4_path = CAMPAIGN / "staging/timing_window/v2/results/e450_seed4_i16_v2/timing_report.json"
    e462 = read_json(e462_path)
    seed4 = read_json(seed4_path)
    if (e462.get("schema") != "e462-public-combo-q8-timing-v1"
            or e462.get("passed") is not True
            or e462.get("rows_per_trial") != 60_000
            or seed4.get("schema") != "e450-seed4-i16-timing-v1"
            or seed4.get("status") != "timing_complete"
            or seed4.get("passed") is not True
            or seed4.get("rows_per_trial") != 60_000):
        raise ValueError("cannot project full-run rows without both completed timing reports")
    candidates = {}
    for label, summary in e462.get("summaries", {}).items():
        if label not in ("i16", "q8r", "q8x"):
            continue
        median_us = float(summary["median_us_per_row"])
        seconds = median_us * 39_000_000 / 1_000_000
        candidates[f"seed3_{label}"] = {
            "median_us_per_row": median_us,
            "projected_39m_rows_seconds": seconds,
            "under_52_minute_projection": seconds <= limit_seconds,
        }
    for label, summary in seed4.get("summaries", {}).items():
        median_us = float(summary["median_us_per_row"])
        seconds = median_us * 39_000_000 / 1_000_000
        candidates[label] = {
            "median_us_per_row": median_us,
            "projected_39m_rows_seconds": seconds,
            "under_52_minute_projection": seconds <= limit_seconds,
        }
    return {
        "source": "completed timing reports only; linear extrapolation from 3 full 20k callback streams",
        "rows_projected": 39_000_000, "threshold_seconds": limit_seconds,
        "timing_reports": {
            "e462_sha256": sha256_file(e462_path), "e450_sha256": sha256_file(seed4_path),
        },
        "candidates": candidates,
        "interpretation": "projection only; not actual competition runtime or upload readiness",
    }


def validate_attempt_outputs(spec: dict, record: dict) -> dict[str, object]:
    artifacts = {item["path"]: item["sha256"] for item in record.get("artifacts", [])}
    for relative in _artifact_paths(spec):
        path = ROOT / relative
        if not path.is_file() or relative not in artifacts:
            raise FileNotFoundError(f"worker artifact was not recorded: {relative}")
        if sha256_file(path) != artifacts[relative]:
            raise ValueError(f"worker artifact hash differs from its runner record: {relative}")
    return {"attempt": record["id"], "exit_code": record["exit_code"],
            "teardown_verified": record["teardown_verified"],
            "spec_sha256": record["spec_sha256"], "artifacts": artifacts}


def launch_or_observe(spec_name: str, attempt_name: str) -> dict[str, object]:
    guard()
    sealed_path = CAMPAIGN / f"specs/{spec_name}.sealed.json"
    if not sealed_path.is_file():
        raise FileNotFoundError(f"root must seal timing spec before execution: {sealed_path}")
    spec = read_json(sealed_path)
    validate_timing_spec(spec, spec_name)
    record_path = RUNNER / f"attempts/{attempt_name}.json"
    if record_path.is_file():
        record = read_json(record_path)
        if record.get("spec") != spec:
            raise ValueError(f"existing timing attempt is bound to a different sealed spec: {attempt_name}")
        if record.get("status") not in ("reserved", "running", "finished"):
            raise RuntimeError(f"refusing to retry stopped timing attempt {attempt_name}: {record}")
        if record.get("status") == "finished":
            return validate_attempt_outputs(spec, require_success(attempt_name))
    else:
        outputs = [ROOT / item for item in _artifact_paths(spec)]
        if any(path.exists() for path in outputs):
            raise FileExistsError(f"timing outputs exist without matching attempt {attempt_name}")
        subprocess.run(
            [sys.executable, "scripts/run_campaign.py", "launch", str(sealed_path),
             "--attempt", attempt_name],
            cwd=ROOT, check=True,
        )
        if not record_path.is_file():
            raise FileNotFoundError(f"run_campaign did not create attempt record {attempt_name}")

    deadline = time.monotonic() + int(spec["limits"]["timeout_seconds"]) + 240
    while True:
        guard()
        record = read_json(record_path)
        if record.get("spec") != spec:
            raise ValueError(f"timing attempt record changed identity: {attempt_name}")
        if record.get("status") == "finished":
            return validate_attempt_outputs(spec, require_success(attempt_name))
        if record.get("status") not in ("reserved", "running"):
            raise RuntimeError(f"timing attempt stopped: {attempt_name}: {record.get('status')}")
        unit = record.get("unit")
        if not isinstance(unit, str) or not unit:
            raise RuntimeError(f"timing attempt lacks exact worker service handle: {attempt_name}")
        observed = systemd_properties(unit)
        if observed.get("ActiveState") not in ("active", "activating"):
            raise RuntimeError(f"timing worker service disappeared: {attempt_name}: {observed}")
        cuda_quiet, cuda_state = cuda_quiet_snapshot()
        if not cuda_quiet:
            subprocess.run(["systemctl", "--user", "stop", unit], cwd=ROOT, check=False)
            raise RuntimeError(
                f"CUDA compute appeared during CPU timing; stopped exact worker {unit}: {cuda_state}"
            )
        if time.monotonic() > deadline:
            raise TimeoutError(f"observation deadline exceeded; refusing retry: {attempt_name}")
        time.sleep(5)


def run() -> int:
    if OUTPUT.exists():
        previous = read_json(OUTPUT)
        if previous.get("status") in ("timing_complete", "timing_deferred"):
            return 0
        raise RuntimeError(f"refusing blind restart of an interrupted timing window: {previous}")
    write_state("waiting_for_e426", controller="timing_window_v2")
    try:
        predecessor = wait_for_e426()
        prerequisites = required_prerequisites()
        if prerequisites is None:
            write_state("timing_deferred", predecessor=predecessor,
                        reason="E462/E450 readiness or root-sealed timing specs are pending",
                        quiet_wait_seconds=0)
            return 0
        deadline = time.monotonic() + QUIET_WAIT_SECONDS
        snapshots = []
        while time.monotonic() < deadline:
            guard()
            quiet, snapshot = quiet_snapshot()
            snapshots.append({"unix_time": time.time(), **snapshot})
            if quiet:
                break
            write_state("waiting_for_quiet_window", predecessor=predecessor,
                        prerequisite_sha256=prerequisites, polls=len(snapshots),
                        last_snapshot=snapshot)
            time.sleep(QUIET_POLL_SECONDS)
        else:
            write_state("timing_deferred", predecessor=predecessor,
                        prerequisite_sha256=prerequisites,
                        reason="load1 remained above 0.3 or CUDA compute/utilization was nonzero",
                        quiet_wait_seconds=QUIET_WAIT_SECONDS, quiet_polls=snapshots)
            return 0

        results = {}
        for spec_name, attempt_name in TIMINGS:
            guard()
            quiet, snapshot = quiet_snapshot()
            if not quiet:
                write_state("timing_deferred", predecessor=predecessor,
                            prerequisite_sha256=prerequisites, completed=results,
                            reason="quiet window ended before the next timing worker",
                            quiet_wait_seconds=QUIET_WAIT_SECONDS, quiet_snapshot=snapshot)
                return 0
            results[attempt_name] = launch_or_observe(spec_name, attempt_name)
            write_state("running", predecessor=predecessor,
                        prerequisite_sha256=prerequisites, completed=results)
        projections = project_39m_rows()
        write_state("timing_complete", predecessor=predecessor,
                    prerequisite_sha256=prerequisites, completed=results,
                    projected_39m_rows=projections)
        return 0
    except BaseException as error:
        write_state("failed_closed", error=f"{type(error).__name__}: {error}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    raise SystemExit(run())


if __name__ == "__main__":
    main()
