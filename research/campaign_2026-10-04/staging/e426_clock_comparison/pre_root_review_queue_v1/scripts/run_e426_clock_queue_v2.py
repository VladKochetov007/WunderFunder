"""Durably queue the fixed E426 epoch-3 exports and paired fold-0 screen.

The controller consumes only root-sealed path-only specs. It waits for the
timing-v2 and fit-provenance gates, exports the three fixed checkpoints in
order, validates each full VALID cache before continuing, and scores only once
all exports are verified. It never selects epochs or retries failed attempts.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
RUNNER = CAMPAIGN / "runner"
OUTPUT = CAMPAIGN / "checks/e426_clock_queue_v2.json"
PREDECESSOR_SERVICE = "wnn-e426-sequence-v1.service"
TIMING_SERVICE = "wnn-timing-window-v2.service"
E426_SEQUENCE = CAMPAIGN / "checks/e426_sequence_v1.json"
TIMING_RESULT = CAMPAIGN / "staging/timing_window/v2/timing_window_v2.json"
PREFLIGHT_ATTEMPT = "e426-fit-provenance-preflight-v1"
PREFLIGHT_REPORT = CAMPAIGN / "checks/e426_fit_provenance_preflight_v1.json"
EXPORTS = (
    ("e426_clock_export_constant_ep3_fp32_v2", "e426-clock-export-constant-ep3-fp32-v2",
     "e426_clock_constant_fp32", "e426_f0_constant_s3_4ep_v2",
     "e426-f0-constant-s3-4ep-v2"),
    ("e426_clock_export_aligned_ep3_fp32_v2", "e426-clock-export-aligned-ep3-fp32-v2",
     "e426_clock_aligned_fp32", "e426_f0_aligned_s3_4ep_v2",
     "e426-f0-aligned-s3-4ep-v2"),
    ("e426_clock_export_shuffled_ep3_fp32_v2", "e426-clock-export-shuffled-ep3-fp32-v2",
     "e426_clock_shuffled_fp32", "e426_f0_shuffled_s3_4ep_v2",
     "e426-f0-shuffled-s3-4ep-v2"),
)
SCORE_SPEC = "e426_clock_fold0_paired_score_v2"
SCORE_ATTEMPT = "e426-clock-fold0-paired-score-v2"
LOCAL_E426_GPU_CAP_SECONDS = 24 * 60 * 60
ROW_COUNT = 1_873
SEQUENCE_LENGTH = 20_000
OUTPUTS = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save(status: str, completed: list[str], **details: object) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"schema": "e426-clock-queue-v2", "status": status,
                                    "completed": completed, **details}, indent=2,
                                    sort_keys=True) + "\n")
    temporary.replace(OUTPUT)


def guard() -> None:
    if (RUNNER / "PAUSE.json").exists():
        raise RuntimeError("campaign resource pause requires review")
    if not (CAMPAIGN / "CONTINUATION_AUTHORIZATION_20261004.md").is_file():
        raise FileNotFoundError("campaign continuation authorization is absent")


def properties(unit: str) -> dict[str, str]:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "LoadState", "-p", "ActiveState",
         "-p", "ExecMainStatus"], cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def successful_attempt(name: str, spec: dict) -> dict:
    record = read_json(RUNNER / f"attempts/{name}.json")
    if (record.get("id") != name or record.get("status") != "finished"
            or record.get("exit_code") != 0 or record.get("teardown_verified") is not True
            or record.get("spec") != spec):
        raise RuntimeError(f"attempt is not terminal-successful with exact identity: {name}")
    unit = f"wnn-c04-{name}.service"
    if record.get("unit") != unit:
        raise RuntimeError(f"attempt has the wrong worker service name: {name}")
    observed = properties(unit)
    if (observed.get("LoadState") != "loaded" or observed.get("ActiveState") != "inactive"
            or observed.get("ExecMainStatus") != "0"):
        raise RuntimeError(f"exact worker service has not exited cleanly: {unit}: {observed}")
    return record


def require_artifacts(spec: dict, record: dict) -> dict[str, str]:
    recorded = {item["path"]: item["sha256"] for item in record.get("artifacts", [])}
    expected = {item["path"] if isinstance(item, dict) else item
                for item in spec.get("artifacts", [])}
    if set(recorded) != expected:
        raise ValueError(f"attempt artifact set changed for {spec.get('name')}")
    for relative, digest in recorded.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"attempt artifact hash mismatch: {relative}")
    return recorded


def wait_for_gates() -> dict[str, object]:
    while True:
        guard()
        fit = read_json(E426_SEQUENCE)
        fit_service = properties(PREDECESSOR_SERVICE)
        if fit_service.get("LoadState") != "loaded":
            raise RuntimeError(f"E426 fit controller unit is missing: {fit_service}")
        if fit.get("status") != "fits_complete":
            if fit.get("status") not in ("waiting_for_seed4", "running"):
                raise RuntimeError(f"E426 fit sequence did not complete successfully: {fit}")
            if fit_service.get("ActiveState") not in ("active", "activating"):
                raise RuntimeError(f"E426 fit service disappeared: {fit_service}")
            save("waiting_for_fits", [])
            time.sleep(15)
            continue
        if fit_service.get("ActiveState") in ("active", "activating", "deactivating"):
            time.sleep(5)
            continue
        if fit_service.get("ExecMainStatus") != "0":
            raise RuntimeError(f"E426 fit controller exited unsuccessfully: {fit_service}")

        timing = read_json(TIMING_RESULT)
        timing_service = properties(TIMING_SERVICE)
        if timing_service.get("LoadState") != "loaded":
            raise RuntimeError(f"timing-v2 service is missing: {timing_service}")
        if timing.get("status") not in ("timing_complete", "timing_deferred"):
            if timing.get("status") in ("waiting_for_e426", "waiting_for_quiet_window", "running",
                                         "waiting_for_e426_teardown"):
                save("waiting_for_timing_window", [], timing_status=timing.get("status"))
                time.sleep(10)
                continue
            raise RuntimeError(f"timing-v2 sequence failed closed: {timing}")
        if timing_service.get("ActiveState") in ("active", "activating", "deactivating"):
            time.sleep(5)
            continue
        if timing_service.get("ExecMainStatus") != "0":
            raise RuntimeError(f"timing-v2 service exited unsuccessfully: {timing_service}")

        preflight_record = read_json(RUNNER / f"attempts/{PREFLIGHT_ATTEMPT}.json")
        preflight_spec = read_json(CAMPAIGN / "specs/e426_fit_provenance_preflight_v1.sealed.json")
        record = successful_attempt(PREFLIGHT_ATTEMPT, preflight_spec)
        artifacts = require_artifacts(preflight_spec, record)
        report_relative = PREFLIGHT_REPORT.relative_to(ROOT).as_posix()
        if artifacts.get(report_relative) != sha256_file(PREFLIGHT_REPORT):
            raise ValueError("preflight report is not bound by its successful runner record")
        if (read_json(PREFLIGHT_REPORT).get("schema") != "e426-fit-provenance-preflight-v1"
                or read_json(PREFLIGHT_REPORT).get("status") != "passed"):
            raise ValueError("E426 fixed-checkpoint provenance preflight did not pass")
        return {
            "fit_service": fit_service,
            "fit_sequence_sha256": sha256_file(E426_SEQUENCE),
            "timing_service": timing_service,
            "timing_result_sha256": sha256_file(TIMING_RESULT),
            "timing_status": timing["status"],
            "preflight_attempt_sha256": record["spec_sha256"],
            "preflight_report_sha256": sha256_file(PREFLIGHT_REPORT),
        }


def local_e426_gpu_charge() -> float:
    """Conservatively count charged time or reserved timeout for all E426 GPU attempts."""
    total = 0.0
    for path in (RUNNER / "attempts").glob("*.json"):
        record = read_json(path)
        if record.get("kind") != "gpu" or not str(record.get("id", "")).startswith("e426-"):
            continue
        if record.get("status") == "finished":
            total += float(record.get("charged_seconds", record.get("elapsed_seconds", 0)))
        else:
            total += float(record.get("reserved_seconds", record.get("timeout_seconds", 0)))
    return total


def require_gpu_reservation(spec: dict) -> dict[str, object]:
    timeout = int(spec.get("limits", {}).get("timeout_seconds", 0))
    charged = local_e426_gpu_charge()
    if timeout <= 0 or charged + timeout > LOCAL_E426_GPU_CAP_SECONDS:
        raise RuntimeError(
            f"E426 local GPU reservation would exceed the 24-hour cap: "
            f"charged_or_reserved={charged:.1f}s next_timeout={timeout}s"
        )
    return {"charged_or_reserved_before_seconds": charged,
            "next_timeout_seconds": timeout,
            "cap_seconds": LOCAL_E426_GPU_CAP_SECONDS}


def require_sealed_job_set() -> dict[str, dict]:
    """Fail before any export unless every downstream immutable spec is sealed."""
    result = {}
    outputs: set[str] = set()
    preflight_path = PREFLIGHT_REPORT.relative_to(ROOT).as_posix()
    for spec_name, attempt_name, output_name, run, _fit_attempt in EXPORTS:
        path = CAMPAIGN / f"specs/{spec_name}.sealed.json"
        spec = read_json(path)
        argv = spec.get("argv", [])
        expected = ["{python}", "scripts/predict_run.py", run, "valid", "--epoch", "3",
                    "--inference-precision", "fp32", "--output-name", output_name]
        if (spec.get("name") != spec_name or spec.get("kind") != "gpu" or argv != expected
                or preflight_path not in spec.get("inputs", [])):
            raise ValueError(f"fixed FP32 export spec changed or lacks provenance input: {spec_name}")
        limits = spec.get("limits", {})
        if (limits.get("cpus") != 8 or limits.get("ram_bytes") != 25 * 1024**3
                or limits.get("gpu_memory_bytes") != 4_200_000_000
                or limits.get("gpu_stop_bytes") != 4_800_000_000
                or limits.get("timeout_seconds") != 900):
            raise ValueError(f"E426 export resource envelope changed: {spec_name}")
        expected_artifacts = {
            f"cache/pred_{output_name}_valid.npy",
            f"cache/pred_{output_name}_valid.json",
        }
        artifact_paths = {item["path"] if isinstance(item, dict) else item
                          for item in spec.get("artifacts", [])}
        if artifact_paths != expected_artifacts or outputs.intersection(artifact_paths):
            raise ValueError(f"export artifact paths collide or changed: {spec_name}")
        outputs.update(artifact_paths)
        result[spec_name] = spec

    score_path = CAMPAIGN / f"specs/{SCORE_SPEC}.sealed.json"
    score = read_json(score_path)
    score_argv = ["{python}",
                  "research/campaign_2026-10-04/staging/e426_clock_comparison/scripts/score_e426_clocks.py",
                  "--output",
                  "research/campaign_2026-10-04/staging/e426_clock_comparison/results/e426_clock_screen_v2.json"]
    if (score.get("name") != SCORE_SPEC or score.get("kind") != "cpu"
            or score.get("argv") != score_argv or preflight_path not in score.get("inputs", [])
            or not outputs.issubset(set(score.get("inputs", [])))):
        raise ValueError("paired-score spec identity, argv, or input closure changed")
    score_limits = score.get("limits", {})
    if (score_limits.get("cpus") != 1 or score_limits.get("ram_bytes") != 2 * 1024**3
            or score_limits.get("gpu_memory_bytes") != 0
            or score_limits.get("gpu_stop_bytes") != 0
            or score_limits.get("timeout_seconds") != 900):
        raise ValueError("paired-score resource envelope changed")
    score_artifacts = {item["path"] if isinstance(item, dict) else item
                       for item in score.get("artifacts", [])}
    if score_artifacts != {score_argv[-1]}:
        raise ValueError("paired-score output path changed")
    result[SCORE_SPEC] = score
    return result


def npy_header(path: Path) -> tuple[dict, int]:
    """Read a .npy header without importing NumPy or mapping its large payload."""
    with path.open("rb") as source:
        if source.read(6) != b"\x93NUMPY":
            raise ValueError(f"not a NumPy .npy file: {path}")
        version = source.read(2)
        if version == b"\x01\x00":
            length = struct.unpack("<H", source.read(2))[0]
        elif version in (b"\x02\x00", b"\x03\x00"):
            length = struct.unpack("<I", source.read(4))[0]
        else:
            raise ValueError(f"unsupported NumPy format version in {path}: {version!r}")
        if not 1 <= length <= 1_000_000:
            raise ValueError(f"invalid NumPy header size: {length}")
        header = ast.literal_eval(source.read(length).decode("latin1").strip())
        offset = source.tell()
    return header, offset


def validate_export(clock: str, output_name: str, run: str, fit_attempt: str,
                    spec: dict, record: dict) -> dict[str, object]:
    artifact_hashes = require_artifacts(spec, record)
    checkpoint_relative = f"artifacts/runs/{run}/model_ep3.pt"
    fit_record = read_json(RUNNER / f"attempts/{fit_attempt}.json")
    fit_spec = read_json(CAMPAIGN / f"specs/{run}.sealed.json")
    fit_artifacts = {item["path"]: item["sha256"] for item in fit_record.get("artifacts", [])}
    checkpoint = ROOT / checkpoint_relative
    if (fit_record.get("status") != "finished" or fit_record.get("exit_code") != 0
            or fit_record.get("teardown_verified") is not True or fit_record.get("spec") != fit_spec
            or fit_artifacts.get(checkpoint_relative) != sha256_file(checkpoint)):
        raise ValueError(f"source checkpoint is not bound by its successful fit: {clock}")
    pred_relative = f"cache/pred_{output_name}_valid.npy"
    sidecar_relative = f"cache/pred_{output_name}_valid.json"
    if set(artifact_hashes) != {pred_relative, sidecar_relative}:
        raise ValueError(f"unexpected prediction artifact paths: {clock}")
    pred = ROOT / pred_relative
    sidecar_path = ROOT / sidecar_relative
    sidecar = read_json(sidecar_path)
    manifest = read_json(ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json")
    header, offset = npy_header(pred)
    if (tuple(header.get("shape", ())) != (ROW_COUNT, SEQUENCE_LENGTH, OUTPUTS)
            or header.get("fortran_order") is not False
            or header.get("descr") not in ("<f4", "=f4")):
        raise ValueError(f"prediction array shape, order, or dtype changed: {clock}: {header}")
    expected_bytes = offset + ROW_COUNT * SEQUENCE_LENGTH * OUTPUTS * 4
    if pred.stat().st_size != expected_bytes:
        raise ValueError(f"prediction file payload length changed: {clock}")
    groups = sidecar.get("row_groups")
    if (not isinstance(groups, list) or len(groups) != ROW_COUNT
            or any(type(group) is not int for group in groups)
            or groups != list(range(ROW_COUNT))):
        raise ValueError(f"prediction sidecar does not list typed canonical VALID IDs: {clock}")
    if (sidecar.get("run") != run or sidecar.get("split") != "valid"
            or sidecar.get("epoch") != 3
            or sidecar.get("checkpoint_sha256") != sha256_file(checkpoint)
            or sidecar.get("prediction_sha256") != sha256_file(pred)
            or sidecar.get("split_manifest_sha256") != hashlib.sha256(
                (ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_bytes()
            ).hexdigest()
            or sidecar.get("inference_precision") != "fp32"
            or manifest.get("evaluation", {}).get("valid") is None):
        raise ValueError(f"prediction sidecar does not bind fixed checkpoint/export: {clock}")
    return {"attempt": record["id"], "spec_sha256": record["spec_sha256"],
            "checkpoint_sha256": sha256_file(checkpoint),
            "prediction_sha256": sha256_file(pred),
            "sidecar_sha256": sha256_file(sidecar_path),
            "shape": [ROW_COUNT, SEQUENCE_LENGTH, OUTPUTS], "dtype": header["descr"],
            "row_groups": ROW_COUNT, "clock": clock}


def validate_score_output(spec: dict, record: dict) -> dict[str, object]:
    paths = require_artifacts(spec, record)
    relative = next(iter(paths))
    report_path = ROOT / relative
    report = read_json(report_path)
    if (report.get("metric_id") != "global_wp_clip2_v1"
            or report.get("n_sequences") != 468
            or report.get("screen") != read_json(
                CAMPAIGN / "staging/e426_clock_comparison/specs/e426_clock_screen_v1.json"
            )):
        raise ValueError("paired fold-0 report failed its fixed metric/panel contract")
    return {"attempt": record["id"], "spec_sha256": record["spec_sha256"],
            "report_sha256": sha256_file(report_path), "report_path": relative}


def run_job(spec_name: str, attempt_name: str, kind: str,
            completed: list[str]) -> tuple[dict, dict[str, object]]:
    guard()
    spec_path = CAMPAIGN / f"specs/{spec_name}.sealed.json"
    spec = read_json(spec_path)
    if spec.get("name") != spec_name or spec.get("kind") != kind:
        raise ValueError(f"sealed job identity changed: {spec_name}")
    record_path = RUNNER / f"attempts/{attempt_name}.json"
    if record_path.is_file():
        record = read_json(record_path)
        if record.get("spec") != spec:
            raise ValueError(f"existing attempt uses different sealed spec: {attempt_name}")
        if record.get("status") not in ("reserved", "running", "finished"):
            raise RuntimeError(f"refusing to retry stopped attempt {attempt_name}")
    else:
        output_paths = [ROOT / (item["path"] if isinstance(item, dict) else item)
                        for item in spec.get("artifacts", [])]
        if any(path.exists() for path in output_paths):
            raise FileExistsError(f"outputs exist without an attempt record: {attempt_name}")
        if kind == "gpu":
            budget = require_gpu_reservation(spec)
            save("reserving_export", completed, next_attempt=attempt_name,
                 gpu_budget=budget)
        subprocess.run([sys.executable, "scripts/run_campaign.py", "launch",
                        str(spec_path), "--attempt", attempt_name], cwd=ROOT, check=True)
        if not record_path.is_file():
            raise FileNotFoundError(f"runner did not create attempt record: {attempt_name}")
        record = read_json(record_path)

    deadline = time.monotonic() + int(spec["limits"]["timeout_seconds"]) + 240
    while True:
        guard()
        record = read_json(record_path)
        if record.get("spec") != spec:
            raise ValueError(f"attempt spec identity changed while running: {attempt_name}")
        if record.get("status") == "finished":
            record = successful_attempt(attempt_name, spec)
            break
        if record.get("status") not in ("reserved", "running"):
            raise RuntimeError(f"attempt stopped; refusing retry: {attempt_name}")
        expected = f"wnn-c04-{attempt_name}.service"
        if record.get("unit") != expected:
            raise RuntimeError(f"worker unit identity changed for {attempt_name}")
        observed = properties(expected)
        if (observed.get("LoadState") != "loaded"
                or observed.get("ActiveState") not in ("active", "activating")):
            raise RuntimeError(f"worker service disappeared: {expected}: {observed}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"observation timed out; do not restart: {attempt_name}")
        save("running", completed, active_attempt=attempt_name, observed=observed)
        time.sleep(5)
    if kind == "gpu":
        _spec_name, _attempt, output_name, run, fit_attempt = next(
            item for item in EXPORTS if item[1] == attempt_name
        )
        clock = run.split("_")[2]
        result = validate_export(clock, output_name, run, fit_attempt, spec, record)
    else:
        result = validate_score_output(spec, record)
    if attempt_name not in completed:
        completed.append(attempt_name)
    save("running", completed)
    return record, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.resume:
        raise FileExistsError(f"existing queue state requires explicit resume: {OUTPUT}")
    state = read_json(OUTPUT) if OUTPUT.exists() else {"completed": []}
    if state.get("status") in ("complete", "stopped"):
        raise RuntimeError(f"queue is terminal and will not retry: {state}")
    completed = list(state.get("completed", []))
    try:
        sealed_specs = require_sealed_job_set()
        gates = wait_for_gates()
        save("gates_passed", completed, gates=gates)
        for spec_name, attempt_name, *_ in EXPORTS:
            if attempt_name in completed:
                spec = sealed_specs[spec_name]
                record = successful_attempt(attempt_name, spec)
                _spec_name, _attempt, output_name, run, fit_attempt = next(
                    item for item in EXPORTS if item[1] == attempt_name
                )
                clock = run.split("_")[2]
                result = validate_export(clock, output_name, run, fit_attempt, spec, record)
                continue
            _, result = run_job(spec_name, attempt_name, "gpu", completed)
            save("running", completed, last_export=result)
        if SCORE_ATTEMPT not in completed:
            _, score = run_job(SCORE_SPEC, SCORE_ATTEMPT, "cpu", completed)
        else:
            spec = sealed_specs[SCORE_SPEC]
            record = successful_attempt(SCORE_ATTEMPT, spec)
            score = validate_score_output(spec, record)
        save("complete", completed, gates=gates, score=score)
        return 0
    except BaseException as error:
        save("stopped", completed,
             reason=f"{type(error).__name__}: {error}",
             no_automatic_retry=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
