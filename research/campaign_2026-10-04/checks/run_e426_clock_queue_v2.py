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
PREFLIGHT_SPEC = "e426_fit_provenance_preflight_v1"
PREFLIGHT_REPORT = CAMPAIGN / "checks/e426_fit_provenance_preflight_v1.json"
TIMING_RELEASE = CAMPAIGN / "checks/ROOT_TIMING_INCIDENT_RELEASE.json"
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
EXPECTED_SCORE_DRAFT_SHA256 = "b07f124853b540460247c1bd503b222dbeeb50e000d81e7bacc31c60b2f7d66a"
LOCAL_E426_GPU_CAP_SECONDS = 24 * 60 * 60
ROW_COUNT = 1_873
SEQUENCE_LENGTH = 20_000
OUTPUTS = 2
QUEUE_STAGES = ("timing_gate", "provenance_preflight", "three_exports", "score_seal", "score")


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
         "-p", "SubState", "-p", "ExecMainStatus", "-p", "MainPID"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def terminal_service_success(observed: dict[str, str]) -> bool:
    """Accept a completed RemainAfterExit oneshot or an inactive clean worker."""
    if observed.get("LoadState") != "loaded" or observed.get("ExecMainStatus") != "0":
        return False
    return (observed.get("ActiveState") == "inactive"
            or (observed.get("ActiveState") == "active" and observed.get("SubState") == "exited"))


def require_attempt_status(status: str) -> None:
    if status not in ("reserved", "running", "finished"):
        raise RuntimeError(f"refusing to retry stopped attempt status: {status}")


def require_pinned_file(path: Path, expected_sha256: str) -> str:
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"pinned file changed: {path}")
    return digest


def require_paths_ready(paths: set[str], root: Path = ROOT) -> None:
    missing = sorted(path for path in paths if not (root / path).is_file())
    if missing:
        raise FileNotFoundError(f"required future inputs are not ready: {missing}")


def verify_timing_incident_release(timing: dict, timing_service: dict[str, str]) -> dict:
    """Allow this science-only queue after the single root-released wrapper failure."""
    attempt_id = "e450-seed4-i16-timing-v2"
    if (RUNNER / "attempts/e462-public-q8-timing-v2.json").exists():
        raise RuntimeError("incident release cannot authorize a prior or retried q8 timing attempt")
    attempt_path = RUNNER / f"attempts/{attempt_id}.json"
    log_path = RUNNER / "logs/e450-seed4-i16-timing-v2.log"
    attempt = read_json(attempt_path)
    worker_unit = f"wnn-c04-{attempt_id}.service"
    worker = properties(worker_unit)
    if (timing.get("status") != "failed_closed"
            or timing_service.get("ExecMainStatus") != "1"
            or attempt.get("id") != attempt_id or attempt.get("status") != "failed"
            or attempt.get("exit_code") != 1 or attempt.get("teardown_verified") is not True
            or attempt.get("failure_category") != "command_exit"
            or attempt.get("peak_observed_vram_bytes") != 0
            or attempt.get("unit") != worker_unit or not terminal_worker_failure(worker)
            or not log_path.is_file()):
        raise RuntimeError("timing failure does not match the root releaseable pre-native incident")
    events = attempt.get("final_memory_events", {})
    if any(int(events.get(key, 0)) != 0 for key in ("oom", "oom_group_kill", "oom_kill")):
        raise RuntimeError("timing incident shows OOM; root release cannot authorize this failure")
    release = read_json(TIMING_RELEASE)
    require_pinned_file(
        TIMING_RELEASE,
        "46cc9ac549765a46510a1e86d6303525c334c12052735015322cb8118d443547",
    )
    failed = release.get("failed_worker", {})
    terminal = release.get("terminal_services", {})
    expected_root_worker = {"LoadState": "loaded", "ActiveState": "failed",
                            "SubState": "failed", "MainPID": "0", "ExecMainStatus": "1"}
    expected = {
        "format": "wnn.root-timing-incident-release-v1",
        "version": 1,
        "status": "approved",
        "classification": "relative_path_before_native_timing",
        "permitted_scope": "package_timing_deferred_science_only",
        "timing_result": {"path": TIMING_RESULT.relative_to(ROOT).as_posix(),
                           "sha256": sha256_file(TIMING_RESULT)},
        "failed_worker": {
            "attempt_id": attempt_id,
            "record_path": attempt_path.relative_to(ROOT).as_posix(),
            "record_sha256": sha256_file(attempt_path),
            "log_path": log_path.relative_to(ROOT).as_posix(),
            "log_sha256": sha256_file(log_path),
            "spec_sha256": attempt.get("spec_sha256"),
            "unit": worker_unit,
            "exit_code": 1,
            "teardown_verified": True,
            "oom_delta": 0,
            "observed_vram_bytes": 0,
        },
        "zip_readiness_authorized": False,
        "old_attempt_retry_authorized": False,
        "science_release_only": (
            "May run root-reviewed fixed-checkpoint provenance, exports and paired scores "
            "after verifying these exact incident bytes and no live timing worker; no "
            "automatic timing retry or submission promotion."
        ),
    }
    actual = {key: release.get(key) for key in expected if key not in ("failed_worker", "timing_result")}
    if (actual != {key: value for key, value in expected.items()
                   if key not in ("failed_worker", "timing_result")}
            or release.get("timing_result") != expected["timing_result"]
            or any(failed.get(key) != value for key, value in expected["failed_worker"].items())
            or terminal.get(TIMING_SERVICE) != expected_root_worker
            or terminal.get(worker_unit) != expected_root_worker
            or timing_service.get("LoadState") != "loaded"
            or timing_service.get("ActiveState") != "failed"
            or timing_service != expected_root_worker):
        raise RuntimeError("root timing incident release does not match exact failure bytes/scope")
    return {"release_sha256": sha256_file(TIMING_RELEASE),
            "timing_attempt_sha256": expected["failed_worker"]["record_sha256"],
            "timing_log_sha256": expected["failed_worker"]["log_sha256"],
            "scope": expected["permitted_scope"], "zip_readiness_authorized": False,
            "old_attempt_retry_authorized": False}


def terminal_worker_failure(observed: dict[str, str]) -> bool:
    return (observed.get("LoadState") == "loaded"
            and observed.get("ActiveState") == "failed"
            and observed.get("SubState") == "failed"
            and observed.get("MainPID") == "0"
            and observed.get("ExecMainStatus") == "1")


def successful_attempt(name: str, spec: dict) -> dict:
    record = read_json(RUNNER / f"attempts/{name}.json")
    if (record.get("id") != name or record.get("status") != "finished"
            or record.get("exit_code") != 0 or record.get("teardown_verified") is not True
            or record.get("spec") != spec):
        raise RuntimeError(f"attempt is not terminal-successful with exact identity: {name}")
    unit = f"wnn-c04-{name}.service"
    if record.get("unit") != unit:
        raise RuntimeError(f"attempt has the wrong worker service name: {name}")
    deadline = time.monotonic() + 30
    while True:
        observed = properties(unit)
        if observed.get("LoadState") != "loaded":
            raise RuntimeError(f"exact worker service is missing: {unit}: {observed}")
        if observed.get("ActiveState") == "inactive":
            if observed.get("ExecMainStatus") != "0":
                raise RuntimeError(f"exact worker service exited unsuccessfully: {unit}: {observed}")
            break
        if observed.get("ActiveState") not in ("active", "activating", "deactivating"):
            raise RuntimeError(f"exact worker service did not reach clean teardown: {unit}: {observed}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"worker service teardown remained pending: {unit}: {observed}")
        time.sleep(1)
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
        if (fit_service.get("ActiveState") in ("activating", "deactivating")
                or (fit_service.get("ActiveState") == "active"
                    and fit_service.get("SubState") != "exited")):
            time.sleep(5)
            continue
        if fit_service.get("ExecMainStatus") != "0":
            raise RuntimeError(f"E426 fit controller exited unsuccessfully: {fit_service}")

        timing = read_json(TIMING_RESULT)
        timing_service = properties(TIMING_SERVICE)
        if timing_service.get("LoadState") != "loaded":
            raise RuntimeError(f"timing-v2 service is missing: {timing_service}")
        timing_status = timing.get("status")
        incident_release = None
        if timing_status not in ("timing_complete", "timing_deferred", "failed_closed"):
            if timing.get("status") in ("waiting_for_e426", "waiting_for_quiet_window", "running",
                                         "waiting_for_e426_teardown"):
                save("waiting_for_timing_window", [], timing_status=timing.get("status"))
                time.sleep(10)
                continue
            raise RuntimeError(f"timing-v2 sequence failed closed: {timing}")
        if timing_status in ("timing_complete", "timing_deferred"):
            if not terminal_service_success(timing_service):
                time.sleep(5)
                continue
        else:
            incident_release = verify_timing_incident_release(timing, timing_service)

        return {
            "fit_service": fit_service,
            "fit_sequence_sha256": sha256_file(E426_SEQUENCE),
            "timing_service": timing_service,
            "timing_result_sha256": sha256_file(TIMING_RESULT),
            "timing_status": timing_status,
            "incident_release": incident_release,
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


def require_sealed_export_specs() -> dict[str, dict]:
    """Verify the three fixed exporters are sealed after preflight, before GPU use."""
    result = {}
    outputs: set[str] = set()
    preflight_path = PREFLIGHT_REPORT.relative_to(ROOT).as_posix()
    preflight_report = read_json(PREFLIGHT_REPORT)
    preflight_rows = {row.get("run"): row for row in preflight_report.get("fits", [])}
    for spec_name, attempt_name, output_name, run, _fit_attempt in EXPORTS:
        path = CAMPAIGN / f"specs/{spec_name}.sealed.json"
        spec = read_json(path)
        argv = spec.get("argv", [])
        input_hashes = {item["path"]: item["sha256"] for item in spec.get("inputs", [])
                        if isinstance(item, dict)}
        expected = ["{python}", "scripts/predict_run.py", run, "valid", "--epoch", "3",
                    "--inference-precision", "fp32", "--output-name", output_name]
        if (spec.get("name") != spec_name or spec.get("kind") != "gpu" or argv != expected
                or preflight_path not in input_hashes):
            raise ValueError(f"fixed FP32 export spec changed or lacks provenance input: {spec_name}")
        source_row = preflight_rows.get(run)
        if source_row is None:
            raise ValueError(f"provenance report lacks expected fit: {run}")
        source_artifacts = source_row.get("artifacts", {})
        required_source_paths = {
            f"artifacts/runs/{run}/model_ep3.pt",
            f"artifacts/runs/{run}/provenance.json",
            f"artifacts/runs/{run}/training_state.pt",
        }
        bound_sources = {preflight_path: sha256_file(PREFLIGHT_REPORT)}
        bound_sources.update({relative: source_artifacts[relative]
                              for relative in required_source_paths})
        if any(input_hashes.get(relative) != digest for relative, digest in bound_sources.items()):
            raise ValueError(f"export spec does not bind the completed provenance/checkpoint bytes: {spec_name}")
        limits = spec.get("limits", {})
        if (limits.get("cpus") != 8 or limits.get("ram_bytes") != 24 * 1024**3
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

    return result


def seal_score_after_exports() -> dict:
    """Seal the pinned PATH-only score spec only after all exports pass gates."""
    draft_path = CAMPAIGN / f"specs/{SCORE_SPEC}.draft.json"
    sealed_path = CAMPAIGN / f"specs/{SCORE_SPEC}.sealed.json"
    require_pinned_file(draft_path, EXPECTED_SCORE_DRAFT_SHA256)
    draft = read_json(draft_path)
    required_paths = set(draft.get("inputs", []))
    export_outputs = {
        f"cache/pred_{output}_valid.{extension}"
        for _, _, output, _, _ in EXPORTS
        for extension in ("npy", "json")
    }
    preflight_relative = PREFLIGHT_REPORT.relative_to(ROOT).as_posix()
    if not export_outputs.issubset(required_paths) or preflight_relative not in required_paths:
        raise ValueError("pinned score spec omits preregistered prediction/preflight inputs")
    require_paths_ready(required_paths)
    if not sealed_path.exists():
        subprocess.run([sys.executable, "scripts/run_campaign.py", "seal", str(draft_path),
                        str(sealed_path)], cwd=ROOT, check=True)
    score = read_json(sealed_path)
    score_argv = ["{python}",
                  "research/campaign_2026-10-04/staging/e426_clock_comparison/scripts/score_e426_clocks.py",
                  "--output",
                  "research/campaign_2026-10-04/staging/e426_clock_comparison/results/e426_clock_screen_v2.json"]
    expected_fields = {"version", "name", "kind", "argv", "code", "code_sha256",
                       "inputs", "artifacts", "limits", "environment"}
    if set(score) != expected_fields:
        raise ValueError("sealed score spec fields differ from the campaign schema")
    scalar_fields = ("version", "name", "kind", "argv", "artifacts", "limits", "environment")
    if any(draft.get(field) != score.get(field) for field in scalar_fields):
        raise ValueError("sealed score spec differs from the pinned path-only draft")
    declared_code = set(draft.get("code", []))
    code_items = {item["path"]: item for item in score.get("code", [])}
    if (set(code_items) != declared_code
            or "research/campaign_2026-10-04/staging/e426_clock_comparison/tests/test_e426_packet_schema.py"
            not in declared_code):
        raise ValueError("sealed score source closure differs from the pinned draft")
    for relative, item in code_items.items():
        path = ROOT / relative
        if (not path.is_file() or item.get("size") != path.stat().st_size
                or item.get("sha256") != sha256_file(path)):
            raise ValueError(f"sealed score source hash changed: {relative}")
    input_items = {item["path"]: item for item in score.get("inputs", [])}
    if set(input_items) != required_paths:
        raise ValueError("sealed score input closure differs from the pinned draft")
    for relative, item in input_items.items():
        path = ROOT / relative
        if (not path.is_file() or item.get("size") != path.stat().st_size
                or item.get("sha256") != sha256_file(path)):
            raise ValueError(f"sealed score input hash changed: {relative}")
    code_sha = hashlib.sha256(json.dumps(
        score["code"], sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if code_sha != score.get("code_sha256"):
        raise ValueError("sealed score code closure digest is inconsistent")
    if (score.get("name") != SCORE_SPEC or score.get("kind") != "cpu"
            or score.get("argv") != score_argv):
        raise ValueError("paired-score spec identity or argv changed")
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
    return score


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


def validate_preflight_output(spec: dict, record: dict) -> dict[str, object]:
    paths = require_artifacts(spec, record)
    relative = PREFLIGHT_REPORT.relative_to(ROOT).as_posix()
    report = read_json(PREFLIGHT_REPORT)
    if (set(paths) != {relative} or report.get("schema") != "e426-fit-provenance-preflight-v1"
            or report.get("status") != "passed"):
        raise ValueError("E426 fixed-checkpoint provenance preflight failed")
    expected_runs = {
        "e426_f0_constant_s3_4ep_v2",
        "e426_f0_aligned_s3_4ep_v2",
        "e426_f0_shuffled_s3_4ep_v2",
    }
    observed_runs = {row.get("run") for row in report.get("fits", [])}
    if len(observed_runs) != 3 or observed_runs != expected_runs:
        raise ValueError("provenance preflight report does not contain all three fixed arms")
    return {"attempt": record["id"], "spec_sha256": record["spec_sha256"],
            "report_sha256": sha256_file(PREFLIGHT_REPORT), "report_path": relative}


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
        require_attempt_status(record.get("status", ""))
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
    elif attempt_name == PREFLIGHT_ATTEMPT:
        result = validate_preflight_output(spec, record)
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
        gates = wait_for_gates()
        save("gates_passed", completed, gates=gates)
        preflight_spec_path = CAMPAIGN / f"specs/{PREFLIGHT_SPEC}.sealed.json"
        preflight_spec = read_json(preflight_spec_path)
        if PREFLIGHT_ATTEMPT in completed:
            preflight_record = successful_attempt(PREFLIGHT_ATTEMPT, preflight_spec)
            preflight_result = validate_preflight_output(preflight_spec, preflight_record)
        else:
            _, preflight_result = run_job(PREFLIGHT_SPEC, PREFLIGHT_ATTEMPT, "cpu", completed)
        save("preflight_passed", completed, gates=gates,
             preflight=preflight_result)

        sealed_specs = require_sealed_export_specs()
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
        score_spec = seal_score_after_exports()
        save("score_spec_sealed", completed,
             score_spec_sha256=sha256_file(CAMPAIGN / f"specs/{SCORE_SPEC}.sealed.json"))
        if SCORE_ATTEMPT not in completed:
            _, score = run_job(SCORE_SPEC, SCORE_ATTEMPT, "cpu", completed)
        else:
            record = successful_attempt(SCORE_ATTEMPT, score_spec)
            score = validate_score_output(score_spec, record)
        save("complete", completed, gates=gates, score=score)
        return 0
    except BaseException as error:
        save("stopped", completed,
             reason=f"{type(error).__name__}: {error}",
             no_automatic_retry=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
