"""Build unsealed, hash-resolved E462/E450 quiet-window timing specs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
STAGING = CAMPAIGN / "staging/timing_window"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def code_paths_from_spec(path: Path) -> set[str]:
    spec = json.loads(path.read_text())
    return {item["path"] if isinstance(item, dict) else item for item in spec.get("code", [])}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def cpu_spec(name: str, argv: list[str], code: set[str], inputs: set[str],
             artifacts: list[str]) -> dict[str, object]:
    return {
        "version": 1, "name": name, "kind": "cpu", "argv": ["{python}", *argv],
        "code": sorted(code), "inputs": sorted(inputs), "artifacts": artifacts,
        "limits": {"cpus": 1, "ram_bytes": 2 * 1024**3, "gpu_memory_bytes": 0,
                   "gpu_stop_bytes": 0, "timeout_seconds": 1800},
        "environment": {},
    }


def build() -> dict[str, object]:
    e462_timing = CAMPAIGN / "staging/e462/time_public_q8.py"
    e462_build_dir = CAMPAIGN / "E462_public_q8_build_v2"
    e462_parity = CAMPAIGN / "E462_public_q8_parity_v2/parity_report.json"
    e462_score = CAMPAIGN / "E462_public_q8_score_v4/report/score_report.json"
    e462_build = e462_build_dir / "build_report.json"
    e457_data = ROOT / "cache/valid_subset20.parquet"
    e457_ready = CAMPAIGN / "E457_head_transfer/readiness.json"
    e462_parity_spec = CAMPAIGN / "specs/e462_public_q8_parity_v2.sealed.json"
    e462_score_spec = CAMPAIGN / "specs/e462_public_q8_score_v4.sealed.json"
    e462_build_spec = CAMPAIGN / "specs/e462_public_q8_build_v3.sealed.json"
    e462_attempts = (
        CAMPAIGN / "runner/attempts/e462-public-q8-build-v3.json",
        CAMPAIGN / "runner/attempts/e462-public-q8-parity-v2.json",
        CAMPAIGN / "runner/attempts/e462-public-q8-score-v4.json",
    )
    e462_outputs = [
        "research/campaign_2026-10-04/staging/timing_window/results/e462_q8_v1/trials.jsonl",
        "research/campaign_2026-10-04/staging/timing_window/results/e462_q8_v1/timing_report.json",
    ]
    e462_code = code_paths_from_spec(e462_score_spec)
    e462_code.update(code_paths_from_spec(e462_parity_spec))
    e462_code.add(rel(e462_timing))
    e462_inputs = {
        "submissions/c20261004_combo_s3_i16/submission.zip",
        "research/campaign_2026-10-04/E462_public_q8_build_v2/q8r/submission.zip",
        "research/campaign_2026-10-04/E462_public_q8_build_v2/q8x/submission.zip",
        rel(e462_build), rel(e462_parity), rel(e462_score), rel(e457_data), rel(e457_ready),
        rel(e462_build_spec), rel(e462_parity_spec), rel(e462_score_spec),
        *(rel(path) for path in e462_attempts),
    }
    e462_argv = [
        "research/campaign_2026-10-04/staging/e462/time_public_q8.py",
        "--control", "submissions/c20261004_combo_s3_i16/submission.zip",
        "--q8r", "research/campaign_2026-10-04/E462_public_q8_build_v2/q8r/submission.zip",
        "--q8x", "research/campaign_2026-10-04/E462_public_q8_build_v2/q8x/submission.zip",
        "--build-dir", "research/campaign_2026-10-04/E462_public_q8_build_v2",
        "--parity-report", rel(e462_parity), "--score-report", rel(e462_score),
        "--data", rel(e457_data), "--output", e462_outputs[0],
        "--cpu", "15", "--max-load1", "1.0", "--paired-repeats", "5",
    ]
    e462_spec = cpu_spec("e462_public_q8_timing_v1", e462_argv, e462_code,
                         e462_inputs, e462_outputs)

    seed4_script = STAGING / "time_seed4_i16.py"
    seed4_ready = CAMPAIGN / "checks/e450_seed4_native_parity_v1/readiness.json"
    seed4_attempt_path = CAMPAIGN / "runner/attempts/e450-seed4-native-parity-v1.json"
    seed4_parity_spec = CAMPAIGN / "specs/e450_seed4_native_parity_v1.sealed.json"
    seed4_zip = ROOT / "submissions/c20261004_combo_s4_i16/submission.zip"
    seed3_zip = ROOT / "submissions/c20261004_combo_s3_i16/submission.zip"
    seed4_outputs = [
        "research/campaign_2026-10-04/staging/timing_window/results/e450_seed4_i16_v1/trials.jsonl",
        "research/campaign_2026-10-04/staging/timing_window/results/e450_seed4_i16_v1/timing_report.json",
    ]
    seed4_code = code_paths_from_spec(seed4_parity_spec)
    seed4_code.update(e462_code)
    seed4_code.add(rel(seed4_script))
    seed4_inputs = {
        rel(seed3_zip), rel(seed4_zip), rel(seed4_ready), rel(seed4_attempt_path),
        rel(seed4_parity_spec), rel(e457_data), rel(e457_ready), rel(e462_timing),
    }
    seed4_argv = [
        rel(seed4_script), "--control", rel(seed3_zip), "--seed4", rel(seed4_zip),
        "--readiness", rel(seed4_ready), "--parity-attempt", rel(seed4_attempt_path),
        "--data", rel(e457_data), "--output", seed4_outputs[0],
        "--expected-e462-timing-sha256", sha256_file(e462_timing),
        "--cpu", "15", "--max-load1", "1.0", "--repeats", "5",
        "--projection-limit-seconds", "3120",
    ]
    seed4_spec = cpu_spec("e450_seed4_i16_timing_v1", seed4_argv, seed4_code,
                          seed4_inputs, seed4_outputs)

    e462_draft = STAGING / "specs/e462_public_q8_timing_v1.draft.json"
    seed4_draft = STAGING / "specs/e450_seed4_i16_timing_v1.draft.json"
    write_json(e462_draft, e462_spec)
    write_json(seed4_draft, seed4_spec)
    return {
        "status": "draft_unsealed",
        "e462_timing_source_sha256": sha256_file(e462_timing),
        "seed4_timing_source_sha256": sha256_file(seed4_script),
        "timing_controller_sha256": sha256_file(STAGING / "run_timing_window.py"),
        "e425_controller_sha256_unchanged": sha256_file(CAMPAIGN / "checks/run_e425_sequence.py"),
        "oneshot_unit_sha256": sha256_file(STAGING / "systemd/wnn-timing-window-v1.service"),
        "e425_dependency_dropin_sha256": sha256_file(
            STAGING / "systemd/wnn-e425-sequence-v1.service.d/20-timing-window.conf"),
        "seed3_i16_zip_sha256": sha256_file(seed3_zip),
        "seed4_i16_zip_sha256": sha256_file(seed4_zip),
        "e450_readiness_sha256": sha256_file(seed4_ready),
        "e450_parity_attempt_sha256": sha256_file(seed4_attempt_path),
        "e462_parity_report_sha256": sha256_file(e462_parity),
        "e462_score_report_sha256": sha256_file(e462_score),
        "e462_spec": rel(e462_draft),
        "seed4_spec": rel(seed4_draft),
        "worker_limits": {"cpus": 1, "ram_bytes": 2 * 1024**3,
                          "timeout_seconds": 1800, "gpu_memory_bytes": 0},
        "timing_design": {
            "cpu": 15, "quiet_load1_max": 1.0, "rows_per_trial": 60_000,
            "full_streams_per_trial": 3, "alternating_pairs": 5,
            "e462_timing_trials": 20, "seed4_timing_trials": 10,
            "full_data_projection_rows": 39_000_000,
            "projection_limit_seconds": 3120,
        },
    }


def main() -> None:
    summary = build()
    write_json(STAGING / "timing_specs.draft.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
