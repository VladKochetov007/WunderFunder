"""Independent scalar/lineage audit; hashes output bytes, never loads tensor values."""
from __future__ import annotations

import hashlib
import json
import math
import resource
import stat
import time
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
R6 = HERE.parent
ROOT = HERE.parents[4]
ACTOR = R6 / "state_drift_v2"
OUTPUT = ACTOR / "real_probe_2222r"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def finite(value: Any) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def classify(effects: dict, floors: dict, rule: dict) -> tuple[str, dict, dict]:
    detectable = {key: value["rms_difference"] > rule["each_required_effect_must_exceed_independent_floor_times"] * floors[key]
                  for key, value in effects.items()}
    terms = {
        "hidden": detectable["hidden"] and effects["hidden"]["relative_l2"] >= rule["hidden_relative_rms_at_least"],
        "prediction": detectable["prediction"] and effects["prediction"]["rms_difference"] >= rule["main_prediction_rms_at_least"],
        "gradient_or_update": any(detectable[key] and effects[key]["relative_l2"] >= rule["whole_gradient_relative_l2_at_least_OR_whole_update_relative_l2_at_least"] for key in ("gradient", "update")),
    }
    outcome = "MATERIAL_LOCAL_DRIFT" if all(terms.values()) else "DETECTABLE_SUBTHRESHOLD" if any(detectable.values()) else "UNRESOLVED_AT_NUMERIC_FLOOR"
    return outcome, detectable, terms


def main() -> None:
    started = time.perf_counter()
    receipt_path = R6 / "state_drift_approved_2222r.json"
    receipt = read(receipt_path)
    inventory = read(R6 / "state_drift_review_v2_binding/candidate_runtime_hashes.json")
    comparison = read(OUTPUT / "comparison.json")
    frozen = read(OUTPUT / "independent_floors.json")
    policy = read(OUTPUT / "default_baselines/frozen_floor.json")
    inputs = read(OUTPUT / "input_hashes.json")
    prereg = read(ACTOR / "PREREG.json")
    job_path = ROOT / "research/campaign_2026-09-29/jobs/c20261003_state_drift_v2_2222r.json"
    job = read(job_path)
    incident_dir = R6.parent / "runtime_incidents/payload_permission_2220"
    incident = read(incident_dir / "REPORT.json")
    failed = read(incident_dir / "failed_job.json")
    assert receipt["status"] == "approved" and receipt["default_v4_gate"] == receipt["independent_state_drift_review"] == "PASS"
    assert receipt["candidate_sha256"] == inventory["candidate_sha256"]
    assert receipt["runtime_sha256"] == inventory["runtime_sha256"]
    assert receipt["runtime_limits"] == inventory["runtime_limits"]
    for name, expected in receipt["candidate_sha256"].items():
        assert sha(ACTOR / name) == expected
    for name, expected in receipt["runtime_sha256"].items():
        assert sha(ROOT / name) == expected
    for name, expected in inventory["source"].items():
        assert sha(ROOT / name) == expected == job["source_sha256"][name]
    review = R6 / "state_drift_review_v2_binding"
    assert receipt["review_bindings"] == read(review / "HASHES.json")
    assert all(sha(review / name) == value for name, value in receipt["review_bindings"].items())
    for key in ("pre", "post"):
        path = receipt["launch"][key]
        assert inputs[path] == receipt["capture_sha256"][key] == inventory["declared_real_normal_capture_sha256"][key]
    trace = Path(receipt["launch"]["trajectory"])
    assert inputs[str(trace)] == sha(trace) == receipt["capture_sha256"]["trajectory"] == inventory["trajectory_sha256"]
    assert inputs[str(receipt_path)] == sha(receipt_path)
    assert Path(receipt["launch"]["output"]) == OUTPUT
    assert receipt["launch"]["receipt"] == str(receipt_path)
    assert Path(receipt["launch"]["entrypoint"]) == ACTOR / "run_real.py"
    assert job["command"][:2] == ["research/campaign_2026-10-03/continuation/run_gpu.sh", "research/campaign_2026-10-03/continuation/round6/controller_state_drift_payload.sh"]
    args = dict(zip(job["command"][2::2], job["command"][3::2], strict=True))
    assert all(args["--" + key] == receipt["launch"][key] for key in ("pre", "post", "trajectory", "receipt", "output"))
    assert job["status"] == "finished" and job["exit_code"] == 0 and job["failure"] is None
    assert job["timeout_seconds"] == 320 and job["elapsed_seconds"] < 300
    assert job["peak_observed_vram_bytes"] <= 5500000000
    assert datetime.fromisoformat(job["finished_at"]).timestamp() < 1791057600
    assert datetime.fromisoformat(receipt["controller_attested_at"]).timestamp() < job["started_unix"]
    properties = job["runtime_provenance"]["effective_slice_properties"]
    assert properties["CPUQuotaPerSecUSec"] == "8s" and properties["MemoryMax"] == str(24 * 1024**3) and properties["MemorySwapMax"] == "0"
    assert incident["status"] == "DIAGNOSED_PERMISSION_FIXED_BYTES_UNCHANGED"
    assert incident["Python_started"] is False and incident["CUDA_work_started"] is False and incident["new_OOM"] is False
    assert failed["exit_code"] == 126
    assert "Permission denied" in (incident_dir / "failed_job.log").read_text()
    wrapper = ROOT / "research/campaign_2026-10-03/continuation/round6/controller_state_drift_payload.sh"
    assert stat.S_IMODE(wrapper.stat().st_mode) == 0o755
    assert sha(wrapper) == incident["wrapper_sha256"] == receipt["runtime_sha256"][str(wrapper.relative_to(ROOT))]
    assert comparison["independent_floors"] == frozen
    assert sha(OUTPUT / "default_baselines/frozen_floor.json") == frozen["policy_sha256"]
    assert policy["repeats"] == 6 and policy["automatic_relaxation"] is False and policy["multiplier"] == 2
    assert policy["ceiling_rule"] == {"abs": 1e-5, "rel": 1e-4}
    assert all(value == 0 for value in policy["limits"].values())
    assert all(value == 0 for value in policy["empirical_floors"].values())
    assert all(value == 0 for value in frozen["floors_rms"].values())
    finite(comparison)
    finite(frozen)
    finite(policy)
    outcome, detectable, terms = classify(comparison["effects"], frozen["floors_rms"], prereg["material_local_drift_rule"])
    assert outcome == comparison["outcome"] == "DETECTABLE_SUBTHRESHOLD"
    assert detectable == comparison["detectable"] and all(detectable.values())
    assert not any(terms.values())
    for key in ("prediction", "gradient", "update"):
        effect = comparison["effects"][key]
        assert math.isclose(effect["relative_l2"], effect["rms_difference"] / max(effect["reference_rms"], 1e-12), rel_tol=1e-14)
    planted = {key: {"rms_difference": 1, "relative_l2": 1} for key in detectable}
    assert classify(planted, frozen["floors_rms"], prereg["material_local_drift_rule"])[0] == "MATERIAL_LOCAL_DRIFT"
    zeros = {key: {"rms_difference": 0, "relative_l2": 0} for key in detectable}
    assert classify(zeros, frozen["floors_rms"], prereg["material_local_drift_rule"])[0] == "UNRESOLVED_AT_NUMERIC_FLOOR"
    output_hashes = {}
    output_bytes = 0
    for path in sorted(OUTPUT.rglob("*")):
        if not path.is_file():
            continue
        before = path.stat()
        digest = sha(path)  # bounded raw byte hash; no tensor load/deserialization
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        name = str(path.relative_to(OUTPUT))
        output_hashes[name] = {"sha256": digest, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns}
        output_bytes += before.st_size
    assert len(policy["sample_hashes"]) == 6
    assert all(output_hashes["default_baselines/" + name]["sha256"] == value for name, value in policy["sample_hashes"].items())
    mtime = {name: value["mtime_ns"] for name, value in output_hashes.items()}
    assert max(mtime["default_baselines/" + name] for name in policy["sample_hashes"]) <= mtime["default_baselines/frozen_floor.json"] <= mtime["independent_floors.json"] <= mtime["binary.pt"] <= mtime["comparison.json"] <= mtime["input_hashes.json"]
    source = (ACTOR / "protocol.py").read_text()
    floor_write = source.index('(output / "independent_floors.json").write_text')
    counterfactual = source.index("alternative, intervention_pre = local_update")
    assert floor_write < counterfactual
    assert 'cap.compare(baseline["target"], alternative["target"])' in source
    assert 'for key in ("rng", "clock", "scheduler", "scaler", "h_ema"):' in source
    assert 'if key != "h":' in source
    assert 'cap.compare(payload[key], observer.pre[key]' in source
    log_path = ROOT / job["log"]
    assert read(log_path) == comparison
    panel_path = R6.parent / "round4/live_capture_v4/fit_panel.json"
    panel = read(panel_path)
    trace_data = read(trace)
    assert sha(panel_path) == trace_data["panel_sha256"]
    ids = {"rows": panel["rows"], "selection_rule": panel["rule"],
           "prior_exposure": "same28TRAIN+4fitVALID fresh seed3 panel exposed in default-v4 capture and prior independent measurement review; not a new holdout",
           "prefix": [0, 4000], "local_chunk": [4000, 5000]}
    (HERE / "IDs.json").write_text(json.dumps(ids, indent=2) + "\n")
    rows = panel["rows"]
    assert len(rows) == 32 and sum(row["split"] == "train" for row in rows) == 28
    record = {
        "verdict": "PASS_RECORDED_LOCAL_DETECTABLE_SUBTHRESHOLD",
        "outcome": outcome, "materiality_terms": terms, "detectable": detectable,
        "effects": comparison["effects"], "floor_values": frozen["floors_rms"],
        "job": {key: job[key] for key in ("name", "status", "exit_code", "elapsed_seconds", "peak_observed_vram_bytes", "started_at", "finished_at")},
        "permission_retry_verified": True, "new_OOM_reported": False,
        "output_byte_hashes": output_hashes, "bytes_streamed_once": output_bytes,
        "input_capture_rehashed": False, "input_hash_agreement_with_receipt_and_prior_review": True,
        "preservation_evidence": "hash-bound executed source assertions and success; no independent tensor comparison/replay",
        "floor_order_evidence": "hash-bound source plus consistent artifact mtimes, no independent event trace",
        "scientific_limits": "single early exposed panel; warmed empirical floors; no component attribution/fullfit/WP benefit",
        "provenance": {str(path.relative_to(ROOT)): sha(path) for path in
                       [receipt_path, job_path, log_path, incident_dir / "REPORT.json", incident_dir / "failed_job.json", incident_dir / "failed_job.log", trace, ACTOR / "PREREG.json", ACTOR / "protocol.py", ACTOR / "run_real.py", review / "HASHES.json", review / "candidate_runtime_hashes.json"]},
        "seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "GPU_calls": 0, "GRU_forwards": 0, "tensor_deserializations": 0,
    }
    (HERE / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verdict": record["verdict"], "materiality_terms": terms,
                      "bytes_streamed_once": output_bytes, "seconds": record["seconds"],
                      "peak_rss_kib": record["peak_rss_kib"]}, indent=2))


if __name__ == "__main__":
    main()
