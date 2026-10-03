"""Post-freeze leaf guards, runtime schema, lineage metadata; no capture/model reads."""
from __future__ import annotations

import ast
import copy
import json
import math
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from identity_controls import extract, sha

HERE = Path(__file__).resolve().parent
R6 = HERE.parent
ROOT = HERE.parents[4]
ACTOR = R6 / "state_drift_v2"


def expect_reject(function: Any, *args: Any) -> bool:
    try:
        function(*args)
    except (ValueError, RuntimeError):
        return True
    return False


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(1)
    state = json.loads((ACTOR / "STATE.json").read_text())
    if "FROZEN" not in json.dumps(state):
        raise RuntimeError("Author not terminal frozen")
    actor_result = json.loads((ACTOR / "results.json").read_text())
    assert actor_result["passed"] and all(actor_result["checks"].values())
    manifest = json.loads((ACTOR / "HASHES.json").read_text())
    names = ["PREREG.json", "protocol.py", "run_real.py", "runtime.py", "controls.py",
             "STATE.json", "results.json", "REPORT.md"]
    actor_hashes = {name: sha(ACTOR / name) for name in names}
    assert all(manifest[name] == value for name, value in actor_hashes.items())
    checks = {}
    nameset = {"finite_tensor", "finite_metrics", "finite_hidden", "finite_post", "vector", "observable", "rms"}
    ns = extract(ACTOR / "protocol.py", nameset, {"torch": torch, "math": math})
    checks["nan_prediction_rejected"] = expect_reject(ns["observable"], {"out": torch.tensor([float("nan")])}, "prediction")
    checks["nan_rms_rejected"] = expect_reject(ns["rms"], torch.tensor([float("nan")]))
    checks["empty_prediction_rejected"] = expect_reject(ns["observable"], {"out": torch.tensor([])}, "prediction")
    checks["nonfinite_stale_hidden_rejected"] = expect_reject(ns["finite_hidden"], {"h": torch.tensor([float("inf")]), "h_ema": torch.zeros(1)})
    checks["nested_nan_metric_rejected"] = expect_reject(ns["finite_metrics"], {"effects": [{"rms": float("nan")} ]})
    checks["empty_vector_rejected"] = expect_reject(ns["vector"], {})
    assert ns["rms"](torch.tensor([3.0, 4.0])) == math.sqrt(12.5)
    checks["finite_rms_positive"] = True
    warm = extract(ACTOR / "run_real.py", {"warm_counts"}, {"torch": torch})["warm_counts"]
    payload = {"clock": {"skips": torch.tensor(0, dtype=torch.int64), "attempts": 4,
                         "step_count": 4, "ema_updates": 4}, "ema_n": 4,
               "scheduler": {"last_epoch": 4}, "optimizer": {"state": {0: {"step": 4}}}}
    checks["four_completed_original_eligibility"] = warm(payload) == (4, 0)
    wrong = copy.deepcopy(payload)
    wrong["clock"]["skips"] = torch.tensor(2, dtype=torch.int64)
    wrong["optimizer"]["state"][0]["step"] = 2
    checks["warm_skip_amendment_not_admitted_v2"] = expect_reject(warm, wrong)
    runtime_tree = ast.parse((ACTOR / "runtime.py").read_text())
    constants = {}
    # Controlled literal assignments only. No module imports or code execution.
    for node in runtime_tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in {"DEADLINE", "ALLOCATOR_BYTES", "OBSERVED_BYTES", "PAYLOAD"}:
                constants[key] = ast.literal_eval(node.value)
    const_ns = dict(constants)
    for node in runtime_tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"RUNTIME_FILES", "LIMITS"}:
            const_ns[node.targets[0].id] = eval(compile(ast.Expression(node.value), "reviewed_constants", "eval"), {"__builtins__": {}}, const_ns)
    files, limits = const_ns["RUNTIME_FILES"], const_ns["LIMITS"]
    runtime_hashes = {name: sha(ROOT / name) for name in files}
    receipt = {"runtime_sha256": runtime_hashes, "runtime_limits": limits}
    env = {"WNN_IN_CAMPAIGN_JOB": "1", "WNN_SLICE": "wnn-campaign-gpu.slice",
           "WNN_RESEARCH_DEADLINE_UTC": str(constants["DEADLINE"]),
           "WNN_CAMPAIGN_VRAM_BYTES": str(constants["OBSERVED_BYTES"]),
           "WNN_STATE_DRIFT_PAYLOAD_SHA256": runtime_hashes[constants["PAYLOAD"]]}
    runtime_ns = {**const_ns, "ROOT": ROOT, "math": math,
                  "os": SimpleNamespace(environ=env),
                  "time": SimpleNamespace(time=lambda: constants["DEADLINE"] - 326)}
    old_protocol = sys.modules.get("protocol")
    sys.modules["protocol"] = SimpleNamespace(cap=SimpleNamespace(sha=sha))
    try:
        rt = extract(ACTOR / "runtime.py", {"verify_contract", "scope_guard", "check_time", "check_memory", "allocator_fraction"}, runtime_ns)
        rt["scope_guard"](receipt)
        checks["current_payload_handshake_positive"] = True
        bad = copy.deepcopy(receipt)
        bad["runtime_sha256"][constants["PAYLOAD"]] = "0" * 64
        checks["wrong_payload_hash_rejected"] = expect_reject(rt["verify_contract"], bad)
        bad = copy.deepcopy(receipt)
        bad["runtime_limits"]["work_timeout_seconds"] = 301
        checks["changed_runtime_limit_rejected"] = expect_reject(rt["verify_contract"], bad)
        env["WNN_STATE_DRIFT_PAYLOAD_SHA256"] = "wrong"
        checks["wrong_execution_handshake_rejected"] = expect_reject(rt["scope_guard"], receipt)
        env["WNN_STATE_DRIFT_PAYLOAD_SHA256"] = runtime_hashes[constants["PAYLOAD"]]
        checks["absolute_deadline_rejected"] = expect_reject(rt["check_time"], constants["DEADLINE"])
        checks["device_overcap_rejected"] = expect_reject(rt["check_memory"], constants["OBSERVED_BYTES"] + 1)
        fraction = rt["allocator_fraction"](8 * 1024 ** 3)
        checks["original_allocator_decimal_cap"] = fraction * (8 * 1024 ** 3) == 4_800_000_000
    finally:
        if old_protocol is None:
            del sys.modules["protocol"]
        else:
            sys.modules["protocol"] = old_protocol
    trace_path = R6 / "real_trajectory.json"
    trace = json.loads(trace_path.read_text())
    review = R6 / "capture_measurement_review/STATE.json"
    review_state = json.loads(review.read_text())
    checks["independent_default_terminal_pass"] = review_state["verdict"] == "PASS_MEASURED_PARITY_FOR_RECORDED_WARM_V4_DEFAULT_TRAJECTORIES_ONLY" and not review_state["pending"]
    checks["trajectory_bound_review_state"] = trace["measured_capture_review_sha256"] == sha(review)
    checks["metadata_original_four_completed_zero_skips"] = trace["completed_warm_updates"] == 4 and trace["warm_skips"] == 0
    checks["metadata_source_inferred_initial_none_four_starts"] = trace["initial_student_h"] is None and trace["initial_ema_h"] is None and trace["warm_starts"] == [0, 1000, 2000, 3000]
    checks["metadata_discloses_no_intermediate_weight_trace"] = trace["provenance_limit"] == "Source-inferred initial states and loop; no saved intermediate weight trace"
    current_sources = {name: sha(ROOT / name) for name in trace["source"]}
    checks["trajectory_native_source_hashes_current"] = current_sources == trace["source"]
    v4 = R6.parents[0] / "round4/live_capture_v4"
    checks["trajectory_adapter_runner_fitpanel_hashes_current"] = all(
        trace[key] == sha(v4 / name) for key, name in
        [("capture_runner_sha256", "runner.py"), ("fit_panel_source_sha256", "fit_panel.py"), ("adapter_sha256", "live_capture.py")])
    # Recheck the tiny byte grammar on the terminal source, not its draft hash.
    import hashlib
    identity_ns = {"torch": torch, "json": json, "hashlib": hashlib}
    helper_identity = extract(R6 / "prepare_capture_metadata.py", {"identity"}, identity_ns.copy())["identity"]
    actor_identity = extract(ACTOR / "protocol.py", {"tensor_identity"}, identity_ns.copy())["tensor_identity"]
    checks["terminal_identity_grammar_equivalence"] = all(helper_identity(x) == actor_identity(x) for x in
        [torch.arange(12, dtype=torch.float32).reshape(3, 4), torch.arange(262145, dtype=torch.float32)])
    assert all(checks.values()), checks
    reviewed = [R6 / "prepare_capture_metadata.py", trace_path, review,
                R6 / "capture_measurement_review/REPORT.md", R6 / "controller_state_drift_payload.sh"]
    result = {"passed": True, "checks": checks, "actor_terminal_state": state,
              "actor_stored_control_checks": actor_result["checks"], "actor_hashes_verified": actor_hashes,
              "runtime_hashes": runtime_hashes, "runtime_limits": limits,
              "metadata_and_helper_hashes": {str(p.relative_to(ROOT)): sha(p) for p in reviewed},
              "reviewed_native_sources": current_sources,
              "trajectory_identity_verification": "metadata strings and synthetic grammar only; no real tensor-byte rehash",
              "seconds": time.perf_counter() - started,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "actual_tensor_reads": 0, "GRU_forwards": 0, "GPU_calls": 0, "actor_control_reruns": 0}
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "checks": checks, "seconds": result["seconds"], "peak_rss_kib": result["peak_rss_kib"]}, indent=2))


if __name__ == "__main__":
    main()
