"""Narrow AST-extracted source adversaries; no model or capture imports."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ACTOR = HERE.parent / "state_drift"
V4 = HERE.parents[1] / "round4/live_capture_v4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    """Compile selected function bodies only; never execute module import statements."""
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Selected source functions missing")
    future = ast.parse("from __future__ import annotations").body
    module = ast.fix_missing_locations(ast.Module(body=future + nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(1)
    gate = extract(ACTOR / "run_real.py", {"warm_counts"}, {"torch": torch})["warm_counts"]
    def warm_payload(skips: int, step: int) -> dict[str, Any]:
        return {"clock": {"skips": torch.tensor(skips, dtype=torch.int64),
                          "attempts": 4, "step_count": 4, "ema_updates": 4},
                "ema_n": 4, "scheduler": {"last_epoch": 4},
                "optimizer": {"state": {0: {"step": step}}}}
    complete = gate(warm_payload(0, 4))
    amended = gate(warm_payload(2, 2))
    assert complete == (4, 0) and amended == (2, 2)
    try:
        gate(warm_payload(2, 4))
    except ValueError:
        mismatched_rejected = True
    else:
        mismatched_rejected = False
    assert mismatched_rejected
    ns = extract(ACTOR / "protocol.py", {"vector", "observable", "rms"}, {"torch": torch})
    nan_prediction = ns["observable"]({"out": torch.tensor([[[float("nan"), 0.0]]])}, "prediction")
    nan_rms = ns["rms"](nan_prediction)
    finite_violation_accepted = math.isnan(nan_rms)
    assert finite_violation_accepted
    # These are the actual outcome decision comparisons, using invalid prediction
    # and otherwise zero identity residuals: invalid silently falls through.
    effect_rms = {"hidden": 0.0, "prediction": nan_rms, "gradient": 0.0, "update": 0.0}
    detectable = {k: v > 0.0 for k, v in effect_rms.items()}
    nan_outcome = "DETECTABLE_SUBTHRESHOLD" if any(detectable.values()) else "UNRESOLVED_AT_NUMERIC_FLOOR"
    assert nan_outcome == "UNRESOLVED_AT_NUMERIC_FLOOR"
    try:
        ns["observable"]({"preclip": {"p": torch.tensor([float("nan")])}}, "gradient")
    except ValueError:
        gradient_nonfinite_rejected = True
    else:
        gradient_nonfinite_rejected = False
    assert gradient_nonfinite_rejected
    runtime_cases = {}
    for name, now, env in [
        ("just_before", 1791057599, {"WNN_IN_CAMPAIGN_JOB": "1", "WNN_SLICE": "wnn-campaign-gpu.slice"}),
        ("at_deadline", 1791057600, {"WNN_IN_CAMPAIGN_JOB": "1", "WNN_SLICE": "wnn-campaign-gpu.slice"}),
        ("outside_scope", 1791057599, {}),
    ]:
        guard = extract(V4 / "runner.py", {"runtime_guard"},
                        {"os": SimpleNamespace(environ=env), "time": SimpleNamespace(time=lambda: now)})
        try:
            guard["runtime_guard"]()
        except (RuntimeError, ValueError):
            runtime_cases[name] = "BLOCK"
        else:
            runtime_cases[name] = "PASS_ENTRY_ONLY"
    assert runtime_cases == {"just_before": "PASS_ENTRY_ONLY", "at_deadline": "BLOCK", "outside_scope": "BLOCK"}
    manifest = json.loads((ACTOR / "HASHES.json").read_text())
    actor_names = ["PREREG.json", "PREREG_WARM_CLOCK_CLARIFICATION.json", "STATE.json",
                   "results.json", "REPORT.md", "SCHEMA.json", "protocol.py", "run_real.py", "controls.py"]
    actor_hashes = {name: sha(ACTOR / name) for name in actor_names}
    assert all(manifest[name] == value for name, value in actor_hashes.items())
    actor_results = json.loads((ACTOR / "results.json").read_text())
    assert len(actor_results["checks"]) == 13 and all(actor_results["checks"].values())
    source_binding = json.loads((V4 / "source_binding.json").read_text())
    current_source = {name: sha(ROOT / name) for name in source_binding}
    assert current_source == source_binding == actor_results["source_hashes"]
    adapters = {name: sha(V4 / name) for name in actor_results["adapter_hashes"]}
    assert adapters == actor_results["adapter_hashes"]
    runtime_files = ["scripts/campaign_job.py", "scripts/limited",
                     "research/campaign_2026-10-03/continuation/run_gpu.sh",
                     "research/campaign_2026-10-03/continuation/round4/controller_gpu_payload.sh"]
    result = {
        "four_completed_fixture": list(complete), "amended_two_completed_fixture": list(amended),
        "inconsistent_warm_clock_rejected": mismatched_rejected,
        "nan_prediction_not_rejected": finite_violation_accepted,
        "nan_identity_outcome": nan_outcome,
        "nonfinite_gradient_rejected": gradient_nonfinite_rejected,
        "runtime_entry_controls": runtime_cases,
        "stored_actor_checks_passed": 13,
        "consumed_actor_artifact_hashes_verified": actor_hashes,
        "current_source_binding_verified": current_source,
        "adapter_hashes_verified": adapters,
        "runtime_hashes": {name: sha(ROOT / name) for name in runtime_files},
        "test_plan_sha256": sha(HERE / "TEST_PLAN.json"),
        "seconds_including_hashes": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "actual_capture_reads": 0, "actual_result_reads": 0, "model_forwards": 0,
        "gpu_calls": 0, "actor_controls_rerun": 0,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if "hash" not in k and "binding" not in k}, indent=2))


if __name__ == "__main__":
    main()
