"""Bounded source/metadata-only terminal supplement; never execute receipt helper."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
R6 = HERE.parent
ROOT = HERE.parents[4]
PRIOR = R6 / "state_drift_review_v2"
ACTOR = R6 / "state_drift_v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    options = parser.parse_args()
    started = time.perf_counter()
    prior = json.loads((PRIOR / "results.json").read_text())
    assert prior["passed"] and len(prior["checks"]) == 24 and all(prior["checks"].values())
    actor_state = json.loads((ACTOR / "STATE.json").read_text())
    actor_result = json.loads((ACTOR / "results.json").read_text())
    assert actor_state["state"] == "FROZEN_V2_CPU_PASS_TERMINAL_READY_GPU_UNAPPROVED"
    assert len(actor_result["checks"]) == 49 and all(actor_result["checks"].values())
    actor_manifest = json.loads((ACTOR / "HASHES.json").read_text())
    for name, expected in prior["actor_hashes_verified"].items():
        assert sha(ACTOR / name) == expected == actor_manifest[name]
    runtime = {name: sha(ROOT / name) for name in prior["runtime_hashes"]}
    source_map = {name: sha(ROOT / name) for name in prior["reviewed_native_sources"]}
    assert runtime == prior["runtime_hashes"] == actor_state["runtime_sha256"]
    assert source_map == prior["reviewed_native_sources"]
    assert len(runtime) == 4 and len(source_map) == 9
    tree = ast.parse((ACTOR / "run_real.py").read_text())
    own = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "own_hashes")
    module = ast.fix_missing_locations(ast.Module(body=ast.parse("from __future__ import annotations").body + [own], type_ignores=[]))
    ns = {"HERE": ACTOR, "cap": SimpleNamespace(sha=sha)}
    exec(compile(module, "isolated_own_hashes", "exec"), ns)
    candidate = ns["own_hashes"]()
    assert candidate == actor_state["candidate_sha256"] and len(candidate) == 5
    constructor = R6 / "approve_state_drift.py"
    text = constructor.read_text()
    ast.parse(text)  # source syntax only; constructor/main/imports never executed
    assert 'review = HERE / "state_drift_review_v2_binding"' in text
    receipt_at = text.index("receipt = {")
    metadata_raise = text.index('raise ValueError("Reviewed trajectory, measurement review or provenance helper changed")')
    fields = ["trajectory_sha256", "independent_measurement_review_state_sha256", "metadata_helper_sha256"]
    assert all(text.index('inventory["' + key + '"]') < metadata_raise < receipt_at for key in fields)
    output_raise = text.index('raise ValueError("Independently reviewed capture manifest changed")')
    capture_raise = text.index('raise ValueError("Capture files differ from measured manifest")')
    assert text.index('inventory["output_manifest_sha256"]') < output_raise < text.index("observed_capture =") < capture_raise < receipt_at
    measurement = R6 / "capture_measurement_review/STATE.json"
    measure_manifest = json.loads((measurement.parent / "HASHES.json").read_text())
    assert sha(measurement) == measure_manifest["STATE.json"]
    measured = json.loads(measurement.read_text())
    assert measured["verdict"] == "PASS_MEASURED_PARITY_FOR_RECORDED_WARM_V4_DEFAULT_TRAJECTORIES_ONLY"
    assert not measured["pending"]
    trajectory = R6 / "real_trajectory.json"
    helper = R6 / "prepare_capture_metadata.py"
    for path in (measurement, trajectory, helper):
        assert sha(path) == prior["metadata_and_helper_hashes"][str(path.relative_to(ROOT))]
    trace = json.loads(trajectory.read_text())
    assert trace["measured_capture_review_sha256"] == sha(measurement)
    assert trace["completed_warm_updates"] == 4 and trace["warm_skips"] == 0
    assert "no saved intermediate weight trace" in trace["provenance_limit"]
    output = R6.parent / "round4/gpu_capture_v4_default_2120r/output_hashes.json"
    verified_file = measurement.parent / "output_byte_verification.json"
    assert sha(verified_file) == measure_manifest[verified_file.name]
    verified = json.loads(verified_file.read_text())
    declared = json.loads(output.read_text())["files"]
    declared_capture = {kind: declared[f"real_fit_panel/normal/{kind}.pt"] for kind in ("pre", "post")}
    assert all(value == verified["files"][f"real_fit_panel/normal/{kind}.pt"]["sha256"] for kind, value in declared_capture.items())
    state = json.loads((HERE / "STATE.json").read_text())
    assert state["verdict"] == "PASS"
    inventory = {
        "verdict": state["verdict"], "candidate_sha256": candidate,
        "source": source_map, "runtime_sha256": runtime,
        "trajectory_sha256": sha(trajectory),
        "independent_measurement_review_state_sha256": sha(measurement),
        "metadata_helper_sha256": sha(helper), "output_manifest_sha256": sha(output),
        "controller_constructor_sha256": sha(constructor),
        "controller_constructor_path": str(constructor.relative_to(ROOT)),
        "controller_sha_helpers_sha256": sha(R6.parent / "round4/approve_capture.py"),
        "runtime_limits": prior["runtime_limits"],
        "independent_state_drift_review": "PASS",
        "declared_real_normal_capture_sha256": declared_capture,
        "capture_payloads_rehashed_by_critic": False,
        "prior_review_results_sha256": sha(PRIOR / "results.json"),
        "author_terminal_state_sha256": sha(ACTOR / "STATE.json"),
        "author_terminal_manifest_sha256": sha(ACTOR / "HASHES.json"),
        "GPU_authorization": False,
        "trajectory_limit": trace["provenance_limit"],
    }
    if options.freeze:
        for name in ("REPORT.txt", "STATE.json"):
            assert sha(PRIOR / name) == sha(PRIOR / name.replace(".", "_DRAFT_BLOCK.", 1))
        old_manifest = PRIOR / "HASHES.json"
        if not old_manifest.exists():
            write_new(old_manifest, {p.name: sha(p) for p in sorted(PRIOR.iterdir()) if p.is_file() and p != old_manifest})
        else:
            assert all(sha(PRIOR / name) == expected for name, expected in json.loads(old_manifest.read_text()).items())
        write_new(HERE / "candidate_runtime_hashes.json", inventory)
        write_new(HERE / "results.json", {
            "passed": True, "review_type": "current constructor source/metadata binding supplement",
            "reused_tests": 24, "reused_actor_controls": 49, "controls_rerun": 0,
            "current_constructor_sha256": sha(constructor),
            "all_four_required_metadata_pins_verified_in_current_source": True,
            "old_BLOCK_manifest_sha256": sha(old_manifest),
            "actual_tensor_reads": 0, "payload_rehashes": 0, "GPU_calls": 0,
            "constructor_executed": False,
        })
        write_new(HERE / "COST.json", {
            "CPU_bodies": 1, "seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "resource_wrapper": "run_cpu.sh reviewer 1core2GiB900s",
        })
        write_new(HERE / "HASHES.json", {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "HASHES.json"})
    else:
        assert inventory == json.loads((HERE / "candidate_runtime_hashes.json").read_text())
    assert all(sha(HERE / name) == expected for name, expected in json.loads((HERE / "HASHES.json").read_text()).items())
    print("PASS: current constructor's metadata and output-manifest pins verified; exact inventory and supplemental manifest frozen. No payload reads or receipts.")


if __name__ == "__main__":
    main()
