from __future__ import annotations

import ast
import argparse
import hashlib
import json
import math
import resource
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any  # Saved heterogeneous capture/JSON structures require dynamic traversal.

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[4]
ROUND4 = ROOT / "research/campaign_2026-10-03/continuation/round4"
OUTPUT = ROUND4 / "gpu_capture_v4_default_2120r"
ACTOR = ROUND4 / "live_capture_v4"
JOB = ROOT / "research/campaign_2026-09-29/jobs/c20261003_capture_v4_default_2120r.json"
RECEIPT = ROUND4 / "controller_receipts/capture_v4_2120.json"


def load_json(path: Path) -> Any:
    """Load only small JSON metadata."""
    return json.loads(path.read_text())


def finite_json(value: Any) -> None:
    """Reject every nonfinite JSON number, including nested numeric leaves."""
    if type(value) is float:
        assert math.isfinite(value), "nonfinite JSON number"
    elif isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)


def main() -> None:
    """Audit bytes once, then inspect authenticated capture metadata on CPU only."""
    started = time.monotonic()
    hashes, sizes, mtimes = {}, {}, {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-verified-byte-pass", action="store_true")
    options = parser.parse_args()
    if options.resume_verified_byte_pass:
        saved = load_json(OWN / "output_byte_verification.json")
        assert saved["all_match"]
        for name, record in saved["files"].items():
            path = OUTPUT / name
            stat = path.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtime_ns"])
            key = str(path.resolve())
            hashes[key], sizes[key], mtimes[key] = (
                record["sha256"],
                record["bytes"],
                record["mtime_ns"],
            )

    def digest(path: Path) -> str:
        key = str(path.resolve())
        if key not in hashes:
            initial = path.stat()
            value = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    value.update(chunk)
            final = path.stat()
            assert (initial.st_size, initial.st_mtime_ns) == (final.st_size, final.st_mtime_ns)
            hashes[key] = value.hexdigest()
            sizes[key], mtimes[key] = final.st_size, final.st_mtime_ns
        return hashes[key]

    job, receipt, manifest = (
        load_json(JOB),
        load_json(RECEIPT),
        load_json(OUTPUT / "output_hashes.json"),
    )
    assert receipt["status"] == "approved" and receipt["queue_finished"] is True
    assert job["status"] == "finished" and job["exit_code"] == 0 and job["failure"] is None
    expected_command = [
        "research/campaign_2026-10-03/continuation/run_gpu.sh",
        "research/campaign_2026-10-03/continuation/round4/controller_gpu_payload.sh",
        str(RECEIPT.relative_to(ROOT)),
        str(OUTPUT.relative_to(ROOT)),
    ]
    assert job["command"] == expected_command and job["kind"] == "gpu"
    assert datetime.fromisoformat(receipt["controller_attested_at"]) <= datetime.fromisoformat(
        job["started_at"]
    )
    assert job["elapsed_seconds"] < 590 and job["timeout_seconds"] == 600
    assert job["peak_observed_vram_bytes"] <= 5_500_000_000
    assert datetime.fromisoformat(job["finished_at"]).timestamp() < 1791057600
    recorded = manifest["files"]
    actual_names = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name != "output_hashes.json"
    }
    assert set(recorded) == actual_names, "missing/unindexed output artifact"
    verified_output = {}
    for name, expected in recorded.items():
        path = OUTPUT / name
        assert path.resolve().is_relative_to(OUTPUT.resolve())
        actual = digest(path)
        assert actual == expected, "output hash mismatch: " + name
        verified_output[name] = {
            "sha256": actual,
            "bytes": sizes[str(path.resolve())],
            "mtime_ns": mtimes[str(path.resolve())],
        }
    # Record the single byte pass immediately, before optional mmap inspection.
    if not options.resume_verified_byte_pass:
        (OWN / "output_byte_verification.json").write_text(
            json.dumps(
                {
                    "all_match": True,
                    "chunk_bytes": 1048576,
                    "files": verified_output,
                    "total_output_bytes": sum(row["bytes"] for row in verified_output.values()),
                    "manifest_sha256": digest(OUTPUT / "output_hashes.json"),
                },
                indent=2,
            )
            + "\n"
        )
    else:
        assert digest(OUTPUT / "output_hashes.json") == saved["manifest_sha256"]
    request = load_json(ACTOR / "review_request.json")
    assert receipt["candidate_sha256"] == request["candidate_sha256"]
    assert receipt["source"] == request["source"] == manifest["source"]
    bindings = {}
    for group, base in (("candidate_sha256", ACTOR), ("source", ROOT), ("runtime_sha256", ROOT)):
        for name, expected in receipt[group].items():
            actual = digest(base / name)
            assert actual == expected, group + " drift: " + name
            bindings[group + ":" + name] = actual
    for name, expected in receipt["source"].items():
        assert job["source_sha256"][name] == expected, "job source binding differs"
    for name, expected in receipt["review_bindings"].items():
        assert digest(ROUND4 / "capture_review_v4" / name) == expected
    for name, expected in load_json(ACTOR / "generated_hashes.json").items():
        assert digest(ACTOR / name) == expected
    panel = load_json(ACTOR / "fit_panel.json")
    panel_sha = digest(ACTOR / "fit_panel.json")
    assert manifest["panel_sha256"] == panel_sha
    provenance_path = ROOT / "artifacts/runs/c20261003_combo_s3_f0/provenance.json"
    assert digest(provenance_path) == panel["provenance_sha256"]
    split_path = Path(panel["manifest_path"])
    assert digest(split_path) == panel["manifest_sha256"]
    provenance, split = load_json(provenance_path), load_json(split_path)
    assert len(panel["rows"]) == 32 and panel["scored"] is False
    for name, binding in panel["bindings"].items():
        path = ROOT / name
        if path.suffix == ".json":
            assert digest(path) == binding["sha256"]
        if "size" in binding:
            assert path.stat().st_size == binding["size"]
        if "mtime_ns" in binding:
            assert path.stat().st_mtime_ns == binding["mtime_ns"]
    for kind, count in (("train", 28), ("valid", 4)):
        selected = [row["row_group"] for row in panel["rows"] if row["split"] == kind]
        assert len(selected) == count
        eligible = set(provenance["fit_row_groups"][kind])
        assert selected == sorted(eligible - ({528} if kind == "train" else set()))[:count]
        assert set(selected) <= set(split["fit"][kind])
        # This manifest records evaluation VALID groups; linked TRAIN exclusions
        # need the component construction, not an invented evaluation.train key.
        if kind in split["evaluation"]:
            assert not set(selected) & set(split["evaluation"][kind])
    results, policies = {}, {}
    for branch, result_path in (
        ("synthetic_actual_architecture", OUTPUT / "synthetic_actual_architecture/results.json"),
        ("real_fit_panel", OUTPUT / "results.json"),
    ):
        result = load_json(result_path)
        finite_json(result)
        assert result["source"] == receipt["source"]
        for key in (
            "normal_max_delta",
            "repeat_max_delta",
            "skip_max_delta",
            "uninstrumented_default_max_delta",
        ):
            assert type(result[key]) in (int, float) and result[key] == 0
        for key in (
            "normal_deltas",
            "repeat_deltas",
            "skip_deltas",
            "uninstrumented_default_deltas",
        ):
            assert result[key] and all(
                type(value) in (int, float) and value == 0 for value in result[key].values()
            )
        assert result["all_four_components"] == ["ema", "main", "next", "struct"]
        assert result["gpu_parity_established"] is True
        floor_summary = result["independent_floor"]
        assert floor_summary["online_difference_used_to_calibrate"] is False
        assert floor_summary["repeats_per_control"] == 6
        for control, label in (("normal_baselines", "normal"), ("skip_baselines", "skip")):
            directory = OUTPUT / branch / control
            policy = load_json(directory / "frozen_floor.json")
            finite_json(policy)
            assert policy["device"] == "cuda" and policy["repeats"] == 6
            assert policy["source"] == receipt["source"]
            assert policy["automatic_relaxation"] is False and policy["multiplier"] == 2
            assert policy["ceiling_rule"] == {"abs": 1e-5, "rel": 1e-4}
            assert all(value == 0 for value in policy["limits"].values())
            assert all(value == 0 for value in policy["empirical_floors"].values())
            assert set(policy["sample_hashes"]) == {f"original_default_{i}.pt" for i in range(6)}
            for name, expected in policy["sample_hashes"].items():
                assert recorded[str((directory / name).relative_to(OUTPUT))] == expected
            floor_name = str((directory / "frozen_floor.json").relative_to(OUTPUT))
            assert floor_summary[label + "_sha256"] == recorded[floor_name]
            assert floor_summary[label + "_max_floor"] == 0
            ordered = all(
                (directory / name).stat().st_mtime_ns
                <= (directory / "frozen_floor.json").stat().st_mtime_ns
                for name in policy["sample_hashes"]
            )
            assert ordered
            policies[branch + ":" + control] = {
                "samples": policy["sample_hashes"],
                "policy_sha256": recorded[floor_name],
                "original_mtimes_before_floor": ordered,
                "floor_before_results_mtime": (directory / "frozen_floor.json").stat().st_mtime_ns
                <= result_path.stat().st_mtime_ns,
            }
        results[branch] = {
            key: value for key, value in result.items() if not key.endswith("deltas")
        }
    # Only source-order inspection: no imports of actor validators/private CUDA engines.
    runner_tree = ast.parse((ACTOR / "runner.py").read_text())
    function = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_probe"
    )
    lines = [
        (node.lineno, ast.unparse(node))
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    ]
    normal_freeze = min(line for line, call in lines if call.startswith("calibrate(observed.pre,"))
    normal_compare = min(
        line for line, call in lines if call.startswith("compare_policy(observed.post,")
    )
    skip_freeze = min(line for line, call in lines if call.startswith("calibrate(skipped.pre,"))
    skip_compare = min(
        line for line, call in lines if call.startswith("compare_policy(skipped.post,")
    )
    assert normal_freeze < normal_compare and skip_freeze < skip_compare
    audit = {
        "job": {
            key: job[key]
            for key in (
                "name",
                "status",
                "exit_code",
                "command",
                "elapsed_seconds",
                "peak_observed_vram_bytes",
                "started_at",
                "finished_at",
            )
        },
        "binding_checks_pass": True,
        "bindings": bindings,
        "panel_sha256": panel_sha,
        "rows": panel["rows"],
        "policies": policies,
        "results": results,
        "ordering_evidence": "hash-bound source and file mtimes; no independent comparison-event timestamp trace",
        "normal_freeze_source_line": normal_freeze,
        "normal_compare_source_line": normal_compare,
        "skip_freeze_source_line": skip_freeze,
        "skip_compare_source_line": skip_compare,
        "external_full_cache_rehashed_by_reviewer": False,
        "external_cache_limit": "small sidecars/stat checked; full cache hashes enforced by source-bound successful real_environment entry, not independently reread",
        "full_output_byte_passes": 1,
        "metadata_resumed_after_schema_key_correction": options.resume_verified_byte_pass,
        "wall_seconds": time.monotonic() - started,
    }
    (OWN / "lineage_and_numeric_results.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {
                "bindings_pass": True,
                "output_files": len(recorded),
                "output_bytes": sum(verified_output[name]["bytes"] for name in verified_output),
                "controls": len(policies),
                "hash_wall_seconds": time.monotonic() - started,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
    )


if __name__ == "__main__":
    main()
