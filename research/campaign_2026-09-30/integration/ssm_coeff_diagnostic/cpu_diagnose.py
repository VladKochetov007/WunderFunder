"""Bounded saved-synthetic sensitivity proof and optional captured-coefficient replay."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

from support import ATOL, MODEL_HASH, RTOL, coefficients, cpu_tree, difference, literal, sha
from wnn.structured_state import StructuredStateModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def coordinates(states, target) -> list[dict[str, object]]:
    return [{"block": 1, "index": index, "state": float(states[1][0, index]), "saved_gpu": float(target[1][0, index]), "error": float(states[1][0, index] - target[1][0, index])} for index in [22, 61]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic output exists")
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert sha(ROOT / "src/wnn/structured_state.py") == MODEL_HASH
    sys.path.insert(0, str(ROOT / "research/campaign_2026-09-30/audit"))
    from grade_structured_state import reference
    capture = torch.load(args.capture, weights_only=True, map_location="cpu") if args.capture else None
    cases = []
    for arm in ["multi", "single"]:
        stem = HERE.parent / f"gpu_probe_{arm}_v2"
        profile = json.loads(stem.with_suffix(".json").read_text())
        assert sha(stem.with_suffix(".cpu_replay.pt")) == profile["cpu_replay_sha256"]
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved = torch.load(stem.with_suffix(".cpu_replay.pt"), weights_only=True, map_location="cpu")
        model = StructuredStateModel(saved["n_in"], saved["n_out"], arm)
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
        x = saved["input"]
        with torch.no_grad():
            base_coeff = cpu_tree(coefficients(model))
            baseline, states, traces = literal(model, x, base_coeff)
            independent, independent_states = reference(model, x, model.reset(1))
            assert torch.equal(baseline, independent)
            assert all(torch.equal(a, b) for a, b in zip(states, independent_states, strict=True))
            case = {"arm": arm, "replay_sha256": sha(stem.with_suffix(".cpu_replay.pt")), "literal_exact_to_independent_reference": True,
                    "baseline_saved_GPU_comparison": {"prediction": difference(saved["prediction"], baseline), "states": [difference(a, b) for a, b in zip(saved["state"], states, strict=True)]}, "baseline_failed_coordinates": coordinates(states, saved["state"]), "one_ULP_sensitivity": []}
            for block in [0, 1]:
                for field in ["scalar_rho", "scale", "rotation_a", "rotation_b"]:
                    for direction in [-1, 1]:
                        perturbed = cpu_tree(base_coeff)
                        original = perturbed[block][field]
                        perturbed[block][field] = torch.nextafter(original, torch.full_like(original, direction * torch.inf))
                        out, changed, _ = literal(model, x, perturbed)
                        case["one_ULP_sensitivity"].append({"block": block, "field": field, "direction": direction, "coefficient_max_abs_change": float((perturbed[block][field] - original).abs().max()), "state1_vs_baseline": difference(changed[1], states[1]), "prediction_vs_baseline": difference(out, baseline), "failed_coordinates": coordinates(changed, saved["state"])})
            if capture is not None:
                actual = next(item for item in capture["cases"] if item["arm"] == arm)
                assert actual["input_replay_sha256"] == case["replay_sha256"] and capture["model_source_sha256"] == MODEL_HASH
                assert torch.equal(actual["input"], x)
                assert all(torch.equal(value, actual["model"][name]) for name, value in saved["model"].items())
                assert all(torch.equal(value, actual["cpu_coefficients"][i][name]) for i, c in enumerate(base_coeff) for name, value in c.items())
                variants = [
                    ("GPU_coefficients_CPU_maps", actual["gpu_coefficients"], None),
                    ("CPU_coefficients_GPU_fixed_drives", base_coeff, [trace["drive"] for trace in actual["gpu_block_traces"]]),
                    ("GPU_coefficients_GPU_fixed_drives", actual["gpu_coefficients"], [trace["drive"] for trace in actual["gpu_block_traces"]]),
                    ("GPU_coefficients_CPU_fixed_drives", actual["gpu_coefficients"], [trace["drive"] for trace in traces]),
                ]
                results = []
                for name, coeff, drives in variants:
                    out, changed, _ = literal(model, x, coeff, drives)
                    results.append({"variant": name, "GPU_prediction_vs_CPU": difference(actual["gpu_prediction"], out), "GPU_states_vs_CPU": [difference(a, b) for a, b in zip(actual["gpu_state"], changed, strict=True)], "failed_coordinates": coordinates(changed, actual["gpu_state"])})
                case["captured_coefficients_replay"] = results
                case["GPU_input_drive_timeline_vs_CPU_baseline"] = [{name: difference(gpu[name], cpu[name]) for name in ["input", "drive", "timeline"]} for gpu, cpu in zip(actual["gpu_block_traces"], traces, strict=True)]
            cases.append(case)
    assert not torch.cuda.is_initialized()
    cgroup = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
    scope = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    limits = {name: (scope / name).read_text().strip() for name in ["memory.max", "memory.swap.max", "cpu.max"]}
    assert limits["memory.max"] == "2147483648" and limits["memory.swap.max"] == "0"
    quota, period = map(int, limits["cpu.max"].split())
    assert quota <= period
    report = {"diagnostic_only": True, "model_source_unchanged": MODEL_HASH, "tolerances_unchanged": {"atol": ATOL, "rtol": RTOL}, "cases": cases,
              "CUDA_initialized": False, "synthetic_only": True, "competition_data_read": False, "new_optimizer_updates": 0,
              "capture_sha256": sha(args.capture) if args.capture else None, "support_sha256": sha(HERE / "support.py"), "script_sha256": sha(Path(__file__)),
              "independent_reference_sha256": sha(ROOT / "research/campaign_2026-09-30/audit/grade_structured_state.py"),
              "duration_seconds": time.monotonic() - started, "peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "cgroup": cgroup, "verified_limits": limits,
              "interpretation": "Sensitivity only cannot establish CPU/CUDA mismatch cause. Captured-coefficient/fixed-drive variants diagnose coefficient and map/FFT contributions; no model/tolerance change or inference serialization approved."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "duration_seconds": report["duration_seconds"], "peak_rss_KiB": report["peak_rss_KiB"]}))


if __name__ == "__main__":
    main()
