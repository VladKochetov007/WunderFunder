"""UNLAUNCHED: capture final synthetic SSM GPU coefficients without fitting."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from support import MODEL_HASH, coefficients, cpu_tree, difference, sha
from wnn.resources import configure_torch
from wnn.structured_state import StructuredStateModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def backend_flags() -> dict[str, object]:
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "cuda_matmul_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("exclusive diagnostic output exists")
    started = time.monotonic()
    assert sha(ROOT / "src/wnn/structured_state.py") == MODEL_HASH
    configure_torch(gpu_fraction=0.70)
    flags = backend_flags()
    cases = []
    # Both arms are captured in one root-owned GPU job; neither takes a new step.
    for arm in ["multi", "single"]:
        stem = HERE.parent / f"gpu_probe_{arm}_v2"
        replay_path, checkpoint_path = stem.with_suffix(".cpu_replay.pt"), stem.with_suffix(".pt")
        profile = json.loads(stem.with_suffix(".json").read_text())
        assert sha(replay_path) == profile["cpu_replay_sha256"]
        assert sha(checkpoint_path) == profile["checkpoint_sha256"]
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved = torch.load(replay_path, weights_only=True, map_location="cpu")
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        assert flags == saved["backend_flags"], "fresh-process backend flags differ from failed replay"
        assert saved["reset"] and not saved["autocast_enabled"] and saved["optimizer_updates"] == 10
        assert all(torch.equal(value, checkpoint["model"][name]) for name, value in saved["model"].items())
        model = StructuredStateModel(saved["n_in"], saved["n_out"], arm)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        with torch.no_grad():
            cpu_coeff = cpu_tree(coefficients(model))
        model = model.cuda()
        x = saved["input"].cuda()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            gpu_coeff = cpu_tree(coefficients(model))
            prediction, states = model(x, None)
            z = model.encode(x)
            u, traces = z, []
            for block in model.blocks:
                _, scale = block.dynamics()
                drive = block._drive(u, scale)
                timeline = block.fft_states(drive, drive.new_zeros(x.shape[0], 128))
                traces.append(cpu_tree({"input": u, "drive": drive, "timeline": timeline}))
                u = block._mix(u, timeline)
            prediction, states = cpu_tree((prediction, states))
        assert all(torch.equal(value.cpu(), checkpoint["model"][name]) for name, value in model.state_dict().items())
        coeff_differences = []
        for original, actual in zip(cpu_coeff, gpu_coeff, strict=True):
            diff = {}
            for name in original:
                delta = actual[name] - original[name]
                positions = (delta != 0).nonzero().flatten().tolist()
                diff[name] = {"max_abs": float(delta.abs().max()), "different_entries": len(positions), "entries": [{"index": i, "CPU": float(original[name][i]), "GPU": float(actual[name][i]), "CPU_bits": int(original[name].view(torch.int32)[i]), "GPU_bits": int(actual[name].view(torch.int32)[i])} for i in positions]}
            coeff_differences.append(diff)
        cases.append({"arm": arm, "input_checkpoint_sha256": sha(checkpoint_path), "input_replay_sha256": sha(replay_path), "model": saved["model"], "input": saved["input"], "cpu_coefficients": cpu_coeff, "gpu_coefficients": gpu_coeff, "gpu_block_traces": traces, "gpu_prediction": prediction, "gpu_state": states, "original_gpu_prediction": saved["prediction"], "original_gpu_state": saved["state"], "coefficient_differences": coeff_differences, "rerun_vs_original": {"prediction": difference(prediction, saved["prediction"]), "states": [difference(a, b) for a, b in zip(states, saved["state"], strict=True)]}, "weights_unchanged": True, "new_optimizer_updates": 0})
        del model, x
    artifact = {"synthetic_only": True, "competition_data_read": False, "diagnostic_only": True, "model_source_sha256": MODEL_HASH, "runner_sha256": sha(Path(__file__)), "support_sha256": sha(HERE / "support.py"), "backend_flags": flags, "torch": str(torch.__version__), "python": platform.python_version(), "cases": cases, "new_optimizer_updates": 0, "wall_seconds": time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(artifact, stream)
    summary = {name: value for name, value in artifact.items() if name != "cases"}
    summary["artifact_sha256"] = sha(args.output)
    summary["cases"] = [{key: value for key, value in case.items() if key not in {"model", "input", "cpu_coefficients", "gpu_coefficients", "gpu_block_traces", "gpu_prediction", "gpu_state", "original_gpu_prediction", "original_gpu_state"}} for case in cases]
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"artifact": str(args.output), "sha256": summary["artifact_sha256"], "wall_seconds": artifact["wall_seconds"], "new_optimizer_updates": 0}))


if __name__ == "__main__":
    main()
