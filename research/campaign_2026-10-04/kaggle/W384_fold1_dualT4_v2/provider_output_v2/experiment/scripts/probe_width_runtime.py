from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wnn.campaign_spec import file_hash  # noqa: E402

CASES = ("d256_head1", "d256_head2", "w384_head1", "w384_head2")


def duplicate_head(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Duplicate the same trained head only to size a two-head workload."""
    result = dict(weights)
    result["head.0.weight"] = np.concatenate([weights["head.0.weight"]] * 2)
    result["head.0.bias"] = np.concatenate([weights["head.0.bias"]] * 2)
    result["head.2.weight"] = np.concatenate([weights["head.2.weight"]] * 2, axis=1) * np.float32(
        0.5
    )
    return {name: np.ascontiguousarray(value) for name, value in result.items()}


def export_checkpoint(path: Path) -> dict[str, np.ndarray]:
    """Export only checkpoint inference state, retaining its trained normalization."""
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    weights = {name: tensor.detach().float().numpy() for name, tensor in value["model"].items()}
    for name in ("head.2.weight", "head.2.bias"):
        weights[name] = np.ascontiguousarray(weights[name][:2])
    return weights


def prepare(args: argparse.Namespace) -> None:
    """Bind real checkpoints and an existing engine snapshot before inference."""
    args.out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in (
        "rnn_fast.rs",
        "half.rs",
        "approx.rs",
        "python.rs",
        "kernels.rs",
        "stream.rs",
        "rnn.rs",
    ):
        current = file_hash(args.repository / "native/src" / name)
        snapshot = file_hash(args.build / "src/src" / name)
        if current != snapshot:
            raise ValueError(f"existing native snapshot differs from current runtime: {name}")
        sources[name] = current
    originals = {"d256": export_checkpoint(args.d256), "w384": export_checkpoint(args.w384)}
    cases = {}
    for prefix, weights in originals.items():
        hidden = weights["inp.bias"].size
        if hidden != {"d256": 256, "w384": 384}[prefix] or weights["head.0.bias"].size != hidden:
            raise ValueError("checkpoint does not have the expected ordinary trained head")
        for heads in (1, 2):
            name = f"{prefix}_head{heads}"
            exported = weights if heads == 1 else duplicate_head(weights)
            path = args.out / f"{name}.npz"
            np.savez(path, **exported)
            cases[name] = {
                "hidden": hidden,
                "heads": heads,
                "sha256": file_hash(path),
                "head2_source": "duplicate identical trained head for workload sizing"
                if heads == 2
                else None,
            }
    manifest = {
        "experiment": "E409",
        "purpose": "actual payload width/head runtime, no public-time extrapolation",
        "checkpoints": {str(path): file_hash(path) for path in (args.d256, args.w384)},
        "native_source_sha256": sources,
        "libraries": {
            level: file_hash(args.build / level / "wnn_native.abi3.so")
            for level in ("v2", "avx2", "v3")
        },
        "data": {"path": str(args.data), "sha256": file_hash(args.data)},
        "probe_source_sha256": file_hash(Path(__file__)),
        "harness_source_sha256": file_hash(Path(__file__).with_name("engine_fp16_check.py")),
        "row_groups": list(range(args.seqs)),
        "cases": cases,
        "gate": {
            "max_abs": 3e-4,
            "abs_global_wp_delta": 5e-6,
            "repeat_bitwise": True,
            "v2_v3_bitwise": True,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def score(predictions: np.ndarray, sequences: list[tuple]) -> dict[str, object]:
    """Record pooled and legacy metrics only as numerical parity diagnostics."""
    from utils import BlockAccumulator, GlobalAccumulator

    pooled, block = GlobalAccumulator(), BlockAccumulator()
    for pred, (_, _, target, _, mask) in zip(predictions, sequences, strict=True):
        pooled.add(target, pred, mask)
        block.add(target, pred, mask)
    return {"global": pooled.result(), "block": block.result()}


def run(args: argparse.Namespace) -> None:
    """Execute parity or paired timing in one torch-free, pinned CPU process."""
    import engine_fp16_check as engine

    manifest = json.loads((args.out / "manifest.json").read_text())
    if file_hash(Path(manifest["data"]["path"])) != manifest["data"]["sha256"]:
        raise ValueError("data changed after preparation")
    outputs = (
        [args.out / f"parity_{args.lib.name}.{suffix}" for suffix in ("json", "npz")]
        if args.phase == "parity"
        else [args.out / "timing_trials.jsonl", args.out / "result.json"]
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("phase output exists; preserve it and use a fresh probe identity")
    if file_hash(args.lib / "wnn_native.abi3.so") != manifest["libraries"][args.lib.name]:
        raise ValueError("native library changed after preparation")
    if file_hash(Path(__file__)) != manifest["probe_source_sha256"]:
        raise ValueError("probe changed after preparation")
    if (
        file_hash(Path(__file__).with_name("engine_fp16_check.py"))
        != manifest["harness_source_sha256"]
    ):
        raise ValueError("timing harness changed after preparation")
    os.sched_setaffinity(0, {args.cpu})
    native = engine.load_native(args.lib)
    sequences = list(engine.read_sequences(Path(manifest["data"]["path"]), manifest["row_groups"]))
    weights = {}
    for name in CASES:
        path = args.out / f"{name}.npz"
        if file_hash(path) != manifest["cases"][name]["sha256"]:
            raise ValueError(f"weight artifact changed: {name}")
        weights[name] = engine.load_npz(path)

    def inference(name: str, precision: str) -> tuple[np.ndarray, dict[str, object]]:
        model = native.EnsemblePredictor(
            [weights[name]],
            [np.float32(1.0)],
            fast_activations=True,
            precision=precision,
            budget_s=None,
        )
        cycles = [0]
        elapsed = 0.0
        arrays = []
        for seq, features, _, need, _ in sequences:
            seconds, pred = engine.harness(model, seq, features, need, cycles)
            elapsed += seconds
            arrays.append(pred)
        if args.phase == "parity" and precision == "q16x":
            model.reset()
            first = sequences[0]
            _, reset = engine.harness(model, first[0], first[1], first[3])
            if not np.array_equal(reset, arrays[0], equal_nan=True):
                raise ValueError(f"sequence reset mismatch: {name}")
        rows = sum(len(sequence[1]) for sequence in sequences)
        return np.stack(arrays), {
            "seconds": elapsed,
            "us_per_row": elapsed * 1e6 / rows,
            "kcycles_per_row": cycles[0] / rows / 1000 if engine.CYCLES.fd >= 0 else None,
            "effective_precision": model.precision,
        }

    hardware = {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpuinfo": Path("/proc/cpuinfo").read_text(),
        "loadavg_start": Path("/proc/loadavg").read_text(),
        "cgroup": Path("/proc/self/cgroup").read_text(),
    }
    if args.phase == "parity":
        arrays, results = {}, {}
        for name in CASES:
            fp32, _ = inference(name, "f32")
            quantized, _ = inference(name, "q16x")
            repeated, _ = inference(name, "q16x")
            selected = np.stack([sequence[3] for sequence in sequences])
            diff = np.abs(quantized[selected] - fp32[selected])
            f32_score, q16_score = score(fp32, sequences), score(quantized, sequences)
            delta = q16_score["global"]["wp"] - f32_score["global"]["wp"]
            repeat = np.array_equal(quantized, repeated, equal_nan=True)
            passed = bool(diff.max() <= 3e-4 and abs(delta) <= 5e-6 and repeat)
            results[name] = {
                "max_abs": float(diff.max()),
                "rms": float(np.sqrt(np.mean(diff**2))),
                "global_wp_delta": delta,
                "repeat_equal": repeat,
                "passed": passed,
                "fp32_scores": f32_score,
                "q16x_scores": q16_score,
            }
            arrays[name] = quantized
            print(json.dumps({"level": args.lib.name, "case": name, **results[name]}), flush=True)
        np.savez(args.out / f"parity_{args.lib.name}.npz", **arrays)
        (args.out / f"parity_{args.lib.name}.json").write_text(
            json.dumps({"hardware": hardware, "cases": results}, indent=2)
        )
        if not all(result["passed"] for result in results.values()):
            raise RuntimeError("precision parity gate failed; timings remain blocked")
        return
    for level in ("v2", "v3"):
        report = json.loads((args.out / f"parity_{level}.json").read_text())
        if not all(report["cases"][name]["passed"] for name in CASES):
            raise ValueError("timing requires passed v2 and v3 parity gates")
    with np.load(args.out / "parity_v2.npz") as v2, np.load(args.out / "parity_v3.npz") as v3:
        if not all(np.array_equal(v2[name], v3[name], equal_nan=True) for name in CASES):
            raise ValueError("v2/v3 predictions are not bitwise identical")
    trials = []
    for repeat in range(args.repeats):
        order = CASES if repeat % 2 == 0 else tuple(reversed(CASES))
        for position, name in enumerate(order):
            _, timing = inference(name, "q16x")
            trial = {
                "case": name,
                "repeat": repeat,
                "position": position,
                **timing,
                "loadavg": Path("/proc/loadavg").read_text(),
            }
            trials.append(trial)
            with (args.out / "timing_trials.jsonl").open("a") as output:
                output.write(json.dumps(trial) + "\n")
            print(json.dumps(trial), flush=True)
    summary = {}
    for name in CASES:
        summary[name] = {}
        for metric in ("us_per_row", "kcycles_per_row"):
            values = [
                trial[metric]
                for trial in trials
                if trial["case"] == name and trial[metric] is not None
            ]
            summary[name][metric] = (
                None
                if not values
                else {
                    "min": min(values),
                    "median": statistics.median(values),
                    "max": max(values),
                    "all": values,
                }
            )
    (args.out / "result.json").write_text(
        json.dumps(
            {
                "experiment": "E409",
                "hardware": hardware,
                "summary": summary,
                "trials": trials,
                "public_runtime_projection": None,
            },
            indent=2,
        )
    )


def main() -> None:
    """Prepare frozen artifacts, then run parity before measuring payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "parity", "timing"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--d256", type=Path)
    parser.add_argument("--w384", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--cpu", type=int, default=15)
    parser.add_argument("--seqs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.seqs <= 5 or not 3 <= args.repeats <= 7:
        parser.error("bounded probe requires 1..5 sequences and 3..7 timing repeats")
    if args.phase == "prepare":
        if any(
            getattr(args, name) is None for name in ("repository", "build", "d256", "w384", "data")
        ):
            parser.error("prepare requires repository/build/checkpoints/data")
        prepare(args)
    else:
        if args.lib is None:
            parser.error("execution requires --lib")
        run(args)


if __name__ == "__main__":
    main()
