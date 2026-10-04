from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np

from wnn.campaign_spec import file_hash
from wnn.quantization import PRESETS, merge_frozen_heads, quantize_model

CASES = ("original", *PRESETS)


def write_json(path: Path, value: object) -> None:
    """Write machine-readable evidence without silently replacing a result."""
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2) + "\n")


def prepare(args: argparse.Namespace) -> None:
    """Freeze an exact merged predictor and a deterministic component panel."""
    import engine_fp16_check as engine
    from probe_width_runtime import export_checkpoint

    from wnn.splits import load_campaign_split

    split = load_campaign_split(args.split, args.repository)
    components = {
        group: component for source, group, component in split.components if source == "valid"
    }
    groups = sorted(split.evaluation_groups, key=lambda group: (components[group], group))[:64]
    if len(groups) != 64 or split.fold != 0:
        raise ValueError("E411 requires exactly 64 sequences from the frozen fold-0 evaluation")
    weights = merge_frozen_heads(export_checkpoint(args.parent), export_checkpoint(args.head))
    if weights["mean"].size != 211 or weights["inp.bias"].size != 256:
        raise ValueError("E411 expects the ordinary D256/event-feature current combo")
    if shutil.disk_usage(args.out.parent).free < 21 * 1024**3:
        raise ValueError("E411 requires 1 GiB output allowance plus 20 GiB free-disk reserve")
    args.out.mkdir(parents=True, exist_ok=False)
    reports = {}
    for case in CASES:
        if case == "original":
            represented = weights
        else:
            represented, encoded, reports[case] = quantize_model(weights, case)
            np.savez(args.out / f"{case}_codes.npz", **encoded)
        np.savez(args.out / f"{case}_weights.npz", **represented)
    x = np.lib.format.open_memmap(
        args.out / "features.npy", mode="w+", dtype=np.float32, shape=(64, 20_000, 112)
    )
    y = np.lib.format.open_memmap(
        args.out / "targets.npy", mode="w+", dtype=np.float32, shape=(64, 20_000, 2)
    )
    need = np.lib.format.open_memmap(
        args.out / "need.npy", mode="w+", dtype=np.bool_, shape=(64, 20_000)
    )
    mask = np.lib.format.open_memmap(
        args.out / "mask.npy", mode="w+", dtype=np.bool_, shape=(64, 20_000)
    )
    sequences = []
    for index, (seq, features, target, required, scored) in enumerate(
        engine.read_sequences(args.data, groups)
    ):
        if features.shape != (20_000, 112) or not np.isfinite(features).all():
            raise ValueError("invalid sequence feature matrix")
        x[index], y[index], need[index], mask[index] = features, target, required, scored
        sequences.append(
            {"row_group": groups[index], "component": components[groups[index]], "seq_ix": seq}
        )
    for array in (x, y, need, mask):
        array.flush()
    del x, y, need, mask
    sources = [
        args.parent,
        args.head,
        args.split,
        Path(__file__),
        args.repository / "src/wnn/quantization.py",
        args.repository / "src/wnn/features.py",
        args.repository / "src/wnn/nn.py",
        args.repository / "src/wnn/gmetric.py",
        args.repository / "src/wnn/data.py",
        args.repository / "src/wnn/splits.py",
        args.repository / "scripts/engine_fp16_check.py",
        args.repository / "utils.py",
    ]
    if portable := os.environ.get("WNN_PORTABLE_DATA_MANIFEST"):
        sources.extend([args.repository / portable, args.repository / "src/wnn/portability.py"])
    manifest = {
        "experiment": "E411",
        "purpose": "fixed weight-only q8 degradation pilot; existing float32 native arithmetic on represented weights",
        "deployment_shape": "one frozen D256 trunk with equal parent and tuned heads",
        "panel": "first 64 fold-0 evaluation groups sorted by component then row group; exposed evaluation data",
        "sequences": sequences,
        "quantization": reports,
        "source_sha256": {str(path): file_hash(path) for path in sources},
        "data_identity": {
            "path": str(args.data),
            "size": args.data.stat().st_size,
            "mtime_ns": args.data.stat().st_mtime_ns,
        },
        "native_library_sha256": file_hash(args.lib / "wnn_native.abi3.so"),
        "files_sha256": {path.name: file_hash(path) for path in sorted(args.out.iterdir())},
        "parity": {
            "sequences": 2,
            "max_abs": 0.001,
            "absolute_wp_delta": 0.00002,
            "repeat_exact": True,
        },
        "bootstrap": {"replicates": 2000, "seed": 271828, "unit": "selected component"},
        "gates": "No scientific promotion from this small, nonrandom panel; do not tune presets on its outcomes",
    }
    write_json(args.out / "manifest.json", manifest)


def torch_reference(weights: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    """Evaluate the exact represented weights with the independent Torch GRU."""
    import torch
    from torch import nn

    from wnn.features import event_features
    from wnn.nn import RNNModel

    hidden, head = weights["inp.bias"].size, weights["head.0.bias"].size
    layers = sum(name.startswith("gru.weight_ih_l") for name in weights)
    model = RNNModel(211, hidden, layers, n_out=2).eval()
    model.head = nn.Sequential(nn.Linear(2 * hidden, head), nn.GELU(), nn.Linear(head, 2))
    model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
    inputs = np.concatenate([features, event_features(features)], axis=1)
    predictions, state = [], None
    with torch.inference_mode():
        for start in range(0, len(inputs), 512):
            out, state = model(torch.from_numpy(inputs[start : start + 512])[None], state)
            predictions.append(out[0].numpy())
    return np.concatenate(predictions)


def bootstrap_delta(
    original: dict[int, np.ndarray], variant: dict[int, np.ndarray]
) -> dict[str, object]:
    """Report paired component resampling, without calling this random panel independent."""
    from wnn.gmetric import wp_from_sums

    keys = sorted(original)
    left, right = np.stack([original[k] for k in keys]), np.stack([variant[k] for k in keys])
    rng = np.random.default_rng(271828)
    deltas = []
    for _ in range(2000):
        sample = rng.integers(0, len(keys), size=len(keys))
        deltas.append(
            float((wp_from_sums(right[sample].sum(0)) - wp_from_sums(left[sample].sum(0))).mean())
        )
    return {
        "components": len(keys),
        "sd": float(np.std(deltas, ddof=1)),
        "ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
    }


def run(args: argparse.Namespace) -> None:
    """Check independent references first, then score every frozen preset."""
    import engine_fp16_check as engine
    import torch

    from utils import BlockAccumulator
    from wnn.data import market_class
    from wnn.gmetric import GlobalAccumulator

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0, {args.cpu})
    manifest = json.loads((args.out / "manifest.json").read_text())
    if (args.out / "result.json").exists() or (args.out / "progress.jsonl").exists():
        raise FileExistsError("preserve the previous attempt and use a new output identity")
    for name, digest in manifest["files_sha256"].items():
        if file_hash(args.out / name) != digest:
            raise ValueError(f"prepared artifact changed: {name}")
    for name, digest in manifest["source_sha256"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError(f"bound source changed: {name}")
    if file_hash(args.lib / "wnn_native.abi3.so") != manifest["native_library_sha256"]:
        raise ValueError("native library changed")
    native = engine.load_native(args.lib)
    x, y, need, mask = [
        np.load(args.out / f"{name}.npy", mmap_mode="r")
        for name in ("features", "targets", "need", "mask")
    ]
    weights = {case: engine.load_npz(args.out / f"{case}_weights.npz") for case in CASES}
    parity = {}
    for case in CASES:
        checks = []
        for index in range(manifest["parity"]["sequences"]):
            predictor = native.EnsemblePredictor(
                [weights[case]],
                [np.float32(1)],
                fast_activations=False,
                precision="f32",
                budget_s=None,
            )
            _, prediction = engine.harness(
                predictor, manifest["sequences"][index]["seq_ix"], x[index], need[index]
            )
            reference = torch_reference(weights[case], x[index])
            difference = np.abs(prediction[need[index]] - reference[need[index]])
            pscore, rscore = GlobalAccumulator(), GlobalAccumulator()
            pscore.add(y[index], prediction, mask[index])
            rscore.add(y[index], reference, mask[index])
            delta = float(pscore.result()["wp"] - rscore.result()["wp"])
            predictor.reset()
            _, repeated = engine.harness(
                predictor, manifest["sequences"][index]["seq_ix"], x[index], need[index]
            )
            repeat = np.array_equal(prediction, repeated, equal_nan=True)
            passed = bool(
                difference.max() <= manifest["parity"]["max_abs"]
                and abs(delta) <= manifest["parity"]["absolute_wp_delta"]
                and repeat
            )
            checks.append(
                {
                    "row_group": manifest["sequences"][index]["row_group"],
                    "max_abs": float(difference.max()),
                    "wp_delta": delta,
                    "repeat": repeat,
                    "passed": passed,
                }
            )
        parity[case] = checks
    write_json(args.out / "parity.json", parity)
    if not all(check["passed"] for checks in parity.values() for check in checks):
        raise RuntimeError("native/Torch represented-weight parity failed; quality probe blocked")
    sums, results = {}, {}
    for case in CASES:
        predictor = native.EnsemblePredictor(
            [weights[case]], [np.float32(1)], fast_activations=False, precision="f32", budget_s=None
        )
        pooled, block = GlobalAccumulator(), BlockAccumulator()
        classes = {label: GlobalAccumulator() for label in range(-1, 4)}
        predictions = np.lib.format.open_memmap(
            args.out / f"{case}_predictions.npy", mode="w+", dtype=np.float32, shape=(64, 20_000, 2)
        )
        for index, sequence in enumerate(manifest["sequences"]):
            _, prediction = engine.harness(predictor, sequence["seq_ix"], x[index], need[index])
            predictions[index] = prediction
            pooled.add(y[index], prediction, mask[index], group=sequence["component"])
            block.add(y[index], prediction, mask[index])
            classes[market_class(y[index])].add(y[index], prediction, mask[index])
            with (args.out / "progress.jsonl").open("a") as output:
                output.write(json.dumps({"case": case, "index": index, **sequence}) + "\n")
        predictions.flush()
        del predictions
        sums[case] = pooled.groups
        results[case] = {
            "global": pooled.result(),
            "block": block.result(),
            "classes": {
                str(label): accumulator.result()
                for label, accumulator in classes.items()
                if accumulator.selected_rows
            },
        }
        if case != "original":
            results[case]["global_wp_delta"] = (
                results[case]["global"]["wp"] - results["original"]["global"]["wp"]
            )
            results[case]["paired_component_bootstrap"] = bootstrap_delta(
                sums["original"], sums[case]
            )
        write_json(args.out / f"{case}_result.json", results[case])
        print(json.dumps({"case": case, **results[case]}), flush=True)
    write_json(
        args.out / "result.json",
        {
            "cases": results,
            "hardware": {
                "platform": platform.platform(),
                "python": sys.version,
                "torch": torch.__version__,
                "affinity": sorted(os.sched_getaffinity(0)),
                "cpuinfo": Path("/proc/cpuinfo").read_text(),
                "cgroup": Path("/proc/self/cgroup").read_text(),
            },
            "limitations": "Weight-only pilot using float32 native compute on reconstructed q8 weights. No byte-kernel recurrent runtime claim, no independent holdout or public-score claim.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded E411 fixed-preset weight-only q8 pilot")
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--cpu", type=int, default=15)
    args = parser.parse_args()
    if args.phase == "prepare" and any(
        getattr(args, name) is None for name in ("repository", "parent", "head", "split", "data")
    ):
        parser.error("prepare requires repository, parent, head, split and data")
    (prepare if args.phase == "prepare" else run)(args)


if __name__ == "__main__":
    main()
