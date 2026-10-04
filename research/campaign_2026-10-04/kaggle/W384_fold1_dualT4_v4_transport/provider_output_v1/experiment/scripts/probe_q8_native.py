from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

from wnn.campaign_spec import file_hash
from wnn.quantization_export import read_q8_export, write_q8_export

CASES = tuple(
    f"{model}_{precision}" for model in ("d256", "w384") for precision in ("q16x", "q8r", "q8x")
)


def write_json(path: Path, value: object) -> None:
    """Publish one immutable stage result."""
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2) + "\n")


def prepare(args: argparse.Namespace) -> None:
    """Bind real payloads, exact byte exports and fixed float-reference predictions."""
    import torch
    from probe_q8_model import torch_reference

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    e411 = json.loads((args.e411 / "manifest.json").read_text())
    e409 = json.loads((args.e409 / "manifest.json").read_text())
    required = (
        "features.npy",
        "targets.npy",
        "need.npy",
        "mask.npy",
        "original_weights.npz",
        "recurrent_codes.npz",
        "recurrent_head_codes.npz",
    )
    for name in required:
        if file_hash(args.e411 / name) != e411["files_sha256"][name]:
            raise ValueError(f"E411 artifact changed: {name}")
    if file_hash(args.e409 / "w384_head1.npz") != e409["cases"]["w384_head1"]["sha256"]:
        raise ValueError("W384 trained one-head payload changed")
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("features", "targets", "need", "mask"):
        np.save(args.out / f"{name}.npy", np.load(args.e411 / f"{name}.npy", mmap_mode="r")[:3])
    originals = {}
    for model, path in (
        ("d256", args.e411 / "original_weights.npz"),
        ("w384", args.e409 / "w384_head1.npz"),
    ):
        with np.load(path, allow_pickle=False) as archive:
            originals[model] = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        np.savez(args.out / f"{model}_weights.npz", **originals[model])
        for preset, precision in (("recurrent", "q8r"), ("recurrent_head", "q8x")):
            preserved = None
            if model == "d256":
                with np.load(args.e411 / f"{preset}_codes.npz", allow_pickle=False) as archive:
                    preserved = {
                        name: np.ascontiguousarray(archive[name]) for name in archive.files
                    }
            write_q8_export(
                args.out / f"{model}_{precision}.npz", originals[model], preset, preserved
            )
    features = np.load(args.out / "features.npy", mmap_mode="r")
    references = {}
    for case in CASES:
        model, precision = case.split("_")
        weights = (
            originals[model] if precision == "q16x" else read_q8_export(args.out / f"{case}.npz")[0]
        )
        references[case] = np.stack([torch_reference(weights, sequence) for sequence in features])
    np.savez(args.out / "torch_references.npz", **references)
    sources = [
        Path(__file__),
        Path(__file__).with_name("probe_q8_model.py"),
        Path(__file__).with_name("engine_fp16_check.py"),
    ]
    sources.extend(
        args.repository / name
        for name in (
            "src/wnn/quantization.py",
            "src/wnn/quantization_export.py",
            "src/wnn/campaign_spec.py",
            "src/wnn/features.py",
            "src/wnn/data.py",
            "src/wnn/event_content.py",
            "src/wnn/nn.py",
            "src/wnn/gmetric.py",
            "utils.py",
            "native/Cargo.toml",
            "native/Cargo.lock",
        )
    )
    sources.extend(sorted((args.repository / "native/src").glob("*.rs")))
    manifest = {
        "experiment": "E416",
        "models": {
            "d256": "actual equal parent/head fused predictor from E411",
            "w384": "one trained ordinary head from E409; no duplicate or ensemble",
        },
        "q8_arithmetic": "int8 weights, transient int16 activation codes, float32 recurrent state; input projection and final head remain float32",
        "cases": CASES,
        "sequences": e411["sequences"][:3],
        "input_evidence": {
            "E411_manifest": file_hash(args.e411 / "manifest.json"),
            "E409_manifest": file_hash(args.e409 / "manifest.json"),
        },
        "source_sha256": {str(path): file_hash(path) for path in sources},
        "library_sha256": {
            level: file_hash(args.build / level / "wnn_native.abi3.so") for level in ("v2", "v3")
        },
        "files_sha256": {path.name: file_hash(path) for path in sorted(args.out.iterdir())},
        "gates": {
            "max_abs": 3e-4,
            "absolute_wp_delta": 5e-6,
            "portable_bitwise": True,
            "v2_v3_bitwise": True,
            "repeat_reset_bitwise": True,
        },
        "timing": {
            "repeats": 5,
            "order": "forward/reverse alternating",
            "sequences": 3,
            "no_public_time_conversion": True,
        },
        "reference_stack": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    write_json(args.out / "manifest.json", manifest)


def run(args: argparse.Namespace) -> None:
    """Gate all trained-model arithmetic before paired workload timings."""
    import engine_fp16_check as engine

    from wnn.gmetric import GlobalAccumulator

    os.sched_setaffinity(0, {args.cpu})
    manifest = json.loads((args.out / "manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        if file_hash(Path(name)) != digest:
            raise ValueError(f"source changed after preparation: {name}")
    for name, digest in manifest["files_sha256"].items():
        if file_hash(args.out / name) != digest:
            raise ValueError(f"prepared artifact changed: {name}")
    if file_hash(args.lib / "wnn_native.abi3.so") != manifest["library_sha256"][args.lib.name]:
        raise ValueError("native library changed after preparation")
    native = engine.load_native(args.lib)
    features, targets, need, mask = [
        np.load(args.out / f"{name}.npy", mmap_mode="r")
        for name in ("features", "targets", "need", "mask")
    ]
    inputs = {}
    for case in CASES:
        model, precision = case.split("_")
        if precision == "q16x":
            with np.load(args.out / f"{model}_weights.npz") as archive:
                inputs[case] = (
                    {name: np.ascontiguousarray(archive[name]) for name in archive.files},
                    None,
                )
        else:
            weights, codes, _ = read_q8_export(args.out / f"{case}.npz")
            inputs[case] = (weights, codes)

    def inference(case: str, portable: bool = False) -> tuple[np.ndarray, dict[str, object]]:
        weights, codes = inputs[case]
        precision = case.split("_")[1] + ("_portable" if portable else "")
        predictor = native.EnsemblePredictor(
            [weights],
            [np.float32(1)],
            precision=precision,
            fast_activations=True,
            budget_s=None,
            **({"q8_codes": [codes]} if codes is not None else {}),
        )
        if predictor.precision != precision:
            raise ValueError(
                f"unexpected native precision fallback: {precision} -> {predictor.precision}"
            )
        if codes is not None:
            exported = predictor.quantization_codes()[0]
            if exported.keys() != codes.keys() or any(
                exported[name].dtype != codes[name].dtype
                or not np.array_equal(exported[name].view(np.uint8), codes[name].view(np.uint8))
                for name in codes
            ):
                raise ValueError("native import did not preserve exact code/scale bytes")
        arrays, elapsed, cycles = [], 0.0, [0]
        for index, sequence in enumerate(manifest["sequences"]):
            seconds, prediction = engine.harness(
                predictor, sequence["seq_ix"], features[index], need[index], cycles
            )
            arrays.append(prediction)
            elapsed += seconds
        if args.phase == "parity" and not portable:
            predictor.reset()
            _, repeated = engine.harness(
                predictor, manifest["sequences"][0]["seq_ix"], features[0], need[0]
            )
            if not np.array_equal(repeated, arrays[0], equal_nan=True):
                raise ValueError(f"reset/repeat mismatch: {case}")
        rows = len(features) * features.shape[1]
        return np.stack(arrays), {
            "us_per_row": elapsed * 1e6 / rows,
            "kcycles_per_row": cycles[0] / rows / 1000 if engine.CYCLES.fd >= 0 else None,
            "weight_bytes": predictor.weight_bytes[0],
            "q8_weight_bytes": predictor.q8_weight_bytes[0],
        }

    hardware = {
        "platform": platform.platform(),
        "python": sys.version,
        "affinity": sorted(os.sched_getaffinity(0)),
        "cpuinfo": Path("/proc/cpuinfo").read_text(),
        "cgroup": Path("/proc/self/cgroup").read_text(),
        "loadavg_start": Path("/proc/loadavg").read_text(),
    }
    if args.phase == "parity":
        if (args.out / f"parity_{args.lib.name}.json").exists():
            raise FileExistsError("preserve previous parity evidence")
        results, predictions = {}, {}
        with np.load(args.out / "torch_references.npz") as references:
            for case in CASES:
                prediction, workload = inference(case)
                portable, _ = inference(case, portable=True)
                reference = references[case]
                error = np.abs(prediction[need] - reference[need])
                pscore, rscore = GlobalAccumulator(), GlobalAccumulator()
                pscore.add(targets, prediction, mask)
                rscore.add(targets, reference, mask)
                delta = float(pscore.result()["wp"] - rscore.result()["wp"])
                exact = np.array_equal(prediction, portable, equal_nan=True)
                passed = bool(
                    error.max() <= manifest["gates"]["max_abs"]
                    and abs(delta) <= manifest["gates"]["absolute_wp_delta"]
                    and exact
                )
                results[case] = {
                    "max_abs": float(error.max()),
                    "wp_delta": delta,
                    "portable_bitwise": exact,
                    "passed": passed,
                    **workload,
                }
                predictions[case] = prediction
                print(json.dumps({"case": case, **results[case]}), flush=True)
        np.savez(args.out / f"parity_{args.lib.name}.npz", **predictions)
        write_json(
            args.out / f"parity_{args.lib.name}.json", {"hardware": hardware, "cases": results}
        )
        if not all(result["passed"] for result in results.values()):
            raise RuntimeError("trained-model arithmetic gate failed; timings blocked")
        return
    if (args.out / "trials.jsonl").exists():
        raise FileExistsError("preserve previous timing attempt")
    for level in ("v2", "v3"):
        if not all(
            value["passed"]
            for value in json.loads((args.out / f"parity_{level}.json").read_text())[
                "cases"
            ].values()
        ):
            raise ValueError("timing requires both passed parity levels")
    with np.load(args.out / "parity_v2.npz") as v2, np.load(args.out / "parity_v3.npz") as v3:
        if not all(np.array_equal(v2[case], v3[case], equal_nan=True) for case in CASES):
            raise ValueError("v2/v3 predictions differ")
    trials = []
    for repeat in range(5):
        for position, case in enumerate(CASES if repeat % 2 == 0 else tuple(reversed(CASES))):
            _, result = inference(case)
            workload = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "list-units",
                    "--no-legend",
                    "--plain",
                    "--state=active",
                    "wnn-*",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
            trial = {
                "repeat": repeat,
                "position": position,
                "case": case,
                **result,
                "loadavg": Path("/proc/loadavg").read_text(),
                "concurrent_campaign_units": workload,
            }
            trials.append(trial)
            with (args.out / "trials.jsonl").open("a") as output:
                output.write(json.dumps(trial) + "\n")
            print(json.dumps(trial), flush=True)
    summary = {}
    for case in CASES:
        summary[case] = {}
        for metric in ("us_per_row", "kcycles_per_row"):
            values = [
                trial[metric]
                for trial in trials
                if trial["case"] == case and trial[metric] is not None
            ]
            summary[case][metric] = (
                {"min": min(values), "median": statistics.median(values), "max": max(values)}
                if values
                else None
            )
        summary[case]["weight_bytes"] = next(
            trial["weight_bytes"] for trial in trials if trial["case"] == case
        )
        summary[case]["q8_weight_bytes"] = next(
            trial["q8_weight_bytes"] for trial in trials if trial["case"] == case
        )
    write_json(
        args.out / "result.json", {"hardware": hardware, "summary": summary, "trials": trials}
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E416 actual-payload native q8 accuracy and runtime"
    )
    parser.add_argument("phase", choices=("prepare", "parity", "timing"))
    for name in ("out", "lib", "repository", "build", "e409", "e411"):
        parser.add_argument(f"--{name}", type=Path, required=name == "out")
    parser.add_argument("--cpu", type=int, default=15)
    args = parser.parse_args()
    if args.phase == "prepare" and any(
        getattr(args, name) is None for name in ("repository", "build", "e409", "e411")
    ):
        parser.error("prepare requires repository, build, e409 and e411")
    if args.phase != "prepare" and args.lib is None:
        parser.error("parity/timing require lib")
    (prepare if args.phase == "prepare" else run)(args)


if __name__ == "__main__":
    main()
