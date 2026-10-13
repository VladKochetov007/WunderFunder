from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from wnn.gmetric import GlobalAccumulator
from wnn.quantization import represented_rows
from wnn.quantization_export import (
    read_packed_quantized_export,
    write_packed_quantized_export,
)
from wnn.splits import file_sha256, load_campaign_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = ("fp32", "native_q8", "packed_q8", "recurrent_w4", "input_w4_recurrent_w8")


def write_json(path: Path, value: object) -> None:
    """Create an immutable receipt; never overwrite an earlier attempt."""
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def native_module(library: Path) -> object:
    """Import exactly the reviewed native library rather than an installed sibling."""
    import importlib

    sys.path.insert(0, str(library.resolve()))
    native = importlib.import_module("wnn_native")
    if Path(native.__file__).resolve().parent != library.resolve():
        raise ValueError("native import resolved outside the requested directory")
    return native


def prepare(args: argparse.Namespace) -> None:
    """Freeze parent weights, exact transports, full evaluation and fit-only profiling."""
    from package_submission import load_checkpoint_member

    split = load_campaign_split(args.split, args.root)
    raw_split = json.loads(args.split.read_text())
    weights, parent = load_checkpoint_member(args.run, args.root / "artifacts/runs", True)
    if weights["mean"].size != 211 or weights["inp.bias"].size != 256:
        raise ValueError("initial deployment recipe requires the 211-input H256 student")
    native = native_module(args.library)
    original = native.Predictor(weights, precision="q8vnni", fast_activations=True)
    codes = original.quantization_codes()
    if not codes:
        raise ValueError("native build lacks exact dynamic-Q8 export")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "parent.npz").open("xb") as output:
        np.savez_compressed(output, **weights)
    effective = dict(weights)
    selected = [
        name for name in weights if name.startswith(("gru.weight_ih_", "gru.weight_hh_"))
    ] + ["head.0.weight"]
    for name in selected:
        effective[name] = np.ascontiguousarray(
            np.concatenate(
                [
                    represented_rows(
                        codes[f"{name}.{index}.codes"], codes[f"{name}.{index}.scales"]
                    )
                    for index in range(2 if name == "head.0.weight" else 1)
                ],
                axis=1,
            )
        )
    write_packed_quantized_export(args.output / "packed_q8.npz", effective, "recurrent_head", codes)
    for case in CASES[3:]:
        write_packed_quantized_export(args.output / f"{case}.npz", weights, case)
    components = {
        group: component for source, group, component in split.components if source == "valid"
    }
    sequence_ids = {
        int(node["row_group"]): int(node["seq_ix"]) for node in raw_split["sequences"]["valid"]
    }
    ordered = sorted(
        split.valid_groups,
        key=lambda group: hashlib.sha256(f"qat-profile-13:{group}".encode()).digest(),
    )
    profile_groups: list[int] = []
    seen: set[int] = set()
    for group in ordered:
        if components[group] not in seen:
            profile_groups.append(group)
            seen.add(components[group])
        if len(profile_groups) == 2:
            break
    if len(profile_groups) != 2:
        raise ValueError("need two distinct fitting components for runtime/state diagnostics")
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(args.output.glob("*.npz"))
    }
    sources = [
        Path(__file__),
        ROOT / "scripts/engine_fp16_check.py",
        ROOT / "src/wnn/quantization.py",
        ROOT / "src/wnn/quantization_export.py",
        ROOT / "src/wnn/gmetric.py",
        ROOT / "src/wnn/splits.py",
        ROOT / "utils.py",
    ]
    write_json(
        args.output / "manifest.json",
        {
            "version": 1,
            "experiment": "E-QAT-DEPLOYMENT-20261013-R1",
            "root": str(args.root.resolve()),
            "parent": parent,
            "split": str(args.split.resolve()),
            "split_sha256": split.sha256,
            "fold": split.fold,
            "evaluation_groups": list(split.evaluation_groups),
            "profile_groups": profile_groups,
            "sequence_ids": sequence_ids,
            "components": components,
            "artifacts": artifacts,
            "library": str(args.library.resolve()),
            "library_sha256": file_sha256(Path(native.__file__)),
            "source_sha256": file_sha256(Path(__file__)),
            "source_files": {str(path.resolve()): file_sha256(path) for path in sources},
            "data_source": raw_split["sources"]["datasets/valid.parquet"],
            "cases": CASES,
            "metric": "global_wp_clip2_v1",
            "exposure": "exposed development fold, exact parent ancestry audited separately",
            "semantics": "Native fast activations; q8vnni uses existing I16 projection, A8 matvec inputs and FP32 recurrent state. PackedQ4 expands once to byte kernels. FP32 uses unquantized projection.",
            "profile_scope": "Two hashed fitting components; complete20000-row trajectories, callback and state diagnostics; no validation tuning",
        },
    )


def case_weights(directory: Path, case: str) -> tuple[dict, dict | None]:
    """Read the frozen transport once; preserve codes rather than requantizing weights."""
    if case in ("fp32", "native_q8"):
        with np.load(directory / "parent.npz", allow_pickle=False) as archive:
            weights = {name: archive[name] for name in archive.files}
        codes = None
    else:
        weights, codes, _ = read_packed_quantized_export(directory / f"{case}.npz")
    return weights, codes


def load_case(native: object, directory: Path, case: str) -> object:
    """Construct one deterministic endpoint with exact quantized rows when applicable."""
    weights, codes = case_weights(directory, case)
    return native.EnsemblePredictor(
        [weights],
        [np.float32(1)],
        fast_activations=True,
        precision="f32" if case == "fp32" else "q8vnni",
        q8_codes=None if codes is None else [codes],
        budget_s=None,
    )


def checked_manifest(directory: Path) -> tuple[dict, object]:
    """Check the frozen source, split and transport bytes before any callback."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if file_sha256(Path(__file__)) != manifest["source_sha256"]:
        raise ValueError("deployment evaluator changed after preparation")
    for name, digest in manifest["source_files"].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"deployment source changed: {name}")
    split = load_campaign_split(Path(manifest["split"]), Path(manifest["root"]))
    if split.sha256 != manifest["split_sha256"]:
        raise ValueError("split changed")
    for name, identity in manifest["artifacts"].items():
        if file_sha256(directory / name) != identity["sha256"]:
            raise ValueError(f"transport changed: {name}")
    native = native_module(Path(manifest["library"]))
    if file_sha256(Path(native.__file__)) != manifest["library_sha256"]:
        raise ValueError("native library changed")
    return manifest, native


def state_diagnostic(args: argparse.Namespace) -> None:
    """Measure recurrent error through full fitting trajectories with fixed sample times."""
    from engine_fp16_check import read_sequences

    manifest, native = checked_manifest(args.output)
    destination = args.output / "state_diagnostic"
    destination.mkdir(exist_ok=False)
    models = {}
    for case in CASES:
        weights, codes = case_weights(args.output, case)
        models[case] = native.Predictor(
            weights,
            fast_activations=True,
            precision="f32" if case == "fp32" else "q8vnni",
            q8_codes=codes,
        )
    sample_steps = np.arange(999, 20_000, 1000)
    all_states = {case: [] for case in CASES}
    data = Path(manifest["root"]) / "datasets/valid.parquet"
    for _, features, _, _, _ in read_sequences(data, manifest["profile_groups"]):
        for case, model in models.items():
            model.reset()
            if np.any(model.hidden_state):
                raise ValueError("reset did not zero recurrent state")
            saved = []
            for step, row in enumerate(features):
                model.step(row)
                if (step + 1) % 1000 == 0:
                    saved.append(model.hidden_state)
            all_states[case].append(np.stack(saved))
    states = {case: np.stack(values) for case, values in all_states.items()}
    if not np.array_equal(states["native_q8"], states["packed_q8"]):
        raise ValueError("exact native-Q8 transport changed recurrent states")
    np.savez(destination / "states.npz", sample_steps=sample_steps, **states)
    errors = {}
    for case in CASES[1:]:
        difference = states[case].astype(np.float64) - states["fp32"]
        errors[case] = {
            "rms_by_sequence_time_layer": np.sqrt(np.mean(difference**2, axis=-1)).tolist(),
            "max_abs_by_sequence_time_layer": np.max(np.abs(difference), axis=-1).tolist(),
        }
    write_json(
        destination / "result.json",
        {
            "profile_groups": manifest["profile_groups"],
            "sample_steps": sample_steps.tolist(),
            "state_shape": list(states["fp32"].shape),
            "native_q8_packed_q8_state_exact": True,
            "errors_against_native_fp32": errors,
            "scope": "Fitting components only; compression diagnostic, not quality evidence",
            "manifest_sha256": file_sha256(args.output / "manifest.json"),
        },
    )


def run(args: argparse.Namespace) -> None:
    """Export and score complete trajectories in frozen full-fold order, one at a time."""
    from engine_fp16_check import harness, read_sequences

    manifest, native = checked_manifest(args.output)
    groups = manifest["profile_groups"] if args.profile else manifest["evaluation_groups"]
    destination = args.output / (f"profile_{args.case}" if args.profile else args.case)
    destination.mkdir(exist_ok=False)
    model = load_case(native, args.output, args.case)
    accumulator = GlobalAccumulator()
    predictions = np.lib.format.open_memmap(
        destination / "predictions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(groups), 20_000, 2),
    )
    timing = []
    replay = None
    data = Path(manifest["root"]) / "datasets/valid.parquet"
    start = time.monotonic()
    for index, (seq, features, targets, need, mask) in enumerate(read_sequences(data, groups)):
        group = groups[index]
        if seq != manifest["sequence_ids"][str(group)] or features.shape != (20_000, 112):
            raise ValueError("actual sequence disagrees with frozen group identity/shape")
        elapsed, prediction = harness(model, seq, features, need)
        if index == 0:
            model.reset()
            _, repeated = harness(model, seq, features, need)
            replay = bool(np.array_equal(prediction, repeated, equal_nan=True))
            if not replay:
                raise ValueError("full sequence reset/replay failed")
        predictions[index] = prediction
        accumulator.add(targets, prediction, mask, group=manifest["components"][str(group)])
        timing.append(elapsed)
        with (destination / "progress.jsonl").open("a") as log:
            log.write(json.dumps({"index": index, "group": group, "seconds": elapsed}) + "\n")
    predictions.flush()
    del predictions
    keys = sorted(accumulator.groups)
    np.savez(
        destination / "component_sums.npz",
        components=np.asarray(keys, dtype=np.int64),
        sums=np.stack([accumulator.groups[key] for key in keys]),
    )
    write_json(
        destination / "result.json",
        {
            "case": args.case,
            "profile_only": args.profile,
            "score": accumulator.result(),
            "groups": groups,
            "components": len(keys),
            "full_sequence_replay_exact": replay,
            "callback_seconds": timing,
            "callback_seconds_total": sum(timing),
            "elapsed_seconds": time.monotonic() - start,
            "storage_bytes": model.storage_bytes,
            "predictions_sha256": file_sha256(destination / "predictions.npy"),
            "manifest_sha256": file_sha256(args.output / "manifest.json"),
            "hardware": platform.processor(),
            "scope": "Fit-only runtime diagnostic" if args.profile else manifest["exposure"],
            "runtime_limit": "Local CPU callback timing; no EPYC or final-submission guarantee",
        },
    )


def main() -> None:
    """Prepare immutable transports or evaluate a fixed complete native endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "states"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        if not all((args.run, args.library, args.split)):
            parser.error("prepare requires --run, --library and --split")
        prepare(args)
    elif args.phase == "states":
        state_diagnostic(args)
    else:
        if args.case is None:
            parser.error("run requires --case")
        run(args)


if __name__ == "__main__":
    main()
