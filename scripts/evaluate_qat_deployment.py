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
from wnn.quantization import (
    PRESETS,
    policy_metadata,
    quantize_model,
    represented_rows,
    represented_rows_q4,
)
from wnn.quantization_export import (
    read_packed_quantized_export,
    write_packed_quantized_export,
)
from wnn.splits import file_sha256, load_campaign_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = ("fp32", "native_q8", "packed_q8", "recurrent_w4", "input_w4_recurrent_w8")
CASE_PRESETS = {
    "packed_q8": "recurrent_head",
    "recurrent_w4": "recurrent_w4",
    "input_w4_recurrent_w8": "input_w4_recurrent_w8",
}


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


def checkpoint_quantization(checkpoint: dict, weights: dict[str, np.ndarray]) -> dict[str, object]:
    """Validate and normalize the exact representation stored by an inference checkpoint."""
    from collections.abc import Mapping

    import torch

    arguments = checkpoint.get("args", {})
    if not isinstance(arguments, dict):
        raise ValueError("checkpoint args must be a mapping")
    qat_mode = arguments.get("qat", "none")
    if qat_mode != "none" and qat_mode not in PRESETS:
        raise ValueError(f"unsupported checkpoint QAT mode: {qat_mode!r}")
    learn_scales = arguments.get("qat_learn_scales", False)
    if not isinstance(learn_scales, bool):
        raise ValueError("checkpoint learned-scale flag must be boolean")
    from wnn.training_qat import QAT_LEARN_SCALE_MODES, scale_learning_metadata

    if learn_scales and qat_mode not in QAT_LEARN_SCALE_MODES:
        raise ValueError("checkpoint learned scales require a versioned Q4 QAT preset")
    metadata = checkpoint.get("quantization")
    if metadata is None:
        if qat_mode != "none":
            raise ValueError("QAT checkpoint is missing its quantization metadata")
        return {
            "kind": "float_checkpoint",
            "qat_mode": "none",
            "preset": None,
            "metadata": None,
            "encoded": None,
        }
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint quantization metadata must be a mapping")
    preset = metadata.get("preset")
    if preset not in PRESETS:
        raise ValueError("checkpoint quantization metadata has an unsupported preset")
    if qat_mode != "none" and qat_mode != preset:
        raise ValueError("checkpoint QAT args disagree with quantization metadata preset")
    expected_policy = policy_metadata(preset)
    for field in ("version", "preset", "scheme"):
        if metadata.get(field) != expected_policy[field]:
            raise ValueError(f"checkpoint quantization {field} disagrees with the versioned policy")
    if metadata.get("representation") != "effective_reconstructed_float32":
        raise ValueError("checkpoint quantization representation is not effective float32")
    snapshot_source = metadata.get("snapshot_source")
    if not isinstance(snapshot_source, str) or not snapshot_source:
        raise ValueError("checkpoint quantization snapshot source is missing")
    provenance = checkpoint.get("provenance", {})
    control = provenance.get("quantization_control") if isinstance(provenance, dict) else None
    training_rule = metadata.get("training_rule")
    if qat_mode == "none":
        if (
            not isinstance(control, dict)
            or control.get("preset") != preset
            or control.get("source_sha256") != metadata.get("source_checkpoint_sha256")
            or metadata.get("training_updates") != 0
            or training_rule != "none; post-training quantization without updates"
        ):
            raise ValueError(
                "quantization metadata without QAT args lacks no-finetune control provenance"
            )
        if learn_scales or "scale_learning" in metadata:
            raise ValueError("post-training quantization control cannot declare learned scales")
        kind = "post_training_quantization_control"
    else:
        expected_training_rule = (
            "identity weight STE; learned row scales with LSQ-inspired clipped scale derivative"
            if learn_scales
            else "identity STE; detached row scales; no scale, rounding or clipping derivative"
        )
        if control is not None or training_rule != expected_training_rule:
            raise ValueError("fitted QAT args disagree with checkpoint quantization semantics")
        if learn_scales:
            if metadata.get("scale_learning") != scale_learning_metadata():
                raise ValueError("fitted QAT checkpoint has missing or incompatible learned-scale semantics")
            expected_optimizer_semantics = scale_learning_metadata(
                arguments.get("optim"), arguments.get("wd")
            )
            qat_provenance = provenance.get("qat") if isinstance(provenance, dict) else None
            if (
                not isinstance(qat_provenance, dict)
                or qat_provenance.get("scale_learning") != expected_optimizer_semantics
            ):
                raise ValueError("learned-scale QAT optimizer-group provenance is missing or incompatible")
        elif "scale_learning" in metadata:
            raise ValueError("fixed-scale QAT checkpoint unexpectedly declares learned-scale semantics")
        kind = "fitted_qat"

    _, expected_encoded, report = quantize_model(weights, preset)
    selected = metadata.get("selected")
    if not isinstance(selected, dict) or set(selected) != set(report["selected"]):
        raise ValueError("checkpoint quantization selected matrices disagree with the preset")
    for name, details in report["selected"].items():
        recorded = selected[name]
        if (
            not isinstance(recorded, dict)
            or recorded.get("shape") != details["shape"]
            or recorded.get("block_bits") != details["block_bits"]
        ):
            raise ValueError(
                f"checkpoint quantization matrix metadata disagrees for {name}"
            )

    raw_encoded = metadata.get("encoded")
    if not isinstance(raw_encoded, Mapping) or set(raw_encoded) != set(expected_encoded):
        raise ValueError("checkpoint quantization exact codes/scales are missing or unexpected")
    encoded: dict[str, np.ndarray] = {}
    for name, value in raw_encoded.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint encoded field is not a tensor: {name}")
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        expected = expected_encoded[name]
        if array.dtype != expected.dtype or array.shape != expected.shape:
            raise ValueError(f"checkpoint encoded field has the wrong dtype or shape: {name}")
        encoded[name] = array

    for name, details in report["selected"].items():
        blocks = []
        block_count = 2 if name == "head.0.weight" else 1
        for index, bits in enumerate(details["block_bits"]):
            codes = encoded[f"{name}.{index}.codes"]
            scales = encoded[f"{name}.{index}.scales"]
            represented = represented_rows_q4 if bits == 4 else represented_rows
            blocks.append(represented(codes, scales))
        actual = np.concatenate(blocks, axis=1) if block_count == 2 else blocks[0]
        if not np.array_equal(actual, np.asarray(weights[name], dtype=np.float32)):
            raise ValueError(
                f"checkpoint effective model disagrees with exact codes/scales for {name}"
            )
    summary = {
        "kind": kind,
        "qat_mode": qat_mode,
        "preset": preset,
        "version": metadata["version"],
        "representation": metadata["representation"],
        "snapshot_source": snapshot_source,
        "training_rule": training_rule,
        "scale_learning": metadata.get("scale_learning"),
        "encoded_sha256": {
            name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
            for name, value in encoded.items()
        },
    }
    return {
        "kind": kind,
        "qat_mode": qat_mode,
        "preset": preset,
        "metadata": summary,
        "encoded": encoded,
    }


def export_case_archives(
    output: Path,
    weights: dict[str, np.ndarray],
    native_codes: dict[str, np.ndarray],
    quantization: dict[str, object],
) -> dict[str, str]:
    """Write transports, preserving checkpoint codes only for their matching preset."""
    exact = quantization["encoded"]
    checkpoint_preset = quantization["preset"]
    semantics: dict[str, str] = {}

    q8_encoded = exact if checkpoint_preset == "recurrent_head" else native_codes
    if checkpoint_preset == "recurrent_head":
        q8_effective = dict(weights)
        semantics["packed_q8"] = "exact_checkpoint_codes_scales"
    else:
        q8_effective = dict(weights)
        selected = [
            name for name in weights if name.startswith(("gru.weight_ih_", "gru.weight_hh_"))
        ] + ["head.0.weight"]
        for name in selected:
            q8_effective[name] = np.ascontiguousarray(
                np.concatenate(
                    [
                        represented_rows(
                            native_codes[f"{name}.{index}.codes"],
                            native_codes[f"{name}.{index}.scales"],
                        )
                        for index in range(2 if name == "head.0.weight" else 1)
                    ],
                    axis=1,
                )
            )
        semantics["packed_q8"] = "exact_native_q8_transport_from_effective_checkpoint"
    write_packed_quantized_export(
        output / "packed_q8.npz", q8_effective, "recurrent_head", q8_encoded
    )

    for case in CASES[3:]:
        use_exact = checkpoint_preset == CASE_PRESETS[case]
        write_packed_quantized_export(
            output / f"{case}.npz",
            weights,
            case,
            exact if use_exact else None,
        )
        semantics[case] = (
            "exact_checkpoint_codes_scales" if use_exact else "ptq_from_effective_checkpoint"
        )
    return semantics


def endpoint_semantics(
    quantization: dict[str, object], case_semantics: dict[str, str]
) -> dict[str, str]:
    """Name the actual source of each comparison endpoint in the frozen manifest."""
    return {
        "fp32": (
            "effective_reconstructed_diagnostic_not_latent_weights"
            if quantization["metadata"] is not None
            else "stored_float_checkpoint_weights"
        ),
        "native_q8": "dynamic_q8_from_effective_checkpoint",
        **case_semantics,
    }


def prepare(args: argparse.Namespace) -> None:
    """Freeze parent weights, exact transports, full evaluation and fit-only profiling."""
    from package_submission import load_checkpoint_member

    split = load_campaign_split(args.split, args.root)
    raw_split = json.loads(args.split.read_text())
    weights, parent = load_checkpoint_member(args.run, args.root / "artifacts/runs", True)
    import torch

    checkpoint_path = Path(parent["checkpoint_path"])
    if file_sha256(checkpoint_path) != parent["checkpoint_sha256"]:
        raise ValueError("checkpoint changed while preparing deployment artifacts")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    quantization = checkpoint_quantization(checkpoint, weights)
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
    case_semantics = export_case_archives(args.output, weights, codes, quantization)
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
        ROOT / "scripts/package_submission.py",
        ROOT / "src/wnn/quantization.py",
        ROOT / "src/wnn/quantization_export.py",
        ROOT / "src/wnn/training_qat.py",
        ROOT / "scripts/train_rnn.py",
        ROOT / "scripts/export_qat_control.py",
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
            "checkpoint_quantization": quantization["metadata"],
            "case_semantics": endpoint_semantics(quantization, case_semantics),
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
        weights, codes, metadata = read_packed_quantized_export(directory / f"{case}.npz")
        expected = "recurrent_head" if case == "packed_q8" else case
        if metadata["preset"] != expected:
            raise ValueError(f"archive preset disagrees with requested case: {case}")
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


def checked_manifest(directory: Path, expected_sha256: str) -> tuple[dict, object]:
    """Check the frozen source, split and transport bytes before any callback."""
    if file_sha256(directory / "manifest.json") != expected_sha256:
        raise ValueError("deployment manifest differs from the admitted hash")
    manifest = json.loads((directory / "manifest.json").read_text())
    if file_sha256(Path(__file__)) != manifest["source_sha256"]:
        raise ValueError("deployment evaluator changed after preparation")
    for name, digest in manifest["source_files"].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"deployment source changed: {name}")
    split = load_campaign_split(Path(manifest["split"]), Path(manifest["root"]))
    if split.sha256 != manifest["split_sha256"]:
        raise ValueError("split changed")
    if manifest["evaluation_groups"] != list(split.evaluation_groups):
        raise ValueError("evaluation membership/order differs from the frozen full fold")
    if not set(manifest["profile_groups"]).issubset(split.valid_groups):
        raise ValueError("runtime profile includes a group outside fitting membership")
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

    manifest, native = checked_manifest(args.output, args.manifest_sha256)
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
    for group, (seq, features, _, _, _) in zip(
        manifest["profile_groups"],
        read_sequences(data, manifest["profile_groups"]),
        strict=True,
    ):
        if seq != manifest["sequence_ids"][str(group)] or features.shape != (20_000, 112):
            raise ValueError("profile sequence disagrees with frozen group identity/shape")
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

    manifest, native = checked_manifest(args.output, args.manifest_sha256)
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
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.phase != "prepare" and args.manifest_sha256 is None:
        parser.error("run/states requires --manifest-sha256 frozen at admission")
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
