"""Compare the fixed native callback with its deterministic PyTorch reference."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.package_final_tl3 import EXPECTED_CHECKPOINT_SHA256, ROLES, RUNS  # noqa: E402


@dataclass(frozen=True)
class DataPoint:
    """Minimal object implementing the scorer callback fields used by the extension."""

    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_paths(root: Path) -> tuple[Path, ...]:
    return (
        root / RUNS[0] / "checkpoints" / "model.pt",
        root / RUNS[1] / "checkpoints" / "model.pt",
        root / RUNS[2] / "checkpoints" / "model.pt",
        root / "models" / RUNS[3] / "model.pt",
    )


def _load_reference_models(
    paths: tuple[Path, ...],
) -> tuple[list[Any], list[dict[str, np.ndarray]]]:
    import torch

    from wnn.nn import RNNModel
    from wnn.training.sequence_models import build_sequence_model

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models: list[Any] = []
    native_weights: list[dict[str, np.ndarray]] = []
    for role, path, expected_sha in zip(ROLES, paths, EXPECTED_CHECKPOINT_SHA256, strict=True):
        if _sha256(path) != expected_sha:
            raise ValueError(f"{role} checkpoint SHA-256 differs from the pinned candidate")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model"]
        if role == "source4_lstm":
            model = build_sequence_model(checkpoint["config"]["model"])
        else:
            args = checkpoint["args"]
            width = (211, 263, 263)[ROLES.index(role)]
            model = RNNModel(
                n_in=width,
                hidden=args["hidden"],
                layers=args["layers"],
                dropout=args["dropout"],
                n_out=state["head.2.weight"].shape[0],
                cell=args["cell"],
                enc=args["enc"],
                readout_activation=args["readout_activation"],
                numeric_map=args["numeric_map"],
                ple_intervals=args["ple_intervals"],
                head_layers=args["head_layers"],
            )
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)

        exported = {
            name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
            for name, value in state.items()
        }
        exported["head.2.weight"] = np.ascontiguousarray(exported["head.2.weight"][:2])
        exported["head.2.bias"] = np.ascontiguousarray(exported["head.2.bias"][:2])
        native_weights.append(exported)
    return models, native_weights


def _synthetic_sequences(raw_mean: np.ndarray, raw_std: np.ndarray) -> tuple[np.ndarray, ...]:
    """Create two finite sequences with quiet rows and staged price-event updates."""
    rng = np.random.default_rng(20261031)
    scales = np.maximum(np.abs(raw_std.astype(np.float32)), np.float32(1e-3))

    def make(seed_offset: float) -> np.ndarray:
        rows = np.repeat(raw_mean[None, :].astype(np.float32), 48, axis=0)
        rows[0] += np.float32(seed_offset) * scales * np.float32(0.01)
        for step in range(1, len(rows)):
            rows[step] = rows[step - 1]
            rows[step, 22:52] += rng.normal(0.0, 0.025, 30).astype(np.float32) * scales[22:52]
            rows[step, 74:104] += rng.normal(0.0, 0.025, 30).astype(np.float32) * scales[74:104]
            if step % 5 == 1:
                rows[step, 52] += np.float32(0.2) * scales[52]
            if step % 9 == 3:
                rows[step, 0] += np.float32(0.15) * scales[0]
                rows[step, 11] -= np.float32(0.12) * scales[11]
            if step % 13 == 7:
                rows[step, 63] += np.float32(0.1) * scales[63]
        return np.ascontiguousarray(rows, dtype=np.float32)

    return make(0.0), make(1.0)


def _reference_predictions(models: list[Any], raw: np.ndarray) -> np.ndarray:
    import torch

    from wnn.features import event_features_torch
    from wnn.tiedlead import tiedlead_i1_at_last_i0_price_event

    raw_tensor = torch.from_numpy(np.ascontiguousarray(raw[None]))
    with torch.inference_mode():
        events = event_features_torch(raw_tensor)
        w18_inputs = torch.cat((raw_tensor, events), dim=-1)
        lead = tiedlead_i1_at_last_i0_price_event(raw_tensor, events)
        tiedlead_inputs = torch.cat((raw_tensor, events, lead), dim=-1)
        outputs = [models[0](w18_inputs)[0][..., :2]]
        outputs.extend(model(tiedlead_inputs)[0][..., :2] for model in models[1:3])
        outputs.append(models[3](raw_tensor)[0][..., :2])
    return np.stack([value[0].cpu().numpy() for value in outputs], axis=1).astype(
        np.float32, copy=False
    )


def _blend_reference(members: np.ndarray) -> np.ndarray:
    result = np.empty(2, dtype=np.float32)
    for target in range(2):
        baseline64 = 0.0
        baseline64 += float(members[0, target]) * 0.34
        baseline64 += float(members[1, target]) * 0.33
        baseline64 += float(members[2, target]) * 0.33
        baseline32 = np.float32(baseline64)
        outer64 = 0.0
        outer64 += float(baseline32) * 0.9
        outer64 += float(members[3, target]) * 0.1
        result[target] = np.float32(outer64)
    return result


def _native_rows(
    predictor: Any,
    sequence_id: int,
    raw: np.ndarray,
    warmup: int,
    *,
    callback: bool = False,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    members: list[np.ndarray] = []
    predictions: list[np.ndarray | None] = []
    for step, row in enumerate(raw):
        point = DataPoint(sequence_id, step, step >= warmup, row)
        if callback:
            value = predictor.predict(point)
            predictions.append(None if value is None else np.asarray(value, dtype=np.float32))
        else:
            value = predictor.step_members(point)
            members.append(np.asarray(value, dtype=np.float32))
    return np.stack(members) if members else np.empty((0, 4, 2), np.float32), predictions


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if not args.native_dir.is_dir():
        raise FileNotFoundError(f"native extension directory does not exist: {args.native_dir}")
    sys.path.insert(0, str(args.native_dir.resolve()))
    import wnn_native

    paths = _checkpoint_paths(args.checkpoints_root.resolve(strict=True))
    models, native_weights = _load_reference_models(paths)
    raw_mean = models[3].mean.detach().cpu().numpy()
    raw_std = models[3].std.detach().cpu().numpy()
    sequences = _synthetic_sequences(raw_mean, raw_std)
    reference = tuple(_reference_predictions(models, sequence) for sequence in sequences)

    # Run A, B, then A again on one object to exercise sequence resets and replay stability.
    predictor = wnn_native.FinalTl3Predictor(native_weights)
    outputs: list[np.ndarray] = []
    for index, (sequence_id, sequence) in enumerate(
        ((101, sequences[0]), (202, sequences[1]), (101, sequences[0]))
    ):
        rows, _ = _native_rows(predictor, sequence_id, sequence, args.warmup)
        outputs.append(rows)
        compare = reference[index % 2]
        errors = np.abs(rows - compare)
        if not np.allclose(rows, compare, atol=args.atol, rtol=args.rtol):
            member = np.unravel_index(int(np.argmax(errors)), errors.shape)
            raise AssertionError(
                f"native/PyTorch parity failed at replay {index}, member/row/target={member}: "
                f"max_abs={float(errors.max()):.8g}, atol={args.atol}, rtol={args.rtol}"
            )
    if not np.array_equal(outputs[0], outputs[2]):
        raise AssertionError(
            "native sequence reset/replay did not reproduce identical member outputs"
        )

    # Independent future perturbation must not change any prediction already emitted.
    prefix = 19
    changed = sequences[0].copy()
    changed[prefix:, 0] += np.float32(0.7)
    changed[prefix:, 11] -= np.float32(0.5)
    changed[prefix:, 52] += np.float32(0.9)
    reference_changed = _reference_predictions(models, changed)
    if not np.array_equal(reference[0][:prefix], reference_changed[:prefix]):
        raise AssertionError("PyTorch reference uses future rows before the current prefix")
    future_predictor = wnn_native.FinalTl3Predictor(native_weights)
    left, _ = _native_rows(future_predictor, 303, sequences[0], 0)
    future_predictor.reset()
    right, _ = _native_rows(future_predictor, 304, changed, 0)
    if not np.array_equal(left[:prefix], right[:prefix]):
        raise AssertionError(
            "native callback changed an emitted prefix after future rows were altered"
        )

    # Warm-up must still advance all members, then expose the exact nested FP64/FP32 blend.
    callback_predictor = wnn_native.FinalTl3Predictor(native_weights)
    _, callback_values = _native_rows(
        callback_predictor, 404, sequences[0], args.warmup, callback=True
    )
    for step, actual in enumerate(callback_values):
        if step < args.warmup:
            if actual is not None:
                raise AssertionError(f"callback emitted prediction during warm-up row {step}")
        else:
            expected = np.asarray(wnn_native.FinalTl3Predictor.blend_members(outputs[0][step]))
            if actual is None or not np.array_equal(actual, expected):
                raise AssertionError(
                    f"callback blend differs from nested native blend at row {step}"
                )

    probe = np.array(
        [
            [0.1234567, -0.7654321],
            [1.234567, -1.345678],
            [-0.2222222, 0.3333333],
            [2.111111, -2.222222],
        ],
        dtype=np.float32,
    )
    native_blend = np.asarray(wnn_native.FinalTl3Predictor.blend_members(probe))
    if not np.array_equal(native_blend, _blend_reference(probe)):
        raise AssertionError("native nested blend differs from its explicit FP64/FP32 reference")

    # The opt-in serving fallback must run only W18 and return exactly its raw member score.
    full_q8 = wnn_native.FinalTl3Predictor(
        native_weights, precision="q8vnni", fast_activations=True
    )
    w18_only = wnn_native.FinalTl3Predictor(
        native_weights,
        precision="q8vnni",
        fast_activations=True,
        serving_policy="w18_only",
    )
    if tuple(w18_only.active_member_indices) != (0,) or w18_only.serving_policy != "w18_only":
        raise AssertionError("W18-only policy does not report its active member truthfully")
    if not all(name.startswith("q8vnni") for name in w18_only.effective_precisions):
        raise AssertionError("W18-only policy did not preserve requested Q8 precision")
    full_q8_rows: list[np.ndarray] = []
    w18_replays: list[np.ndarray] = []
    for sequence_id, sequence in (
        (501, sequences[0]),
        (502, sequences[1]),
        (503, sequences[0]),
    ):
        full_rows, _ = _native_rows(full_q8, sequence_id, sequence, 0)
        single_rows, _ = _native_rows(w18_only, sequence_id, sequence, 0)
        full_q8_rows.append(full_rows)
        w18_replays.append(single_rows)
        if not np.array_equal(single_rows[:, 0, :], full_rows[:, 0, :]):
            raise AssertionError("W18-only member output differs from full-mix member zero")
        if np.any(single_rows[:, 1:, :] != 0):
            raise AssertionError("W18-only step_members did not leave inactive roles empty")
    if not np.array_equal(w18_replays[0], w18_replays[2]):
        raise AssertionError("W18-only reset/replay did not reproduce identical outputs")
    w18_callback = wnn_native.FinalTl3Predictor(
        native_weights,
        precision="q8vnni",
        fast_activations=True,
        serving_policy="w18_only",
    )
    for step, row in enumerate(sequences[0]):
        value = w18_callback.predict(DataPoint(601, step, step >= args.warmup, row))
        if step < args.warmup:
            if value is not None:
                raise AssertionError("W18-only callback emitted during warm-up")
        elif value is None or not np.array_equal(
            np.asarray(value, dtype=np.float32), w18_replays[0][step, 0]
        ):
            raise AssertionError("W18-only callback differs from the raw W18 member output")

    max_member_abs = np.max(np.abs(outputs[0] - reference[0]), axis=(0, 1)).tolist()
    max_member_rel = np.max(
        np.abs(outputs[0] - reference[0]) / np.maximum(np.abs(reference[0]), 1e-6), axis=(0, 1)
    ).tolist()
    return {
        "schema": "wnn.final-tl3-synthetic-parity.v1",
        "checkpoint_sha256": {role: _sha256(path) for role, path in zip(ROLES, paths, strict=True)},
        "sequence_rows": [len(sequence) for sequence in sequences],
        "warmup_rows": args.warmup,
        "torch_device": "cpu",
        "torch_version": torch.__version__,
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "max_abs_error_by_member": max_member_abs,
        "max_relative_error_by_member": max_member_rel,
        "replay_identical": True,
        "future_prefix_invariant": True,
        "blend_bitwise_match": True,
        "w18_only_raw_member_match": True,
        "w18_only_reset_replay_identical": True,
        "w18_only_active_member_indices": [0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    args = parser.parse_args()
    if args.warmup < 0 or args.atol < 0 or args.rtol < 0:
        parser.error("warmup and parity tolerances must be nonnegative")
    import json

    print(json.dumps(run(args), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
