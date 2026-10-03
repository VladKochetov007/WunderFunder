"""Actual CPU recurrent initial-state instrument; target-free selected windows."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from wnn.nn import RNNModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PANEL = ROOT / "research/campaign_2026-10-03/continuation/round3/matched_usage"
LAGS = [0, 8, 32, 96, 256, 288]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1 << 20), b""):
            sha.update(piece)
    return sha.hexdigest()


def rms(value: Tensor) -> Tensor:
    return value.double().square().mean().sqrt()


def state_digest(model: RNNModel) -> str:
    sha = hashlib.sha256()
    for name, value in model.state_dict().items():
        sha.update(name.encode())
        sha.update(value.numpy().tobytes())
    return sha.hexdigest()


def direction(hidden: Tensor, seed: int) -> tuple[Tensor, dict[str, object]]:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(hidden.shape, generator=generator)
    perturbation = torch.zeros_like(hidden)
    norms, direction_norms, zeros = [], [], []
    for layer in range(hidden.shape[0]):
        h_rms, d_rms = float(rms(hidden[layer])), float(rms(raw[layer]))
        zero = h_rms == 0 or d_rms == 0
        if not zero:
            perturbation[layer] = raw[layer] * (0.01 * h_rms / d_rms)
        norms.append(h_rms)
        direction_norms.append(d_rms)
        zeros.append(zero)
    return perturbation, {
        "hidden_rms": norms,
        "raw_direction_rms": direction_norms,
        "zero_direction_layers": zeros,
        "desired_delta_rms": [v * 0.01 for v in norms],
        "constructed_delta_rms": [float(rms(d)) for d in perturbation],
    }


def rollout(model: RNNModel, inputs: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
    outputs, states = [], []
    start = 0
    for lag in LAGS:
        output, hidden = model(inputs[:, start : lag + 1], hidden)
        outputs.append(output)
        states.append(hidden.clone())
        start = lag + 1
    return torch.cat(outputs, 1), torch.stack(states)


def toy_controls() -> dict[str, object]:
    tests: dict[str, object] = {}
    for name, beta, visible in (
        ("contractive", 0.5, True),
        ("nonforgetting", 1.0, True),
        ("unobservable_head", 1.0, False),
    ):
        h = np.array([0.0, 0.01])
        rows = []
        for step in range(289):
            h = beta * h
            if step in LAGS:
                rows.append(
                    {
                        "lag": step,
                        "hidden_l2": float(np.linalg.norm(h)),
                        "output_abs": float(abs(h[1]) if visible else abs(h[0])),
                    }
                )
        if name == "contractive":
            assert all(
                np.isclose(
                    row["hidden_l2"], 0.01 * 0.5 ** (row["lag"] + 1), atol=1e-100, rtol=1e-12
                )
                for row in rows
            )
        elif name == "nonforgetting":
            assert all(row["output_abs"] == 0.01 for row in rows)
        else:
            assert all(row["hidden_l2"] == 0.01 and row["output_abs"] == 0 for row in rows)
        tests[name] = rows
    zero, metadata = direction(torch.zeros(2, 1, 256), 614407)
    assert torch.count_nonzero(zero) == 0 and all(metadata["zero_direction_layers"])
    tests["zero_norm_retained"] = metadata
    tests["all_pass"] = True
    return tests


def run_case(
    model: RNNModel, case: dict[str, int], expected_hash: str
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Read ONLY declared input/state/reference members; never targets/masks."""
    index, start, end = case["panel_index"], case["start"], case["end"]
    path = PANEL / f"seq_{index:02d}.npz"
    assert digest(path) == expected_hash
    with np.load(path) as stored:
        inputs = torch.from_numpy(stored["packed"][start : end + 1].copy())[None]
        boundaries = stored["state_boundaries"]
        position = np.flatnonzero(boundaries == start)
        assert len(position) == 1
        hidden = torch.from_numpy(stored["original_states"][position[0]].copy())
        reference = torch.from_numpy(stored["reference"][start : end + 1].copy())[None]
        assert case["query"] in stored["queries"]
    assert inputs.shape == (1, 289, 211) and hidden.shape == (2, 1, 256)
    original_inputs = inputs.clone()
    base, states = rollout(model, inputs, hidden.clone())
    repeated, repeat_states = rollout(model, inputs, hidden.clone())
    direct, direct_end_state = model(inputs, hidden.clone())
    direct_prefix_states = torch.stack(
        [model(inputs[:, : lag + 1], hidden.clone())[1] for lag in LAGS]
    )
    eps = torch.finfo(torch.float32).eps
    floors = []
    for j, lag in enumerate(LAGS):
        out_error = max(
            float((base[:, lag, :2] - repeated[:, lag, :2]).abs().max()),
            float((base[:, lag, :2] - direct[:, lag, :2]).abs().max()),
            float((base[:, lag, :2] - reference[:, lag]).abs().max()),
        )
        hidden_error = torch.maximum(
            (states[j] - repeat_states[j]).abs().flatten(1).amax(1),
            (states[j] - direct_prefix_states[j]).abs().flatten(1).amax(1),
        )
        hidden_scale = states[j].abs().flatten(1).amax(1).clamp_min(1)
        hidden_floor = torch.maximum(10 * hidden_error, eps * hidden_scale)
        floors.append(
            {
                "lag": lag,
                "output": max(10 * out_error, eps * max(1.0, float(base[:, lag, :2].abs().max()))),
                "hidden_per_layer": hidden_floor.tolist(),
                "unchanged_output_maxabs": out_error,
                "unchanged_hidden_maxabs_per_layer": hidden_error.tolist(),
            }
        )
    assert torch.allclose(base[..., :2], reference, atol=2e-5, rtol=2e-5)
    delta, scale_metadata = direction(hidden, case["direction_seed"])
    records = []
    arrays = {
        "inputs": inputs.numpy(),
        "incoming_hidden": hidden.numpy(),
        "delta": delta.numpy(),
        "base_outputs": base.numpy(),
        "base_hidden_at_lags": states.numpy(),
    }
    for sign, name in ((1, "plus"), (-1, "minus")):
        initial = hidden + sign * delta
        output, arm_states = rollout(model, inputs, initial)
        effective_delta = initial - hidden
        arrays[name + "_outputs"] = output.numpy()
        arrays[name + "_hidden_at_lags"] = arm_states.numpy()
        initial_l2 = float(effective_delta.norm())
        for j, lag in enumerate(LAGS):
            out_response = output[:, lag, :2] - base[:, lag, :2]
            hidden_response = arm_states[j] - states[j]
            records.append(
                {
                    "sign": name,
                    "lag": lag,
                    "updates": lag + 1,
                    "output_l2": float(out_response.norm()),
                    "output_rms": float(rms(out_response)),
                    "output_maxabs": float(out_response.abs().max()),
                    "aux_all23_maxabs": float((output[:, lag] - base[:, lag]).abs().max()),
                    "hidden_l2": float(hidden_response.norm()),
                    "hidden_maxabs_per_layer": hidden_response.abs().flatten(1).amax(1).tolist(),
                    "hidden_rms_per_layer": [float(rms(h)) for h in hidden_response],
                    "initial_effective_l2": initial_l2,
                    "effective_delta_rms_per_layer": [float(rms(d)) for d in effective_delta],
                    "effective_zero_layers": [
                        bool(torch.count_nonzero(d) == 0) for d in effective_delta
                    ],
                    "initial_outside_gru_unit_interval": int((initial.abs() > 1).sum()),
                    "output_above_floor": float(out_response.abs().max()) > floors[j]["output"],
                    "hidden_above_floor_per_layer": (
                        hidden_response.abs().flatten(1).amax(1)
                        > torch.tensor(floors[j]["hidden_per_layer"])
                    ).tolist(),
                    "output_gain_l2_over_initial_l2": float(out_response.norm()) / initial_l2
                    if initial_l2
                    else None,
                }
            )
    future = inputs.clone()
    future[:, 33:] += 0.137 * model.std
    future_output, future_hidden = model(future, hidden.clone())
    future_error = float((future_output[:, :33] - direct[:, :33]).abs().max())
    future_floor = max(
        10 * float((base[:, :33] - direct[:, :33]).abs().max()),
        eps * max(1.0, float(direct[:, :33].abs().max())),
    )
    assert future_error <= future_floor
    assert torch.equal(inputs, original_inputs) and torch.equal(future[:, :33], inputs[:, :33])
    finite_arrays = all(np.isfinite(value).all() for value in arrays.values())
    assert finite_arrays
    normalized = (inputs - model.mean) / model.std
    activation = model.encode(normalized)
    assert all(
        torch.isfinite(value).all()
        for value in (
            direct,
            direct_end_state,
            direct_prefix_states,
            future_output,
            future_hidden,
            normalized,
            activation,
        )
    )
    envelope = {
        "input_rms": float(rms(inputs)),
        "input_maxabs": float(inputs.abs().max()),
        "normalized_rms": float(rms(normalized)),
        "normalized_maxabs": float(normalized.abs().max()),
        "encoded_gelu_rms": float(rms(activation)),
        "encoded_gelu_min": float(activation.min()),
        "encoded_gelu_max": float(activation.max()),
        "incoming_hidden_maxabs": float(hidden.abs().max()),
        "hidden_at_lags_maxabs": float(states.abs().max()),
        "hidden_abs_over_099_fraction": float((states.abs() > 0.99).float().mean()),
        "baseline_primary_output_maxabs": float(base[..., :2].abs().max()),
        "std_min": float(model.std.min()),
        "std_max": float(model.std.max()),
    }
    return {
        "case": case,
        "scale": scale_metadata,
        "floors": floors,
        "responses": records,
        "numerical_envelope": envelope,
        "identity_reference_maxabs": float((base[..., :2] - reference).abs().max()),
        "repeat_output_maxabs": float((base - repeated).abs().max()),
        "repeat_hidden_maxabs": float((states - repeat_states).abs().max()),
        "segmented_vs_direct_end_hidden_maxabs": float((states[-1] - direct_end_state).abs().max()),
        "future_canary_maxabs": future_error,
        "future_canary_floor": future_floor,
        "all_finite": finite_arrays,
        "inputs_identical_all_arms": True,
    }, arrays


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    selection = json.loads((HERE / "selection.json").read_text())
    assert digest(HERE / "PREREG.json") == selection["prereg_sha256"]
    assert digest(PANEL / "samples.json") == selection["panel_samples_sha256"]
    assert all(
        digest(ROOT / path) == expected for path, expected in selection["source_sha256"].items()
    )
    checkpoint_path = ROOT / selection["checkpoint"]
    assert digest(checkpoint_path) == selection["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    assert (
        args["arch"] == "gru"
        and args.get("enc", "linear") == "linear"
        and args.get("cell", "gru") == "gru"
    )
    assert args["hidden"] == 256 and args["layers"] == 2 and args["dropout"] == 0
    assert checkpoint["n_in"] == 211
    model = RNNModel(211, hidden=256, layers=2, dropout=0, n_out=23).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"], strict=True)
    weight_identity = state_digest(model)
    registry = json.loads((PANEL / "artifact_provenance.json").read_text())["artifacts"]
    cases = []
    controls = toy_controls()
    with torch.inference_mode():
        for case in selection["cases"]:
            print(f"case component={case['component']} query={case['query']}", flush=True)
            result, arrays = run_case(
                model, case, registry[f"seq_{case['panel_index']:02d}.npz"]["sha256"]
            )
            cases.append(result)
            np.savez(HERE / f"case_{case['panel_index']:02d}.npz", **arrays)
    assert state_digest(model) == weight_identity
    # Heterogeneous case/direction envelopes, not a fitted contraction rate.
    envelope = []
    for lag in LAGS:
        observations = [row for case in cases for row in case["responses"] if row["lag"] == lag]
        envelope.append(
            {
                "lag": lag,
                "case_signs": len(observations),
                "output_above_floor": sum(row["output_above_floor"] for row in observations),
                "hidden_above_floor_any_layer": sum(
                    any(row["hidden_above_floor_per_layer"]) for row in observations
                ),
                "output_maxabs_min": min(row["output_maxabs"] for row in observations),
                "output_maxabs_max": max(row["output_maxabs"] for row in observations),
                "hidden_l2_min": min(row["hidden_l2"] for row in observations),
                "hidden_l2_max": max(row["hidden_l2"] for row in observations),
            }
        )
    result = {
        "scope": "local initial-state instrument; no targets/scoring/uniform stability/gain claim",
        "cases": cases,
        "controls": controls,
        "envelope": envelope,
        "model_state_digest_unchanged": weight_identity,
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "array_members_read": [
            "packed",
            "state_boundaries",
            "original_states",
            "reference",
            "queries",
        ],
        "full_sequence_forwards": 0,
        "target_or_mask_arrays_read": False,
        "gpu_used": False,
        "cost": {
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time(),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"envelope": envelope, "cost": result["cost"]}, indent=2))


if __name__ == "__main__":
    main()
