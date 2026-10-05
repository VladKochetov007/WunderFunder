"""Actual-data B4x256 CUDA probe for E474; never launches full fits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(HERE))

import fit_activation_optimizer as e470  # noqa: E402
from fit_activation_placement import (  # noqa: E402
    ARMS,
    CHUNK_ROWS,
    DATASET_SHA256,
    EXPECTED_PARAMETERS,
    SEEDS,
    file_hash,
    full_ieee_float32,
    initialize_arm,
    make_optimizer,
    make_seed_states,
)

from wnn.nn import RNNModel  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402


def _finite(parameters: list[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in parameters)


def main() -> None:
    """Check actual CUDA shape, parity, and one optimizer update for all arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {args.output}")
    args.output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA probe requested but CUDA is unavailable")
    torch.set_num_threads(1)

    manifest, x, y, need, _, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("probe must load fit rows only")
    if x.shape != (24, 20_000, 42) or y.shape != (24, 20_000, 2):
        raise ValueError(f"unexpected fit cohort shape: {x.shape}, {y.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite raw fit-only probe data")
    seed_order = np.arange(24)
    np.random.default_rng(SEEDS[0]).shuffle(seed_order)
    ids = seed_order[:4].tolist()
    row_groups = [nodes[index]["row_group"] for index in ids]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    xx = torch.from_numpy(x[ids, :CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    yy = torch.from_numpy(y[ids, :CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    mm = torch.from_numpy(need[ids, :CHUNK_ROWS]).to(device=device)
    states = make_seed_states()
    result: dict = {
        "kind": "E474 actual-data CUDA shape probe",
        "status": "running",
        "fit_only": True,
        "test_rows_loaded": False,
        "device": str(device),
        "fit_row_groups": row_groups,
        "batch": 4,
        "chunk_rows": CHUNK_ROWS,
        "seeds": list(SEEDS),
        "trainable_parameters_per_arm": EXPECTED_PARAMETERS,
        "precision_policy": "FP32 IEEE, autocast disabled for baseline and candidates",
        "parity_tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "cohort_sha256": file_hash(e470.COHORT_PATH),
        "dataset_sha256": DATASET_SHA256,
        "dataset_bytes": e470.DATA_PATH.stat().st_size,
        "source_sha256": {
            name: file_hash(HERE / name)
            for name in ("fit_activation_placement.py", "tests.py", "gpu_probe.py")
        },
        "arms": [],
    }

    with full_ieee_float32():
        # The real baseline parity check uses the exact same tensors and state.
        baseline = initialize_arm(SEEDS[0], "gelu", "gelu", states[SEEDS[0]]).to(device)
        reference = RNNModel(42, hidden=64, layers=2, dropout=0.0, n_out=2).to(device)
        reference.load_state_dict(states[SEEDS[0]], strict=True)
        baseline_prediction, baseline_state = baseline(xx)
        reference_prediction, reference_state = reference(xx)
        torch.testing.assert_close(baseline_prediction, reference_prediction, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(baseline_state, reference_state, atol=1e-6, rtol=1e-5)
        baseline_loss = baseline_prediction.square().mean()
        reference_loss = reference_prediction.square().mean()
        baseline_loss.backward()
        reference_loss.backward()
        gradient_deltas = []
        for (name_a, param_a), (name_b, param_b) in zip(
            baseline.named_parameters(), reference.named_parameters(), strict=True
        ):
            if name_a != name_b or param_a.grad is None or param_b.grad is None:
                raise ValueError("baseline and reference parameter/gradient structures differ")
            gradient_deltas.append((param_a.grad - param_b.grad).abs().max().item())
        max_gradient_delta = max(gradient_deltas)
        if max_gradient_delta > 1e-5:
            raise ValueError(f"baseline/reference gradient mismatch: {max_gradient_delta}")
        result["baseline_parity"] = {
            "max_output_abs": (baseline_prediction - reference_prediction).abs().max().item(),
            "max_state_abs": (baseline_state - reference_state).abs().max().item(),
            "max_parameter_gradient_abs": max_gradient_delta,
            "status": "passed",
        }

        for seed, name, embedding, head in ARMS:
            model = initialize_arm(seed, embedding, head, states[seed]).to(device).float()
            if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
                raise ValueError(f"parameter count mismatch in {seed}/{name}")
            optimizer = make_optimizer(model)
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(xx)
            if not (torch.isfinite(prediction).all() and torch.isfinite(state).all()):
                raise ValueError(f"nonfinite forward output in {seed}/{name}")
            loss = e470.loss_for_batch(prediction, yy, mm)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss in {seed}/{name}")
            loss.backward()
            gradients = [
                parameter.grad for parameter in model.parameters() if parameter.grad is not None
            ]
            if not _finite(gradients):
                raise ValueError(f"nonfinite gradients in {seed}/{name}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            optimizer.step()
            params = list(model.parameters())
            if not _finite(params):
                raise ValueError(f"nonfinite updated parameters in {seed}/{name}")
            changed = any(
                not torch.equal(old, parameter)
                for old, parameter in zip(before, params, strict=True)
            )
            if not changed:
                raise ValueError(f"optimizer update changed no parameters in {seed}/{name}")
            result["arms"].append(
                {
                    "seed": seed,
                    "arm": name,
                    "embedding_activation": embedding,
                    "head_activation": head,
                    "loss": float(loss.detach().cpu()),
                    "state_shape": list(state.shape),
                    "parameters": sum(parameter.numel() for parameter in params),
                    "finite_updated_parameters": True,
                    "parameters_changed": True,
                    "updates": 1,
                }
            )
    result["status"] = "passed"
    result["cuda_peak_memory_bytes"] = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    result["cuda_peak_reserved_bytes"] = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    )
    (args.output / "probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
