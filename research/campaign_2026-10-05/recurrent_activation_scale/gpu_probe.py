"""Fit-only production-shape alpha-zero parity and one-step probe for E478."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E470_DIR = ROOT / "research/campaign_2026-10-05/activation_optimizer"
E471_DIR = ROOT / "research/campaign_2026-10-05/recurrent_activation"
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E471_DIR))
sys.path.insert(0, str(E474_DIR))
sys.path.insert(0, str(HERE))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
from fit_recurrent_activation_scale import (  # noqa: E402
    ALPHAS,
    EXPECTED_PARAMETERS,
    GRAD_ATOL,
    GRAD_RTOL,
    OUTPUT_ATOL,
    OUTPUT_RTOL,
    SEEDS,
    UPDATE_ATOL,
    UPDATE_RTOL,
    check_parent_sources,
    initialize_arm,
    make_initial_states,
    make_muon_optimizer,
)

from wnn.nn import RNNModel  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402


def assert_finite(values: list[torch.Tensor], label: str) -> None:
    """Reject nonfinite tensors before any metric/loss clipping or update."""
    for index, value in enumerate(values):
        if not torch.isfinite(value).all().item():
            raise ValueError(f"nonfinite {label}[{index}]")


def baseline_parity(
    x: torch.Tensor,
    target: torch.Tensor,
    need: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> dict:
    """Compare actual RNNModel with custom alpha-zero scan through one Muon step."""
    reference = RNNModel(42, hidden=64, layers=2, dropout=0.0, n_out=2).to(device)
    reference.load_state_dict(state_dict, strict=True)
    candidate = initialize_arm(0.0, state_dict).to(device)
    x_reference = x.detach().clone().requires_grad_(True)
    x_candidate = x.detach().clone().requires_grad_(True)
    output_reference, state_reference = reference(x_reference)
    output_candidate, state_candidate = candidate(x_candidate)
    assert_finite(
        [output_reference, state_reference, output_candidate, state_candidate], "parity forward"
    )
    torch.testing.assert_close(
        output_candidate, output_reference, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL
    )
    torch.testing.assert_close(state_candidate, state_reference, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    loss_reference = e470.loss_for_batch(output_reference, target, need)
    loss_candidate = e470.loss_for_batch(output_candidate, target, need)
    assert_finite([loss_reference, loss_candidate], "parity loss")
    loss_reference.backward()
    loss_candidate.backward()
    if x_reference.grad is None or x_candidate.grad is None:
        raise ValueError("alpha-zero input gradient is missing")
    assert_finite([x_reference.grad, x_candidate.grad], "parity input gradient")
    torch.testing.assert_close(x_candidate.grad, x_reference.grad, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if reference_parameters.keys() != candidate_parameters.keys():
        raise ValueError("baseline and alpha-zero parameter names differ")
    max_parameter_gradient = 0.0
    for name in reference_parameters:
        grad_reference = reference_parameters[name].grad
        grad_candidate = candidate_parameters[name].grad
        if grad_reference is None or grad_candidate is None:
            raise ValueError(f"parameter gradient is missing at {name}")
        assert_finite([grad_reference, grad_candidate], f"parameter gradient {name}")
        torch.testing.assert_close(
            grad_candidate, grad_reference, atol=GRAD_ATOL, rtol=GRAD_RTOL, msg=name
        )
        max_parameter_gradient = max(
            max_parameter_gradient,
            (grad_candidate - grad_reference).abs().max().item(),
        )
    ref_optimizer = make_muon_optimizer(reference)
    candidate_optimizer = make_muon_optimizer(candidate)
    nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
    nn.utils.clip_grad_norm_(candidate.parameters(), 1.0, error_if_nonfinite=True)
    ref_optimizer.step()
    candidate_optimizer.step()
    max_update_difference = 0.0
    for name, parameter_reference in reference.named_parameters():
        parameter_candidate = candidate_parameters[name]
        assert_finite([parameter_reference, parameter_candidate], f"updated parameter {name}")
        torch.testing.assert_close(
            parameter_candidate,
            parameter_reference,
            atol=UPDATE_ATOL,
            rtol=UPDATE_RTOL,
            msg=name,
        )
        max_update_difference = max(
            max_update_difference,
            (parameter_candidate - parameter_reference).abs().max().item(),
        )
    return {
        "passed": True,
        "batch": 4,
        "chunk_rows": 256,
        "output_state_tolerances": {"atol": OUTPUT_ATOL, "rtol": OUTPUT_RTOL},
        "gradient_tolerances": {"atol": GRAD_ATOL, "rtol": GRAD_RTOL},
        "optimizer_update_tolerances": {"atol": UPDATE_ATOL, "rtol": UPDATE_RTOL},
        "max_output_abs": (output_candidate - output_reference).abs().max().item(),
        "max_state_abs": (state_candidate - state_reference).abs().max().item(),
        "max_input_gradient_abs": (x_candidate.grad - x_reference.grad).abs().max().item(),
        "max_parameter_gradient_abs": max_parameter_gradient,
        "max_post_step_parameter_abs": max_update_difference,
    }


def one_update(
    seed: int,
    alpha: float,
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    ids: list[int],
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> dict:
    """Run one finite fit-only actual-data Muon update for an arm."""
    torch.manual_seed(seed)
    model = initialize_arm(alpha, state_dict).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise ValueError("probe parameter count mismatch")
    optimizer = make_muon_optimizer(model)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    xx = torch.from_numpy(x[ids, :256]).to(device=device, dtype=torch.float32)
    yy = torch.from_numpy(y[ids, :256]).to(device=device, dtype=torch.float32)
    mm = torch.from_numpy(need[ids, :256]).to(device=device)
    assert_finite([xx, yy], "raw fit batch")
    optimizer.zero_grad(set_to_none=True)
    prediction, final_state = model(xx)
    assert_finite([prediction, final_state], "arm forward")
    loss = e470.loss_for_batch(prediction, yy, mm)
    assert_finite([loss], "arm loss")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert_finite(gradients, "arm gradient")
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    params = list(model.parameters())
    assert_finite(params, "updated parameters")
    changed = any(not torch.equal(old, new) for old, new in zip(before, params, strict=True))
    if not changed:
        raise ValueError(f"optimizer update changed no parameter: seed={seed}, alpha={alpha}")
    return {
        "seed": seed,
        "alpha": alpha,
        "loss": float(loss.detach().cpu()),
        "state_shape": list(final_state.shape),
        "parameters_changed": changed,
        "parameters_finite": True,
        "updates": 1,
    }


def main() -> None:
    """Run parity and one-update checks; never launches a full fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite probe directory {args.output}")
    args.output.mkdir(parents=True)
    parent_hashes = check_parent_sources()
    device = torch.device(args.device)
    if device.type == "cuda":
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    manifest, x, y, need, _, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("probe must load only the 24 fit sequences")
    if x.shape != (24, 20_000, 42) or y.shape != (24, 20_000, 2):
        raise ValueError("unexpected fit-only dataset shape")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite fit-only cohort values")
    states = make_initial_states()
    probe_order = np.arange(24)
    np.random.default_rng(SEEDS[0]).shuffle(probe_order)
    ids = probe_order[:4].tolist()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    xx = torch.from_numpy(x[ids, :256]).to(device=device, dtype=torch.float32)
    yy = torch.from_numpy(y[ids, :256]).to(device=device, dtype=torch.float32)
    mm = torch.from_numpy(need[ids, :256]).to(device=device)
    parent_hashes = check_parent_sources()
    result: dict = {
        "kind": "E478 actual-data full-topology alpha-zero parity and Muon probe",
        "status": "running",
        "fit_only": True,
        "test_rows_loaded": False,
        "device": str(device),
        "fit_row_groups": [nodes[index]["row_group"] for index in ids],
        "seeds": list(SEEDS),
        "alphas": list(ALPHAS),
        "parameters_per_arm": EXPECTED_PARAMETERS,
        "precision_policy": "FP32 IEEE, autocast disabled, cuDNN enabled",
        "parent_source_sha256": parent_hashes,
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "dataset_sha256": e474.DATASET_SHA256,
        "source_sha256": {
            name: e474.file_hash(HERE / name)
            for name in ("fit_recurrent_activation_scale.py", "tests.py", "gpu_probe.py")
        },
        "alpha_zero_parity": {},
        "arms": [],
    }
    with e474.full_ieee_float32():
        result["alpha_zero_parity"] = baseline_parity(xx, yy, mm, states[SEEDS[0]], device)
        for seed in SEEDS:
            order = np.arange(24)
            np.random.default_rng(seed).shuffle(order)
            ids = order[:4].tolist()
            for alpha in ALPHAS:
                result["arms"].append(
                    one_update(seed, alpha, x, y, need, ids, states[seed], device)
                )
    result["status"] = "passed"
    result["cuda_peak_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    result["cuda_peak_reserved_bytes"] = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
