"""Production-batch CUDA parity and update probe for the E471 candidate.

The probe uses one shuffled B4 x 256 batch from the frozen fit-only cohort.
It compares the alpha-zero custom scan to E470 before making one update in
each fixed arm. ``--device cpu`` runs the same checks as a bounded dry check.
No test rows are loaded or scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_optimizer"))

from fit_recurrent_candidate import (  # noqa: E402
    ALPHAS,
    OPTIMIZERS,
    CandidateGRUModel,
    e470,
    make_optimizer,
)

from wnn.resources import configure_torch  # noqa: E402

OUTPUT_ATOL = 2e-5
OUTPUT_RTOL = 2e-4
GRAD_ATOL = 5e-5
GRAD_RTOL = 5e-4
DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"


def assert_finite(tensor: torch.Tensor, label: str) -> None:
    """Reject nonfinite probe values with a field-specific diagnostic."""
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"nonfinite {label}")


def file_hash(path: Path) -> str:
    """Return a content hash for the probe receipt."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alpha_zero_parity(
    x: torch.Tensor, target: torch.Tensor, need: torch.Tensor, device: torch.device
) -> dict:
    """Compare E470 and scripted alpha-zero output, state and full-model grads."""
    torch.manual_seed(271828)
    reference = e470.ActivationGRU("tanh").to(device)
    torch.manual_seed(271828)
    candidate = CandidateGRUModel(0.0).to(device)
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if reference_parameters.keys() != candidate_parameters.keys():
        raise ValueError("E470/candidate parameter names do not match")
    for name, parameter in reference_parameters.items():
        if not torch.equal(parameter, candidate_parameters[name]):
            raise ValueError(f"E470/candidate initialization differs at {name}")

    x_reference = x.detach().clone().requires_grad_(True)
    x_candidate = x.detach().clone().requires_grad_(True)
    output_reference, state_reference = reference(x_reference)
    output_candidate, state_candidate = candidate(x_candidate)
    assert_finite(output_reference, "E470 output")
    assert_finite(output_candidate, "candidate output")
    assert_finite(state_reference, "E470 state")
    assert_finite(state_candidate, "candidate state")
    torch.testing.assert_close(
        output_candidate, output_reference, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL
    )
    torch.testing.assert_close(state_candidate, state_reference, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    loss_reference = e470.loss_for_batch(output_reference, target, need)
    loss_candidate = e470.loss_for_batch(output_candidate, target, need)
    assert_finite(loss_reference, "E470 parity loss")
    assert_finite(loss_candidate, "candidate parity loss")
    loss_reference.backward()
    loss_candidate.backward()

    max_gradient_difference = 0.0
    for name, parameter_reference in reference_parameters.items():
        grad_reference = parameter_reference.grad
        grad_candidate = candidate_parameters[name].grad
        if grad_reference is None or grad_candidate is None:
            if grad_reference is not grad_candidate:
                raise ValueError(f"gradient presence differs at {name}")
            continue
        assert_finite(grad_reference, f"E470 gradient {name}")
        assert_finite(grad_candidate, f"candidate gradient {name}")
        torch.testing.assert_close(
            grad_candidate, grad_reference, atol=GRAD_ATOL, rtol=GRAD_RTOL, msg=name
        )
        max_gradient_difference = max(
            max_gradient_difference,
            float((grad_candidate - grad_reference).abs().max().detach().cpu()),
        )
    if x_reference.grad is None or x_candidate.grad is None:
        raise ValueError("input gradient is missing from alpha-zero parity")
    assert_finite(x_reference.grad, "E470 input gradient")
    assert_finite(x_candidate.grad, "candidate input gradient")
    torch.testing.assert_close(x_candidate.grad, x_reference.grad, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    prediction_difference = float((output_candidate - output_reference).abs().max().detach().cpu())
    state_difference = float((state_candidate - state_reference).abs().max().detach().cpu())
    input_gradient_difference = float(
        (x_candidate.grad - x_reference.grad).abs().max().detach().cpu()
    )
    del reference, candidate
    return {
        "passed": True,
        "batch": 4,
        "chunk_rows": 256,
        "output_atol": OUTPUT_ATOL,
        "output_rtol": OUTPUT_RTOL,
        "gradient_atol": GRAD_ATOL,
        "gradient_rtol": GRAD_RTOL,
        "max_output_abs_difference": prediction_difference,
        "max_state_abs_difference": state_difference,
        "max_input_gradient_abs_difference": input_gradient_difference,
        "max_parameter_gradient_abs_difference": max_gradient_difference,
    }


def update_arm(
    alpha_name: str,
    alpha: float,
    optimizer_name: str,
    x: torch.Tensor,
    target: torch.Tensor,
    need: torch.Tensor,
    device: torch.device,
    duty: object,
) -> dict:
    """Apply and validate one real fit-only optimizer update for an arm."""
    torch.manual_seed(271828)
    model = CandidateGRUModel(alpha).to(device)
    optimizer = make_optimizer(model, optimizer_name)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    update_started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    prediction, final_state = model(x)
    assert_finite(prediction, f"{alpha_name}/{optimizer_name} prediction")
    assert_finite(final_state, f"{alpha_name}/{optimizer_name} state")
    loss = e470.loss_for_batch(prediction, target, need)
    assert_finite(loss, f"{alpha_name}/{optimizer_name} loss")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    for index, gradient in enumerate(gradients):
        assert_finite(gradient, f"{alpha_name}/{optimizer_name} gradient {index}")
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()
    for index, parameter in enumerate(model.parameters()):
        assert_finite(parameter, f"{alpha_name}/{optimizer_name} parameter {index}")
    changed = any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )
    if not changed:
        raise ValueError(f"no parameters changed for {alpha_name}/{optimizer_name}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    update_seconds = time.monotonic() - update_started
    if duty is not None and device.type == "cuda":
        duty.tick()
    return {
        "arm": f"{alpha_name}__{optimizer_name}",
        "alpha": alpha,
        "optimizer": optimizer_name,
        "loss": float(loss.detach().cpu()),
        "one_update": True,
        "parameters_changed": changed,
        "parameters_finite": True,
        "update_seconds": update_seconds,
        "estimated_474_update_seconds": 474 * update_seconds,
    }


def main() -> None:
    """Check parity and resource/update behavior on one fixed fit-only batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E471 CUDA probe requested but CUDA is unavailable")
    device = torch.device(args.device)
    duty = configure_torch() if device.type == "cuda" else None
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _, x_all, y_all, need_all, _, nodes = e470.load_cohort(include_test=False)
    fit_ids = list(range(len(nodes)))
    np.random.default_rng(271828).shuffle(fit_ids)
    ids = fit_ids[:4]
    row_groups = [nodes[index]["row_group"] for index in ids]
    x_np = x_all[ids, :256]
    target_np = y_all[ids, :256]
    need_np = need_all[ids, :256]
    if x_np.shape != (4, 256, 42) or target_np.shape != (4, 256, 2):
        raise ValueError("unexpected real fit-only probe batch shape")
    if not (np.isfinite(x_np).all() and np.isfinite(target_np).all()):
        raise ValueError("nonfinite real fit-only batch")
    x = torch.from_numpy(x_np).to(device)
    target = torch.from_numpy(target_np).to(device)
    need = torch.from_numpy(need_np).to(device)
    assert_finite(x, "real fit-only features")
    assert_finite(target, "real fit-only targets")
    started = time.monotonic()

    parity_started = time.monotonic()
    parity = alpha_zero_parity(x, target, need, device)
    parity_seconds = time.monotonic() - parity_started
    arms = [
        update_arm(alpha_name, alpha, optimizer_name, x, target, need, device, duty)
        for alpha_name, alpha in ALPHAS
        for optimizer_name in OPTIMIZERS
    ]
    probe = {
        "kind": "E471 fit-only B4 x 256 parity and one-update-per-arm probe",
        "device": str(device),
        "data_scope": "frozen fit role only; no test row groups loaded or scored",
        "fit_row_groups": row_groups,
        "dataset_sha256": DATASET_SHA256,
        "cohort_sha256": file_hash(e470.COHORT_PATH),
        "source_sha256": {
            "gpu_probe.py": file_hash(Path(__file__).resolve()),
            "fit_recurrent_candidate.py": file_hash(HERE / "fit_recurrent_candidate.py"),
            "tests.py": file_hash(HERE / "tests.py"),
            "fit_activation_optimizer.py": file_hash(Path(e470.__file__).resolve()),
        },
        "elapsed_seconds": time.monotonic() - started,
        "alpha_zero_parity_seconds": parity_seconds,
        "full_fit_time_estimate_method": "sum of per-arm measured first real B4x256 update seconds times 474; excludes cohort load and parity",
        "estimated_six_arm_full_fit_seconds": sum(
            arm["estimated_474_update_seconds"] for arm in arms
        ),
        "alpha_zero_parity": parity,
        "arms": arms,
    }
    if device.type == "cuda":
        probe["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        probe["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    args.output.mkdir(parents=True)
    (args.output / "result.json").write_text(json.dumps(probe, indent=2))
    print(json.dumps(probe, indent=2))


if __name__ == "__main__":
    main()
