"""Compare fixed TF32/backend precision policies for the E471 alpha-zero parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_optimizer"))

from fit_recurrent_candidate import CandidateGRUModel, e470  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

DATASET_SHA256 = "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782"
EXPECTED_ORIGINAL_PROBE_SHA256 = "aeac54f44ade8370845746a65c3e5d7c7285d3c4241e96c768d73f1a7c5a5b5d"
# Frozen from the source above; the hash check prevents silent tolerance drift.
OUTPUT_ATOL = 2e-5
OUTPUT_RTOL = 2e-4
GRAD_ATOL = 5e-5
GRAD_RTOL = 5e-4
SEED = 271828


def sha256_file(path: Path) -> str:
    """Hash one content-bound source or output file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite(tensor: torch.Tensor, label: str) -> None:
    """Fail closed on a nonfinite numerical value before recording diagnostics."""
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"nonfinite {label}")


def compare_array(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    absolute_tolerance: float,
    relative_tolerance: float,
    key: str,
    differences: dict[str, np.ndarray],
) -> dict:
    """Return fixed-tolerance diagnostics and preserve the raw difference array."""
    require_finite(reference, f"reference {key}")
    require_finite(candidate, f"candidate {key}")
    reference_array = reference.detach().to(torch.float32).cpu().numpy()
    candidate_array = candidate.detach().to(torch.float32).cpu().numpy()
    difference = candidate_array - reference_array
    if not np.isfinite(difference).all():
        raise ValueError(f"nonfinite difference array: {key}")
    allowed = absolute_tolerance + relative_tolerance * np.abs(reference_array)
    failed = np.abs(difference) > allowed
    differences[key] = difference
    denominator = np.maximum(np.abs(reference_array), np.finfo(np.float32).tiny)
    return {
        "passed_original_tolerance": bool(not failed.any()),
        "atol": absolute_tolerance,
        "rtol": relative_tolerance,
        "elements": int(difference.size),
        "mismatched_elements": int(failed.sum()),
        "max_abs_difference": float(np.max(np.abs(difference), initial=0.0)),
        "max_relative_difference": float(np.max(np.abs(difference) / denominator, initial=0.0)),
        "difference_array_key": key,
        "shape": list(difference.shape),
    }


def evaluate_policy(
    name: str,
    flags: dict[str, bool],
    x: torch.Tensor,
    target: torch.Tensor,
    need: torch.Tensor,
    differences: dict[str, np.ndarray],
    device: torch.device,
) -> dict:
    """Compare same-initialization E470 and candidate output, state and gradients."""
    torch.backends.cuda.matmul.allow_tf32 = flags["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = flags["cudnn_allow_tf32"]
    torch.backends.cudnn.enabled = flags["cudnn_enabled"]
    torch.manual_seed(SEED)
    reference = e470.ActivationGRU("tanh").to(device)
    torch.manual_seed(SEED)
    candidate = CandidateGRUModel(0.0).to(device)
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if reference_parameters.keys() != candidate_parameters.keys():
        raise ValueError(f"parameter names differ in policy {name}")
    for parameter_name, parameter in reference_parameters.items():
        if not torch.equal(parameter, candidate_parameters[parameter_name]):
            raise ValueError(f"initialization differs at {parameter_name} in policy {name}")

    x_reference = x.detach().clone().requires_grad_(True)
    x_candidate = x.detach().clone().requires_grad_(True)
    output_reference, state_reference = reference(x_reference)
    output_candidate, state_candidate = candidate(x_candidate)
    policy_key = name.replace("-", "_")
    output = compare_array(
        output_reference,
        output_candidate,
        OUTPUT_ATOL,
        OUTPUT_RTOL,
        f"{policy_key}_output",
        differences,
    )
    state = compare_array(
        state_reference,
        state_candidate,
        OUTPUT_ATOL,
        OUTPUT_RTOL,
        f"{policy_key}_state",
        differences,
    )
    loss_reference = e470.loss_for_batch(output_reference, target, need)
    loss_candidate = e470.loss_for_batch(output_candidate, target, need)
    require_finite(loss_reference, f"{name} E470 loss")
    require_finite(loss_candidate, f"{name} candidate loss")
    loss_difference = float((loss_candidate - loss_reference).detach().cpu())
    if not np.isfinite(loss_difference):
        raise ValueError(f"nonfinite loss difference in policy {name}")
    loss_passed = bool(
        abs(loss_difference)
        <= OUTPUT_ATOL + OUTPUT_RTOL * abs(float(loss_reference.detach().cpu()))
    )
    loss_reference.backward()
    loss_candidate.backward()
    if x_reference.grad is None or x_candidate.grad is None:
        raise ValueError(f"missing input gradient in policy {name}")
    input_gradient = compare_array(
        x_reference.grad,
        x_candidate.grad,
        GRAD_ATOL,
        GRAD_RTOL,
        f"{policy_key}_input_gradient",
        differences,
    )
    reference_gradient_parts, candidate_gradient_parts = [], []
    for parameter_name, parameter_reference in reference_parameters.items():
        gradient_reference = parameter_reference.grad
        gradient_candidate = candidate_parameters[parameter_name].grad
        if gradient_reference is None or gradient_candidate is None:
            if gradient_reference is not gradient_candidate:
                raise ValueError(f"gradient presence differs at {parameter_name} in {name}")
            continue
        require_finite(gradient_reference, f"{name} E470 gradient {parameter_name}")
        require_finite(gradient_candidate, f"{name} candidate gradient {parameter_name}")
        reference_gradient_parts.append(gradient_reference.detach().reshape(-1))
        candidate_gradient_parts.append(gradient_candidate.detach().reshape(-1))
    if not reference_gradient_parts:
        raise ValueError(f"no parameter gradients were produced in policy {name}")
    parameter_gradient = compare_array(
        torch.cat(reference_gradient_parts),
        torch.cat(candidate_gradient_parts),
        GRAD_ATOL,
        GRAD_RTOL,
        f"{policy_key}_parameter_gradient",
        differences,
    )
    result = {
        "policy": name,
        "backend_flags": dict(flags),
        "output": output,
        "state": state,
        "loss": {
            "reference": float(loss_reference.detach().cpu()),
            "candidate": float(loss_candidate.detach().cpu()),
            "difference": loss_difference,
            "passed_original_tolerance": loss_passed,
            "atol": OUTPUT_ATOL,
            "rtol": OUTPUT_RTOL,
        },
        "input_gradient": input_gradient,
        "parameter_gradient": parameter_gradient,
    }
    result["all_original_tolerances_passed"] = bool(
        output["passed_original_tolerance"]
        and state["passed_original_tolerance"]
        and loss_passed
        and input_gradient["passed_original_tolerance"]
        and parameter_gradient["passed_original_tolerance"]
    )
    del reference, candidate
    return result


def main() -> None:
    """Run the fixed precision-policy matrix; mismatches are recorded, never hidden."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite precision probe output: {output}")
    if sha256_file(HERE / "gpu_probe.py") != EXPECTED_ORIGINAL_PROBE_SHA256:
        raise ValueError("original E471 parity tolerances/source identity changed")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA precision-policy probe requested but CUDA is unavailable")

    device = torch.device(args.device)
    duty = configure_torch() if device.type == "cuda" else None
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    original_flags = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    (cohort, x_all, y_all, need_all, _, nodes) = e470.load_cohort(include_test=False)
    fit_ids = list(range(len(nodes)))
    np.random.default_rng(SEED).shuffle(fit_ids)
    ids = fit_ids[:4]
    row_groups = [nodes[index]["row_group"] for index in ids]
    x_np = x_all[ids, :256]
    target_np = y_all[ids, :256]
    need_np = need_all[ids, :256]
    if x_np.shape != (4, 256, 42) or target_np.shape != (4, 256, 2):
        raise ValueError("unexpected frozen fit-only batch shape")
    if not (np.isfinite(x_np).all() and np.isfinite(target_np).all()):
        raise ValueError("nonfinite raw fit batch before any clipping or cast")
    x = torch.from_numpy(x_np).to(device)
    target = torch.from_numpy(target_np).to(device)
    need = torch.from_numpy(need_np).to(device)
    require_finite(x, "fit features")
    require_finite(target, "fit targets")

    policies = [
        {"name": "runtime_default", "flags": dict(original_flags)},
        {
            "name": "explicit_full_fp32_cudnn_enabled",
            "flags": {
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "cudnn_enabled": True,
            },
        },
        {
            "name": "explicit_full_fp32_cudnn_disabled",
            "flags": {
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "cudnn_enabled": False,
            },
        },
    ]
    differences: dict[str, np.ndarray] = {}
    results = []
    started = time.monotonic()
    try:
        for item in policies:
            case_started = time.monotonic()
            result = evaluate_policy(
                item["name"], item["flags"], x, target, need, differences, device
            )
            result["elapsed_seconds"] = time.monotonic() - case_started
            results.append(result)
            if duty is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
                duty.tick()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_flags["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = original_flags["cudnn_allow_tf32"]
        torch.backends.cudnn.enabled = original_flags["cudnn_enabled"]

    output.mkdir(parents=True)
    difference_path = output / "precision_differences.npz"
    np.savez_compressed(difference_path, **differences)
    result = {
        "kind": "E471 fixed-batch backend precision policy diagnostic",
        "device": str(device),
        "data_scope": "same frozen fit-only B4x256 batch; no test rows loaded or scored",
        "fit_row_groups": row_groups,
        "dataset_sha256": DATASET_SHA256,
        "cohort_sha256": sha256_file(e470.COHORT_PATH),
        "pytorch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "seed": SEED,
        "original_gpu_probe_sha256": sha256_file(HERE / "gpu_probe.py"),
        "original_tolerances": {
            "output_atol": OUTPUT_ATOL,
            "output_rtol": OUTPUT_RTOL,
            "gradient_atol": GRAD_ATOL,
            "gradient_rtol": GRAD_RTOL,
        },
        "original_runtime_backend_flags": original_flags,
        "elapsed_seconds": time.monotonic() - started,
        "policies": results,
        "all_cases_reported": len(results) == len(policies),
        "difference_arrays": {
            "path": difference_path.name,
            "sha256": sha256_file(difference_path),
            "arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                for name, array in differences.items()
            },
        },
    }
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
