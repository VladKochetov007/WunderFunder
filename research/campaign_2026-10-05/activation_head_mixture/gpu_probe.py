"""Actual fit-only B4x256 CUDA admission probe for E477; no full fits."""

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
E474_DIR = ROOT / "research/campaign_2026-10-05/activation_placement"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(E470_DIR))
sys.path.insert(0, str(E474_DIR))
sys.path.insert(0, str(HERE))

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
from fit_activation_head_mixture import (  # noqa: E402
    ACTIVATIONS,
    DROPOUTS,
    E470_SOURCE_SHA256,
    E474_SOURCE_SHA256,
    EXPECTED_PARAMETERS,
    SEEDS,
    arm_name,
    check_parent_sources,
    initialize_arm,
    make_initial_states,
)

from wnn.resources import configure_torch  # noqa: E402


def finite(values: list[torch.Tensor]) -> bool:
    """Return whether all supplied tensors contain finite values."""
    return all(bool(torch.isfinite(value).all().item()) for value in values)


def main() -> None:
    """Check E474 parity and one real fit-only optimizer step per E477 arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {args.output}")
    args.output.mkdir(parents=True)
    check_parent_sources()
    device = torch.device(args.device)
    if device.type == "cuda":
        configure_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    manifest, x, y, need, _, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("probe must load fit groups only")
    if x.shape != (24, 20_000, 42) or y.shape != (24, 20_000, 2):
        raise ValueError(f"unexpected fit-only data shape {x.shape}/{y.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("nonfinite fit-only probe arrays")
    order = np.arange(24)
    np.random.default_rng(SEEDS[0]).shuffle(order)
    ids = order[:4].tolist()
    groups = [nodes[index]["row_group"] for index in ids]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    xx = torch.from_numpy(x[ids, :256]).to(device=device, dtype=torch.float32)
    states = make_initial_states()
    result: dict = {
        "kind": "E477 actual-data B4x256 CUDA probe",
        "status": "running",
        "fit_only": True,
        "test_rows_loaded": False,
        "device": str(device),
        "row_groups": groups,
        "parameters_per_arm": EXPECTED_PARAMETERS,
        "precision_policy": "FP32 IEEE, autocast disabled",
        "parent_sources": {
            "e470": E470_SOURCE_SHA256,
            "e474": E474_SOURCE_SHA256,
        },
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "dataset_sha256": e474.DATASET_SHA256,
        "source_sha256": {
            name: e474.file_hash(HERE / name)
            for name in ("fit_activation_head_mixture.py", "tests.py", "gpu_probe.py")
        },
        "baseline_parity_tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "arms": [],
    }
    with e474.full_ieee_float32():
        candidate = initialize_arm("gelu", 0.0, states[SEEDS[0]]).to(device).eval()
        reference = (
            e474.initialize_arm(SEEDS[0], "gelu", "gelu", states[SEEDS[0]]).to(device).eval()
        )
        candidate_output, candidate_state = candidate(xx)
        reference_output, reference_state = reference(xx)
        torch.testing.assert_close(candidate_output, reference_output, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(candidate_state, reference_state, atol=1e-6, rtol=1e-5)
        candidate_output.square().mean().backward()
        reference_output.square().mean().backward()
        gradient_max = max(
            (left.grad - right.grad).abs().max().item()
            for (_, left), (_, right) in zip(
                candidate.named_parameters(), reference.named_parameters(), strict=True
            )
        )
        if gradient_max > 1e-5:
            raise ValueError(f"E474 baseline gradient parity failed: {gradient_max}")
        result["gelu_p0_e474_parity"] = {
            "output_max_abs": (candidate_output - reference_output).abs().max().item(),
            "state_max_abs": (candidate_state - reference_state).abs().max().item(),
            "gradient_max_abs": gradient_max,
            "status": "passed",
        }

        for seed in SEEDS:
            for activation in ACTIVATIONS:
                for dropout in DROPOUTS:
                    torch.manual_seed(seed)
                    model = initialize_arm(activation, dropout, states[seed]).to(device).train()
                    if (
                        sum(parameter.numel() for parameter in model.parameters())
                        != EXPECTED_PARAMETERS
                    ):
                        raise ValueError("probe model parameter count mismatch")
                    optimizer = e474.make_optimizer(model)
                    before = [parameter.detach().clone() for parameter in model.parameters()]
                    loss, telemetry = e474.fit_one_update(model, optimizer, x, y, need, ids, device)
                    params = list(model.parameters())
                    changed = any(
                        not torch.equal(previous, current)
                        for previous, current in zip(before, params, strict=True)
                    )
                    if not changed or not finite(params):
                        raise ValueError(f"invalid update for {seed}/{activation}/{dropout}")
                    result["arms"].append(
                        {
                            "seed": seed,
                            "arm": arm_name(activation, dropout),
                            "activation": activation,
                            "head_dropout": dropout,
                            "loss": loss,
                            "telemetry_step0": telemetry,
                            "parameters_changed": changed,
                            "parameters_finite": True,
                            "updates": 1,
                        }
                    )
    result["status"] = "passed"
    result["cuda_peak_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    result["cuda_peak_reserved_bytes"] = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    )
    (args.output / "probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
