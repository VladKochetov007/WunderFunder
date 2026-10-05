"""Representative actual-fit-data alpha/parity/update probe; never launched here."""

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

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
import fit_recurrent_activation_adaptive as e479  # noqa: E402

from wnn.resources import configure_torch  # noqa: E402


def main() -> None:
    """Run one actual B4×256 probe across the complete planned cell matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA probe requested but unavailable")
    configure_torch()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    device = torch.device("cuda")
    manifest, x, y, need, _score, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("probe loader must contain fit-role rows only")
    order = np.arange(24)
    np.random.default_rng(e479.SEEDS[0]).shuffle(order)
    ids = order[:4].tolist()
    groups = [nodes[index]["row_group"] for index in ids]
    batch_x = torch.from_numpy(x[ids, : e479.CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    batch_y = torch.from_numpy(y[ids, : e479.CHUNK_ROWS]).to(device=device, dtype=torch.float32)
    batch_need = torch.from_numpy(need[ids, : e479.CHUNK_ROWS]).to(device)
    if not (torch.isfinite(batch_x).all() and torch.isfinite(batch_y).all()):
        raise ValueError("nonfinite actual fit-only probe batch")
    states = e479.seed_states()
    output_atol, output_rtol = 2e-5, 2e-4
    grad_atol, grad_rtol = 5e-5, 5e-4
    update_atol, update_rtol = 1e-4, 1e-3
    torch.cuda.reset_peak_memory_stats(device)
    parity = {}
    with e474.full_ieee_float32():
        reference = e474.PlacementRNN("gelu", "gelu").to(device).float()
        reference.load_state_dict(states[e479.SEEDS[0]], strict=True)
        candidate = e479.initialize_arm("fixed0", states[e479.SEEDS[0]]).to(device).float()
        x_ref = batch_x.detach().clone().requires_grad_(True)
        x_new = batch_x.detach().clone().requires_grad_(True)
        ref_output, ref_state = reference(x_ref)
        new_output, new_state = candidate(x_new)
        torch.testing.assert_close(new_output, ref_output, atol=output_atol, rtol=output_rtol)
        torch.testing.assert_close(new_state, ref_state, atol=output_atol, rtol=output_rtol)
        (ref_output.square().mean() + ref_state.square().mean()).backward()
        (new_output.square().mean() + new_state.square().mean()).backward()
        torch.testing.assert_close(x_new.grad, x_ref.grad, atol=grad_atol, rtol=grad_rtol)
        ref_parameters = dict(reference.named_parameters())
        new_parameters = dict(candidate.named_parameters())
        for name, parameter in ref_parameters.items():
            torch.testing.assert_close(
                new_parameters[name].grad, parameter.grad, atol=grad_atol, rtol=grad_rtol
            )
        ref_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=e479.LR, weight_decay=e479.WEIGHT_DECAY, betas=e479.BETAS
        )
        new_optimizer = e479.make_optimizer(candidate, "adamw")
        torch.nn.utils.clip_grad_norm_(
            reference.parameters(), e479.CLIP_NORM, error_if_nonfinite=True
        )
        torch.nn.utils.clip_grad_norm_(
            candidate.parameters(), e479.CLIP_NORM, error_if_nonfinite=True
        )
        ref_optimizer.step()
        new_optimizer.step()
        updated_candidate = dict(candidate.named_parameters())
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(
                updated_candidate[name], parameter, atol=update_atol, rtol=update_rtol
            )
        parity["alpha_zero_outputs_states_gradients_and_clipped_adamw_update"] = "passed"

        # Equal-start learnable/fixed pairs must be functionally identical.
        for fixed_name, learned_name in (("fixed0", "learned0"), ("fixed01", "learned01")):
            fixed = e479.initialize_arm(fixed_name, states[e479.SEEDS[0]]).to(device).float()
            learned = e479.initialize_arm(learned_name, states[e479.SEEDS[0]]).to(device).float()
            fixed_output, fixed_state = fixed(batch_x)
            learned_output, learned_state = learned(batch_x)
            torch.testing.assert_close(
                learned_output, fixed_output, atol=output_atol, rtol=output_rtol
            )
            torch.testing.assert_close(
                learned_state, fixed_state, atol=output_atol, rtol=output_rtol
            )
        parity["same_initial_alpha_pairs"] = "passed"

        update_receipts = []
        for seed, optimizer_name, condition in e479.ARMS:
            model = e479.initialize_arm(condition, states[seed]).to(device).float()
            optimizer = e479.make_optimizer(model, optimizer_name)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            optimizer.zero_grad(set_to_none=True)
            prediction, state = model(batch_x)
            if not (torch.isfinite(prediction).all() and torch.isfinite(state).all()):
                raise ValueError(f"nonfinite GPU forward: {seed}/{optimizer_name}/{condition}")
            loss = e470.loss_for_batch(prediction, batch_y, batch_need)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite GPU loss: {seed}/{optimizer_name}/{condition}")
            loss.backward()
            alpha_gradient = None
            alpha_before = None
            if model.raw_alpha is not None:
                alpha_gradient = model.raw_alpha.grad.detach().clone()
                alpha_before = model.raw_alpha.detach().clone()
                if not torch.isfinite(alpha_gradient).all() or not (alpha_gradient.abs() > 0).any():
                    raise ValueError(
                        f"alpha gradient missing/zero: {seed}/{optimizer_name}/{condition}"
                    )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), e479.CLIP_NORM, error_if_nonfinite=True
            )
            optimizer.step()
            e470.ensure_finite_parameters(model, f"GPU probe {seed}/{optimizer_name}/{condition}")
            changed = any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters(), strict=True)
            )
            if not changed:
                raise ValueError(f"no parameter change: {seed}/{optimizer_name}/{condition}")
            alpha_delta = None
            if model.raw_alpha is not None:
                alpha_delta = (model.raw_alpha.detach() - alpha_before).cpu().tolist()
                if not (torch.tensor(alpha_delta).abs() > 0).any():
                    raise ValueError(f"alpha did not update: {seed}/{optimizer_name}/{condition}")
            update_receipts.append(
                {
                    "seed": seed,
                    "optimizer": optimizer_name,
                    "condition": condition,
                    "loss": float(loss.detach().cpu()),
                    "state_abs_max": float(state.detach().abs().max().cpu()),
                    "alpha_gradient": None
                    if alpha_gradient is None
                    else alpha_gradient.cpu().tolist(),
                    "alpha_update": alpha_delta,
                    "parameters_changed": changed,
                }
            )
    if len(update_receipts) != len(e479.ARMS):
        raise ValueError("GPU probe did not cover all 16 cells")
    result = {
        "kind": "E479 actual-fit-data CUDA shape probe",
        "fit_only": True,
        "test_rows_loaded": False,
        "device": torch.cuda.get_device_name(device),
        "cohort_sha256": e474.file_hash(e470.COHORT_PATH),
        "dataset_sha256": e479.DATASET_SHA256,
        "fit_row_groups": groups,
        "batch": 4,
        "chunk_rows": e479.CHUNK_ROWS,
        "tolerances": {
            "output_state": {"atol": output_atol, "rtol": output_rtol},
            "input_parameter_gradients": {"atol": grad_atol, "rtol": grad_rtol},
            "post_update": {"atol": update_atol, "rtol": update_rtol},
        },
        "parity": parity,
        "updates": update_receipts,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "fit_only_manifest_roles": sorted({node["role"] for node in nodes}),
    }
    (args.output / "probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
