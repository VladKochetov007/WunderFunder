"""Callable E489 CUDA probe: three-seed native parity and six timed cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "research/campaign_2026-10-05/activation_optimizer"),
    str(ROOT / "research/campaign_2026-10-05/activation_placement"),
    str(ROOT / "research/campaign_2026-10-05/activation_duration"),
    str(ROOT / "research/campaign_2026-10-05/muon_orthogonalization_precision"),
    str(ROOT / "src"),
]

import fit_activation_optimizer as e470  # noqa: E402
import fit_activation_placement as e474  # noqa: E402
import fit_fp32_duration as actor  # noqa: E402
import precision_replay as e487  # noqa: E402

from wnn.nn import RNNModel  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402

OUTPUT_ATOL, OUTPUT_RTOL = 2e-5, 2e-4
GRAD_ATOL, GRAD_RTOL = 5e-5, 5e-4
DELTA_RELATIVE_LIMIT, DELTA_MAX_ABS = 0.02, 2e-5
ALLOCATOR_CAP, PHYSICAL_STOP = 2 * 1024**3, 3_000_000_000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash named initialization tensors in a stable, unambiguous order."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def memory_snapshot() -> dict[str, int]:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.max_memory_allocated()
    observed = total - free
    if allocated > ALLOCATOR_CAP or observed > PHYSICAL_STOP:
        raise RuntimeError(f"E489 memory stop exceeded: allocator={allocated}, device={observed}")
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "peak_allocated_bytes": int(allocated),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "observed_device_bytes": int(observed),
        "total_device_bytes": int(total),
    }


def check_initial_bytes(model: torch.nn.Module, initial: dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    if state.keys() != initial.keys():
        raise ValueError("E489 GPU model state keys differ from common initialization")
    for name, value in state.items():
        first, second = value.detach().cpu().contiguous(), initial[name].detach().cpu().contiguous()
        if (
            first.dtype != second.dtype
            or first.shape != second.shape
            or first.numpy().tobytes() != second.numpy().tobytes()
        ):
            raise ValueError(f"E489 device transfer changed initialization bytes: {name}")


def grouped_deltas(
    model_before: dict[str, torch.Tensor], model_after: torch.nn.Module
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    parameters = dict(model_after.named_parameters())
    if set(parameters) != set(model_before):
        raise ValueError("E489 delta parameter keys differ")
    deltas = {
        name: parameter.detach() - model_before[name].to(parameter.device)
        for name, parameter in parameters.items()
    }
    names = {"matrix:" + name: [name] for name in sorted(actor.MUON_NAMES)}
    names["adamw"] = sorted(set(parameters) - actor.MUON_NAMES)
    covered = {name for group in names.values() for name in group}
    if covered != set(parameters) or sum(map(len, names.values())) != len(parameters):
        raise ValueError("E489 update groups do not exactly cover all parameters")
    return deltas, names


def compare_update_deltas(
    native_before: dict[str, torch.Tensor],
    native_after: torch.nn.Module,
    custom_before: dict[str, torch.Tensor],
    custom_after: torch.nn.Module,
) -> dict[str, Any]:
    n_delta, groups = grouped_deltas(native_before, native_after)
    c_delta, custom_groups = grouped_deltas(custom_before, custom_after)
    if groups != custom_groups:
        raise ValueError("E489 native/custom optimizer update groups differ")
    report: dict[str, Any] = {}
    for group_name, names in groups.items():
        ref = torch.cat([n_delta[name].detach().float().reshape(-1) for name in names])
        err = torch.cat(
            [(c_delta[name] - n_delta[name]).detach().float().reshape(-1) for name in names]
        )
        reference_rms = float(ref.square().mean().sqrt().cpu())
        error_rms = float(err.square().mean().sqrt().cpu())
        max_abs = max(
            float((c_delta[name] - n_delta[name]).detach().abs().max().cpu()) for name in names
        )
        relative = error_rms / max(reference_rms, 1e-10)
        passed = relative <= DELTA_RELATIVE_LIMIT and max_abs <= DELTA_MAX_ABS
        report[group_name] = {
            "reference_update_rms": reference_rms,
            "error_update_rms": error_rms,
            "relative_error_rms": relative,
            "max_abs_error": max_abs,
            "relative_limit": DELTA_RELATIVE_LIMIT,
            "max_abs_limit": DELTA_MAX_ABS,
            "passed": passed,
        }
        if not passed:
            raise ValueError(
                f"E489 strict update delta gate failed: {group_name}: {report[group_name]}"
            )
    return report


def native_parity(
    seed: int,
    initial: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    need: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    native = (
        RNNModel(actor.FEATURES, hidden=actor.HIDDEN, layers=2, dropout=0.0, n_out=2)
        .to(device)
        .float()
        .train()
    )
    native.load_state_dict(initial, strict=True)
    candidate = actor.initialize_arm(0.0, initial).to(device).float().train()
    check_initial_bytes(native, initial)
    check_initial_bytes(candidate, initial)
    nx = x.detach().clone().requires_grad_(True)
    cx = x.detach().clone().requires_grad_(True)
    native_prediction, native_state = native(nx)
    candidate_prediction, candidate_state, candidate_hidden = candidate.forward_details(cx)
    if not torch.isfinite(candidate_hidden).all():
        raise ValueError("E489 alpha-zero candidate hidden output is nonfinite")
    torch.testing.assert_close(
        candidate_prediction, native_prediction, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL
    )
    torch.testing.assert_close(candidate_state, native_state, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL)
    native_loss = e470.loss_for_batch(native_prediction, y, need)
    candidate_loss = e470.loss_for_batch(candidate_prediction, y, need)
    native_loss.backward()
    candidate_loss.backward()
    torch.testing.assert_close(cx.grad, nx.grad, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    native_parameters = dict(native.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if native_parameters.keys() != candidate_parameters.keys():
        raise ValueError("E489 native and custom parameter keys differ")
    for name in native_parameters:
        torch.testing.assert_close(
            candidate_parameters[name].grad,
            native_parameters[name].grad,
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
            msg=f"E489 gradient mismatch {name}",
        )
    native_before = {name: p.detach().clone() for name, p in native_parameters.items()}
    candidate_before = {name: p.detach().clone() for name, p in candidate_parameters.items()}
    native_optimizer = e487.make_optimizer(native, "fp32_diagnostic")
    candidate_optimizer = e487.make_optimizer(candidate, "fp32_diagnostic")
    native_norm = torch.nn.utils.clip_grad_norm_(
        native.parameters(), actor.CLIP_NORM, error_if_nonfinite=True
    )
    candidate_norm = torch.nn.utils.clip_grad_norm_(
        candidate.parameters(), actor.CLIP_NORM, error_if_nonfinite=True
    )
    if native_norm <= actor.CLIP_NORM or candidate_norm <= actor.CLIP_NORM:
        raise ValueError("E489 native alpha-zero parity update must activate clipping")
    native_optimizer.step()
    candidate_optimizer.step()
    e470.ensure_finite_parameters(native, f"parity/native/{seed}")
    e470.ensure_finite_parameters(candidate, f"parity/candidate/{seed}")
    e487.require_optimizer_state_finite(native_optimizer, f"parity/native/{seed}")
    e487.require_optimizer_state_finite(candidate_optimizer, f"parity/candidate/{seed}")
    delta_report = compare_update_deltas(native_before, native, candidate_before, candidate)
    return {
        "seed": seed,
        "output_max_abs_error": float((candidate_prediction - native_prediction).abs().max().cpu()),
        "state_max_abs_error": float((candidate_state - native_state).abs().max().cpu()),
        "all_input_and_parameter_gradients_within_tolerance": True,
        "clipping_active": True,
        "native_preclip_norm": float(native_norm.detach().cpu()),
        "candidate_preclip_norm": float(candidate_norm.detach().cpu()),
        "strict_update_delta_gates": delta_report,
    }


def timed_cell(
    seed: int,
    alpha: float,
    initial: dict[str, torch.Tensor],
    x: np.ndarray,
    y: np.ndarray,
    need: np.ndarray,
    ids: list[int],
    device: torch.device,
) -> dict[str, Any]:
    model = actor.initialize_arm(alpha, initial).to(device).float().train()
    check_initial_bytes(model, initial)
    optimizer = actor.make_optimizer(model)
    times = []
    loss_values = []
    state = None
    for chunk_index in range(2):
        row_start = chunk_index * actor.CHUNK_ROWS
        torch.cuda.synchronize()
        started = time.perf_counter()
        xx = torch.from_numpy(x[ids, row_start : row_start + actor.CHUNK_ROWS]).to(
            device=device, dtype=torch.float32
        )
        yy = torch.from_numpy(y[ids, row_start : row_start + actor.CHUNK_ROWS]).to(
            device=device, dtype=torch.float32
        )
        mm = torch.from_numpy(need[ids, row_start : row_start + actor.CHUNK_ROWS]).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction, state, candidate_hidden = model.forward_details(xx, state)
        if not (
            torch.isfinite(prediction).all()
            and torch.isfinite(state).all()
            and torch.isfinite(candidate_hidden).all()
        ):
            raise ValueError(
                f"E489 nonfinite forward for seed={seed}/alpha={alpha}/chunk={chunk_index}"
            )
        loss = e470.loss_for_batch(prediction, yy, mm)
        if not torch.isfinite(loss):
            raise ValueError("E489 probe loss is nonfinite")
        loss.backward()
        clipped, _ = actor.optimizer_update(
            model,
            optimizer,
            chunk_index,
            chunk_index // actor.UPDATES_PER_EPOCH,
            f"probe/{seed}/{alpha}/{chunk_index}",
            batch_index=0,
            chunk_start=row_start,
            alpha=alpha,
            state=state,
            prediction=prediction,
            candidate=candidate_hidden,
        )
        if not clipped:
            raise ValueError("E489 actual-shape probe chunk did not exercise clipping")
        state = state.detach()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        loss_values.append(float(loss.detach().cpu()))
    state_steps = sorted(
        {
            int(value["step"].item() if isinstance(value["step"], torch.Tensor) else value["step"])
            for value in optimizer.state.values()
            if "step" in value
        }
    )
    if state_steps != [2]:
        raise ValueError(f"E489 optimizer did not persist across two chunks: {state_steps}")
    record = {
        "seed": seed,
        "alpha": alpha,
        "update_seconds_per_chunk": times,
        "losses": loss_values,
        "optimizer_steps": state_steps,
        "final_state_shape": list(state.shape),
        "candidate_abs_max": float(candidate_hidden.detach().abs().max().cpu()),
        "prediction_abs_max": float(prediction.detach().abs().max().cpu()),
        "state_abs_max": float(state.detach().abs().max().cpu()),
        "changed_parameters": sum(
            not torch.equal(initial[name].to(p.device), p) for name, p in model.named_parameters()
        ),
        "memory": memory_snapshot(),
    }
    if record["changed_parameters"] == 0:
        raise ValueError(f"E489 actual-shape updates changed no parameters: {seed}/{alpha}")
    return record


def run_probe(output: Path, proof_path: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    probe_started = time.perf_counter()
    actor.validate_e487_saved_delta_gate(proof_path)
    source_hashes = actor.parent_source_hashes()
    os.environ["WNN_GPU_MEMORY_BYTES"] = str(2 * 1024**3)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    configure_torch()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E489 actual probe requires exactly one visible CUDA device")
    torch.set_num_threads(1)
    device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info()
    if total - free >= PHYSICAL_STOP or torch.cuda.memory_allocated() > ALLOCATOR_CAP:
        raise RuntimeError("E489 CUDA device is already at or above the registered memory stop")
    states = actor.make_initial_states()
    cohort, x, y, need, _, nodes = e470.load_cohort(include_test=False)
    if len(nodes) != 24 or any(node["role"] != "fit" for node in nodes):
        raise ValueError("E489 probe may access only the 24 fit-role groups")
    torch.manual_seed(actor.SEEDS[0])
    torch.cuda.reset_peak_memory_stats(device)
    parity_records = []
    with e474.full_ieee_float32():
        active_precision = e474.precision_metadata()
        for seed in actor.SEEDS:
            ids = actor.canonical_epoch_order(seed, 0)[: actor.BATCH_SIZE].tolist()
            xx = torch.from_numpy(x[ids, : actor.CHUNK_ROWS]).to(device=device, dtype=torch.float32)
            yy = torch.from_numpy(y[ids, : actor.CHUNK_ROWS]).to(device=device, dtype=torch.float32)
            mask = torch.from_numpy(need[ids, : actor.CHUNK_ROWS]).to(device=device)
            parity_records.append(native_parity(seed, states[seed], xx, yy, mask, device))
        parity_memory = memory_snapshot()
        cell_records = []
        for seed in actor.SEEDS:
            ids = actor.canonical_epoch_order(seed, 0)[: actor.BATCH_SIZE].tolist()
            for alpha in actor.ALPHAS:
                torch.cuda.reset_peak_memory_stats(device)
                cell_records.append(timed_cell(seed, alpha, states[seed], x, y, need, ids, device))
        overall_memory = memory_snapshot()
        if len(cell_records) != 6 or len(parity_records) != 3:
            raise ValueError("E489 probe omitted one or more registered seed/arm cells")
    output.mkdir(parents=True)
    probe_elapsed = time.perf_counter() - probe_started
    initial_hashes = {str(seed): state_sha256(states[seed]) for seed in actor.SEEDS}
    max_update_seconds = max(max(item["update_seconds_per_chunk"]) for item in cell_records)
    projected_total = max_update_seconds * 11_376 * 1.5 + 120.0 + probe_elapsed
    report = {
        "kind": "E489 three-seed FP32 Muon actual-shape probe (v2 entrypoint fix)",
        "status": "passed",
        "fit_only": True,
        "test_rows_loaded": False,
        "device_name": torch.cuda.get_device_name(device),
        "device_count_visible": torch.cuda.device_count(),
        "allocator_cap_bytes": ALLOCATOR_CAP,
        "physical_stop_bytes": PHYSICAL_STOP,
        "root_e487_saved_delta_gate_sha256": sha256_file(proof_path),
        "parent_source_sha256": source_hashes,
        "initial_state_sha256": initial_hashes,
        "fit_row_groups": [node["row_group"] for node in nodes],
        "fit_normalization": cohort["normalization"],
        "native_alpha0_parity_all_three_seeds": parity_records,
        "parity_memory": parity_memory,
        "actual_fit_cells_two_successive_chunks": cell_records,
        "overall_memory": overall_memory,
        "probe_body_seconds": probe_elapsed,
        "active_precision_inside_full_ieee_context": active_precision,
        "max_update_seconds_including_input_transfer": max_update_seconds,
        "projected_11376_updates_with_1_5_margin_plus_120_and_probe_seconds": projected_total,
        "source_sha256": {
            name: sha256_file(HERE / name)
            for name in (
                "fit_fp32_duration.py",
                "tests_fp32_duration.py",
                "gpu_probe_fp32_duration_v2.py",
                "PROVENANCE_ADMISSION_ADDENDUM.md",
            )
        },
    }
    temporary = output / "probe.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, output / "probe.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--e487-root-proof",
        type=Path,
        default=ROOT / "research/campaign_2026-10-05/checks/e487_saved_delta_reproduction.json",
    )
    args = parser.parse_args()
    print(json.dumps(run_probe(args.output, args.e487_root_proof), indent=2), flush=True)


if __name__ == "__main__":
    main()
