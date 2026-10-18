"""Probe synthetic episode shapes and second-order autograd on an assigned GPU."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import torch

from wnn.data import N_FEATURES
from wnn.features import N_EVENT_FEATURES
from wnn.nn import RNNModel

from .fast_adapter import CausalFastAdapter, adapted_forward


def build_parser() -> argparse.ArgumentParser:
    """Build the synthetic profile CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--initial-support-rows", type=int, default=99)
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--tbptt-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--device", default="cuda")
    return parser


def profile(args: argparse.Namespace) -> dict[str, object]:
    """Run a synthetic shape probe, not a task-data or quality evaluation."""
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("run this profile on the assigned CUDA device")
    if min(args.hidden, args.layers, args.rank, args.batch_size) < 1:
        raise ValueError("model and batch dimensions must be positive")
    query_rows = args.tbptt_rows - args.initial_support_rows
    if args.initial_support_rows < 2 or args.block_rows < 2 or query_rows < 1:
        raise ValueError("support/query rows and block size must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    input_width = N_FEATURES + N_EVENT_FEATURES
    model = RNNModel(input_width, hidden=args.hidden, layers=args.layers).to(device)
    adapter = CausalFastAdapter(
        args.hidden, feature_size=N_FEATURES, rank=args.rank
    ).to(device)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *adapter.parameters()], lr=1e-4, weight_decay=0.01
    )

    total_rows = args.tbptt_rows
    # The profile checks tensor shapes and the second-order graph only. Inputs
    # and task labels are synthetic; task labels are intentionally independent
    # of the model inputs and cannot say anything about predictive quality.
    standardized_raw = torch.randn(
        args.batch_size, total_rows, N_FEATURES, device=device
    )
    stream_inputs = torch.randn(args.batch_size, total_rows, input_width, device=device)
    stream_inputs[..., :N_FEATURES] = standardized_raw
    query_targets = torch.randn(args.batch_size, query_rows, 2, device=device)
    recurrent_state = torch.zeros(
        args.layers, args.batch_size, args.hidden, device=device
    )
    query_blocks = [
        (start, min(start + args.block_rows, query_rows))
        for start in range(0, query_rows, args.block_rows)
    ]

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    support_inputs = stream_inputs[:, : args.initial_support_rows]
    support_encoded = model.encode((support_inputs - model.mean) / model.std)
    support_recurrent, recurrent_state = model.gru(support_encoded, recurrent_state)
    fast_state, diagnostics = adapter.update(
        support_recurrent,
        standardized_raw[:, : args.initial_support_rows],
        adapter.initial_state(args.batch_size, differentiable=True),
        create_graph=True,
    )
    query_loss_numerators: list[torch.Tensor] = []
    query_loss_denominators: list[torch.Tensor] = []
    update_diagnostics: list[dict[str, torch.Tensor]] = [diagnostics]
    for block_index, (query_start, query_stop) in enumerate(query_blocks):
        start = args.initial_support_rows + query_start
        stop = args.initial_support_rows + query_stop
        predictions, recurrent_state, query_recurrent = adapted_forward(
            model,
            adapter,
            stream_inputs[:, start:stop],
            recurrent_state,
            fast_state,
        )
        labels = query_targets[:, query_start:query_stop].clamp(-2.0, 2.0)
        weights = labels.abs()
        query_loss_numerators.append(
            (weights * (predictions[..., :2].float() - labels).square()).sum()
        )
        query_loss_denominators.append(weights.sum())
        if block_index + 1 < len(query_blocks):
            fast_state, diagnostics = adapter.update(
                query_recurrent,
                standardized_raw[:, start:stop],
                fast_state,
                create_graph=True,
            )
            update_diagnostics.append(diagnostics)
    query_loss = torch.stack(query_loss_numerators).sum() / torch.stack(
        query_loss_denominators
    ).sum().clamp_min(1e-8)
    query_loss.backward()
    task_gradient_norm = torch.nn.utils.clip_grad_norm_(
        [*model.parameters(), *adapter.parameters()], max_norm=1.0
    )
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in [*model.parameters(), *adapter.parameters()]
    )
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    properties = torch.cuda.get_device_properties(device)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    host_peak_rss_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    mean_diagnostics = {
        key: torch.stack([item[key] for item in update_diagnostics]).mean()
        for key in update_diagnostics[0]
    }
    return {
        "profile_kind": "synthetic_shape_and_second_order_autograd_probe_only",
        "profile_scope": (
            "random_inputs_and_labels; task_labels_are_independent_of_inputs; "
            "no_data_or_quality_evidence"
        ),
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "hidden": args.hidden,
        "layers": args.layers,
        "rank": args.rank,
        "batch_size": args.batch_size,
        "initial_support_rows": args.initial_support_rows,
        "tbptt_rows": args.tbptt_rows,
        "query_rows": query_rows,
        "query_block_rows": args.block_rows,
        "query_block_lengths": [stop - start for start, stop in query_blocks],
        "meta_unroll_query_blocks": len(query_blocks),
        "model_input_width": input_width,
        "raw_feature_width": N_FEATURES,
        "elapsed_seconds": elapsed,
        "host_peak_rss_bytes": host_peak_rss_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "query_loss": float(query_loss.detach().cpu()),
        "fast_update_count": len(update_diagnostics),
        "mean_input_prediction_loss_over_updates": float(
            mean_diagnostics["input_prediction_loss"].cpu()
        ),
        "mean_fast_gradient_norm_preclip_over_updates": float(
            mean_diagnostics["fast_gradient_norm_preclip"].cpu()
        ),
        "mean_fast_gradient_clip_fraction_over_updates": float(
            mean_diagnostics["fast_gradient_clip_fraction"].cpu()
        ),
        "task_gradient_norm_preclip": float(task_gradient_norm.detach().cpu()),
        "inner_learning_rate": float(mean_diagnostics["inner_learning_rate"].cpu()),
        "finite_gradients": finite_gradients,
        "precision": "strict_fp32_tf32_off_autocast_off",
    }


def main() -> None:
    """Execute the profile and print a machine-readable receipt to stdout."""
    args = build_parser().parse_args()
    print(json.dumps(profile(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
