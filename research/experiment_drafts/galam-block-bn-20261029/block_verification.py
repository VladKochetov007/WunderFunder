from __future__ import annotations

import copy
from dataclasses import replace
from unittest.mock import Mock

import torch
from block_comparison import GRAD_ATOL, GRAD_RTOL, compare, compare_modules, pair
from torch import nn

from wnn.galam.config import GALAMConfig
from wnn.galam.mixer import PerTimeBatchNorm1d


def check_chunks(
    config: GALAMConfig, device: torch.device, sizes: list[int], batch: int, training: bool
) -> dict[str, object]:
    """Compare full outputs/state, input/parameter gradients, buffers and AdamW updates."""
    step, block = pair(config, device)
    step.train(training)
    block.train(training)
    states = [model.reset_state(batch, device, torch.float32) for model in (step, block)]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-4) for model in (step, block)]
    maxima = {"output": 0.0, "state": 0.0, "input_gradient": 0.0, "gradient": 0.0, "buffer": 0.0}
    offset = 0
    for size in sizes:
        x = torch.randn(batch, size, config.input_features, device=device)
        copies = [x.detach().clone().requires_grad_(training) for _ in range(2)]
        timestamps = None
        if config.timestamp_source == "provided":
            timestamps = (torch.arange(size, device=device).float() + offset).square()[
                None
            ] + torch.arange(batch, device=device).float()[:, None] * 0.37
        predictions = []
        target = torch.randn(batch, size, config.output_features, device=device)
        with torch.set_grad_enabled(training):
            for index, model in enumerate((step, block)):
                predictions_i, states[index] = model.forward_chunk(
                    copies[index], states[index], timestamps
                )
                if training and size:
                    (predictions_i - target).square().mean().backward()
                predictions.append(predictions_i.detach())
                # Release each autograd graph before constructing the other arm.
                # The acceptance batch must fit on one-card L4 receivers too.
                states[index] = states[index].detach()
            maxima["output"] = max(maxima["output"], compare(*predictions, "output"))
            maxima["state"] = max(maxima["state"], compare(*states, "state"))
            if training and size:
                maxima["gradient"] = max(
                    maxima["gradient"], compare_modules(step, block, gradients=True)
                )
                if copies[0].grad is None or copies[1].grad is None:
                    raise AssertionError("Input gradient missing")
                maxima["input_gradient"] = max(
                    maxima["input_gradient"],
                    compare(
                        copies[0].grad,
                        copies[1].grad,
                        "input_gradient",
                        atol=GRAD_ATOL,
                        rtol=GRAD_RTOL,
                    ),
                )
                for model, optimizer in zip((step, block), optimizers, strict=True):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for name, value in step.named_parameters():
                    compare(value, dict(block.named_parameters())[name], f"updated.{name}")
                compare(optimizers[0].state_dict(), optimizers[1].state_dict(), "adamw_state")
        maxima["buffer"] = max(maxima["buffer"], compare_modules(step, block, gradients=False))
        states = [state.detach() for state in states]
        offset += size
    # A strict reload of all saved weights/buffers plus detached explicit streaming state.
    fresh_step, fresh_block = pair(config, device)
    fresh_step.load_state_dict(step.state_dict(), strict=True)
    fresh_block.load_state_dict(step.state_dict(), strict=True)
    fresh_step.eval()
    fresh_block.eval()
    tail = torch.randn(batch, 3, config.input_features, device=device)
    times = None
    if config.timestamp_source == "provided":
        times = (torch.arange(3, device=device).float() + offset).square().expand(batch, -1)
    with torch.no_grad():
        out_a, state_a = fresh_step.forward_chunk(tail, states[0], times)
        out_b, state_b = fresh_block.forward_chunk(tail, states[0], times)
        compare(out_a, out_b, "reload.output")
        compare(state_a, state_b, "reload.state")
    return {"training": training, "batch": batch, "chunks": sizes, "max_absolute_errors": maxima}


def check_causality(config: GALAMConfig, device: torch.device, training: bool) -> None:
    """Change only future input rows and require equal predictions in the prefix."""
    _, model = pair(config, device)
    model.train(training)
    alternative = copy.deepcopy(model)
    x = torch.randn(4, 19, config.input_features, device=device)
    changed = x.clone()
    changed[:, 7:] = 10 * torch.randn_like(changed[:, 7:])
    with torch.no_grad():
        first = model(x)
        other = alternative(changed)
        compare(first[:, :7], other[:, :7], "causal_prefix", atol=1e-6, rtol=1e-5)
    model.eval()
    with torch.no_grad():
        compare(model(x), model(x), "fresh_sequence_reset", atol=0.0, rtol=0.0)


def check_batch_norm(device: torch.device) -> int:
    """Cover per-time normalization, unusual momentum, no running stats and batch-one eval."""
    count = 0
    for momentum in (None, 0.0, 0.1, 1.0):
        for tracking in (False, True):
            for training in (False, True):
                step = PerTimeBatchNorm1d(7).to(device)
                step.norm = nn.BatchNorm1d(7, momentum=momentum, track_running_stats=tracking).to(
                    device
                )
                if tracking:
                    step.norm.running_mean.fill_(0.2)
                    step.norm.running_var.fill_(1.7)
                    step.norm.num_batches_tracked.fill_(11)
                block = copy.deepcopy(step)
                step.train(training)
                block.train(training)
                batch = 1 if not training and tracking else 4
                x = torch.randn(batch, 13, 7, device=device)
                a = x.detach().clone().requires_grad_()
                b = x.detach().clone().requires_grad_()
                left = torch.stack([step(a[:, index]) for index in range(x.shape[1])], dim=1)
                right = block(b)
                compare(left, right, "bn.output")
                weights = torch.randn_like(left)
                (left * weights).sum().backward()
                (right * weights).sum().backward()
                compare(a.grad, b.grad, "bn.input_gradient", atol=GRAD_ATOL, rtol=GRAD_RTOL)
                compare_modules(step, block, gradients=True)
                compare_modules(step, block, gradients=False)
                count += 1
    return count


def check_observers(config: GALAMConfig, device: torch.device) -> None:
    """Preserve all legacy per-step attention diagnostics when probes are attached."""
    step, block = pair(config, device)
    observers = []
    for model in (step, block):
        model.eval()
        spies = []
        for attention in (*model.local_attention, *model.global_attention):
            spy = Mock()
            attention._monitor_observer = spy
            spies.append(spy)
        observers.append(spies)
    x = torch.randn(4, 8, config.input_features, device=device)
    with torch.no_grad():
        compare(step(x), block(x), "diagnostic.output")
    for left, right in zip(*observers, strict=True):
        if left.call_count != 8 or right.call_count != 8:
            raise AssertionError("Diagnostic observer must retain one call per step")
        for a, b in zip(left.call_args_list, right.call_args_list, strict=True):
            compare(a.args, b.args, "diagnostic.arguments")


def run_checks(
    config: GALAMConfig, device: torch.device, batch_size: int, chunk_rows: int
) -> dict[str, object]:
    """Run bounded receiver-only engineering checks, not quality training or scoring."""
    cases = []
    for training in (False, True):
        cases.append(check_chunks(config, device, [0, 1, 7, 13, 71, 3], 4, training))
        cases.append(check_chunks(config, device, [chunk_rows, chunk_rows], batch_size, training))
        check_causality(config, device, training)
    deep = replace(
        config,
        lstm_layers=2,
        global_blocks=2,
        local_attention_blocks=2,
        attention_history=7,
        conv_kernel_size=1,
    )
    cases.append(check_chunks(deep, device, [3, 11, 2], 4, True))
    provided = replace(
        config, timestamp_source="provided", timestamp_scale=13.0, attention_history=9
    )
    cases.append(check_chunks(provided, device, [2, 13, 3], 4, True))
    cases.append(check_chunks(config, device, [1, 5, 3], 1, False))
    check_observers(config, device)
    return {
        "cases": cases,
        "batch_norm_cases": check_batch_norm(device),
        "causality": "passed",
        "reload": "passed",
        "diagnostic_observers": "passed",
    }
