from __future__ import annotations

import torch

from wnn.spectral_cap import maybe_cap_
from wnn.training.state import TrainingRun
from wnn.training_qat import invalidate_qat_cache, project_qat_scales_
from wnn.training_views import update_ema_for_rows


def optimizer_step(run: TrainingRun) -> None:
    """Apply unscale, preclip diagnostics, optimizer, QAT, scaler, scheduler and averaging in order."""
    activation_profile_updates = run.progress.activation_profile_updates
    args = run.options.args
    device = run.options.device
    duty = run.options.duty
    dynamics = run.optimization.dynamics
    ema_model = run.optimization.ema_model
    epoch_counter = run.progress.epoch_counter
    grad_stats = run.optimization.grad_stats
    loss = run.batch.loss
    losses_trace = run.progress.losses_trace
    model = run.network.model
    opt = run.optimization.opt
    qat_cache_identity = run.batch.qat_cache_identity
    scaler = run.optimization.scaler
    sched = run.optimization.sched
    sl = run.batch.sl
    step_count = run.progress.step_count
    swa_model = run.optimization.swa_model
    swa_start = run.optimization.swa_start
    timer = run.progress.timer
    timer.mark("backward")
    scaler.unscale_(opt)
    epoch_counter.observe_unscaled(scaler, opt)
    if grad_stats is not None:
        grad_stats.observe()
    dynamics.perturb_and_clip(step_count)
    scaler.step(opt)
    maybe_cap_(model, args, step_count)
    project_qat_scales_(model)
    invalidate_qat_cache(model)
    if qat_cache_identity is not None:
        parameters = dict(model.named_parameters())
        versions_after = {name: parameters[name]._version for name in model._qat_names}
        activation_profile_updates.append(
            {
                "cache_identity_valid_before_update": qat_cache_identity["valid_before_update"],
                "latent_parameter_versions_before": {
                    name: int(qat_cache_identity["cache"][name][1]) for name in model._qat_names
                },
                "latent_parameter_versions_after": versions_after,
                "latent_versions_advanced": any(
                    versions_after[name] != qat_cache_identity["cache"][name][1]
                    for name in model._qat_names
                ),
                "weight_cache_cleared_after_optimizer_step": not model._qat_cache,
            }
        )
    scaler.update()
    sched.step()
    timer.mark("optimizer")
    if args.parity_dump:
        losses_trace.append(loss.detach().float())
    step_count += 1
    if swa_model is not None and step_count >= swa_start:
        swa_model.update_parameters(model)
        invalidate_qat_cache(swa_model.module)
    if ema_model is not None:
        if args.origin_view != "off":
            update_ema_for_rows(
                ema_model, model, args.ema_decay, sl.stop - sl.start, args.ema_reference_rows
            )
        else:
            ema_model.update_parameters(model)
        invalidate_qat_cache(ema_model.module)
        epoch_counter.ema_updates += 1
    if duty is not None:
        torch.cuda.synchronize()
        duty.tick()
    elif not args.sync_free and device.type == "cuda":
        torch.cuda.synchronize()
    run.progress.step_count = step_count
