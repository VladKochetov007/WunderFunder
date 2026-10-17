from __future__ import annotations

import time

import torch

from wnn.training.forward import detach_state
from wnn.training.state import TrainingRun


def forward_standard(run: TrainingRun) -> None:
    """Run the student and optional EMA teacher under their original autocast policy."""
    environment = run.environment
    amp_dtype = run.optimization.amp_dtype
    args = run.options.args
    device = run.options.device
    ema_model = run.optimization.ema_model
    h = run.batch.h
    h_ema = run.batch.h_ema
    inputs = run.batch.inputs
    model = run.network.model
    out = run.batch.out
    out_ema = run.batch.out_ema
    qat_cache_identity = run.batch.qat_cache_identity
    sl = run.batch.sl
    timer = run.progress.timer
    use_autocast = run.optimization.use_autocast
    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_autocast):
        out, h = environment.forward_chunk(model, inputs, sl, h)
    if args.profile_steps and args.activation_backend != "none" and (args.qat != "none"):
        parameters = dict(model.named_parameters())
        qat_cache_identity = {
            "cache": {
                name: model._qat_cache[name][0]
                for name in model._qat_names
                if name in model._qat_cache
            }
        }
        qat_cache_identity["valid_before_update"] = all(
            qat_cache_identity["cache"].get(name)
            == (id(parameters[name]), parameters[name]._version, parameters[name].data_ptr())
            for name in model._qat_names
        )
    h = detach_state(h)
    timer.mark("forward")
    if args.ema_teacher > 0 or args.td_alpha > 0:
        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=use_autocast):
            out_ema, h_ema = environment.forward_chunk(ema_model.module, inputs, sl, h_ema)
        h_ema = detach_state(h_ema)
    run.batch.h = h
    run.batch.h_ema = h_ema
    run.batch.out = out
    run.batch.out_ema = out_ema
    run.batch.qat_cache_identity = qat_cache_identity


def backward_standard(run: TrainingRun) -> None:
    """Backpropagate the prepared objective without changing scaler or compile timing semantics."""
    device = run.options.device
    loss = run.batch.loss
    model = run.network.model
    opt = run.optimization.opt
    profile_losses = run.progress.profile_losses
    scaler = run.optimization.scaler
    timer = run.progress.timer
    if profile_losses is not None:
        profile_losses.append(loss.detach())
    timer.mark("loss")
    opt.zero_grad(set_to_none=True)
    compile_timings = getattr(model, "_activation_profile_compile_timings", None)
    backward_started = (
        time.perf_counter()
        if compile_timings is not None and compile_timings["backward"] is None
        else None
    )
    scaler.scale(loss).backward()
    if backward_started is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compile_timings["backward"] = time.perf_counter() - backward_started
