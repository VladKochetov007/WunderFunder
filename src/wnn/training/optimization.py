from __future__ import annotations

import json

import torch

from wnn.nn import (
    PooledCorrMemory,
    weighted_corr_loss,
    weighted_huber_loss,
    weighted_mse_loss,
    weighted_pooled_corr_loss,
)
from wnn.optim import GradStats
from wnn.training.averaging import FastEMA
from wnn.training.calibration import calibrate_distill_reference
from wnn.training.reporting import emit
from wnn.training.state import TrainingRun
from wnn.training_dynamics import Dynamics, build_scheduler


def build_optimization(run: TrainingRun) -> bool:
    """Construct optimizer, scheduler, objective and averaging after optional preflight exits. Returns false when preflight/calibration completed without fitting."""
    environment = run.environment
    amp_dtype = run.optimization.amp_dtype
    args = run.options.args
    campaign_split = run.data.campaign_split
    device = run.options.device
    dynamics = run.optimization.dynamics
    ema_model = run.optimization.ema_model
    ema_update_decay = run.options.ema_update_decay
    frontier_cache_binding = run.data.frontier_cache_binding
    frontier_contract = run.data.frontier_contract
    frontier_contract_sha256 = run.options.frontier_contract_sha256
    frontier_order = run.data.frontier_order
    grad_stats = run.optimization.grad_stats
    model = run.network.model
    objective = run.optimization.objective
    opt = run.optimization.opt
    pool_memory = run.optimization.pool_memory
    prop_model = run.data.prop_model
    provenance = run.data.provenance
    scaler = run.optimization.scaler
    sched = run.optimization.sched
    step_count = run.progress.step_count
    swa_model = run.optimization.swa_model
    swa_start = run.optimization.swa_start
    total = run.network.total
    train_loader = run.data.train_loader
    use_autocast = run.optimization.use_autocast
    if args.student_frontier_preflight_only:
        receipt = {
            "format": "wnn.student-frontier-trainer-preflight",
            "version": 1,
            "contract_id": frontier_contract["id"],
            "contract_sha256": frontier_contract_sha256,
            "stage": args.student_frontier_stage,
            "shape": [args.hidden, args.layers],
            "split_manifest_sha256": campaign_split.sha256,
            "cache_binding": frontier_cache_binding,
            "sample_order": frontier_order,
            "normalization_identity": provenance["normalization_identity"],
            "normalization_source": provenance["student_frontier"]["normalization_source"],
            "batch_size": args.batch,
            "batches_per_epoch": len(train_loader),
            "updates_per_epoch": total,
            "optimizer_constructed": False,
            "fit_started": False,
            "quality_claim": None,
        }
        emit("STUDENT_FRONTIER_PREFLIGHT", json.dumps(receipt, sort_keys=True), flush=True)
        run.optimization.amp_dtype = amp_dtype
        run.optimization.dynamics = dynamics
        run.optimization.ema_model = ema_model
        run.optimization.grad_stats = grad_stats
        run.optimization.objective = objective
        run.optimization.opt = opt
        run.optimization.pool_memory = pool_memory
        run.optimization.scaler = scaler
        run.optimization.sched = sched
        run.progress.step_count = step_count
        run.optimization.swa_model = swa_model
        run.optimization.swa_start = swa_start
        run.optimization.use_autocast = use_autocast
        return False
    if args.calibrate_distill_reference:
        calibrate_distill_reference(
            args, train_loader, prop_model, provenance, device, total, environment=environment
        )
        run.optimization.amp_dtype = amp_dtype
        run.optimization.dynamics = dynamics
        run.optimization.ema_model = ema_model
        run.optimization.grad_stats = grad_stats
        run.optimization.objective = objective
        run.optimization.opt = opt
        run.optimization.pool_memory = pool_memory
        run.optimization.scaler = scaler
        run.optimization.sched = sched
        run.progress.step_count = step_count
        run.optimization.swa_model = swa_model
        run.optimization.swa_start = swa_start
        run.optimization.use_autocast = use_autocast
        return False
    opt = environment.build_optimizer(model, args)
    sched = build_scheduler(opt, args, total)
    dynamics = Dynamics(model, args, total)
    provenance["optimizer_switch_events"] = dynamics.events
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and args.precision != "fp32"
    scaler = torch.amp.GradScaler(
        device.type, enabled=args.precision == "fp16", init_scale=args.amp_init_scale
    )
    grad_stats = GradStats(model, args.clip) if args.grad_stats else None
    swa_model = torch.optim.swa_utils.AveragedModel(model) if args.swa_frac > 0 else None
    ema_model = None
    if args.ema_teacher > 0 or args.eval_ema or args.td_alpha > 0:
        if args.sync_free:
            ema_model = FastEMA(model, ema_update_decay)
        else:
            ema_model = torch.optim.swa_utils.AveragedModel(
                model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_update_decay)
            )
            ema_model.eval()
    swa_start = int(total * (1.0 - args.swa_frac))
    step_count = 0
    pool_memory = PooledCorrMemory(args.pool_decay) if args.loss == "gcorr_pool" else None

    def objective(o: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Objective."""
        if pool_memory is not None:
            return pool_memory(o, t, v)
        if args.loss == "corr":
            return weighted_corr_loss(o, t, v)
        if args.loss == "wmse":
            return weighted_mse_loss(o, t, v)
        if args.loss == "mse":
            yv = t.float().clamp(-2, 2)
            vv = v[..., None].float()
            return (vv * (o.float() - yv) ** 2).sum() / (vv.sum() * yv.shape[-1]).clamp_min(1e-08)
        if args.loss == "huber":
            return weighted_huber_loss(o, t, v, args.huber_delta)
        if args.loss == "huber_curvature":
            return 2.0 * weighted_huber_loss(o, t, v, args.huber_delta)
        if args.loss == "gcorr":
            return weighted_pooled_corr_loss(o, t, v)
        if args.loss == "wmse_gcorr":
            return weighted_mse_loss(o, t, v) + args.gcorr_weight * weighted_pooled_corr_loss(
                o, t, v
            )
        return weighted_corr_loss(o, t, v) + 0.5 * weighted_mse_loss(o, t, v)

    run.optimization.amp_dtype = amp_dtype
    run.optimization.dynamics = dynamics
    run.optimization.ema_model = ema_model
    run.optimization.grad_stats = grad_stats
    run.optimization.objective = objective
    run.optimization.opt = opt
    run.optimization.pool_memory = pool_memory
    run.optimization.scaler = scaler
    run.optimization.sched = sched
    run.progress.step_count = step_count
    run.optimization.swa_model = swa_model
    run.optimization.swa_start = swa_start
    run.optimization.use_autocast = use_autocast
    return True
