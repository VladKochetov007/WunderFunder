from __future__ import annotations

import json
import time

import torch

from wnn.splits import file_sha256
from wnn.training.checkpoints import (
    checkpoint_model,
    checkpoint_selected,
    inference_checkpoint_fields,
    serialized_args,
)
from wnn.training.epoch import fit_epoch_summary
from wnn.training.evaluation import evaluate
from wnn.training.reporting import emit
from wnn.training.state import TrainingRun
from wnn.training_state import atomic_save_training_state, capture_training_state


def evaluate_epoch(run: TrainingRun) -> None:
    """Grade the fixed endpoint and retain all original diagnostic and selection fields."""
    environment = run.environment
    args = run.options.args
    cumulative_updates = run.progress.cumulative_updates
    device = run.options.device
    ema_model = run.optimization.ema_model
    epoch = run.progress.epoch
    epoch_counter = run.progress.epoch_counter
    epoch_grad_stats = run.progress.epoch_grad_stats
    epoch_order_identity = run.progress.epoch_order_identity
    eval_model = run.progress.eval_model
    evaluation_score = run.progress.evaluation_score
    history = run.progress.history
    model = run.network.model
    moments = run.network.moments
    n_valid = run.data.n_valid
    provenance = run.data.provenance
    step_count = run.progress.step_count
    swa_model = run.optimization.swa_model
    swa_start = run.optimization.swa_start
    t0 = run.progress.t0
    valid_loader = run.data.valid_loader
    update_counts = epoch_counter.finish(cumulative_updates)
    update_counts["sample_order_identity"] = epoch_order_identity
    update_counts["normalization_identity"] = provenance["normalization_identity"]
    if epoch_grad_stats is not None:
        update_counts["grad_stats"] = epoch_grad_stats
    provenance["update_diagnostics"].append({"epoch_index": epoch, **update_counts})
    provenance["cumulative_updates"] = cumulative_updates.copy()
    train_seconds = round(time.time() - t0, 1)
    eval_model = checkpoint_model(
        model, ema_model, swa_model if step_count >= swa_start else None, args.eval_ema
    )
    if moments is not None:
        environment.readout.tables, environment.readout.tilted = (moments.tables(), args.ce_tilt)
    evaluation_score = None
    if args.no_eval_during_fit:
        res = {"epoch": epoch, "train_seconds": train_seconds, "evaluation": "withheld"}
    else:
        t_eval = time.time()
        capture = (
            args.save_eval_predictions
            and args.checkpoint_policy == "fixed"
            and epoch == args.checkpoint_epoch
        )
        prediction_archive = (
            run.options.out_dir / f"evaluation_epoch{epoch}.npz" if capture else None
        )
        res = evaluate(
            eval_model,
            valid_loader,
            4000,
            device,
            environment=environment,
            capture_path=prediction_archive,
            group_ids=run.data.valid_idx if capture else None,
        )
        if prediction_archive is not None:
            res["prediction_archive"] = prediction_archive.name
        res.update(
            epoch=epoch,
            train_seconds=train_seconds,
            eval_seconds=round(time.time() - t_eval, 1),
            split=f"{args.eval_source}{n_valid}",
        )
        evaluation_score = res["weighted_pearson"]
    res["update_diagnostics"] = update_counts
    res["fit_summary"] = fit_epoch_summary(run, update_counts["epoch"])
    history.append(res)
    emit("FIT" if args.no_eval_during_fit else "VALID", json.dumps(res), flush=True)
    run.progress.eval_model = eval_model
    run.progress.evaluation_score = evaluation_score


def save_epoch(run: TrainingRun) -> None:
    """Write inference snapshots, source provenance and resumable optimizer/stream state."""
    environment = run.environment
    args = run.options.args
    best = run.progress.best
    continuation = run.options.continuation
    cumulative_updates = run.progress.cumulative_updates
    dynamics = run.optimization.dynamics
    ema_model = run.optimization.ema_model
    epoch = run.progress.epoch
    eval_model = run.progress.eval_model
    evaluation_score = run.progress.evaluation_score
    fitting_started = run.progress.fitting_started
    history = run.progress.history
    model = run.network.model
    moments = run.network.moments
    n_in = run.network.n_in
    norm_ema = run.progress.norm_ema
    opt = run.optimization.opt
    out_dir = run.options.out_dir
    pool_memory = run.optimization.pool_memory
    prior_fit_seconds = run.progress.prior_fit_seconds
    provenance = run.data.provenance
    scaler = run.optimization.scaler
    sched = run.optimization.sched
    step_count = run.progress.step_count
    swa_model = run.optimization.swa_model
    t0 = run.progress.t0
    train_loader = run.data.train_loader
    training_contract = run.progress.training_contract
    representation = inference_checkpoint_fields(
        eval_model,
        "ema" if args.eval_ema else "raw" if eval_model is model else "swa",
        environment=environment,
    )
    checkpoint = {
        **representation,
        "args": serialized_args(args),
        "n_in": n_in,
        "epoch": epoch,
        "readout": None
        if environment.readout.tables is None
        else [t.cpu() for t in environment.readout.tables],
        "readout_tilted": environment.readout.tilted,
        "provenance": {
            key: value for key, value in provenance.items() if key != "checkpoint_sha256"
        },
    }
    snapshot = out_dir / f"model_ep{epoch}.pt"
    torch.save(checkpoint, snapshot)
    provenance["checkpoint_sha256"][snapshot.name] = file_sha256(snapshot)
    is_selected = checkpoint_selected(
        epoch, args.checkpoint_policy, args.checkpoint_epoch, evaluation_score, best
    )
    if evaluation_score is not None:
        best = max(best, evaluation_score)
    if is_selected or (
        args.checkpoint_policy == "best"
        and args.include_heldout_valid
        and (epoch == args.epochs - 1)
    ):
        torch.save(checkpoint, out_dir / "model.pt")
        provenance["checkpoint_sha256"]["model.pt"] = file_sha256(out_dir / "model.pt")
        provenance["selected_epoch"] = epoch
    provenance["fitting_seconds"] = round(prior_fit_seconds + time.time() - fitting_started, 1)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if continuation:
        training_state = capture_training_state(
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            loader=train_loader,
            contract=training_contract,
            next_epoch=epoch + 1,
            ema=ema_model,
            swa=swa_model,
            progress={
                "step_count": step_count,
                "best": best,
                "history": history,
                "cumulative_updates": cumulative_updates,
                "update_diagnostics": provenance["update_diagnostics"],
                "fitting_seconds": provenance["fitting_seconds"],
            },
            auxiliary={
                "norm_ema": norm_ema,
                "dynamics": dynamics.state_dict(),
                "pool_memory": None if pool_memory is None else pool_memory.memory,
                "readout_moments": None
                if moments is None
                else {field: getattr(moments, field) for field in ("m1", "m0", "n")},
            },
        )
        atomic_save_training_state(training_state, out_dir / "training_state.pt")
    t0 = time.time()
    run.progress.best = best
    run.progress.t0 = t0


def finalize(run: TrainingRun) -> None:
    """Finish artifacts, including export recovery after a fully completed resumed fit."""
    environment = run.environment
    args = run.options.args
    best = run.progress.best
    cumulative_updates = run.progress.cumulative_updates
    ema_model = run.optimization.ema_model
    eval_model = run.progress.eval_model
    history = run.progress.history
    model = run.network.model
    moments = run.network.moments
    n_in = run.network.n_in
    out_dir = run.options.out_dir
    prior_fit_seconds = run.progress.prior_fit_seconds
    provenance = run.data.provenance
    saved_training = run.data.saved_training
    start_epoch = run.progress.start_epoch
    step_count = run.progress.step_count
    swa_model = run.optimization.swa_model
    swa_start = run.optimization.swa_start
    if saved_training is not None and start_epoch == args.epochs:
        eval_model = checkpoint_model(
            model, ema_model, swa_model if step_count >= swa_start else None, args.eval_ema
        )
        provenance["selected_epoch"] = args.epochs - 1
        provenance["cumulative_updates"] = cumulative_updates.copy()
        provenance["fitting_seconds"] = prior_fit_seconds
        representation = inference_checkpoint_fields(
            eval_model,
            "ema" if args.eval_ema else "raw" if eval_model is model else "swa",
            environment=environment,
        )
        checkpoint = {
            **representation,
            "args": serialized_args(args),
            "n_in": n_in,
            "epoch": args.epochs - 1,
            "readout": None if moments is None else [value.cpu() for value in moments.tables()],
            "readout_tilted": bool(args.ce_tilt),
            "provenance": {
                key: value for key, value in provenance.items() if key != "checkpoint_sha256"
            },
        }
        for filename in (f"model_ep{args.epochs - 1}.pt", "model.pt"):
            torch.save(checkpoint, out_dir / filename)
            provenance["checkpoint_sha256"][filename] = file_sha256(out_dir / filename)
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        atomic_save_training_state(saved_training, out_dir / "training_state.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    if args.checkpoint_policy == "fixed":
        emit("SELECTED_FIXED", args.checkpoint_epoch)
    else:
        emit("BEST", best)
    if args.include_heldout_valid and (not args.no_eval_during_fit):
        emit("SELECTED_FINAL", history[-1]["weighted_pearson"])
    run.progress.eval_model = eval_model
