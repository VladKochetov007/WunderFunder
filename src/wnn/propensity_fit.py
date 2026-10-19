from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wnn.propensity import PropensityNet
from wnn.propensity_failure import create_only_json
from wnn.propensity_fit_checkpoint import _checkpoint_payload, _save_epoch_outputs
from wnn.propensity_fit_setup import prepare_strict_fit
from wnn.propensity_fit_update import _fit_update
from wnn.propensity_progress import UpdateProgress
from wnn.propensity_recovery import atomic_write_json, save_resume_state
from wnn.splits import CampaignSplit, file_sha256
from wnn.training.reporting import emit


def fit_strict_propensity(
    *,
    model: PropensityNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    build_inputs: Callable[[torch.Tensor], torch.Tensor],
    args: dict[str, object],
    provenance: dict[str, object],
    normalization: dict[str, object] | None,
    split: CampaignSplit,
    data_root: Path,
    source_root: Path,
    device: torch.device,
    warmup: int,
    chunk_rows: int,
    resume: bool,
    run_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Fit strict FP32 propensity with update telemetry and epoch recovery."""
    state = prepare_strict_fit(
        model=model,
        loader=loader,
        optimizer=optimizer,
        args=args,
        provenance=provenance,
        normalization=normalization,
        split=split,
        data_root=data_root,
        source_root=source_root,
        device=device,
        resume=resume,
        run_dir=run_dir,
    )
    attempt_start = state.cumulative.optimizer_attempts
    started = time.time()
    try:
        for epoch in range(state.next_epoch, int(args["epochs"])):
            epoch_started = time.time()
            epoch_progress = UpdateProgress()
            expected_updates = 0
            model.train()
            for batch_index, (features, _, mask) in enumerate(loader):
                inputs = build_inputs(features.to(device))
                target = mask.to(device).float()
                hidden = None
                for chunk_start in range(0, inputs.shape[1], chunk_rows):
                    stop = min(chunk_start + chunk_rows, inputs.shape[1])
                    hidden = _fit_update(
                        model=model,
                        optimizer=optimizer,
                        inputs=inputs[:, chunk_start:stop],
                        target=target[:, chunk_start:stop],
                        hidden=hidden,
                        chunk_start=chunk_start,
                        warmup=warmup,
                        epoch=epoch,
                        batch=batch_index,
                        attempt_id=state.attempt_id,
                        journal=state.journal,
                        cumulative=state.cumulative,
                        epoch_progress=epoch_progress,
                    )
                    expected_updates += 1
            if (
                not expected_updates
                or epoch_progress.optimizer_attempts != expected_updates
                or epoch_progress.optimizer_step_calls != expected_updates
                or epoch_progress.optimizer_completed_updates != expected_updates
                or epoch_progress.nonfinite_updates
            ):
                raise RuntimeError("strict propensity epoch update accounting is incomplete")
            elapsed = state.elapsed_before + time.time() - started
            record: dict[str, object] = {
                "epoch": epoch,
                "train_seconds": round(time.time() - epoch_started, 3),
                "evaluation": "withheld",
                "optimizer_progress": epoch_progress.snapshot(),
            }
            state.history.append(record)
            state.provenance["checkpoint_sha256"] = state.provenance.get("checkpoint_sha256", {})
            payload = _checkpoint_payload(
                model, epoch=epoch, args=args, provenance=state.provenance
            )
            snapshot = run_dir / f"model_ep{epoch}.pt"
            save_resume_state(payload, snapshot)
            state.provenance["checkpoint_sha256"][snapshot.name] = file_sha256(snapshot)
            if epoch == int(args["checkpoint_epoch"]):
                final_path = run_dir / "model.pt"
                save_resume_state(payload, final_path)
                state.provenance["checkpoint_sha256"][final_path.name] = file_sha256(final_path)
                state.provenance["selected_epoch"] = epoch
            emit("PROP", json.dumps(record, sort_keys=True), flush=True)
            state.journal.append(
                state.attempt_id,
                {
                    "event": "epoch_completed",
                    "epoch": epoch,
                    "optimizer_progress": epoch_progress.snapshot(),
                    "elapsed_seconds": elapsed,
                },
                durable=True,
            )
            next_epoch = epoch + 1
            _save_epoch_outputs(
                model=model,
                optimizer=optimizer,
                loader=loader,
                run_dir=run_dir,
                args=args,
                provenance=state.provenance,
                contract=state.contract,
                history=state.history,
                cumulative=state.cumulative,
                elapsed_seconds=elapsed,
                next_epoch=next_epoch,
                attempt_id=state.attempt_id,
                journal=state.journal,
                on_checkpoint_saved=lambda cursor: setattr(state, "next_epoch", cursor),
            )
        state.provenance["optimizer_progress"] = state.cumulative.snapshot()
        state.provenance["fit_status"] = "completed"
        state.provenance["current_attempt_status"] = "completed"
        state.provenance["fitting_seconds"] = state.elapsed_before + time.time() - started
        for attempt in state.provenance["training_attempts"]:
            if attempt["attempt_id"] == state.attempt_id:
                attempt["status"] = "completed"
                break
        atomic_write_json(run_dir / "provenance.json", state.provenance)
        atomic_write_json(run_dir / "history.json", {"epochs": state.history})
        state.journal.append(
            state.attempt_id,
            {"event": "run_completed", "next_epoch": int(args["epochs"])},
            durable=True,
        )
    except Exception as error:
        failure = {
            "status": "failed",
            "failure_scope": (
                "after_all_epochs_committed"
                if state.next_epoch == int(args["epochs"])
                else "before_run_completion"
            ),
            "attempt_id": state.attempt_id,
            "committed_next_epoch": state.next_epoch,
            "optimizer_attempts_this_attempt": state.cumulative.optimizer_attempts - attempt_start,
            "optimizer_progress_at_failure": state.cumulative.snapshot(),
            "exception_type": type(error).__name__,
            "message": str(error),
        }
        state.journal.append(state.attempt_id, {"event": "run_failed", **failure}, durable=True)
        create_only_json(run_dir / f"failure-{state.attempt_id}.json", failure)
        state.provenance["fit_status"] = "failed"
        state.provenance["current_attempt_status"] = "failed"
        state.provenance["optimizer_progress"] = state.cumulative.snapshot()
        state.provenance.setdefault("failure_history", []).append(failure)
        for attempt in state.provenance["training_attempts"]:
            if attempt["attempt_id"] == state.attempt_id:
                attempt["status"] = "failed"
                break
        atomic_write_json(run_dir / "provenance.json", state.provenance)
        raise
    return state.history, state.provenance
