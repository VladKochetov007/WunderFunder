from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wnn.propensity import PropensityNet
from wnn.propensity_progress import ProgressJournal, UpdateProgress
from wnn.propensity_recovery import atomic_write_json, make_resume_state, save_resume_state


def _checkpoint_payload(
    model: PropensityNet,
    *,
    epoch: int,
    args: dict[str, object],
    provenance: dict[str, object],
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "hidden": int(args["hidden"]),
        "n_in": int(provenance["n_in"]),
        "epoch": epoch,
        "args": {key: value for key, value in args.items() if key != "resume"},
        "provenance": {
            key: value for key, value in provenance.items() if key != "checkpoint_sha256"
        },
    }


def _save_epoch_outputs(
    *,
    model: PropensityNet,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    run_dir: Path,
    args: dict[str, object],
    provenance: dict[str, object],
    contract: dict[str, object],
    history: list[dict[str, object]],
    cumulative: UpdateProgress,
    elapsed_seconds: float,
    next_epoch: int,
    attempt_id: str,
    journal: ProgressJournal,
    on_checkpoint_saved: Callable[[int], None] | None = None,
) -> str:
    provenance["optimizer_progress"] = cumulative.snapshot()
    provenance["resume_policy"] = "completed_epoch_checkpoint; restart any incomplete epoch"
    provenance["training_state_format"] = "wnn.strict-propensity-training-state.v1"
    provenance["scheduler"] = {"kind": "none", "reason": "fixed learning rate"}
    provenance["fitting_seconds"] = elapsed_seconds
    state = make_resume_state(
        model=model,
        optimizer=optimizer,
        loader=loader,
        contract=contract,
        next_epoch=next_epoch,
        progress=cumulative.__dict__.copy(),
        history=history,
        provenance=provenance,
        elapsed_seconds=elapsed_seconds,
        attempt_id=attempt_id,
        event_sequence=journal.sequence,
    )
    state_path = run_dir / "training_state.pt"

    def commit_cursor() -> None:
        if on_checkpoint_saved is not None:
            on_checkpoint_saved(next_epoch)

    on_saved = commit_cursor if on_checkpoint_saved is not None else None
    state_sha = save_resume_state(state, state_path, on_saved=on_saved)
    journal.append(
        attempt_id,
        {
            "event": "resume_checkpoint_saved",
            "next_epoch": next_epoch,
            "sha256": state_sha,
        },
        durable=True,
    )
    atomic_write_json(run_dir / "provenance.json", provenance)
    atomic_write_json(run_dir / "history.json", {"epochs": history})
    return state_sha
