from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wnn.propensity import PropensityNet
from wnn.propensity_fit_checkpoint import _save_epoch_outputs
from wnn.propensity_progress import ProgressJournal, UpdateProgress
from wnn.propensity_recovery import (
    atomic_write_json,
    build_static_contract,
    load_resume_state,
    make_resume_state,
    restore_resume_state,
    save_resume_state,
)
from wnn.run_identity import normalization_identity
from wnn.splits import CampaignSplit


@dataclass
class StrictFitState:
    """Restored training objects and provenance for one bounded fit attempt."""

    attempt_id: str
    journal: ProgressJournal
    cumulative: UpdateProgress
    history: list[dict[str, object]]
    elapsed_before: float
    next_epoch: int
    provenance: dict[str, object]
    contract: dict[str, object]


def prepare_strict_fit(
    *,
    model: PropensityNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: dict[str, object],
    provenance: dict[str, object],
    normalization: dict[str, object] | None,
    split: CampaignSplit,
    data_root: Path,
    source_root: Path,
    device: torch.device,
    resume: bool,
    run_dir: Path,
) -> StrictFitState:
    """Initialize a new strict run or recover its last complete epoch."""
    static = build_static_contract(
        args=args,
        provenance=provenance,
        split=split,
        data_root=data_root,
        source_root=source_root,
        loader=loader,
        optimizer=optimizer,
        device=device,
    )
    state_path = run_dir / "training_state.pt"
    journal_path = run_dir / "progress.jsonl"
    attempt_id = uuid.uuid4().hex
    if resume:
        saved = load_resume_state(state_path, static)
        cursor = int(saved["event_sequence"])
        journal = ProgressJournal(journal_path, resume_after=cursor)
        recovery = journal.recovery_summary(cursor)
        contract = saved["contract"]
        restore_resume_state(saved, model=model, optimizer=optimizer, loader=loader)
        cumulative = UpdateProgress(**saved["progress"])
        history = list(saved["history"])
        elapsed_before = float(saved["elapsed_seconds"])
        next_epoch = int(saved["next_epoch"])
        provenance = dict(saved["provenance"])
        failure_history = provenance.setdefault("failure_history", [])
        known_failures = {item.get("attempt_id") for item in failure_history}
        for failed in recovery["failed_attempts"]:
            if failed["attempt_id"] not in known_failures:
                failure_history.append(failed)
                known_failures.add(failed["attempt_id"])
            for prior_attempt in provenance.get("training_attempts", []):
                if prior_attempt.get("attempt_id") == failed["attempt_id"]:
                    prior_attempt["status"] = "failed"
        prior_outcome = journal.attempt_outcome(str(saved["attempt_id"]))
        for prior_attempt in provenance.get("training_attempts", []):
            if prior_attempt.get("attempt_id") != saved["attempt_id"]:
                continue
            if prior_outcome == "run_failed":
                prior_attempt["status"] = "failed"
            elif prior_outcome == "run_completed":
                prior_attempt["status"] = "completed"
            elif prior_attempt.get("status") == "running":
                prior_attempt["status"] = "interrupted_recovered"
                prior_attempt["recovered_from_epoch"] = next_epoch
            break
        if recovery["rolled_back_optimizer_attempts"]:
            provenance.setdefault("recovery_history", []).append(
                {
                    "from_attempt_id": saved["attempt_id"],
                    "resumed_at_epoch": next_epoch,
                    **recovery,
                    "policy": "rollback to completed epoch; replay incomplete epoch from saved RNG and loader state",
                }
            )
    else:
        if normalization is None:
            raise ValueError("strict propensity needs initial fit-only normalization")
        contract = {
            "static": static,
            "normalization_identity": normalization_identity(model),
            "normalization_summary": normalization,
        }
        journal = ProgressJournal(journal_path)
        cumulative = UpdateProgress()
        history = []
        elapsed_before = 0.0
        next_epoch = 0
        journal.append(attempt_id, {"event": "run_started", "next_epoch": 0}, durable=True)
        recovery = None
    provenance["normalization"] = contract["normalization_summary"]
    provenance["resume_contract"] = contract
    provenance["resume_attempt_id"] = attempt_id
    provenance["resume_policy"] = "completed_epoch_checkpoint; replay any incomplete epoch"
    provenance["fit_status"] = "running"
    provenance["current_attempt_status"] = "running"
    provenance["current_fit_attempt_id"] = attempt_id
    provenance.setdefault("scheduler", {"kind": "none", "reason": "fixed learning rate"})
    provenance.setdefault("training_attempts", []).append(
        {
            "attempt_id": attempt_id,
            "resumed_from_epoch": next_epoch,
            "recovery": recovery,
            "status": "running",
        }
    )
    if resume:
        journal.append(
            attempt_id,
            {"event": "resume_started", "next_epoch": next_epoch, "recovery": recovery},
            durable=True,
        )
        _save_epoch_outputs(
            model=model,
            optimizer=optimizer,
            loader=loader,
            run_dir=run_dir,
            args=args,
            provenance=provenance,
            contract=contract,
            history=history,
            cumulative=cumulative,
            elapsed_seconds=elapsed_before,
            next_epoch=next_epoch,
            attempt_id=attempt_id,
            journal=journal,
        )
    else:
        initial = make_resume_state(
            model=model,
            optimizer=optimizer,
            loader=loader,
            contract=contract,
            next_epoch=0,
            progress=cumulative.__dict__.copy(),
            history=history,
            provenance=provenance,
            elapsed_seconds=0.0,
            attempt_id=attempt_id,
            event_sequence=journal.sequence,
        )
        state_sha = save_resume_state(initial, state_path)
        journal.append(
            attempt_id,
            {"event": "initial_resume_checkpoint_saved", "next_epoch": 0, "sha256": state_sha},
            durable=True,
        )
    atomic_write_json(run_dir / "provenance.json", provenance)
    atomic_write_json(run_dir / "history.json", {"epochs": history})
    return StrictFitState(
        attempt_id=attempt_id,
        journal=journal,
        cumulative=cumulative,
        history=history,
        elapsed_before=elapsed_before,
        next_epoch=next_epoch,
        provenance=provenance,
        contract=contract,
    )
