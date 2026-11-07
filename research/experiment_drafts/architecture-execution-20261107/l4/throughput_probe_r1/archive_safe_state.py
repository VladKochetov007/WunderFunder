from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
from transition_support import pinned_json, validate_evaluations, write_json_new

from wnn.training.sequence_checkpoint import load_training_state
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_provenance import canonical_sha256, file_sha256

CPU_CONTEXT = DistributedContext(0, 0, 1, torch.device("cpu"), False)


def freeze_checkpoint(source: Path, folder: Path) -> Path:
    """Copy one open atomic checkpoint inode; an adjacent publication cannot change it."""
    if source.is_symlink() or not source.is_file():
        raise ValueError("safe checkpoint must be a regular native file")
    folder.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".safe-state-", suffix=".pt", dir=folder)
    target = Path(name)
    try:
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
            before = os.fstat(incoming.fileno())
            shutil.copyfileobj(incoming, outgoing, length=1 << 20)
            after = os.fstat(incoming.fileno())
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("checkpoint inode changed during archival")
            outgoing.flush()
            os.fsync(outgoing.fileno())
        return target
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def archive_once(
    output: Path,
    archive: Path,
    identity: dict[str, object],
    transition_sha: str,
    expected_next_epoch: int,
) -> tuple[int, int]:
    """Archive exactly the next safe epoch, refusing a missed or reordered boundary."""
    temporary = freeze_checkpoint(output / "training_state.pt", archive)
    return archive_frozen(temporary, archive, identity, transition_sha, expected_next_epoch)


def archive_frozen(
    temporary: Path,
    archive: Path,
    identity: dict[str, object],
    transition_sha: str,
    expected_next_epoch: int,
) -> tuple[int, int]:
    """Publish the already-opened safe boundary, preventing an adjacent-read epoch race."""
    try:
        state = load_training_state(temporary, identity, CPU_CONTEXT)
        epoch = state["next_epoch"]
        if epoch != expected_next_epoch or not 1 <= epoch <= 12:
            raise ValueError("safe-state archiver missed or reordered a whole-epoch boundary")
        sha = file_sha256(temporary)
        filename = f"epoch-{epoch:06d}-{sha}.pt"
        destination = archive / filename
        if destination.exists():
            if destination.is_symlink() or file_sha256(destination) != sha:
                raise FileExistsError("immutable safe-state archive conflicts")
        else:
            os.link(temporary, destination)
        proof = {
            "schema": "wnn.gru-safe-state-archive.v1",
            "completed_epochs": epoch,
            "global_step": state["global_step"],
            "checkpoint_path": filename,
            "checkpoint_sha256": sha,
            "identity_sha256": canonical_sha256(identity),
            "source_sha256": identity["source_sha256"],
            "transition_proof_sha256": transition_sha,
            "retention": "immutable full resume state including model/optimizer/scheduler/rank RNG",
            "checkpoint_selection": "every safe whole epoch; no best/winner selection",
        }
        proof_path = archive / f"epoch-{epoch:06d}.json"
        if proof_path.exists():
            if proof_path.is_symlink() or json.loads(proof_path.read_bytes()) != proof:
                raise FileExistsError("immutable safe-state proof conflicts")
        else:
            write_json_new(proof_path, proof)
        if os.name == "posix":
            descriptor = os.open(archive, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return epoch, state["global_step"]
    finally:
        temporary.unlink(missing_ok=True)


def watch(output: Path, proof_sha: str, poll_seconds: float = 1.0) -> None:
    """Observe safe-state publications until the original bounded12E endpoint."""
    proof = pinned_json({"path": str(output / "transition_proof.json"), "sha256": proof_sha})
    identity = proof["continuation_identity"]
    deadline = float(proof["request"]["not_after_epoch"])
    initial = proof["next_epoch"]
    if not 0.1 <= poll_seconds <= 2.0:
        raise ValueError("archive polling must be bounded between0.1 and2 seconds")
    archive = output / "safe_state_archive"
    if archive.is_symlink():
        raise ValueError("safe-state archive cannot be a symlink")
    completed = initial - 1
    for path in sorted(archive.glob("epoch-*.json")):
        value = json.loads(path.read_bytes())
        if value["completed_epochs"] != completed + 1:
            raise ValueError("existing safe-state archive has an epoch gap")
        checkpoint = archive / value["checkpoint_path"]
        if (
            checkpoint.is_symlink()
            or file_sha256(checkpoint) != value["checkpoint_sha256"]
            or value["transition_proof_sha256"] != proof_sha
            or value["identity_sha256"] != canonical_sha256(identity)
        ):
            raise ValueError("existing archive changed or belongs to another continuation")
        completed += 1
    last_inode = None
    while time.time() < deadline:
        state_path = output / "training_state.pt"
        info = state_path.stat()
        marker = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if marker != last_inode:
            # Read one frozen descriptor even if a new safe state is atomically published meanwhile.
            temporary = freeze_checkpoint(state_path, archive)
            try:
                state = load_training_state(temporary, identity, CPU_CONTEXT)
                epoch = state["next_epoch"]
                if epoch > completed:
                    archive_frozen(temporary, archive, identity, proof_sha, completed + 1)
                    completed = epoch
                elif epoch != completed:
                    raise ValueError("safe checkpoint regressed behind its immutable archive")
                else:
                    saved = json.loads((archive / f"epoch-{epoch:06d}.json").read_bytes())
                    if file_sha256(temporary) != saved["checkpoint_sha256"]:
                        raise ValueError(
                            "safe checkpoint bytes changed at an already archived epoch"
                        )
            finally:
                temporary.unlink(missing_ok=True)
            last_inode = marker
            ready = output / "archive_ready.json"
            if not ready.exists():
                write_json_new(
                    ready,
                    {
                        "schema": "wnn.gru-safe-state-archive-ready.v1",
                        "transition_proof_sha256": proof_sha,
                        "initial_epoch": initial,
                        "not_after_epoch": deadline,
                        "pid": os.getpid(),
                    },
                )
            if completed == 12:
                validate_evaluations(output / "monitor_evaluations.jsonl", 12)
                write_json_new(
                    output / "archive_complete.json",
                    {
                        "schema": "wnn.gru-safe-state-archive-complete.v1",
                        "completed_epochs": 12,
                        "transition_proof_sha256": proof_sha,
                        "initial_epoch": initial,
                        "archive_epochs": list(range(initial, 13)),
                    },
                )
                return
        time.sleep(poll_seconds)
    raise TimeoutError("original L4 deadline reached before all safe epochs were archived")


def main() -> None:
    """Run only CPU checkpoint I/O; the ROOT-owned supervisor must monitor this process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transition-proof-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    watch(args.output, args.transition_proof_sha256, args.poll_seconds)


if __name__ == "__main__":
    main()
