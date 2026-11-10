from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from controller import load_helpers
from receiving_state import process_family_records


def become_subreaper() -> None:
    """Adopt only this relay's orphaned descendants while retaining its owned identity."""
    library = ctypes.CDLL(None, use_errno=True)
    library.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    library.prctl.restype = ctypes.c_int
    if library.prctl(36, 1, 0, 0, 0) != 0:  # Linux PR_SET_CHILD_SUBREAPER, this process only.
        raise OSError(ctypes.get_errno(), "could not enable owned descendant reaping")


def clean_descendant_groups(
    helpers: SimpleNamespace,
    owner: dict[str, object],
    identities: dict[int, int],
    proof: Path,
    *,
    include_leader: bool,
    history: dict[int, dict[str, object]] | None = None,
) -> None:
    """Stop PPID-proven rank sessions before the canonical owned leader group."""
    history = {} if history is None else history
    for _ in range(4):
        records = process_family_records(identities)
        history.update(records)
        helpers.write(
            proof,
            {
                "schema": "wnn.pr67-owned-descendants.v1",
                "owner": owner,
                "processes": list(history.values()),
                "cleanup": True,
            },
        )
        groups = {int(record["pgid"]) for record in records.values()} - {int(owner["pgid"])}
        for group in groups:
            leader = history.get(group)
            if leader is None or int(leader["pgid"]) != group:
                raise RuntimeError("descendant session lacks a PPID-proven leader identity")
            for record in records.values():
                if int(record["pgid"]) != group:
                    continue
                identity = {key: record[key] for key in ("pid", "birth", "pgid")}
                if helpers.same(identity):
                    helpers.terminate(identity, grace_seconds=0.0)
                    break
        if not groups:
            break
        time.sleep(0.05)
    remaining = [
        record
        for pid, record in process_family_records(identities).items()
        if pid != owner["pid"] and record.get("state") != "Z"
    ]
    helpers.write(
        proof,
        {
            "schema": "wnn.pr67-owned-descendants.v1",
            "owner": owner,
            "processes": list(history.values()),
            "cleanup": True,
            "remaining_non_zombie": remaining,
            "cleanup_success": not remaining,
        },
    )
    if include_leader:
        helpers.terminate(owner, grace_seconds=0.0)
    elif remaining:
        raise RuntimeError("owned descendants remain after phase cleanup")


def main() -> int:
    """Observe owned descendants across torchrun sessions and preserve exact-group cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-src", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if os.name != "posix" or os.getpgrp() != os.getpid() or not command:
        raise RuntimeError("owned phase requires an isolated Linux session and wrapper command")
    helpers = load_helpers(args.support_src)
    owner = helpers.owner()
    proof = Path(os.environ["WNN_CONTROLLER_DESCENDANTS_RECEIPT"])
    identities = {int(owner["pid"]): int(owner["birth"])}
    become_subreaper()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda _signal, _frame: stop.set())
    process = subprocess.Popen(command)
    history: dict[int, dict[str, object]] = {}
    try:
        while process.poll() is None and not stop.is_set():
            records = process_family_records(identities)
            history.update(records)
            helpers.write(
                proof,
                {
                    "schema": "wnn.pr67-owned-descendants.v1",
                    "owner": owner,
                    "processes": list(records.values()),
                    "cleanup": False,
                },
            )
            stop.wait(0.2)
        code = process.poll()
        clean_descendant_groups(
            helpers, owner, identities, proof, include_leader=code != 0, history=history
        )
        return int(code or 0)
    except BaseException:
        clean_descendant_groups(
            helpers, owner, identities, proof, include_leader=True, history=history
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
