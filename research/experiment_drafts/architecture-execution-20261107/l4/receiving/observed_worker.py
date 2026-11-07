"""Keep the exact owned process alive if its read-only observer fails unexpectedly."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

from wnn.multistage_search_runtime import current_process_owner, terminate_owned_process


def main() -> int:
    """Run the existing observer and terminate this owned group on observer failure.

    run_owned_command must launch this small entrypoint as its new session leader.
    It keeps that receipt-bound leader alive until observation/training exits;
    an observer I/O exception therefore cannot leave an unowned trainer behind.
    """
    if os.name != "posix" or os.getpgrp() != os.getpid():
        raise RuntimeError("observed worker requires the caller's isolated owned session")
    command = sys.argv[1:]
    command = command[1:] if command[:1] == ["--"] else command
    if not command:
        raise ValueError("observed worker requires the existing observation command")
    owner = current_process_owner()
    process = subprocess.Popen(command)
    # Spawn first so the observer retains its normal TERM behavior. Keeping only
    # this leader alive lets the external owner reach its exact-group KILL path.
    previous_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        code = process.wait()
    except BaseException:
        terminate_owned_process(owner, grace_seconds=0.0)
        raise
    finally:
        if process.poll() == 0:
            signal.signal(signal.SIGTERM, previous_term)
    if code:
        terminate_owned_process(owner, grace_seconds=0.0)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
