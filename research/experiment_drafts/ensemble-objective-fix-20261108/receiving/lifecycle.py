from __future__ import annotations

import sys
import time
from pathlib import Path

from admission import digest, read

ACCEPTED = {
    "controller.py": "cdb319a45fce3c2084825feb515bcf128e6369f4c4623fbd9472df57129c9946",
    "receiving_state.py": "af03c2f095977402ab02c2a664d0f578c8e4028e7e5c4c76deae796033f0b7f3",
    "owned_phase.py": "68884ce6c2f277c3b8bf04ea52e7545929454198f2582be2a9a7d47a88c5a0df",
}


def accepted_helpers(folder: Path, controller: type) -> None:
    """Bind the actually imported controller and relay to the accepted native implementation."""
    imported = Path(sys.modules[controller.__module__].__file__).resolve()
    if imported != (folder / "controller.py").resolve():
        raise ValueError("PYTHONPATH imported another controller; include accepted receiver first")
    if any(digest(folder / name) != sha for name, sha in ACCEPTED.items()):
        raise ValueError("accepted native ownership helpers changed")


def progress(receiver: object, label: str) -> None:
    """Latch every callback/cleanup failure; always return to the executor so it can reap."""
    try:
        receiver.guard()
    except Exception as error:
        receiver.failures[label] = f"{type(error).__name__}: {error}"
        try:
            receiver.terminate()
        except Exception as cleanup_error:
            receiver.failures[label] += (
                f"; cleanup refusal {type(cleanup_error).__name__}: {cleanup_error}"
            )


def wait_for_plots(receiver: object, name: str) -> None:
    """Require all actual epoch image manifests before including them in the result bundle."""
    folder = receiver.run_root / name / "fit/plots"
    deadline = min(time.time() + 120, receiver.authority()["deadline_epoch"])
    while time.time() < deadline:
        receiver.guard()
        markers = [folder / f"epoch-{epoch:02d}.json" for epoch in range(1, 5)]
        if all(marker.exists() for marker in markers):
            for marker in markers:
                for filename, sha in read(marker)["files"].items():
                    if digest(folder / filename) != sha:
                        raise ValueError("rendered actual epoch evidence changed")
            return
        receiver.stop.wait(0.2)
    raise RuntimeError("actual four-epoch PNG/SVG evidence is incomplete")
