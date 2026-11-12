"""Import-check the pinned GRU adapter/runtime contract without starting a fit."""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path

from impact81.bindings import admit_runtime, source_identity, write_new
from impact81.runtime_family import GRU_RUNTIME_COMMIT

_GRU_MODULES = (
    "impact81.fitting",
    "impact81.lifecycle",
    "impact81.lifecycle_support",
    "impact81.engine",
    "impact81.auxiliary_loop",
    "impact81.auxiliary_backward",
    "impact81.gru_models",
    "impact81.gru_objectives",
    "impact81.preparation_later",
    "impact81.receiver_later",
    "impact81.receiver_gru_checks",
)
_RESUME_PARAMETERS = (
    "state_path",
    "identity",
    "model",
    "optimizer",
    "scheduler",
    "context",
    "rank_log",
    "output",
    "monitor",
)


def main() -> None:
    """Check exact imports/signatures and write a one-shot native receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this import preflight is for the receiving Linux runtime")

    runtime_root = admit_runtime(args.runtime_root, GRU_RUNTIME_COMMIT)
    packet_runtime = Path(__file__).resolve().parent
    imported: dict[str, str] = {}
    for name in _GRU_MODULES:
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve(strict=True)
        if not (origin.is_relative_to(runtime_root) or origin.is_relative_to(packet_runtime)):
            raise RuntimeError(f"{name} resolved outside the pinned roots: {origin}")
        imported[name] = str(origin)

    from impact81.lifecycle_support import _load_resume

    signature = inspect.signature(_load_resume)
    if tuple(signature.parameters) != _RESUME_PARAMETERS:
        raise TypeError(f"resume helper signature differs from lifecycle call: {signature}")
    if signature.parameters["monitor"].default is not None:
        raise TypeError("resume helper monitor parameter must remain optional")
    if _load_resume.__module__ != "wnn.training.sequence_fit_resume":
        raise TypeError("GRU runtime must use PR79's public load_resume API")

    identity = source_identity(runtime_root)
    write_new(
        args.output,
        {
            "schema": "impact81.gru-import-preflight.v1",
            "runtime_commit": GRU_RUNTIME_COMMIT,
            "runtime_source_sha256": identity["sha256"],
            "runtime_source_file_count": len(identity["files"]),
            "resume_provider": f"{_load_resume.__module__}.{_load_resume.__name__}",
            "resume_signature": str(signature),
            "imported_modules": imported,
            "cuda_fit_started": False,
        },
    )


if __name__ == "__main__":
    main()
