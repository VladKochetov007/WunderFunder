"""Parse the ten frozen innovation vectors without executing post-parse work."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parents[3]
sys.path.insert(0, str(SOURCE / "scripts"))


class Parsed(Exception):
    """Stop at argparse's boundary before data reads or device initialization."""


def main() -> None:
    """Verify exact actual parsers and write a bounded compatibility receipt."""
    import predict_run
    import torch
    import train_rnn

    assert train_rnn.SOURCE_ROOT == predict_run.SOURCE_ROOT == SOURCE
    assert train_rnn.ROOT == predict_run.ROOT == Path(os.environ["WNN_PROJECT_ROOT"])
    assert not torch.cuda.is_initialized()
    original = argparse.ArgumentParser.parse_args
    namespaces = []

    def stop_after_parse(
        parser: argparse.ArgumentParser,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> NoReturn:
        result = original(parser, args, namespace)
        namespaces.append(result)
        raise Parsed

    argparse.ArgumentParser.parse_args = stop_after_parse
    checks = []
    try:
        for path in sorted(BASE.glob("*COMMANDS_R2.json")):
            packet = json.loads(path.read_text())
            vectors = [
                (stage["part"], stage["trainer_argv"][1:], train_rnn.main)
                for stage in packet["stages"]
            ]
            profile = packet["device_profile"]["worker_argv"]
            offset = profile.index(str(SOURCE / "scripts/train_rnn.py"))
            vectors.extend(
                [
                    ("profile", profile[offset:], train_rnn.main),
                    ("export", packet["export"]["trainer_argv"][1:], predict_run.main),
                ]
            )
            for name, argv, entry in vectors:
                sys.argv = argv
                error = io.StringIO()
                try:
                    with contextlib.redirect_stderr(error):
                        entry()
                except Parsed:
                    ns = namespaces[-1]
                    if name != "export":
                        assert (ns.batch, ns.chunk, ns.epochs, ns.checkpoint_epoch) == (
                            16,
                            2000,
                            6,
                            5,
                        )
                        assert ns.inherit_init_preprocessing
                        assert ns.research_features == "causal_innovations"
                        if name == "profile":
                            assert (ns.profile_warmup, ns.profile_steps) == (5, 5)
                    checks.append({"contract": path.name, "stage": name, "status": "pass"})
                except SystemExit as failure:
                    raise ValueError(error.getvalue().splitlines()[-1]) from failure
                else:
                    raise AssertionError("entry point did not stop at its parser")
    finally:
        argparse.ArgumentParser.parse_args = original
    assert len(checks) == 10 and not torch.cuda.is_initialized()
    record = {
        "created_utc": datetime.now(UTC).isoformat(),
        "checks": checks,
        "source_root": str(SOURCE),
        "data_root": str(train_rnn.ROOT),
        "cuda_initialized": False,
        "post_parse_work_executed": False,
        "memory_bound_bytes": 1200000000,
        "quality_evaluated": False,
    }
    (BASE / "CLI_PREFLIGHT_R2.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"checks": len(checks), "status": "pass", "cuda_initialized": False}))


if __name__ == "__main__":
    main()
