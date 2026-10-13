"""Parse sealed S14 worker arguments with the exact frozen trainer, before device setup."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from typing import Any


class ParsedArguments(Exception):
    """Stop the trainer immediately after its real argparse parser succeeds."""


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of a byte string."""
    return hashlib.sha256(value).hexdigest()


def split_worker_command(
    profile: dict[str, Any], overlay: Path
) -> tuple[Path, list[str], dict[str, str]]:
    """Extract the trainer path, arguments, and environment from the sealed controller vector."""
    command = profile["controller_argv"]
    delimiter = command.index("--")
    worker = command[delimiter + 1 :]
    if not worker or worker[0] != "/usr/bin/env":
        raise ValueError("worker command must start with /usr/bin/env")

    environment: dict[str, str] = {}
    index = 1
    while index < len(worker) and "=" in worker[index] and not worker[index].startswith("/"):
        key, value = worker[index].split("=", 1)
        environment[key] = value
        index += 1
    if index + 2 >= len(worker):
        raise ValueError("worker command is missing its interpreter/wrapper/script")

    interpreter = Path(worker[index]).resolve()
    if interpreter != Path(sys.executable).resolve():
        raise ValueError(
            f"sealed interpreter {interpreter} differs from preflight {sys.executable}"
        )
    wrapper = Path(worker[index + 1]).resolve()
    if wrapper != (overlay / "scripts/cleanfold_exec.py").resolve():
        raise ValueError(f"worker wrapper is outside the exact source overlay: {wrapper}")
    relative_script = worker[index + 2]
    if relative_script != "scripts/train_rnn.py":
        raise ValueError(f"unexpected trainer script token: {relative_script}")
    trainer = (overlay / relative_script).resolve(strict=True)
    arguments = worker[index + 3 :]
    return trainer, arguments, environment


def parse_with_trainer(
    trainer: Path, arguments: list[str], environment: dict[str, str], overlay: Path
) -> dict[str, Any]:
    """Import the frozen trainer, parse the exact argv, and stop before post-parse work."""
    old_argv = sys.argv
    old_path = list(sys.path)
    old_environment = {key: os.environ.get(key) for key in environment}
    sys.path.insert(0, str(overlay / "src"))
    os.environ.update(environment)
    module_name = "_s14_frozen_train_rnn_argparse"
    spec = importlib.util.spec_from_file_location(module_name, trainer)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen trainer at {trainer}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    original_parse = argparse.ArgumentParser.parse_args
    parsed: list[argparse.Namespace] = []

    def parse_then_stop(
        parser: argparse.ArgumentParser,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        result = original_parse(parser, args=args, namespace=namespace)
        parsed.append(result)
        raise ParsedArguments

    stderr = io.StringIO()
    argparse.ArgumentParser.parse_args = parse_then_stop
    sys.argv = [str(trainer), *arguments]
    try:
        with contextlib.redirect_stderr(stderr):
            module.main()
    except ParsedArguments:
        outcome = "accepted"
    except SystemExit as error:
        outcome = "rejected"
        exit_code = error.code
    else:
        raise RuntimeError("trainer continued past argparse without the preflight stop hook")
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        sys.argv = old_argv
        sys.path[:] = old_path
        for key, old_value in old_environment.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during the argparse-only preflight")
    if outcome == "rejected":
        return {
            "outcome": outcome,
            "exit_code": exit_code,
            "stderr": stderr.getvalue().strip().splitlines()[-1:],
            "trainer_path": str(trainer),
            "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
            "cuda_initialized": False,
            "post_parse_training_code_reached": False,
        }
    if len(parsed) != 1:
        raise RuntimeError(f"expected exactly one parser invocation, observed {len(parsed)}")

    args = parsed[0]
    selected = {
        "name": args.name,
        "seed": args.seed,
        "device": args.device,
        "batch": args.batch,
        "epochs": args.epochs,
        "checkpoint_policy": args.checkpoint_policy,
        "checkpoint_epoch": args.checkpoint_epoch,
        "chunk": args.chunk,
        "update_rows": args.update_rows,
        "detach_rows": args.detach_rows,
        "precision": args.precision,
        "train_source": args.train_source,
        "valid_fold": args.valid_fold,
        "profile_steps": args.profile_steps,
        "profile_warmup": args.profile_warmup,
        "distill_alpha": args.distill_alpha,
        "distill_cache_scale": args.distill_cache_scale,
        "init": args.init,
        "distill_cache_root": str(args.distill_cache_root),
        "split_manifest": str(args.split_manifest),
        "effective_profile_batches": args.profile_steps + args.profile_warmup,
    }
    return {
        "outcome": outcome,
        "trainer_path": str(trainer),
        "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
        "selected_arguments": selected,
        "cuda_initialized": False,
        "post_parse_training_code_reached": False,
    }


def main() -> int:
    """Run parser-only checks for every profile in a frozen S14 packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--expect", choices=("accepted", "rejected"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()

    packet_path = cli.packet.resolve(strict=True)
    packet_bytes = packet_path.read_bytes()
    packet = json.loads(packet_bytes)
    overlay = Path(packet["source_overlay"]).resolve(strict=True)
    checks = []
    for profile in packet["profiles"]:
        trainer, arguments, environment = split_worker_command(profile, overlay)
        argv_hash = sha256_bytes(json.dumps(arguments, separators=(",", ":")).encode())
        result = parse_with_trainer(trainer, arguments, environment, overlay)
        if result["outcome"] != cli.expect:
            raise SystemExit(
                f"{profile['attempt_id']}: expected {cli.expect}, got {result['outcome']}"
            )
        result.update(
            attempt_id=profile["attempt_id"],
            profile_name=profile["profile_name"],
            detach_rows=profile["detach_rows"],
            worker_trainer_argv_sha256=argv_hash,
        )
        checks.append(result)

    receipt = {
        "schema": "wnn.s14-trainer-argparse-preflight.r1",
        "packet_path": str(packet_path),
        "packet_sha256": sha256_bytes(packet_bytes),
        "preflight_script": str(Path(__file__).resolve()),
        "preflight_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "expected_outcome": cli.expect,
        "checks": checks,
        "profile_authorized": False,
        "fit_authorized": False,
        "device_or_data_work_performed": False,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
