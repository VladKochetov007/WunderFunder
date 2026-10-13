"""Validate every sealed S14 worker-command copy with the exact trainer parser."""

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


TIMEOUT_COMMAND = [
    "/usr/bin/timeout",
    "--signal=TERM",
    "--kill-after=10s",
    "900s",
]


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of a byte string."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading its full contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_copies(profile: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Require controller, outer-launch, and standalone vectors to agree."""
    controller = profile["controller_argv"]
    delimiter = controller.index("--")
    worker_from_controller = controller[delimiter + 1 :]
    standalone_worker = profile["worker_argv"]
    if standalone_worker != worker_from_controller:
        raise ValueError("standalone worker_argv differs from controller suffix")

    outer = profile["outer_limited_argv"]
    try:
        controller_start = outer.index(controller[0])
    except ValueError as error:
        raise ValueError("outer_limited_argv does not embed controller_argv") from error
    if outer[controller_start:] != controller:
        raise ValueError("outer_limited_argv controller suffix differs from controller_argv")
    return standalone_worker, worker_from_controller, outer


def split_worker_command(
    command: list[str], overlay: Path
) -> tuple[Path, list[str], dict[str, str], dict[str, str]]:
    """Extract trainer arguments and verify the exact finite worker timeout."""
    worker = list(command)
    if not worker or worker[0] != "/usr/bin/env":
        raise ValueError("worker command must start with /usr/bin/env")

    environment: dict[str, str] = {}
    index = 1
    while index < len(worker) and "=" in worker[index] and not worker[index].startswith("/"):
        key, value = worker[index].split("=", 1)
        environment[key] = value
        index += 1

    if worker[index : index + len(TIMEOUT_COMMAND)] != TIMEOUT_COMMAND:
        raise ValueError(f"worker timeout is not the pinned command: {worker[index : index + 4]}")
    timeout_path = Path(TIMEOUT_COMMAND[0]).resolve(strict=True)
    timeout_binding = {
        "path": str(timeout_path),
        "sha256": sha256_file(timeout_path),
        "argv": TIMEOUT_COMMAND,
        "seconds": 900,
        "kill_after_seconds": 10,
    }
    index += len(TIMEOUT_COMMAND)
    if index + 2 >= len(worker):
        raise ValueError("worker command is missing its interpreter/wrapper/script")

    interpreter = Path(worker[index]).resolve(strict=True)
    if interpreter != Path(sys.executable).resolve():
        raise ValueError(
            f"sealed interpreter {interpreter} differs from preflight {sys.executable}"
        )
    wrapper = Path(worker[index + 1]).resolve(strict=True)
    if wrapper != (overlay / "scripts/cleanfold_exec.py").resolve():
        raise ValueError(f"worker wrapper is outside the exact source overlay: {wrapper}")
    relative_script = worker[index + 2]
    if relative_script != "scripts/train_rnn.py":
        raise ValueError(f"unexpected trainer script token: {relative_script}")
    trainer = (overlay / relative_script).resolve(strict=True)
    arguments = worker[index + 3 :]
    return trainer, arguments, environment, timeout_binding


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
            "trainer_sha256": sha256_file(trainer),
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
        "trainer_sha256": sha256_file(trainer),
        "selected_arguments": selected,
        "cuda_initialized": False,
        "post_parse_training_code_reached": False,
    }


def main() -> int:
    """Run actual-parser checks for the standalone worker and controller suffix."""
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
        standalone, controller_suffix, outer = command_copies(profile)
        parsed_vectors = []
        timeout_binding: dict[str, Any] | None = None
        for label, command in (
            ("standalone_worker_argv", standalone),
            ("controller_argv_suffix", controller_suffix),
        ):
            trainer, arguments, environment, timeout = split_worker_command(command, overlay)
            if timeout_binding is not None and timeout != timeout_binding:
                raise ValueError("standalone and controller timeout bindings differ")
            timeout_binding = timeout
            result = parse_with_trainer(trainer, arguments, environment, overlay)
            if result["outcome"] != cli.expect:
                raise SystemExit(
                    f"{profile['attempt_id']} {label}: expected {cli.expect}, "
                    f"got {result['outcome']}"
                )
            parsed_vectors.append({"label": label, **result})
        first = parsed_vectors[0]
        second = parsed_vectors[1]
        if first["selected_arguments"] != second["selected_arguments"]:
            raise ValueError("standalone and controller trainer parse results differ")
        worker_args = split_worker_command(standalone, overlay)[1]
        checks.append(
            {
                "attempt_id": profile["attempt_id"],
                "profile_name": profile["profile_name"],
                "detach_rows": profile["detach_rows"],
                "controller_worker_vector_equals_standalone": True,
                "outer_suffix_embeds_controller_vector": True,
                "worker_vector_sha256": sha256_bytes(
                    json.dumps(standalone, separators=(",", ":")).encode()
                ),
                "trainer_argv_sha256": sha256_bytes(
                    json.dumps(worker_args, separators=(",", ":")).encode()
                ),
                "timeout_binding": timeout_binding,
                "parser_checks": parsed_vectors,
            }
        )

    receipt = {
        "schema": "wnn.s14-trainer-argparse-preflight.r2",
        "packet_path": str(packet_path),
        "packet_sha256": sha256_bytes(packet_bytes),
        "preflight_script": str(Path(__file__).resolve()),
        "preflight_script_sha256": sha256_file(Path(__file__).resolve()),
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
