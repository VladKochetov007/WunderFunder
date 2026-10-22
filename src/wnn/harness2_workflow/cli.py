"""Plan or run bounded task-level Harness² refinement for research handoffs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .task import Task


def main(argv: list[str] | None = None) -> int:
    """Expose offline planning, deterministic verification and explicit model execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "smoke", "refine"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--task", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--mode", choices=("parallel", "sequential"), default="parallel")
    parser.add_argument("--k", type=int, choices=(1, 2), default=1)
    parser.add_argument("--wall-seconds", type=int, default=600)
    parser.add_argument(
        "--worker", type=Path, help="JSON with explicit worker argv and per-call timeout"
    )
    parser.add_argument("--channel", choices=("opencode", "claude_cli"))
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    root = args.repository.resolve(strict=True)
    task_path = args.task or root / "tools/harness2/examples/lifecycle-task.json"
    task = Task.load(task_path.resolve(strict=True), root)
    if not 1 <= args.wall_seconds <= 14400:
        parser.error("--wall-seconds must be in 1..14400")
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "task": task.task_id,
                    "task_fingerprint": task.fingerprint,
                    "mode": args.mode,
                    "k": args.k,
                    "rollouts": 4 * args.k,
                    "improvement_stages": 3 * args.k,
                    "max_editor_calls": 6 * args.k + 2,
                    "wall_seconds": args.wall_seconds,
                    "scientific_fits_started": 0,
                    "model_calls_started": 0,
                },
                indent=2,
            )
        )
        return 0
    if sys.version_info < (3, 12):
        parser.error("Harness² requires an isolated Python 3.12/3.13 research environment")
    try:
        from harness2.core.spec import resolve_channel_model

        from .engine import refine
        from .proposer import BoundedProposer, FixtureProposer, fixture_executor
        from .runtime import Budget, run_worker
    except ImportError as error:
        parser.error(
            f"Install tools/harness2/requirements.txt in the research environment: {error}"
        )
    if args.output is None:
        parser.error("--output is required; interrupted attempts need a new directory")
    budget = Budget(args.wall_seconds, 4 * args.k, 6 * args.k + 2)
    executor_identity: object = "fixture"
    executor = fixture_executor
    proposer = FixtureProposer(budget)
    if args.action == "refine":
        if args.worker is None or not args.channel or not args.model:
            parser.error("refine requires explicit --worker, --channel and --model")
        worker = json.loads(args.worker.read_text(encoding="utf-8"))
        command = worker.get("argv")
        timeout = worker.get("timeout_seconds", 120)
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(x, str) and x for x in command)
        ):
            parser.error("worker argv must be a nonempty string array")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
            parser.error("worker timeout_seconds must be a positive integer")
        if any(
            marker in str(command).lower() for marker in ("hf_", "github_pat_", "ghp_", "api_key=")
        ):
            parser.error(
                "Worker argv must not contain credentials; use existing private authentication"
            )

        def execute(workspace: Path, remaining: int) -> None:
            prompt = (workspace / "task.json").read_text(encoding="utf-8")
            actual = [
                part.replace("{workspace}", str(workspace)).replace("{prompt}", prompt)
                for part in command
            ]
            run_worker(actual, workspace, min(timeout, remaining))

        executor = execute
        executor_identity = worker
        spec = resolve_channel_model(
            args.channel, args.model, timeout=min(timeout, args.wall_seconds)
        )
        proposer = BoundedProposer(spec, budget, args.output / "model_audit")
    summary = refine(
        task,
        args.output,
        args.base or root / "tools/harness2/base",
        args.library or args.output.parent / "harness2-library",
        budget,
        proposer,
        executor,
        mode=args.mode,
        k=args.k,
        executor_identity=executor_identity,
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "task_id",
                    "status",
                    "fixture",
                    "rollouts_attempted",
                    "editor_calls_attempted",
                    "library_entry",
                    "scientific_fits_started",
                    "model_quality_tested",
                )
            },
            indent=2,
        )
    )
    return 0
