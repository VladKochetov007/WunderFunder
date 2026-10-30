from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from scripts.heterogeneous_results_report import (
        INITIAL_INPUT_SHA256,
        atomic_json,
        comment_body,
        read_json,
        ready_snapshot,
        report_ready,
        sha256,
        signature,
        validate_report,
    )
else:
    from heterogeneous_results_report import (
        INITIAL_INPUT_SHA256,
        atomic_json,
        comment_body,
        read_json,
        ready_snapshot,
        report_ready,
        sha256,
        signature,
        validate_report,
    )


if __package__:
    from scripts.heterogeneous_results_github import GitHub, PublicationError, publish
else:
    from heterogeneous_results_github import GitHub, PublicationError, publish


def score(args: argparse.Namespace, directory: Path, deadline: float) -> Path:
    """Run only the two reviewed CPU artifact-evaluation commands in fresh paths."""
    directory.mkdir(parents=True, exist_ok=False)
    manifest, output = directory / "inputs.json", directory / "grade"
    commands = [
        [
            sys.executable,
            "-m",
            "scripts.prepare_heterogeneous_ensemble_inputs",
            "--root",
            str(args.root),
            "--legacy-root",
            str(args.legacy_root),
            "--split",
            str(args.split),
            "--galam-probe",
            str(args.galam_probe),
            "--output",
            str(manifest),
        ],
        [
            sys.executable,
            "-m",
            "scripts.heterogeneous_ensemble_score",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    ]
    environment = os.environ | {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": f"{args.tools}:{args.source / 'src'}:{args.source}",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
    }
    for index, command in enumerate(commands):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("publisher wall-clock allocation exhausted")
        atomic_json(
            directory / f"command-{index}.json",
            {
                "command": command,
                "gpu_visible": False,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "started_epoch": time.time(),
            },
        )
        with (directory / f"command-{index}.log").open("xb") as log:
            result = subprocess.run(
                command,
                cwd=args.tools,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=min(1800, remaining),
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"report command{index} exited with {result.returncode}; no same-signature retry"
            )
    return output / "report.json"


def watch(args: argparse.Namespace) -> int:
    """Reuse the measured R1 table and evaluate only new final artifact signatures."""
    import fcntl

    allowed = set(os.sched_getaffinity(0)) & {58, 59}
    if not allowed:
        raise RuntimeError("publisher requires its existing CPU58/59 allocation")
    os.sched_setaffinity(0, allowed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    lock = (args.root / "results-publisher.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path, initial_path = (
        args.root / "results-publisher-state.json",
        args.root / "ensemble-results-r1/report.json",
    )
    initial = read_json(initial_path)
    weights = validate_report(initial)
    if initial["input_manifest_sha256"] != INITIAL_INPUT_SHA256:
        raise ValueError("initial measured table/input identity differs")
    snapshot = ready_snapshot(args.root, args.legacy_root, args.galam_probe, args.split)
    tools = {
        name: sha256(args.tools / "scripts" / name)
        for name in (
            "prepare_heterogeneous_ensemble_inputs.py",
            "heterogeneous_ensemble_score.py",
            "heterogeneous_ensemble_contract.py",
            "heterogeneous_galam_probe.py",
        )
    }
    if state_path.exists():
        state = read_json(state_path)
        if (
            state.get("schema") != "wnn.heterogeneous-results-publisher.v1"
            or state.get("tools") != tools
            or state.get("max_wall_seconds") != args.max_wall
            or state.get("initial_report_sha256") != sha256(initial_path)
        ):
            raise ValueError("preserved publisher state belongs to another source/input allocation")
    else:
        state = {
            "schema": "wnn.heterogeneous-results-publisher.v1",
            "started_epoch": time.time(),
            "max_wall_seconds": args.max_wall,
            "tools": tools,
            "initial_report_sha256": sha256(initial_path),
            "initial_report_reused": True,
            "last_report": str(initial_path),
            "last_signature": signature(snapshot, report_ready(initial)),
            "tried_signatures": {},
            "next_attempt": 1,
            "posts": {},
            "posts_attempted": 0,
            "last_post_epoch": time.time(),
            "status": "watching",
        }
    state["watcher_pid"] = os.getpid()
    state["watcher_start_ticks"] = (
        Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
    )
    state["watcher_code"] = {
        name: sha256(Path(__file__).with_name(name))
        for name in (
            "watch_heterogeneous_results.py",
            "heterogeneous_results_report.py",
            "heterogeneous_results_github.py",
        )
    }
    state["model_source"] = {
        "path": str(args.source),
        "identity_file_sha256": sha256(args.source.with_name(args.source.name + ".identity.json")),
    }
    if state.get("pending_post"):
        state.setdefault(
            "publication_blocked", {"kind": "unconfirmed_prior_POST", "epoch": time.time()}
        )
    deadline, client = state["started_epoch"] + args.max_wall, None
    if not state.get("publication_blocked"):
        try:
            client = GitHub(args.token_file)
        except (PublicationError, OSError, UnicodeError) as error:
            state["publication_blocked"] = {
                "kind": "GitHub_account_check_failure",
                "message": str(error)
                if isinstance(error, PublicationError)
                else "private credential input failed; content withheld",
                "epoch": time.time(),
            }
    atomic_json(state_path, state)
    while time.time() < deadline:
        snapshot = ready_snapshot(args.root, args.legacy_root, args.galam_probe, args.split)
        current = signature(snapshot)
        if current != state["last_signature"] and current not in state["tried_signatures"]:
            attempt = args.root / "ensemble-results-auto" / f"auto-{state['next_attempt']:04d}"
            state["next_attempt"] += 1
            state["tried_signatures"][current] = {
                "attempt": str(attempt),
                "status": "scoring",
                "ready_aliases": sorted(snapshot["ready"]),
                "epoch": time.time(),
            }
            atomic_json(state_path, state)
            try:
                if any(
                    sha256(args.tools / "scripts" / name) != expected
                    for name, expected in tools.items()
                ):
                    raise ValueError("pinned report grader source changed")
                path = score(args, attempt, deadline)
                report = read_json(path)
                validate_report(report, weights)
                after = ready_snapshot(args.root, args.legacy_root, args.galam_probe, args.split)
                state["last_report"], state["last_signature"] = (
                    str(path),
                    signature(after, report_ready(report)),
                )
                state["tried_signatures"][current].update(
                    status="complete", report_sha256=sha256(path)
                )
                state["pending_result_event"] = "result-" + sha256(path)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                failure = {
                    "kind": type(error).__name__,
                    "message": str(error),
                    "epoch": time.time(),
                }
                state["tried_signatures"][current].update(status="failed", failure=failure)
                state.setdefault("first_score_failure", failure)
            atomic_json(state_path, state)
        report_path = Path(state["last_report"])
        report = read_json(report_path)
        validate_report(report, weights)
        controller_path = args.root / "status/controller-six-r2.json"
        controller = read_json(controller_path) if controller_path.exists() else {}
        event = state.get("pending_result_event")
        hourly = event is None and time.time() - state["last_post_epoch"] >= 3600
        if event is not None or hourly:
            event = event or f"progress-{int((time.time() - state['started_epoch']) // 3600)}"
            body = comment_body(report, report_path, snapshot, controller, event, hourly)
            if publish(client, state, state_path, event, body):
                state.pop("pending_result_event", None)
        state.update(
            observed_epoch=time.time(),
            ready_aliases=sorted(snapshot["ready"]),
            pending_aliases=snapshot["pending"],
            last_report_sha256=sha256(report_path),
        )
        if report["status"] == "complete" and all(
            row["status"] == "complete" for row in report["rows"]
        ):
            state["status"] = (
                "complete"
                if state.get("pending_result_event") is None
                else "complete_publication_pending"
            )
            atomic_json(state_path, state)
            return 0 if state["status"] == "complete" else 1
        atomic_json(state_path, state)
        time.sleep(min(args.poll, max(0, deadline - time.time())))
    state["status"] = "wall_clock_complete_partial_results_preserved"
    atomic_json(state_path, state)
    return 0


def main() -> None:
    """Publish authorized PR52 updates without training, inference or GPU allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "legacy-root", "source", "tools", "split", "galam-probe", "token-file"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--pr", type=int, default=52)
    parser.add_argument("--max-wall", type=int, default=21600)
    parser.add_argument("--poll", type=int, default=60)
    args = parser.parse_args()
    if (
        sys.platform != "linux"
        or args.pr != 52
        or not 0 < args.max_wall <= 21600
        or not 60 <= args.poll <= 300
    ):
        parser.error("require Linux, fixed PR52, at most21600seconds and polling at least60seconds")
    raise SystemExit(watch(args))


if __name__ == "__main__":
    main()
