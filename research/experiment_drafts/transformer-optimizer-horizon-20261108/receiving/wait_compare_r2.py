"""Wait for this receiver to exit, then grade its existing six endpoints once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

PLAN_SHA = "b40ba65c7c2495f1a08a42331800ce8090bf9bbe63e7cb5514af401e784cf1c0"
AUTHORITY_SHA = "24e2847879db39016cfa952eed292a6c6dfa1c5dcf3f5154a09eeca19bca44a5"
SOURCE_COMMIT = "5d73bc2b10ae20e607e0d4b4533da85e76e27bca"
RECEIVER_PID = 1818502
RECEIVER_BIRTH = 158011868
DEADLINE = 1794197564.8507051


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _receiver_observation(proc_root: Path = Path("/proc")) -> dict | None:
    """Observe this exact identity; never signal any process."""
    try:
        fields = (proc_root / str(RECEIVER_PID) / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return {"observed_status": "absent"}
    birth = int(fields[19])
    if birth != RECEIVER_BIRTH:
        return {"observed_status": "pid_reused_different_birth", "current_birth": birth}
    if fields[0] == "Z":
        return {"observed_status": "exited_zombie", "observed_proc_state": "Z"}
    return None


def _completed_cells(plan: dict) -> None:
    """Refuse incomplete endpoints; the grader performs the full byte audit."""
    cells = plan["cells"]
    if len(cells) != 6 or len({cell["cell"] for cell in cells}) != 6:
        raise ValueError("expected exactly six frozen cells")
    for cell in cells:
        folder = Path(plan["runtime"]) / "runs" / cell["cell"] / "fit"
        terminal, score = _read(folder / "terminal_status.json"), _read(folder / "score.json")
        if (
            terminal.get("status") != "completed"
            or terminal.get("completed_epochs") != 4
            or terminal.get("global_updates") != 3280
            or score.get("sequences") != 1873
            or score.get("score", {}).get("selected_rows") != 3528273
        ):
            raise ValueError(f"incomplete fixed endpoint: {cell['cell']}")


def _write_progress(folder: Path, status: str, **details: object) -> None:
    target = folder / "worker-status.json"
    temporary = folder / "worker-status.pending"
    temporary.write_text(
        json.dumps({"status": status, "observed_at_epoch": time.time(), **details}) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _exit_receipt(runtime: Path, observation: dict) -> Path:
    """Create an observation receipt without replacing any previous evidence."""
    proof = runtime / "receiver-exit-r1.json"
    identities = {
        "schema": "wnn.transformer-optimizer-horizon-receiver-exit.v1",
        "observed_absent": True,
        "receiver_pid": RECEIVER_PID,
        "receiver_birth": RECEIVER_BIRTH,
        "plan_sha256": PLAN_SHA,
        "source_commit": SOURCE_COMMIT,
        "authority_sha256": AUTHORITY_SHA,
    }
    if proof.exists():
        existing = _read(proof)
        if any(existing.get(key) != value for key, value in identities.items()):
            raise ValueError("existing receiver-exit receipt belongs to another identity")
        return proof
    with proof.open("x", encoding="utf-8") as stream:
        json.dump({**identities, **observation, "observed_at_epoch": time.time()}, stream, indent=2)
        stream.write("\n")
    return proof


def main() -> int:
    """Bound this one-shot CPU repair by the unchanged cohort deadline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--grader", required=True, type=Path)
    parser.add_argument("--grader-sha256", required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    if _digest(args.plan) != PLAN_SHA or _digest(args.authority) != AUTHORITY_SHA:
        raise ValueError("frozen plan or authority bytes changed")
    plan, authority = _read(args.plan), _read(args.authority)
    if plan["source_commit"] != SOURCE_COMMIT or authority["deadline_epoch"] != DEADLINE:
        raise ValueError("source or original deadline changed")
    folder = args.grader.parent
    try:
        while time.time() < DEADLINE:
            observation = _receiver_observation()
            if observation is not None:
                break
            _write_progress(folder, "waiting_for_original_receiver_exit")
            time.sleep(min(15, max(0, DEADLINE - time.time())))
        else:
            raise TimeoutError("original cohort deadline reached before receiver exit")
        _completed_cells(plan)
        proof = _exit_receipt(Path(plan["runtime"]), observation)
        if _digest(args.grader) != args.grader_sha256:
            raise ValueError("reviewed comparison grader bytes changed")
        seconds = min(900.0, DEADLINE - time.time())
        if seconds <= 0:
            raise TimeoutError("no original collection time remains")
        command = [
            args.python,
            str(args.grader),
            "--plan",
            str(args.plan),
            "--plan-sha256",
            PLAN_SHA,
            "--authority",
            str(args.authority),
            "--authority-sha256",
            AUTHORITY_SHA,
            "--receiver-exit",
            str(proof),
            "--receiver-exit-sha256",
            _digest(proof),
        ]
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        _write_progress(folder, "grading_existing_six_endpoints_once")
        subprocess.run(command, check=True, timeout=seconds, env=environment)
        _write_progress(
            folder,
            "completed",
            comparison_sha256=_digest(Path(plan["runtime"]) / "comparison.json"),
        )
        return 0
    except Exception as error:
        _write_progress(folder, "failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
