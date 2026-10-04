"""Execute one immutable serial screen recipe inside a monitored campaign job."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from campaign_job import JOBS, ROOT, STOP_NEW_JOBS


def sha256(path: Path) -> str:
    """Identify an immutable recipe or source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_controls() -> None:
    """Refuse new mechanisms until every clean reference fit/export succeeded."""
    queue = json.loads((JOBS / "c20260930_controls_queue.json").read_text())
    if queue["status"] != "finished" or queue.get("exit_code") != 0:
        raise RuntimeError("four-fold control queue has not finished successfully")
    for fold in range(4):
        for seed in (3, 4):
            for prefix in ("since", "ft_since"):
                name = f"c20260929_{prefix}_s{seed}_f{fold}"
                job = json.loads((JOBS / f"{name}.json").read_text())
                if job["status"] != "finished" or job.get("exit_code") != 0:
                    raise RuntimeError(f"reference dependency failed: {name}")


def run_recipe(recipe_path: Path, expected_hash: str, arm: str) -> None:
    """Run fixed argv stages, preserving each command, duration and exit code."""
    if sha256(recipe_path) != expected_hash:
        raise ValueError("screen recipe identity differs from launch command")
    recipe = json.loads(recipe_path.read_text())
    if recipe["status"] != "frozen-before-fit":
        raise ValueError("screen recipe was not frozen before fitting")
    for source, expected in recipe["source_sha256"].items():
        if sha256(ROOT / source) != expected:
            raise ValueError(f"screen source changed: {source}")
    require_controls()
    chosen = recipe["arms"][arm]
    log_root = ROOT / chosen["log_directory"]
    log_root.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation makes reruns reviewable and prevents log replacement.
    with (log_root / "recipe_identity.json").open("x") as stream:
        json.dump({"path": str(recipe_path), "sha256": expected_hash, "arm": arm}, stream)
        stream.write("\n")
    started = time.monotonic()
    for index, stage in enumerate(chosen["stages"]):
        if STOP_NEW_JOBS.exists():
            raise RuntimeError("persistent campaign stop barrier is present")
        command = stage["argv"]
        if not command or not all(isinstance(item, str) for item in command):
            raise ValueError("stage argv must be a nonempty list of strings")
        before = time.monotonic()
        with (log_root / f"{index:02d}_{stage['stage']}.log").open("x") as log:
            print(json.dumps({"stage": stage["stage"], "command": command}), flush=True)
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record = {
            "stage": stage["stage"],
            "command": command,
            "duration_seconds": time.monotonic() - before,
            "aggregate_seconds": time.monotonic() - started,
            "exit_code": result.returncode,
        }
        with (log_root / f"{index:02d}_{stage['stage']}.json").open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


def main() -> None:
    """Bind a recipe identity and select a preregistered arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    run_recipe(args.recipe.resolve(), args.recipe_sha256, args.arm)


if __name__ == "__main__":
    main()
