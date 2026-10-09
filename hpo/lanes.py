"""Compute-lane adapters for the optimizer: a Vast host (receipt-owned, via the lane-aware controller) and a fake lane for tests.

A lane accepts job specs (the ``make_refill_jobs`` format), reports their state by name and never decides anything about
the search.  Poll results are plain dicts: ``{"status": running|done|failed|unknown, "score": {...}, "note": str, "rc": int}``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAST_DIR = ROOT / "research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1"
LANE_FOLDS = {"a": {0, 1, 2, 3}, "b": {0, 1, 2, 3}}  # lane B receives the fold0/2/3 assets before an HPO launch


class Lane:
    name = "lane"

    def free_slots(self, inflight: int) -> int:
        raise NotImplementedError

    def can_run(self, fold: int, params: dict) -> bool:
        return True

    def submit(self, job_name: str, job: dict, meta: dict) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        """Make submitted jobs visible to the lane's scheduler (called once per tick)."""

    def poll(self, names: list[str]) -> dict[str, dict]:
        raise NotImplementedError


class FakeLane(Lane):
    """Deterministic in-memory lane for tests: ``objective(params, fold) -> wp | None`` (None = failure note)."""

    def __init__(self, name: str, objective, slots: int = 3, folds=(0, 1, 2, 3), failures=None) -> None:
        self.name, self.objective, self.slots, self.folds = name, objective, slots, set(folds)
        self.jobs: dict[str, dict] = {}
        self.failures = failures or {}
        self.polls: dict[str, int] = {}

    def free_slots(self, inflight: int) -> int:
        return max(0, self.slots - inflight)

    def can_run(self, fold: int, params: dict) -> bool:
        return fold in self.folds

    def submit(self, job_name: str, job: dict, meta: dict) -> None:
        self.jobs[job_name] = meta

    def poll(self, names: list[str]) -> dict[str, dict]:
        out = {}
        for name in names:
            meta = self.jobs.get(name)
            if meta is None:
                out[name] = {"status": "unknown", "note": "job not found on lane"}
                continue
            key = (meta["trial"], meta["fold"])
            self.polls[(key, name)] = self.polls.get((key, name), 0) + 1
            plan = self.failures.get(key)
            if plan and sum(1 for k in self.polls if k[0] == key and k[1] != name) < plan.get("times", 99):
                out[name] = {"status": "failed", "note": plan["note"], "rc": plan.get("rc", 1)}
                continue
            value = self.objective(meta["params"], meta["fold"])
            out[name] = {"status": "done", "score": {"wp": value, "t0": value, "t1": value, "class_wp": {"3": value - 0.2},
                                                      "finite_fraction": 1.0, "train_seconds": 100.0}}
        return out


class VastLane(Lane):
    """One receipt-owned Vast host.  The lane-aware controller module is loaded per lane (it reads its lane from the env)."""

    def __init__(self, letter: str, target_inflight: int = 6, max_pending: int = 8, spend_cap: float = 11.0,
                 folds=(0, 1, 2, 3)) -> None:
        self.name, self.letter = f"vast_{letter}", letter
        self.target, self.max_pending, self.spend_cap, self.folds = target_inflight, max_pending, spend_cap, set(folds)
        os.environ["VT_LANE"] = "" if letter == "a" else letter
        spec = importlib.util.spec_from_file_location(f"vast_controller_{letter}", VAST_DIR / "controller.py")
        self.controller = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(VAST_DIR))
        spec.loader.exec_module(self.controller)
        os.environ.pop("VT_LANE", None)
        self.submitted = False

    def can_run(self, fold: int, params: dict) -> bool:
        return fold in self.folds

    def free_slots(self, inflight: int) -> int:
        try:
            info = self.controller.probe()
            hours, cost = self.controller.spend()
        except Exception:  # noqa: BLE001 - an unreachable host simply offers no slots this tick
            return 0
        if cost >= self.spend_cap:
            return 0
        pending = int(info.get("PENDING", "0 RUNNING 0").split()[0])
        return max(0, min(self.target - inflight, self.max_pending - pending))

    def submit(self, job_name: str, job: dict, meta: dict) -> None:
        self.controller.JOBS_DIR.mkdir(exist_ok=True)
        (self.controller.JOBS_DIR / f"hpo_{job_name}.json").write_text(json.dumps(job, indent=1))
        self.submitted = True

    def flush(self) -> None:
        if self.submitted:
            self.controller.cmd_queue(argparse.Namespace())
            self.controller.cmd_reorder(argparse.Namespace())
            self.submitted = False

    def poll(self, names: list[str]) -> dict[str, dict]:
        if not names:
            return {}
        script = "cd /root/wnn; for n in " + " ".join(names) + "; do " \
            "if [ -f artifacts/runs/$n/score.json ]; then echo \"DONE|$n|$(cat artifacts/runs/$n/score.json)\"; " \
            "else d=$(ls /root/run/refill/done 2>/dev/null | grep -F \"$n\" | head -1); " \
            "if [ -n \"$d\" ]; then echo \"FAILED|$n|$d|$(cat /root/run/logs/$n.*.log 2>/dev/null | tail -n 6 | tr '\\n' ' ' | cut -c1-600)\"; " \
            "elif ls /root/run/refill/running /root/run/refill/pending 2>/dev/null | grep -qF \"$n\"; then echo \"RUNNING|$n\"; " \
            "else echo \"UNKNOWN|$n\"; fi; fi; done"
        out = {}
        result = self.controller.ssh(script, timeout=90)
        if result.returncode != 0:
            return {n: {"status": "unknown", "note": f"ssh failed: {result.stderr[-120:]}", "transient": True} for n in names}
        for line in result.stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) < 2:
                continue
            kind, name = parts[0], parts[1]
            if kind == "DONE":
                try:
                    out[name] = {"status": "done", "score": json.loads(parts[2])}
                except json.JSONDecodeError:
                    out[name] = {"status": "failed", "note": "unreadable score.json"}
            elif kind == "FAILED":
                rc_match = re.search(r"\.rc(-?\d+)\.json", parts[2])
                out[name] = {"status": "failed", "rc": int(rc_match.group(1)) if rc_match else None,
                             "note": parts[3] if len(parts) > 3 else ""}
            elif kind == "RUNNING":
                out[name] = {"status": "running"}
            else:
                out[name] = {"status": "unknown", "note": "job not found on host"}
        for name in names:
            out.setdefault(name, {"status": "unknown", "note": "no poll line", "transient": True})
        return out
