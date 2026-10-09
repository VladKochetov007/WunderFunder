"""Durable study state: Optuna (SQLite) for the sampler history, an append-only JSONL log for everything else.

``events.jsonl`` is the source of truth for orchestration (which trial is on which fold, on which lane, retries, scores);
the SQLite file only carries the sampler/pruner history and can be rebuilt from the log with ``replay``.  Every write is a
single appended line (``flush`` + ``fsync``), so a killed process loses at most the line it was writing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import optuna

from hpo import space

optuna.logging.set_verbosity(optuna.logging.WARNING)

FOLD_ORDER = (1, 0, 2, 3)
PRUNE_AFTER_STAGES = 2  # a trial must pass fold 1 and fold 0 before folds 2 and 3 are spent on it


def make_sampler(mode: str, seed: int):
    if mode == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed, popsize=8, restart_strategy="ipop")
    if mode == "nsga2":
        return optuna.samplers.NSGAIISampler(seed=seed, population_size=16)
    return optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True, n_startup_trials=24, seed=seed)


class Store:
    def __init__(self, root: Path, name: str, sampler: str = "tpe", seed: int = 20261010, directions: int = 1) -> None:
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self.name, self.sampler_name, self.seed = name, sampler, seed
        self.db_path = self.dir / "study.db"
        self.study = self._open(directions)

    def _open(self, directions: int):
        kwargs = {"study_name": self.name, "storage": f"sqlite:///{self.db_path}", "load_if_exists": True,
                  "sampler": make_sampler(self.sampler_name, self.seed)}
        if directions == 1:
            return optuna.create_study(direction="maximize",
                                       pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0), **kwargs)
        return optuna.create_study(directions=["maximize"] * directions, **kwargs)

    # ---- event log -------------------------------------------------------------------------------------------
    def log(self, event: str, **fields) -> dict:
        row = {"t": round(time.time(), 1), "ev": event, **fields}
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return row

    def events(self) -> list[dict]:
        if not self.events_path.exists():
            return []
        rows = []
        for line in self.events_path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from a crash is dropped, never fatal
        return rows

    def ledger(self) -> dict[int, dict]:
        """Fold the event log into per-trial state."""
        trials: dict[int, dict] = {}
        for row in self.events():
            tid = row.get("trial")
            kind = row["ev"]
            if kind == "ask":
                trials[tid] = {"params": row["params"], "mode": row.get("mode", "core"), "stages": {}, "state": "running",
                               "space": row.get("space", space.SPACE_VERSION), "value": None, "asked": row["t"]}
            elif tid in trials:
                trial = trials[tid]
                fold = row.get("fold")
                if kind == "stage_submit":
                    trial["stages"][fold] = {"status": "submitted", "job": row["job"], "lane": row["lane"], "t": row["t"],
                                             "attempts": row.get("attempt", 0)}
                elif kind == "stage_result":
                    trial["stages"][fold].update(status="done", wp=row["wp"], detail=row.get("detail", {}), done_t=row["t"])
                elif kind == "stage_failed":
                    trial["stages"].setdefault(fold, {}).update(status="failed", failure=row.get("failure"), note=row.get("note", ""))
                elif kind == "stage_retry":
                    trial["stages"][fold] = {"status": "pending", "attempts": row.get("attempt", 1), "lane": row.get("lane")}
                elif kind == "stage_pending":
                    trial["stages"][fold] = {"status": "pending", "attempts": 0}
                elif kind == "tell":
                    trial.update(state=row["state"], value=row.get("value"), told=row["t"], reason=row.get("reason", ""))
        return trials

    # ---- replay ---------------------------------------------------------------------------------------------
    def replay(self, mode_for=lambda m: m) -> int:
        """Rebuild an empty study from the log (sampler history only; orchestration state lives in the log)."""
        if self.study.trials:
            return 0
        ledger = self.ledger()
        count = 0
        for tid in sorted(ledger):
            info = ledger[tid]
            self.study.enqueue_trial({k: v for k, v in info["params"].items() if k in _suggested_names(info["mode"])},
                                     skip_if_exists=False)
            trial = self.study.ask()
            space.suggest(trial, mode_for(info["mode"]))
            trial.set_constraint("cost_hours", space.constraint(info["params"]))
            step = 0
            cumulative = []
            for fold in FOLD_ORDER:
                stage = info["stages"].get(fold)
                if stage and stage.get("status") == "done":
                    cumulative.append(stage["wp"])
                    trial.report(sum(cumulative) / len(cumulative), step)
                    step += 1
            state = info["state"]
            if state == "COMPLETE" and info["value"] is not None:
                self.study.tell(trial, info["value"])
            elif state == "PRUNED":
                self.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            elif state in ("FAIL", "INVALID"):
                self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                pass  # still running: leave the Optuna trial open (RUNNING) as it was
            count += 1
        return count


def _suggested_names(mode: str) -> set[str]:
    class Recorder:
        def __init__(self) -> None:
            self.names: set[str] = set()

        def suggest_categorical(self, name, choices):
            self.names.add(name)
            return choices[0]

        def suggest_float(self, name, low, high, log=False):
            self.names.add(name)
            return low

        def suggest_int(self, name, low, high):
            self.names.add(name)
            return low

    recorder = Recorder()
    # Visit every conditional branch so replay enqueues each parameter the original trial could have drawn.
    for sched in ("onecycle", "wsd_sqrt", "sgdr"):
        for init in ("default", "chrono"):
            for loss in ("wmse", "wmse_gcorr"):
                for avg in ("none", "swa"):
                    class Forced(Recorder):
                        def suggest_categorical(self, name, choices, _s=sched, _i=init, _l=loss, _a=avg):
                            self.names.add(name)
                            return {"sched": _s, "rnn_init": _i, "loss": _l, "avg": _a}.get(name, choices[0])
                    forced = Forced()
                    forced.names = recorder.names
                    space.suggest(forced, mode)
    return recorder.names
