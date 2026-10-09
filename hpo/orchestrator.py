"""Resumable hyper-parameter search orchestrator (one stateless ``tick`` per invocation).

Every tick loads the study from disk, ingests finished stages from the compute lanes, decides each trial's next step
(next fold, prune, finish, retry, penalty, skip), dispatches pending stages to lanes with free slots, asks the sampler for new
trials while capacity and budget allow, and appends everything to ``events.jsonl``.  A crash, kill or reboot loses nothing:
the next tick (a systemd timer) continues from the log and the SQLite sampler state.

Trial = one configuration trained from scratch on fold 1, then fold 0, then (if it ranks in the top part) folds 2 and 3; the
objective is the mean official WP over the folds completed.  Seeds are never searched.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
import traceback
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hpo import space  # noqa: E402
from hpo.store import Store  # noqa: E402

STUDIES = ROOT / "hpo/studies"
ALERTS = ROOT / "research/campaign_2026-10-06/bandit_continuation/watchdog/ALERTS.jsonl"
KEEPALIVE = ROOT / ".runtime/hpo_keepalive"
CONTINUE_PERCENTILE = 0.40  # after fold 0, only the top 40% of completed trials spend folds 2 and 3
MIN_PEERS = 8


# ---- failure policy --------------------------------------------------------------------------------------------
def classify_failure(note: str, rc: int | None = None) -> str:
    """infra: retry; oom: retry (later, fewer neighbours); numeric: penalty, no retry; invalid: skip and alert; unknown: one retry."""
    t = (note or "").lower()
    if "out of memory" in t or "cuda error: out of memory" in t:
        return "oom"
    if "no space left" in t or "connection" in t and "refused" in t or "ssh failed" in t or "job not found" in t:
        return "infra"
    if rc in (-9, -15, 137, 143) or "killed" in t or "aggregate working ram" in t:
        return "infra"
    if "nan" in t.split() or "non-finite" in t or "diverg" in t or "loss is nan" in t or "inf loss" in t:
        return "numeric"
    if any(word in t for word in ("error:", "valueerror", "keyerror", "assertionerror", "notimplementederror", "typeerror",
                                  "unrecognized arguments", "invalid choice", "requires --", "unsupported")):
        return "invalid"
    return "unknown"


MAX_ATTEMPTS = {"infra": 2, "oom": 2, "unknown": 1, "numeric": 0, "invalid": 0}


def alert(key: str, severity: str, message: str) -> None:
    ALERTS.parent.mkdir(parents=True, exist_ok=True)
    with ALERTS.open("a") as handle:
        handle.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), "event": "ALERT",
                                 "severity": severity, "key": f"hpo:{key}", "message": message}) + "\n")


# ---- core ------------------------------------------------------------------------------------------------------
class Orchestrator:
    def __init__(self, store: Store, lanes: dict, config: dict) -> None:
        self.store, self.lanes, self.cfg = store, lanes, config
        self.folds = tuple(config.get("folds", (1, 0)))  # search folds: feedback to the sampler
        self.confirm_folds = tuple(config.get("confirm_folds", (2, 3)))  # reserved: finalists only, no sampler feedback
        self.study = store.study

    # job naming: unique per (trial, fold, attempt) so a retry never collides with a half-finished run on the host
    def job_name(self, tid: int, fold: int, attempt: int) -> str:
        return f"hpo_{self.store.name}_t{tid:04d}_f{fold}_a{attempt}"

    def _job_builder(self):
        return self.cfg.get("job_builder", _default_job_builder)

    # ---- fold-level difficulty -----------------------------------------------------------------------------------
    PRIOR_OFFSET = {1: 0.0, 0: 0.0131, 2: 0.004, 3: -0.003}  # fold0 is 0.0131 easier (28-recipe measurement)

    def offsets(self, ledger: dict) -> dict[int, float]:
        """Fold difficulty relative to fold 1, from trials that have both folds (median), else the measured prior."""
        out = dict(self.PRIOR_OFFSET)
        for fold in (0, 2, 3):
            diffs = [t["stages"][fold]["wp"] - t["stages"][1]["wp"] for t in ledger.values()
                     if t["stages"].get(fold, {}).get("status") == "done" and t["stages"].get(1, {}).get("status") == "done"
                     and t["stages"][fold]["wp"] > 0.5 and t["stages"][1]["wp"] > 0.5]
            if len(diffs) >= 4:
                diffs.sort()
                out[fold] = 0.5 * out[fold] + 0.5 * diffs[len(diffs) // 2]
        return out

    # ---- ingest ------------------------------------------------------------------------------------------------
    def ingest(self, ledger: dict) -> None:
        by_lane: dict[str, list[tuple[int, int, dict]]] = {}
        for tid, trial in ledger.items():
            for fold, stage in trial["stages"].items():
                if trial["state"] != "running" and fold not in self.confirm_folds:
                    continue
                if stage["status"] == "submitted":
                    by_lane.setdefault(stage["lane"], []).append((tid, fold, stage))
        for lane_name, items in by_lane.items():
            polled = self.lanes[lane_name].poll([stage["job"] for _, _, stage in items])
            for tid, fold, stage in items:
                result = polled.get(stage["job"], {"status": "unknown"})
                self._handle(ledger, tid, fold, stage, result)

    def _handle(self, ledger: dict, tid: int, fold: int, stage: dict, result: dict) -> None:
        status = result["status"]
        age_h = (time.time() - stage["t"]) / 3600
        if status == "running" or (status == "unknown" and result.get("transient")):
            if age_h > self.cfg.get("stage_timeout_h", 5.0):
                self._failure(ledger, tid, fold, stage, "infra", f"stage timeout after {age_h:.1f} h")
            return
        if status == "done":
            score = result["score"]
            wp = score.get("wp")
            if wp is None or wp != wp or score.get("finite_fraction", 1.0) < 0.99:
                self._failure(ledger, tid, fold, stage, "numeric", f"non-finite predictions (finite {score.get('finite_fraction')})")
                return
            detail = {k: score.get(k) for k in ("t0", "t1", "class_wp", "train_seconds", "amp_skipped")}
            self.store.log("stage_result", trial=tid, fold=fold, wp=wp, detail=detail)
            ledger[tid]["stages"][fold].update(status="done", wp=wp, detail=detail)
            if fold not in self.confirm_folds:
                self._progress(ledger, tid)
            return
        note = result.get("note", "")
        self._failure(ledger, tid, fold, stage, classify_failure(note, result.get("rc")), note)

    def _failure(self, ledger: dict, tid: int, fold: int, stage: dict, kind: str, note: str) -> None:
        attempt = stage.get("attempts", 0)
        if attempt < MAX_ATTEMPTS[kind]:
            self.store.log("stage_retry", trial=tid, fold=fold, attempt=attempt + 1, failure=kind, note=note[:300])
            ledger[tid]["stages"][fold] = {"status": "pending", "attempts": attempt + 1}
            return
        self.store.log("stage_failed", trial=tid, fold=fold, failure=kind, note=note[:300])
        ledger[tid]["stages"][fold].update(status="failed", failure=kind)
        if fold in self.confirm_folds:  # a confirmation stage never changes the trial's search outcome
            alert(f"confirm_lost_t{tid}_f{fold}", "WARN", f"confirmation fold {fold} of trial {tid} lost ({kind})")
            return
        if kind == "numeric":
            penalty = self._penalty()
            self._tell(ledger, tid, "COMPLETE", penalty, f"numeric failure on fold {fold}: penalty")
        else:
            self._tell(ledger, tid, "FAIL", None, f"{kind} failure on fold {fold}: {note[:120]}")
            if kind == "invalid":
                alert(f"invalid_config_t{tid}", "WARN", f"trial {tid} config rejected by the trainer ({note[:160]}); likely a space/validation bug")
            else:
                alert(f"stage_lost_t{tid}", "WARN", f"trial {tid} fold {fold} abandoned after {attempt + 1} attempts ({kind})")

    def _penalty(self) -> float:
        values = [t.value for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        return (min(values) if values else 0.60) - 0.02

    def _progress(self, ledger: dict, tid: int) -> None:
        trial = ledger[tid]
        done_folds = [f for f in self.folds if trial["stages"].get(f, {}).get("status") == "done"]
        if not done_folds:
            return
        offsets = self.offsets(ledger)
        values = [trial["stages"][f]["wp"] - offsets.get(f, 0.0) for f in done_folds]  # fold-difficulty adjusted
        cumulative = sum(values) / len(values)
        step = len(done_folds) - 1
        otrial = self._otrial(tid)
        otrial.report(cumulative, step)
        if len(done_folds) == len(self.folds):
            self._tell(ledger, tid, "COMPLETE", cumulative, "all folds done")
            return
        prune = otrial.should_prune()
        if not prune and step >= 1:
            prune = not self._in_top_part(tid, step, cumulative)
        if prune:
            self._tell(ledger, tid, "PRUNED", cumulative, f"pruned after {len(done_folds)} folds at {cumulative:.5f}")
            return
        nxt = self.folds[len(done_folds)]
        if trial["stages"].get(nxt, {}).get("status") not in ("pending", "submitted", "done"):
            self.store.log("stage_pending", trial=tid, fold=nxt)
            trial["stages"][nxt] = {"status": "pending", "attempts": 0}

    def _otrial(self, tid: int):
        storage = self.study._storage  # private API: Trial() wants the storage id, events carry the trial number
        storage_id = storage.get_trial_id_from_study_id_trial_number(self.study._study_id, tid_to_number(self.study, tid))
        return optuna.trial.Trial(self.study, storage_id)

    def _in_top_part(self, tid: int, step: int, cumulative: float) -> bool:
        peers = [t.intermediate_values[step] for t in self.study.trials
                 if step in t.intermediate_values and t.number != tid_to_number(self.study, tid)]
        if len(peers) < MIN_PEERS:
            return True
        peers.sort(reverse=True)
        cutoff = peers[max(0, int(len(peers) * CONTINUE_PERCENTILE) - 1)]
        return cumulative >= cutoff

    def _tell(self, ledger: dict, tid: int, state: str, value: float | None, reason: str) -> None:
        number = tid_to_number(self.study, tid)
        if state == "COMPLETE":
            self.study.tell(number, value)
        elif state == "PRUNED":
            self.study.tell(number, state=optuna.trial.TrialState.PRUNED)
        else:
            self.study.tell(number, state=optuna.trial.TrialState.FAIL)
        self.store.log("tell", trial=tid, state=state, value=value, reason=reason)
        ledger[tid].update(state=state, value=value)

    # ---- dispatch ----------------------------------------------------------------------------------------------
    def dispatch(self, ledger: dict) -> None:
        inflight = {name: 0 for name in self.lanes}
        for trial in ledger.values():
            if trial["state"] == "running":
                for stage in trial["stages"].values():
                    if stage["status"] == "submitted":
                        inflight[stage["lane"]] = inflight.get(stage["lane"], 0) + 1
        pending = [(tid, fold, stage) for tid, trial in sorted(ledger.items())
                   for fold, stage in trial["stages"].items()
                   if stage["status"] == "pending" and (trial["state"] == "running" or fold in self.confirm_folds)]
        free = {name: lane.free_slots(inflight.get(name, 0)) for name, lane in self.lanes.items()}
        self._fill(ledger, pending, free)
        leftover = sum(1 for tid, fold, _ in pending if ledger[tid]["stages"][fold]["status"] == "pending")
        stop = (self.store.dir / "STOP").exists()
        asked = sum(1 for t in ledger.values() if t["state"] == "running")
        budget = self.cfg.get("max_trials", 10**9) - len(ledger)
        want = max(0, min(sum(free.values()) - leftover, self.cfg.get("max_active_trials", 12) - asked, budget))
        while want > 0 and not stop:
            tid = self._ask(ledger)
            pending = [(tid, self.folds[0], ledger[tid]["stages"][self.folds[0]])]
            self._fill(ledger, pending, free)
            want -= 1
        for lane in self.lanes.values():
            lane.flush()

    def _fill(self, ledger: dict, pending: list, free: dict) -> None:
        for tid, fold, stage in pending:
            params = ledger[tid]["params"]
            options = [n for n, lane in self.lanes.items() if free[n] > 0 and lane.can_run(fold, params)]
            if not options:
                continue
            lane_name = max(options, key=lambda n: free[n])
            attempt = stage.get("attempts", 0)
            name = self.job_name(tid, fold, attempt)
            job = self._job_builder()(params, fold, name)
            self.lanes[lane_name].submit(name, job, {"trial": tid, "fold": fold, "params": params})
            self.store.log("stage_submit", trial=tid, fold=fold, job=name, lane=lane_name, attempt=attempt)
            ledger[tid]["stages"][fold] = {"status": "submitted", "job": name, "lane": lane_name, "t": time.time(), "attempts": attempt}
            free[lane_name] -= 1

    def _ask(self, ledger: dict) -> int:
        mode = self.cfg.get("mode", "core")
        if not self.study.user_attrs.get("baselines_enqueued") and not self.study.trials:
            for override in space.BASELINES:
                self.study.enqueue_trial({**space.DEFAULTS, **override}, skip_if_exists=False)
            self.study.set_user_attr("baselines_enqueued", True)
        trial = self.study.ask()
        params = space.suggest(trial, mode)
        trial.set_constraint("cost_hours", space.constraint(params))
        tid = trial.number
        self.store.log("ask", trial=tid, params=params, mode=mode, space=space.SPACE_VERSION)
        ledger[tid] = {"params": params, "mode": mode, "stages": {}, "state": "running", "value": None, "asked": time.time()}
        self.store.log("stage_pending", trial=tid, fold=self.folds[0])
        ledger[tid]["stages"][self.folds[0]] = {"status": "pending", "attempts": 0}
        return tid

    # ---- tick --------------------------------------------------------------------------------------------------
    def tick(self) -> dict:
        ledger = self.store.ledger()
        for tid, trial in ledger.items():  # a crash between "ask" and "stage_pending" must not strand a trial
            if trial["state"] == "running" and not trial["stages"]:
                self.store.log("stage_pending", trial=tid, fold=self.folds[0])
                trial["stages"][self.folds[0]] = {"status": "pending", "attempts": 0}
        self.ingest(ledger)
        self.dispatch(ledger)
        KEEPALIVE.parent.mkdir(parents=True, exist_ok=True)
        KEEPALIVE.write_text(str(time.time()))
        return summarize(ledger)


def confirm_top(store: Store, cfg: dict, k: int) -> list[int]:
    """Schedule the reserved confirmation folds for the k best completed trials (no sampler feedback, one shot each)."""
    folds = tuple(cfg.get("confirm_folds", (2, 3)))
    ledger = store.ledger()
    done = sorted((t for t in ledger.items() if t[1]["state"] == "COMPLETE" and t[1]["value"] is not None),
                  key=lambda item: -item[1]["value"])[:k]
    chosen = []
    for tid, trial in done:
        fresh = [f for f in folds if f not in trial["stages"]]
        for fold in fresh:
            store.log("stage_pending", trial=tid, fold=fold, role="confirm")
        if fresh:
            chosen.append(tid)
    store.log("confirm_selected", trials=chosen, k=k)
    return chosen


def tid_to_number(study, tid: int) -> int:
    return tid  # events use Optuna's trial.number; kept explicit so a future remap has one place to change


def summarize(ledger: dict) -> dict:
    states: dict[str, int] = {}
    for t in ledger.values():
        states[t["state"]] = states.get(t["state"], 0) + 1
    done = [t for t in ledger.values() if t["state"] == "COMPLETE" and t["value"] is not None]
    best = max(done, key=lambda t: t["value"]) if done else None
    return {"trials": len(ledger), "states": states, "best_value": best["value"] if best else None,
            "best_params": best["params"] if best else None}


def _default_job_builder(params: dict, fold: int, name: str) -> dict:
    sys.path.insert(0, str(ROOT / "research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1"))
    import make_refill_jobs

    return make_refill_jobs.build_job(name, space.to_change(params, fold), "hpo")


# ---- CLI -------------------------------------------------------------------------------------------------------
def load_config(name: str) -> dict:
    return json.loads((STUDIES / name / "config.json").read_text())


def make_lanes(cfg: dict) -> dict:
    from hpo.lanes import VastLane

    lanes = {}
    for entry in cfg["lanes"]:
        lanes[f"vast_{entry['id']}"] = VastLane(entry["id"], entry.get("target", 6), entry.get("max_pending", 8),
                                                entry.get("spend_cap", 11.0), entry.get("folds", (0, 1, 2, 3)))
    return lanes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("tick", "status", "replay", "confirm"):
        node = sub.add_parser(name)
        node.add_argument("--study", required=True)
        if name == "confirm":
            node.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.study)
    store = Store(STUDIES, args.study, cfg.get("sampler", "tpe"), cfg.get("seed", 20261010))
    if args.cmd == "replay":
        print(json.dumps({"replayed": store.replay()}))
        return 0
    if args.cmd == "confirm":
        print(json.dumps({"scheduled": confirm_top(store, cfg, args.k)}))
        return 0
    if args.cmd == "status":
        print(json.dumps(summarize(store.ledger()), indent=1))
        return 0
    lock = (store.dir / "tick.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another tick is running")
        return 0
    if not store.study.trials and store.events_path.exists():
        store.replay()
    try:
        summary = Orchestrator(store, make_lanes(cfg), cfg).tick()
        print(json.dumps(summary))
    except Exception as error:  # noqa: BLE001 - recorded and alerted; the next timer tick retries
        store.log("tick_error", error=f"{type(error).__name__}: {error}", trace=traceback.format_exc()[-1500:])
        alert("tick_error", "WARN", f"hpo tick failed: {type(error).__name__}: {str(error)[:200]}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
