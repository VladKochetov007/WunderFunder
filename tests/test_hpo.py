import math
import shutil

import optuna

from hpo import orchestrator as orch
from hpo import space
from hpo.lanes import FakeLane
from hpo.store import Store

FOLD_OFFSET = {1: 0.0, 0: 0.013, 2: 0.004, 3: -0.003}


def objective(params, fold):
    # smooth bowl in log-lr and muon_mult plus a small penalty for bad choices; deterministic
    score = 0.66 - 0.004 * (math.log10(params["lr"]) + 3.0) ** 2 - 0.002 * (math.log2(params["muon_mult"]) - 1.5) ** 2
    score -= 0.003 * (params["loss"] == "corr") + 0.002 * (params["optim"] == "muon_hidden")
    score += 0.0005 * (params["avg"] == "ema")
    return score + FOLD_OFFSET[fold] + 0.0004 * math.sin(7 * params["lr"] * 1000 + fold)


def fake_job(params, fold, name):
    return {"name": name, "steps": []}


def make(tmp_path, lanes=None, **config):
    cfg = {"mode": "core", "lanes": [], "max_active_trials": 8, "job_builder": fake_job, **config}
    store = Store(tmp_path, "t", "tpe", seed=3)
    return store, orch.Orchestrator(store, lanes or {"fake": FakeLane("fake", objective, slots=4)}, cfg)


def run_ticks(o, n):
    for _ in range(n):
        summary = o.tick()
    return summary


def test_space_maps_to_trainer_flags():
    study = optuna.create_study()
    trial = study.ask()
    params = space.suggest(trial, "full")
    change = space.to_change(params, 1)
    assert change["--checkpoint-epoch"] == str(params["epochs"] - 1)
    assert change["fold"] == 1 and change["--optim"] == "muon"
    assert space.estimate_hours(space.DEFAULTS) == 1.2
    assert space.estimate_hours({**space.DEFAULTS, "hidden": 448, "layers": 4}) > space.COST_CAP_HOURS


def test_search_progresses_prunes_and_improves(tmp_path):
    store, o = make(tmp_path)
    summary = run_ticks(o, 60)
    states = summary["states"]
    assert states.get("COMPLETE", 0) + states.get("PRUNED", 0) >= 20
    assert states.get("PRUNED", 0) > 0, "median / percentile pruning never triggered"
    ledger = store.ledger()
    first = [t["stages"][1]["wp"] for t in ledger.values() if t["stages"].get(1, {}).get("status") == "done"]
    assert max(first) > 0.655
    best = summary["best_value"]
    assert best is not None and best > sorted(first)[len(first) // 2]


def test_baselines_are_enqueued_first_and_once(tmp_path):
    store, o = make(tmp_path)
    run_ticks(o, 3)
    params = [t["params"] for _, t in sorted(store.ledger().items())][:3]
    assert params[0]["muon_mult"] == 1.0 and params[1]["muon_mult"] == 3.0
    store2, o2 = make(tmp_path)  # a fresh process must not enqueue the baselines again
    run_ticks(o2, 3)
    assert sum(1 for t in store2.ledger().values() if t["params"]["muon_mult"] == 3.0 and t["params"]["avg"] == "none") <= 3


def test_failure_policy(tmp_path):
    failures = {(0, 1): {"note": "CUDA out of memory", "times": 1},   # infra-like, succeeds on the retry
                (1, 1): {"note": "train_rnn.py: error: unrecognized arguments: --bogus", "times": 99},  # invalid -> skip
                (2, 1): {"note": "loss is nan at step 120", "times": 99}}                           # numeric -> penalty
    lane = FakeLane("fake", objective, slots=6, failures=failures)
    store, o = make(tmp_path, {"fake": lane}, max_active_trials=3)
    run_ticks(o, 12)
    ledger = store.ledger()
    assert ledger[0]["state"] != "FAIL" and ledger[0]["stages"][1]["attempts"] == 1
    assert ledger[1]["state"] == "FAIL"
    assert ledger[2]["state"] == "COMPLETE" and ledger[2]["value"] < 0.65, "numeric failure must become a penalty"
    kinds = [e.get("failure") for e in store.events() if e["ev"] in ("stage_retry", "stage_failed")]
    assert "oom" in kinds and "invalid" in kinds and "numeric" in kinds


def test_crash_recovery_from_event_log(tmp_path):
    store, o = make(tmp_path)
    run_ticks(o, 25)
    before = store.ledger()
    shutil.rmtree(tmp_path / "t" / "study.db", ignore_errors=True)
    (tmp_path / "t" / "study.db").unlink(missing_ok=True)
    store2 = Store(tmp_path, "t", "tpe", seed=3)  # database gone: rebuild the sampler history from the log
    assert store2.replay() == len(before)
    complete = [t for t in store2.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(complete) == sum(1 for t in before.values() if t["state"] == "COMPLETE")
    o2 = orch.Orchestrator(store2, {"fake": FakeLane("fake", objective, slots=4)},
                           {"mode": "core", "lanes": [], "max_active_trials": 8, "job_builder": fake_job})
    o2.lanes["fake"].jobs = {}  # the lane lost its memory too: in-flight stages must be recognised as lost and retried
    run_ticks(o2, 10)
    after = store2.ledger()
    assert len(after) >= len(before)
    assert all(t["state"] in ("COMPLETE", "PRUNED", "FAIL", "running") for t in after.values())


def test_stop_file_stops_new_trials(tmp_path):
    store, o = make(tmp_path)
    run_ticks(o, 4)
    count = len(store.ledger())
    (tmp_path / "t" / "STOP").write_text("")
    run_ticks(o, 5)
    assert len(store.ledger()) == count
