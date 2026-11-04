from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from wnn.multistage_search import append_event, read_json
from wnn.multistage_search_budget import utc_now
from wnn.multistage_search_ensemble_math import score_ensemble
from wnn.splits import CampaignSplit


def search_wp_weights(
    run_dir: Path,
    data_root: Path,
    split: CampaignSplit,
    prediction_refs: Sequence[Mapping[str, object]],
    candidate_names: Sequence[str],
    warmstart_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, object], int]:
    """Run bounded durable Optuna TPE on the actual clipped pooled WP blend objective."""
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError("stage 4 requires the private research Optuna environment") from error
    admission = read_json(Path(str(read_json(run_dir / "run.json")["admission"]["path"])))
    stage4 = admission.get("stage4", {})
    if not isinstance(stage4, Mapping):
        raise ValueError("stage 4 admission must be an object")
    max_trials = stage4.get("optuna_trials", 24)
    max_trial_wall = stage4.get("max_trial_wall_seconds", 600)
    max_search_wall = stage4.get("max_cpu_wall_seconds", 1800)
    if type(max_trials) is not int or not 8 <= max_trials <= 32:
        raise ValueError("stage 4 Optuna trial cap must be between 8 and 32")
    if type(max_trial_wall) is not int or not 1 <= max_trial_wall <= 1800:
        raise ValueError("stage 4 per-trial CPU wall cap is invalid")
    if type(max_search_wall) is not int or not 1 <= max_search_wall <= 3600:
        raise ValueError("stage 4 CPU search wall cap is invalid")
    deadline = read_json(run_dir / "campaign_deadline.json")
    reserve_start = deadline.get("reserve_start_epoch")
    if not isinstance(reserve_start, (int, float)):
        raise ValueError("stage 4 requires a started campaign deadline")
    start = time.time()
    stop_epoch = min(float(reserve_start), start + max_search_wall)
    if start >= stop_epoch:
        raise RuntimeError("stage 4 CPU search is closed by the collection reserve")
    manifest = read_json(run_dir / "run.json")
    storage_path = run_dir / "stage4" / "optuna.db"
    storage_url = f"sqlite:///{storage_path.as_posix()}"
    study = optuna.create_study(
        study_name=f"{manifest['experiment_id']}-stage4-wp",
        storage=storage_url,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=int(manifest["sampler_seed"]) + 4,
            n_startup_trials=max(5, min(8, max_trials // 3)),
            multivariate=True,
            group=True,
            constant_liar=False,
        ),
        load_if_exists=True,
    )
    if len(study.trials) > max_trials:
        raise RuntimeError("persisted stage 4 study already exceeds its frozen trial cap")
    warmstart_recorded = any(
        isinstance(trial.user_attrs, dict)
        and trial.user_attrs.get("search_role") == "convex_weighted_mse_warmstart"
        for trial in study.trials
    )
    if not warmstart_recorded:
        warmstart_params = _weight_logits(warmstart_weights)
        study.enqueue_trial(
            warmstart_params,
            user_attrs={"search_role": "convex_weighted_mse_warmstart"},
        )

    def objective(trial: object) -> float:
        trial_start = time.time()
        attrs = getattr(trial, "user_attrs", {})
        if isinstance(attrs, Mapping) and attrs.get("search_role") == "convex_weighted_mse_warmstart":
            weights = warmstart_weights.copy()
        else:
            logits = [
                trial.suggest_float(f"logit_{index}", -8.0, 8.0)
                for index in range(len(candidate_names))
            ]
            weights = _softmax(logits)
        trial.set_user_attr("weights", weights.tolist())
        trial_deadline = min(stop_epoch, trial_start + max_trial_wall)
        if trial_deadline - trial_start < 1.0:
            raise optuna.TrialPruned("campaign reserve reached before a complete WP pass")
        try:
            score = score_ensemble(
                data_root, split, prediction_refs, weights, deadline_epoch=trial_deadline
            )
        except TimeoutError as error:
            raise optuna.TrialPruned(str(error)) from error
        trial.set_user_attr("control_score", score)
        trial.set_user_attr("duration_seconds", time.time() - trial_start)
        return float(score["wp"])

    def record_trial(study_object: object, trial: object) -> None:
        state = getattr(getattr(trial, "state", None), "name", "UNKNOWN")
        append_event(
            run_dir / "events.jsonl",
            {
                "event": "stage4_optuna_trial",
                "at_utc": utc_now(),
                "stage": 4,
                "trial_number": int(trial.number),
                "status": state,
                "objective_value": trial.value,
                "duration_seconds": trial.user_attrs.get("duration_seconds"),
                "candidate_weights": trial.user_attrs.get("weights"),
                "search_role": trial.user_attrs.get("search_role", "tpe_wp_trial"),
            },
        )
        from wnn.multistage_plot_ensemble import export_stage4_trial_plots

        export_stage4_trial_plots(run_dir)

    remaining_trials = max_trials - len(study.trials)
    remaining_wall = max(1, int(stop_epoch - time.time()))
    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=remaining_wall,
            n_jobs=1,
            callbacks=[record_trial],
            show_progress_bar=False,
        )
    complete = [trial for trial in study.trials if trial.value is not None]
    if not complete:
        raise RuntimeError("stage 4 Optuna study has no complete actual-WP trials")
    best = study.best_trial
    best_weights = np.asarray(best.user_attrs.get("weights"), dtype=np.float64)
    if best_weights.shape != warmstart_weights.shape or not np.isfinite(best_weights).all():
        raise ValueError("best stage 4 trial does not retain a valid simplex weight vector")
    score = best.user_attrs.get("control_score")
    if not isinstance(score, dict) or abs(float(score.get("wp")) - float(best.value)) > 1e-12:
        raise ValueError("best Optuna objective does not match its pooled clipped WP receipt")
    if abs(float(best_weights.sum()) - 1.0) > 1e-10 or (best_weights < 0).any():
        raise ValueError("best Optuna blend violates nonnegative simplex constraints")
    return best_weights, score, len(study.trials)


def _weight_logits(weights: np.ndarray) -> dict[str, float]:
    clipped = np.clip(weights, 1e-8, 1.0)
    logs = np.log(clipped)
    center = float(logs.mean())
    return {f"logit_{index}": float(np.clip(value - center, -8.0, 8.0))
            for index, value in enumerate(logs)}


def _softmax(logits: Sequence[float]) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - float(values.max())
    exponentials = np.exp(shifted)
    total = float(exponentials.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("TPE logits cannot be normalized into a finite simplex")
    return exponentials / total
