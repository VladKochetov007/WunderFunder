from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from wnn.multistage_search import read_json


class StudyHandle(Protocol):
    """Optuna storage operations used by the bounded controller."""

    trials: list[object]

    def ask(self) -> object: ...

    def enqueue_trial(self, params: Mapping[str, object], **kwargs: object) -> None: ...

    def get_trials(self, deepcopy: bool = True) -> list[object]: ...

    def tell(self, trial: object, value: float | None = None, **kwargs: object) -> None: ...


def open_study(run_dir: Path, worker_id: str, *, stage: int = 1) -> StudyHandle:
    """Open the durable SQLite TPE study with a worker-specific reproducible seed."""
    if stage not in {1, 2, 3}:
        raise ValueError("fitted Optuna studies support stages one through three")
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError(
            "Optuna must be installed in the private research environment"
        ) from error
    manifest = read_json(run_dir / "run.json")
    experiment_id, seed = manifest.get("experiment_id"), manifest.get("sampler_seed")
    if not isinstance(experiment_id, str) or type(seed) is not int:
        raise ValueError("run manifest is missing the Optuna study identity")
    study_path = run_dir / "study.db"
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{study_path.as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    # The sampler seed varies with this worker and the durable trial count so fresh processes
    # do not repeat the same first TPE proposal.
    name_digest = hashlib.sha256(worker_id.encode("utf-8")).hexdigest()
    sampler_seed = seed + stage * 100_003 + int(name_digest[:8], 16)
    sampler = optuna.samplers.TPESampler(
        seed=sampler_seed,
        multivariate=True,
        group=True,
        constant_liar=True,
        n_startup_trials=int(manifest["tpe_startup_trials"]),
    )
    study_name = experiment_id if stage == 1 else f"{experiment_id}-stage{stage}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    trial_count = len(study.trials)
    if trial_count:
        study.sampler = optuna.samplers.TPESampler(
            seed=sampler_seed + trial_count,
            multivariate=True,
            group=True,
            constant_liar=True,
            n_startup_trials=int(manifest["tpe_startup_trials"]),
        )
    return study


def enqueue_anchors(study: StudyHandle, anchors: object, families: Mapping[str, object]) -> int:
    """Queue each exact reviewed warm-start parameter set once at run initialization."""
    if not isinstance(anchors, list):
        raise ValueError("stage1 anchors must be a list")
    count = 0
    seen: set[str] = set()
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            raise ValueError("each stage1 anchor must be an object")
        family, params = anchor.get("family"), anchor.get("params")
        anchor_id = anchor.get("id", anchor.get("anchor_id"))
        if not isinstance(family, str) or family not in families:
            raise ValueError("stage1 anchor names an unadmitted family")
        if not isinstance(params, Mapping) or not isinstance(anchor_id, str) or not anchor_id:
            raise ValueError("stage1 anchor needs id and parameter mapping")
        signature = repr((family, sorted(params.items())))
        if signature in seen:
            raise ValueError("duplicate stage1 anchor")
        seen.add(signature)
        flat = {"family": family} | {f"{family}.{name}": value for name, value in params.items()}
        study.enqueue_trial(flat, user_attrs={"anchor_id": anchor_id})
        count += 1
    return count
