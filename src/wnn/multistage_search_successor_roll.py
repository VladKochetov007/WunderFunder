from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from wnn.multistage_search_successor_supervisor import supervise_successors


def prepare_and_supervise_stage2(
    run_dir: Path,
    parent_results: Sequence[Path],
    *,
    worker_prefix: str = "stage2",
    max_workers: int = 6,
    poll_seconds: float = 5.0,
    max_runtime_seconds: int = 21600,
    lr_multiplier: float | None = None,
) -> dict[str, object]:
    """Prepare parent continuations, then roll reviewed profiles into bounded fits."""
    from wnn.multistage_search_successor_prepare import prepare_stage2_trial

    if not parent_results:
        raise ValueError("stage2 rolling launch requires at least one qualified parent result")
    prepared = [
        prepare_stage2_trial(run_dir, path, lr_multiplier=lr_multiplier) for path in parent_results
    ]
    supervised = supervise_successors(
        run_dir,
        stage=2,
        worker_prefix=worker_prefix,
        max_workers=max_workers,
        poll_seconds=poll_seconds,
        max_runtime_seconds=max_runtime_seconds,
    )
    return {"prepared_trials": prepared, **supervised}


def prepare_and_supervise_stage3(
    run_dir: Path,
    base_config_path: Path,
    teacher_cache_path: Path,
    *,
    family: str,
    pair_count: int,
    worker_prefix: str = "stage3",
    max_workers: int = 6,
    poll_seconds: float = 5.0,
    max_runtime_seconds: int = 21600,
) -> dict[str, object]:
    """Prepare matched label/KD pairs, then process reviewed profiles and bounded fits."""
    from wnn.multistage_search_successor_prepare import prepare_stage3_pair

    if not 1 <= pair_count <= 6:
        raise ValueError("stage3 has at most six matched two-fit pairs under its twelve-fit cap")
    prepared = [
        prepare_stage3_pair(run_dir, base_config_path, teacher_cache_path, family=family)
        for _ in range(pair_count)
    ]
    supervised = supervise_successors(
        run_dir,
        stage=3,
        worker_prefix=worker_prefix,
        max_workers=max_workers,
        poll_seconds=poll_seconds,
        max_runtime_seconds=max_runtime_seconds,
    )
    return {"prepared_pairs": prepared, **supervised}
