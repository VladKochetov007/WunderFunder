from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from wnn.multistage_search_successor_prepare import prepare_stage2_trial


def queue_stage2(
    run_dir: Path, parent_results: Sequence[Path], multipliers: Sequence[float]
) -> list[dict[str, object]]:
    """Queue explicit LR continuations for qualified fixed-final Stage1 results."""
    if not parent_results or len(parent_results) != len(multipliers):
        raise ValueError("supply one explicit LR multiplier for each Stage1 parent result")
    return [
        prepare_stage2_trial(run_dir, result, lr_multiplier=multiplier)
        for result, multiplier in zip(parent_results, multipliers, strict=True)
    ]
