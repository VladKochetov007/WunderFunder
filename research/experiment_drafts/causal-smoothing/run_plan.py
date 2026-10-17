from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from experiment_draft_plan import build_plan  # noqa: E402


def command(phase: str, arm: str, fold: int, seed: int) -> str:
    """Render a portable, nonexecuting cell command for the receiving checkout."""
    return shlex.join(
        [
            "python",
            "research/experiment_drafts/causal-smoothing/run_cell.py",
            phase,
            "--arm",
            arm,
            "--fold",
            str(fold),
            "--seed",
            str(seed),
        ]
    )


def main() -> None:
    """Print all cells and commands; no datasets, imports of ML, or GPU work."""
    recipe = json.loads((PACKET / "recipe.json").read_text())
    plan = build_plan(recipe)
    plan["profiles"] = [command("profile", arm, 0, 1001) for arm in recipe["arms"]]
    plan["commands"] = [
        {
            "arm": arm,
            **cell,
            "stages": {
                phase: command(phase, arm, cell["fold"], cell["seed"])
                for phase in ("fit", "predict", "score")
            },
        }
        for cell in plan["cells"]
        for arm in recipe["arms"]
    ]
    plan["execution"] = (
        "Commands only plan; append --execute on the receiving host after checks. No scheduler."
    )
    print(json.dumps(plan, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
