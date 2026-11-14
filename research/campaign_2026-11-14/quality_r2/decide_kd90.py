"""Apply the frozen two-seed alpha-.9 final-refit admission rule."""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent


def endpoint(path: Path) -> float:
    """Require the predeclared endpoint and read its last official validation score."""
    text = path.read_text()
    if "SELECTED_FIXED 3" not in text:
        raise ValueError(f"Missing completed fixed epoch 3: {path}")
    scores = re.findall(r'"weighted_pearson": ([0-9.]+)', text)
    if len(scores) != 4:
        raise ValueError(f"Expected four validation epochs: {path}")
    return float(scores[-1])


def main() -> None:
    """Save all paired deltas, then stop the final-refit path unless admitted."""
    old = HERE.parent / "local_r1"
    controls = [endpoint(old / f"l14_kd50_s{s}.fit.log") for s in (3, 4)]
    candidates = [endpoint(old / "l14_kd90_s3.fit.log"), endpoint(HERE / "q14_kd90_s4.fit.log")]
    deltas = [candidate - control for candidate, control in zip(candidates, controls, strict=True)]
    mean = sum(deltas) / 2
    admitted = min(deltas) > 0 and mean >= 0.0005
    result = {
        "controls": controls,
        "candidates": candidates,
        "paired_deltas": deltas,
        "mean_delta": mean,
        "final_refits_admitted": admitted,
        "evidence_scope": "two fitted seeds, repeatedly exposed official valid; no public claim",
    }
    (HERE / "KD90_DECISION_R1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not admitted:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
