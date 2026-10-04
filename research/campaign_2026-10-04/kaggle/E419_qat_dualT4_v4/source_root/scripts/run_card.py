"""Write a research run card: args, per-epoch metrics, snapshots, and fold verdicts found in queue logs.

    .venv/bin/python scripts/run_card.py RUN [RUN ...]
Writes research/runs/<RUN>.md (one card per run; re-running refreshes it).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def card(run: str) -> str:
    rd = ROOT / "artifacts" / "runs" / run
    log = ROOT / "artifacts" / "logs" / f"{run}.log"
    args = {}
    if log.exists():
        first = log.read_text().splitlines()[0]
        try:
            args = json.loads(first)
        except json.JSONDecodeError:
            pass
    hist = json.loads((rd / "history.json").read_text()) if (rd / "history.json").exists() else []
    snaps = sorted(p.name for p in rd.glob("model_ep*.pt"))
    fam = run.rsplit("_f", 1)[0]
    verdicts = []
    for q in sorted((ROOT / "artifacts" / "logs").glob("q*.out")):
        verdicts += [ln for ln in q.read_text().splitlines() if ln.startswith(fam + " - ")]
    keys = ["name", "train_source", "valid_fold", "eval_source", "seed", "epochs", "lr", "wd", "enc", "cell", "propensity",
            "prop_eps", "valid_true_mask", "ema_teacher", "ema_decay", "eval_ema", "td_alpha", "loss", "loss_norm",
            "since_weight", "fast", "init", "freeze", "mask_weight"]
    out = [f"# Run card: {run}", "", "## Config", ""]
    out += [f"- {k}: {args[k]}" for k in keys if k in args]
    out += ["", "## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)", "",
            "| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |", "|---|---|---|---|---|---|---|---|---|"]
    for h in hist:
        bc = h.get("by_class", {})
        out.append(f"| {h.get('epoch')} | {h.get('weighted_pearson', float('nan')):.4f} | {h.get('global_t0', float('nan')):.4f} | "
                   f"{h.get('global_t1', float('nan')):.4f} | " + " | ".join(str(bc.get(str(c), bc.get(c, ''))) for c in range(4)) +
                   f" | {h.get('split', '')} |")
    out += ["", f"Snapshots: {', '.join(snaps) if snaps else 'model.pt only (run predates per-epoch snapshots)'}", "",
            "## Fold verdicts (from queue logs)", ""]
    out += [f"- `{v}`" for v in verdicts] or ["- none yet"]
    return "\n".join(out) + "\n"


def main():
    d = ROOT / "research" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    for run in sys.argv[1:]:
        (d / f"{run}.md").write_text(card(run))
        print("wrote", d / f"{run}.md")


if __name__ == "__main__":
    main()
