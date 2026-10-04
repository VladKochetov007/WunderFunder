"""Regenerate research/runs/INDEX.md from artifacts/runs/*, artifacts/logs/*, and research/experiments.jsonl.

    scripts/limited .venv/bin/python scripts/research_index.py

Rebuilds the run cards under research/runs/<run>.md (via scripts/run_card.py's card()
function) for every run directory under artifacts/runs/ that has a model.pt and whose
name does not start with smoke, parity, perf_ladder, or tput, then writes a single
INDEX.md with:
  - a public-leaderboard history table sourced from research/experiments.jsonl
    (entries with kind == "public leaderboard")
  - one row per run: family, fold/eval split, seed, epochs, key recipe flags,
    best/final-epoch WP, #snapshots, ledger E-ids that mention the run (via ripgrep),
    and public LB score for runs that are submission members.

Re-run after each new wave of training runs to refresh the index.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "artifacts" / "runs"
LOGS_DIR = ROOT / "artifacts" / "logs"
RESEARCH_RUNS = ROOT / "research" / "runs"
LEDGER = ROOT / "research" / "experiments.jsonl"

SKIP_PREFIXES = ("smoke", "parity", "perf_ladder", "tput")

# run (submission member) -> public LB score, per the ledger-label -> run mapping:
#   final_ema4_lpft_base35 -> ab60_final_ema4_p1_lpft   0.6618
#   final_s4lpft_base35    -> ab51_final_lpft            0.6601
#   final_s4ft_base35      -> ab40_final_s4_ft           0.6600
#   final_s4_base35        -> ab33_final_s4              0.6586
#   blend_base_final_tv / final_tv_e2 -> ab21 (0.6569 / 0.6548)
#   ab18_wmse_struct                  -> 0.6488
LB_MAP = {
    "ab60_final_ema4_p1_lpft": 0.6618,
    "ab51_final_lpft": 0.6601,
    "ab40_final_s4_ft": 0.6600,
    "ab33_final_s4": 0.6586,
    "ab21_final_tv_e2": 0.6548,  # blend_base_final_tv variant scored 0.6569 (not a standalone run dir)
    "ab18_wmse_struct": 0.6488,
}

RECIPE_KEYS = ["enc", "propensity", "ema_teacher", "td_alpha", "loss_norm", "since_weight", "init", "freeze"]


def list_runs() -> list[str]:
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "model.pt").exists():
            continue
        if d.name.startswith(SKIP_PREFIXES):
            continue
        runs.append(d.name)
    return runs


def load_args(run: str) -> dict:
    log = LOGS_DIR / f"{run}.log"
    if not log.exists():
        return {}
    lines = log.read_text().splitlines()
    if not lines:
        return {}
    try:
        return json.loads(lines[0])
    except json.JSONDecodeError:
        return {}


def load_history(run: str) -> list[dict]:
    hist_path = RUNS_DIR / run / "history.json"
    if not hist_path.exists():
        return []
    try:
        return json.loads(hist_path.read_text())
    except json.JSONDecodeError:
        return []


def family_of(run: str) -> str:
    m = re.match(r"^(.*)_f\d+$", run)
    return m.group(1) if m else run


def snapshot_count(run: str) -> str:
    snaps = sorted((RUNS_DIR / run).glob("model_ep*.pt"))
    if snaps:
        return str(len(snaps))
    return "1 (model.pt only)"


def ledger_ids(run: str) -> str:
    if not LEDGER.exists():
        return ""
    try:
        out = subprocess.run(
            ["rg", "-n", "-F", run, str(LEDGER)],
            capture_output=True, text=True, check=False,
        ).stdout
    except FileNotFoundError:
        return ""
    ids = []
    for line in out.splitlines():
        m = re.search(r'"id":\s*"(E-\d+)"', line)
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    return ", ".join(ids)


def fmt(v) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else str(v)
    return str(v)


def build_run_row(run: str) -> dict:
    args = load_args(run)
    hist = load_history(run)
    fam = family_of(run)

    fold = args.get("valid_fold", "")
    eval_source = args.get("eval_source", "")
    split_label = "/".join(str(x) for x in (fold, eval_source) if x not in ("", None))
    if not split_label and hist:
        split_label = hist[-1].get("split", "")

    seed = args.get("seed", "")
    epochs = args.get("epochs", "")

    flags = []
    for k in RECIPE_KEYS:
        v = args.get(k)
        if v not in (None, "", 0, 0.0, False):
            flags.append(f"{k}={v}")
    flags_str = "; ".join(flags)

    wps = [h.get("weighted_pearson") for h in hist if h.get("weighted_pearson") is not None]
    best_wp = max(wps) if wps else None
    final_wp = hist[-1].get("weighted_pearson") if hist else None

    lb_val = LB_MAP.get(run)
    lb_str = fmt(lb_val) if lb_val is not None else ""

    return {
        "run": run,
        "family": fam,
        "fold_split": split_label,
        "seed": fmt(seed),
        "epochs": fmt(epochs),
        "flags": flags_str,
        "best_wp": fmt(best_wp),
        "final_wp": fmt(final_wp),
        "snapshots": snapshot_count(run),
        "ledger": ledger_ids(run),
        "lb": lb_str,
    }


# generic/meta keys in "public leaderboard" ledger results that are not themselves
# a scored submission (ceilings, ranks, timings, predictions).
_LB_EXCLUDE_KEYS = {
    "lb_top", "rank", "scorer_min", "scorer_duration_min", "predicted", "actual",
    "previous_best", "predicted_delta", "predicted_blend_holdout_delta", "actual_delta",
    "top", "baseline_raw_lb_public",
}


def lb_history_table() -> list[str]:
    if not LEDGER.exists():
        return []
    entries = []
    for line in LEDGER.read_text().splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("kind") == "public leaderboard":
            entries.append(obj)

    out = ["## LB history", "",
           "| E-id | date | submission | LB | delta | lever |",
           "|---|---|---|---|---|---|"]
    running_best = 0.5896  # baseline WP (research/wunder-connectome-project memory)
    for e in entries:
        res = e.get("result", {})
        candidates = []
        for k, v in res.items():
            if k in _LB_EXCLUDE_KEYS:
                continue
            if isinstance(v, dict) and "lb" in v:
                candidates.append((k, v["lb"]))
            elif isinstance(v, (int, float)) and 0.0 <= v <= 1.0:
                candidates.append((k, v))
        lever = (e.get("claim", "") or "")[:140]
        if not candidates:
            out.append(f"| {e.get('id','')} | {e.get('date','')} |  |  |  | {lever} |")
            continue
        for name, val in candidates:
            delta = f"{val - running_best:+.4f}"
            running_best = max(running_best, val)
            out.append(f"| {e.get('id','')} | {e.get('date','')} | {name} | {fmt(val)} | {delta} | {lever} |")
    out.append("")
    return out


def main():
    runs = list_runs()
    run_card_mod_path = ROOT / "scripts" / "run_card.py"
    sys.path.insert(0, str(run_card_mod_path.parent))
    import run_card  # noqa: E402

    RESEARCH_RUNS.mkdir(parents=True, exist_ok=True)
    for run in runs:
        try:
            (RESEARCH_RUNS / f"{run}.md").write_text(run_card.card(run))
        except IndexError:
            # empty log file (predates JSON-line logging); write a minimal card.
            hist = load_history(run)
            out = [f"# Run card: {run}", "", "## Config", "", "(log file empty; args unavailable)", "",
                   "## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)", "",
                   "| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |", "|---|---|---|---|---|---|---|---|---|"]
            for h in hist:
                bc = h.get("by_class", {})
                out.append(f"| {h.get('epoch')} | {h.get('weighted_pearson', float('nan')):.4f} | {h.get('global_t0', float('nan')):.4f} | "
                           f"{h.get('global_t1', float('nan')):.4f} | " + " | ".join(str(bc.get(str(c), bc.get(c, ''))) for c in range(4)) +
                           f" | {h.get('split', '')} |")
            snaps = sorted(p.name for p in (RUNS_DIR / run).glob("model_ep*.pt"))
            out += ["", f"Snapshots: {', '.join(snaps) if snaps else 'model.pt only (run predates per-epoch snapshots)'}", "",
                    "## Fold verdicts (from queue logs)", "", "- none yet"]
            (RESEARCH_RUNS / f"{run}.md").write_text("\n".join(out) + "\n")

    rows = [build_run_row(r) for r in runs]
    rows.sort(key=lambda r: (r["family"], r["fold_split"]))

    lines = ["# Research runs index", "",
             "Regenerated by `scripts/research_index.py`. One row per run under "
             "`artifacts/runs/` with a `model.pt` (excludes smoke/parity/perf_ladder/tput probes).",
             ""]
    lines += lb_history_table()
    lines += ["## Runs", "",
              "| run | family | fold/eval split | seed | epochs | key recipe flags | best WP | final-epoch WP | #snapshots | ledger ids | public LB |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| [{r['run']}]({r['run']}.md) | {r['family']} | {r['fold_split']} | {r['seed']} | {r['epochs']} | "
            f"{r['flags']} | {r['best_wp']} | {r['final_wp']} | {r['snapshots']} | {r['ledger']} | {r['lb']} |"
        )
    lines.append("")

    (RESEARCH_RUNS / "INDEX.md").write_text("\n".join(lines))
    print(f"wrote {RESEARCH_RUNS / 'INDEX.md'} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
