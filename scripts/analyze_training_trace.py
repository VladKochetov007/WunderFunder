"""Plot and summarize training telemetry written by ``train_rnn.py --trace-every``.

For each run directory: loss curve (raw + EMA), learning rate and beta1, pre-clip gradient norms by
parameter family with the clip level, weight-update ratios, second-moment RMS, and the intra-epoch /
epoch-end evaluation curve from ``intra_eval.jsonl`` + ``history.json``.  Several runs overlay on the
same axes.  Prints a compact JSON summary (also written next to the plots).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_rows(run: Path) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines() if line.strip()]
    return [r for r in rows if r["kind"] == "step"], [r for r in rows if r["kind"] == "state"]


def eval_curve(run: Path, steps_per_epoch: int) -> tuple[list[int], list[float]]:
    points: list[tuple[int, float]] = []
    path = run / "intra_eval.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("weighted_pearson") is not None:
                points.append((row["step"], row["weighted_pearson"]))
    history = json.loads((run / "history.json").read_text())
    for row in history:
        if row.get("weighted_pearson") is not None:
            points.append(((row["epoch"] + 1) * steps_per_epoch, row["weighted_pearson"]))
    points.sort()
    return [p[0] for p in points], [p[1] for p in points]


def ema(values: np.ndarray, beta: float = 0.98) -> np.ndarray:
    out, running = np.empty_like(values), values[0]
    for i, value in enumerate(values):
        running = beta * running + (1 - beta) * value
        out[i] = running
    return out


def summarize(name: str, steps: list[dict], states: list[dict], curve: tuple[list[int], list[float]]) -> dict:
    loss = np.array([r["loss"] for r in steps if r.get("loss") is not None])
    gn = np.array([r["gn_total"] for r in steps if r.get("gn_total") is not None])
    summary = {
        "run": name, "trace_rows": len(steps), "state_rows": len(states),
        "loss_first_last_ema": [float(ema(loss)[0]), float(ema(loss)[-1])] if len(loss) else None,
        "grad_norm_median_p90_max": [float(np.median(gn)), float(np.quantile(gn, 0.9)), float(gn.max())] if len(gn) else None,
        "frac_grad_norm_above_1": float((gn > 1.0).mean()) if len(gn) else None,
        "eval_curve": [[s, round(v, 5)] for s, v in zip(*curve, strict=True)],
    }
    if states:
        last = states[-1]
        summary["last_update_ratio"] = {k[3:]: v for k, v in last.items() if k.startswith("ur_")}
        summary["last_weight_norm"] = {k[2:]: v for k, v in last.items() if k.startswith("w_")}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run names under artifacts/runs")
    parser.add_argument("--out", type=Path, required=True, help="directory for PNG and JSON output")
    parser.add_argument("--steps-per-epoch", type=int, default=7460)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 3, figsize=(18, 12))
    summaries = []
    for name in args.runs:
        run = ROOT / "artifacts" / "runs" / name
        steps, states = load_rows(run)
        curve = eval_curve(run, args.steps_per_epoch)
        summaries.append(summarize(name, steps, states, curve))
        x = np.array([r["step"] for r in steps])
        axes[0, 0].plot(x, ema(np.array([r["loss"] for r in steps])), label=name)
        axes[0, 1].plot(x, [r["lr"][0] for r in steps], label=name)
        axes[0, 2].plot(x, [r["beta1"][0] if r["beta1"][0] is not None else np.nan for r in steps], label=name)
        axes[1, 0].plot(x, ema(np.array([r["gn_total"] for r in steps])), label=name)
        families = [k[3:] for k in steps[0] if k.startswith("gn_") and k != "gn_total"]
        for family in families:
            axes[1, 1].plot(x, ema(np.array([r[f"gn_{family}"] for r in steps])), label=f"{name}:{family}")
        if states:
            sx = np.array([r["step"] for r in states])
            for key in (k for k in states[-1] if k.startswith("ur_")):
                axes[1, 2].plot(sx, [r.get(key) for r in states], label=f"{name}:{key[3:]}")
            for key in (k for k in states[-1] if k.startswith("v_rms_")):
                axes[2, 0].plot(sx, [r.get(key) for r in states], label=f"{name}:{key[6:]}")
            for key in (k for k in states[-1] if k.startswith("w_")):
                axes[2, 1].plot(sx, [r.get(key) for r in states], label=f"{name}:{key[2:]}")
        if curve[0]:
            axes[2, 2].plot(*curve, marker="o", label=name)
    titles = ["loss (EMA)", "learning rate", "beta1 / momentum", "pre-clip grad norm (EMA)",
              "grad norm by family (EMA)", "update ratio ||dw||/||w||", "second-moment RMS", "weight norm",
              "evaluation (wp) vs step"]
    for axis, title in zip(axes.ravel(), titles, strict=True):
        axis.set_title(title)
        axis.set_xlabel("optimizer step")
        axis.legend(fontsize=6)
    axes[1, 0].axhline(1.0, color="k", lw=0.5, ls="--")
    figure.tight_layout()
    stem = "_".join(r[-18:] for r in args.runs)[:80]
    figure.savefig(args.out / f"trace_{stem}.png", dpi=110)
    (args.out / f"trace_{stem}.json").write_text(json.dumps(summaries, indent=1))
    print(json.dumps(summaries, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
