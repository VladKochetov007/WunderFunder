"""Parse trainer text logs ('ep E batch B/N loss L lr LR seq/s S') into per-run CSV and an optional comparison plot."""
from __future__ import annotations
import argparse, re
from pathlib import Path

PAT = re.compile(r"ep (\d+) batch (\d+)/(\d+) loss ([0-9.eE+-]+) lr ([0-9.eE+-]+) seq/s ([0-9.]+)")


def parse(path: Path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = PAT.search(line)
        if m:
            e, b, n, loss, lr, s = m.groups()
            rows.append((int(e) + int(b) / int(n), float(loss), float(lr), float(s)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--csv-dir", type=Path)
    ap.add_argument("--plot", type=Path)
    a = ap.parse_args()
    series = {}
    for p in a.logs:
        rows = parse(p)
        series[p.stem] = rows
        print(f"{p.stem}: {len(rows)} points, last loss {rows[-1][1] if rows else None}")
        if a.csv_dir and rows:
            a.csv_dir.mkdir(parents=True, exist_ok=True)
            (a.csv_dir / f"{p.stem}.csv").write_text("epoch_frac,loss,lr,seq_per_s\n" + "\n".join(",".join(map(str, r)) for r in rows))
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for k, rows in series.items():
            if rows:
                x, y = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
                w = 15; ys = np.convolve(y, np.ones(w) / w, mode="valid")
                ax.plot(x[w - 1:], ys, label=k)
        ax.set_xlabel("epoch"); ax.set_ylabel("training loss (smoothed, includes auxiliary terms)"); ax.set_ylim(top=3); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(a.plot, dpi=130)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
