"""Weight-space interpolation / extrapolation between two runs that share normalization buffers.

    .venv/bin/python scripts/interp_runs.py PARENT CHILD ALPHA NAME
Writes artifacts/runs/NAME/model.pt with theta = parent + ALPHA * (child - parent) (WiSE-FT for
0 < ALPHA < 1, task-vector extrapolation for ALPHA > 1); args and readout come from CHILD.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parent, child, alpha, name = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    runs = ROOT / "artifacts" / "runs"
    p = torch.load(runs / parent / "model.pt", map_location="cpu", weights_only=False)
    c = torch.load(runs / child / "model.pt", map_location="cpu", weights_only=False)
    for k in ("mean", "std"):
        if not torch.equal(p["model"][k], c["model"][k]):
            raise SystemExit(f"{k} buffers differ: interpolation is not meaningful")
    out = dict(c)
    out["model"] = {k: p["model"][k] + alpha * (c["model"][k] - p["model"][k]) if c["model"][k].is_floating_point()
                    else c["model"][k] for k in c["model"]}
    (runs / name).mkdir(parents=True, exist_ok=True)
    torch.save(out, runs / name / "model.pt")
    print("wrote", name, "alpha", alpha)


if __name__ == "__main__":
    main()
