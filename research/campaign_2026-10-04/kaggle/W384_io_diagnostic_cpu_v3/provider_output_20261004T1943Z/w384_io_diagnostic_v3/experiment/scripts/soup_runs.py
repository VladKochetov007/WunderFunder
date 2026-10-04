"""Uniform weight soup of runs that share normalization buffers (fine-tune children of one parent).

    .venv/bin/python scripts/soup_runs.py NAME RUN [RUN ...]
Writes artifacts/runs/NAME/model.pt with the mean of the float parameters; args/readout from RUN 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    name, runs = sys.argv[1], sys.argv[2:]
    cks = [torch.load(ROOT / "artifacts" / "runs" / r / "model.pt", map_location="cpu", weights_only=False) for r in runs]
    ref = cks[0]["model"]
    for ck, r in zip(cks[1:], runs[1:]):
        for k in ("mean", "std"):
            if not torch.equal(ck["model"][k], ref[k]):
                raise SystemExit(f"{r}: {k} buffers differ from {runs[0]}")
    out = dict(cks[0])
    out["model"] = {k: (sum(ck["model"][k].float() for ck in cks) / len(cks)).to(ref[k].dtype) if ref[k].is_floating_point()
                    else ref[k] for k in ref}
    (ROOT / "artifacts" / "runs" / name).mkdir(parents=True, exist_ok=True)
    torch.save(out, ROOT / "artifacts" / "runs" / name / "model.pt")
    print("wrote", name, "from", len(runs), "runs")


if __name__ == "__main__":
    main()
