"""Uniform weight soup of one run's fixed-epoch checkpoints, written as a new run (same provenance and fold binding).

    weight_soup.py SRC_RUN DST_RUN EPOCH [EPOCH ...]

Averages every floating tensor of the saved (EMA) model state across the listed epochs; the result is stored as
DST_RUN/model_ep<last>.pt so downstream tools (predict_run, --init RUN@epN) work unchanged.
"""
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    src, dst, epochs = sys.argv[1], sys.argv[2], [int(e) for e in sys.argv[3:]]
    runs = ROOT / "artifacts" / "runs"
    cks = [torch.load(runs / src / f"model_ep{e}.pt", map_location="cpu", weights_only=False) for e in epochs]
    out = cks[-1]
    soup = {}
    for key, last in out["model"].items():
        if torch.is_floating_point(last):
            soup[key] = sum(ck["model"][key].double() for ck in cks).div(len(cks)).to(last.dtype)
        else:
            soup[key] = last
    out["model"] = type(out["model"])(soup)
    out["provenance"] = dict(out["provenance"], soup={"source": src, "epochs": epochs})
    (runs / dst).mkdir(parents=True, exist_ok=False)
    torch.save(out, runs / dst / f"model_ep{epochs[-1]}.pt")
    for name in ("provenance.json", "history.json"):
        if (runs / src / name).exists():
            shutil.copy(runs / src / name, runs / dst / name)
    (runs / dst / "soup.json").write_text(json.dumps({"source": src, "epochs": epochs}) + "\n")
    print("SOUP", dst, epochs)


main()
