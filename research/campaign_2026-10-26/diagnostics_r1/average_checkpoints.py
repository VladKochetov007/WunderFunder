"""Uniform weight average of a run's epoch checkpoints into a new run directory (zero-fit).

Usage: average_checkpoints.py SRC_RUN DST_RUN EPOCH [EPOCH ...]
Writes artifacts/runs/DST_RUN/model_ep{last}.pt with the averaged state dict; args/metadata are
the last epoch's. Float tensors are averaged in float64 and cast back; others must match.
"""

import json
import sys
from pathlib import Path

import torch

torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
src, dst, epochs = sys.argv[1], sys.argv[2], [int(e) for e in sys.argv[3:]]
runs = Path("artifacts/runs")
cks = [torch.load(runs / src / f"model_ep{e}.pt", map_location="cpu", weights_only=False) for e in epochs]
out = cks[-1]
avg = {}
for key, value in out["model"].items():
    if value.is_floating_point():
        avg[key] = (sum(c["model"][key].double() for c in cks) / len(cks)).to(value.dtype)
    else:
        if any(not torch.equal(c["model"][key], value) for c in cks):
            raise ValueError(f"non-float tensor differs across epochs: {key}")
        avg[key] = value
out["model"] = avg
out["weight_average"] = {"source_run": src, "epochs": epochs}
(runs / dst).mkdir(parents=True, exist_ok=False)
torch.save(out, runs / dst / f"model_ep{epochs[-1]}.pt")
print(json.dumps({"dst": dst, "epochs": epochs, "keys": len(avg)}))
