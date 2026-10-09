"""Uniform / weighted weight average ("model soup") of same-architecture checkpoints into a new run directory.

Members must share the architecture and should share the warm-start parent (linear mode connectivity is
only reliable from a common initialization).  The soup checkpoint copies the first member's metadata, adds
``soup`` provenance (members, weights, source hashes) and is written as ``model_ep{E}.pt`` + ``model.pt`` so
``predict_run.py`` scores it like any other run.  No training happens here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--members", required=True, help="comma-separated run[:epoch][*weight], default epoch 1, weight 1")
    parser.add_argument("--epoch", type=int, default=1, help="epoch index the soup is saved as")
    args = parser.parse_args()
    out = ROOT / "artifacts" / "runs" / args.name
    if out.exists():
        raise SystemExit(f"{out} exists")
    members = []
    for token in args.members.split(","):
        weight = 1.0
        if "*" in token:
            token, w = token.split("*")
            weight = float(w)
        run, _, epoch = token.partition(":")
        members.append((run, int(epoch or 1), weight))
    total = sum(w for _, _, w in members)
    checkpoints, sources = [], []
    for run, epoch, _ in members:
        path = ROOT / "artifacts" / "runs" / run / f"model_ep{epoch}.pt"
        checkpoints.append(torch.load(path, map_location="cpu", weights_only=False))
        sources.append({"run": run, "epoch": epoch, "sha256": sha256(path),
                        "init": checkpoints[-1]["args"].get("init")})
    reference = checkpoints[0]
    for other in checkpoints[1:]:
        if set(other["model"]) != set(reference["model"]):
            raise SystemExit("members differ in parameter names")
    averaged = {}
    for key, value in reference["model"].items():
        if value.dtype.is_floating_point:
            acc = sum(w / total * c["model"][key].double() for (_, _, w), c in zip(members, checkpoints, strict=True))
            averaged[key] = acc.to(value.dtype)
        else:
            averaged[key] = value
    soup = dict(reference)
    soup["model"] = averaged
    soup["epoch"] = args.epoch
    soup["soup"] = {"members": sources, "weights": [w / total for _, _, w in members],
                    "note": "weight average; no training"}
    soup["args"] = {**reference["args"], "name": args.name}
    out.mkdir(parents=True)
    torch.save(soup, out / f"model_ep{args.epoch}.pt")
    torch.save(soup, out / "model.pt")
    (out / "provenance.json").write_text(json.dumps({"soup": soup["soup"], "selected_epoch": args.epoch}, indent=1) + "\n")
    (out / "history.json").write_text("[]")
    print(json.dumps({"soup": args.name, "members": [m[0] for m in members]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
