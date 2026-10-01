"""Verify a fixed endpoint without predicting or evaluating competition data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import train_rnn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--stage", choices=["parent", "head"], required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--rho", type=float, required=True)
    a = p.parse_args()
    assert a.run.startswith("on_sam_") and "/" not in a.run and ".." not in a.run
    path = ROOT / "artifacts/runs" / a.run / f"model_ep{a.epoch}.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["epoch"] == a.epoch and ck["args"]["sam_stage"] == a.stage
    assert ck["args"]["valid_fold"] == a.fold and ck["args"]["sam_rho"] == a.rho
    assert a.epoch == {"parent":3, "head":1}[a.stage]
    assert ck["args"]["evaluation_policy"].startswith("none;")
    assert all(torch.isfinite(t).all() for t in ck["model"].values())
    model = train_rnn.make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0])
    model.load_state_dict(ck["model"], strict=True)
    latest = torch.load(path.with_name("model.pt"), map_location="cpu", weights_only=False)
    assert latest["epoch"] == a.epoch
    assert all(torch.equal(t, latest["model"][k]) for k,t in ck["model"].items())
    history = json.loads(path.with_name("history.json").read_text())
    assert len(history) == a.epoch + 1
    assert all(x["evaluation_skipped"] and "weighted_pearson" not in x for x in history)
    print(json.dumps({"status":"PASS", "run":a.run, "epoch":a.epoch,
                      "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                      "shape":{k:list(v.shape) for k,v in ck["model"].items()},
                      "WP_readouts":0}, indent=2))


if __name__ == "__main__":
    main()
