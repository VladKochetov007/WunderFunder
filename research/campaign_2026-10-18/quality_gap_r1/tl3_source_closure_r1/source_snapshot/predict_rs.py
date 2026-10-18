"""Predict valid fold0 (467 sequences) for a run trained by train_rnn_plew19.py; full-size cache with NaN elsewhere."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(os.environ["WUNDER_RESEARCH_ROOT"])
HERE = Path(__file__).resolve().parent
os.environ["WUNDER_RESEARCH_ROOT"] = str(R)
sys.path[:0] = [str(HERE), str(R / "src")]
import train_rnn_rs as T  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main(run, split="valid", full=False):
    out = R / f"cache/pred_{os.environ.get('OUT_TAG', run)}_{split}.npy"
    assert not out.exists()
    ck = torch.load(R / f"artifacts/runs/{run}/model_ep{os.environ.get("EPOCH", "1")}.pt", map_location="cpu", weights_only=False)
    args = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    n_out = ck["model"]["head.2.weight"].shape[0]
    T.LOOKAHEAD = tuple(int(v) for v in str(args.get("lookahead", "") or "").split(",") if v)
    model = T.make_model(args, ck["n_in"], n_out)
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()
    assert T.FAMILY_TEACHERS == ()
    rg = np.load(R / f"cache/{split}_targets.npz")["row_group"]
    cf = np.load(R / "cache/chain_folds.npz")
    f0 = set(cf["row_group"][(cf["split"].astype(str) == "valid") & (cf["fold"] == 0)].tolist())
    positions = np.arange(len(rg)) if full else np.flatnonzero([g in f0 for g in rg])
    loader = DataLoader(T.Sequences(split, rg[positions]), batch_size=16, num_workers=2, prefetch_factor=1)
    tmp = out.with_suffix(".partial.npy")
    dest = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(rg), 20000, 2))
    dest[:] = np.nan
    off = 0
    for x, _, _ in loader:
        pred = T.predict(model, x, 4000, torch.device("cuda"))
        assert np.isfinite(pred).all()
        dest[positions[off:off + len(x)]] = pred
        off += len(x)
    assert off == len(positions)
    dest.flush()
    del dest
    tmp.replace(out)
    print("wrote", out, off, "sequences")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or ["valid"]), full=len(sys.argv) > 3)
