"""Target-blind PLE knots for the w19 lineage: same cohort rule and quantile rule as the research PLE,
but normalized with the w19 ancestor (d256bc1k_f0@ep3) mean/std. Cohort excludes valid fold0 and strict chains."""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

R = Path("/home/grinchenkoeg/WunderFunder-creative")
HERE = Path(__file__).resolve().parent
os.environ["WUNDER_RESEARCH_ROOT"] = str(R)
sys.path[:0] = [str(HERE), str(R / "src")]
import train_rnn_plew19 as T  # noqa: E402
from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, SequenceFile  # noqa: E402


def quantile_knots(samples):
    features, left, right, support, counts = [], [], [], [], []
    for i in range(samples.shape[1]):
        col = samples[:, i]
        other = col[col != col[0]]
        n = 1 if not len(other) else 2 + int(np.any(other != other[0]))
        knots = np.unique(np.quantile(col, np.arange(9) / 8, method="linear").astype(np.float32))
        support.append(n)
        counts.append(len(knots))
        if n < 3 or len(knots) < 3:
            continue
        assert np.all(np.diff(knots) > 0)
        features.extend([i] * (len(knots) - 1))
        left.extend(knots[:-1])
        right.extend(knots[1:])
    return dict(feature=np.asarray(features, np.int64), left=np.asarray(left, np.float32),
                right=np.asarray(right, np.float32), support_count=np.asarray(support, np.int64),
                knot_count=np.asarray(counts, np.int64))


def main():
    out = HERE / "knots_w19.npz"
    assert not out.exists()
    cf = np.load(R / "cache/chain_folds.npz")
    split, groups, folds = cf["split"].astype(str), cf["row_group"], cf["fold"]

    def half(g):
        return int(hashlib.sha256(f"half:{g}".encode()).hexdigest(), 16) % 2

    pool = [("train", int(g)) for s, g in zip(split, groups) if s == "train" and int(g) not in CORRUPT_TRAIN_ROW_GROUPS and half(int(g)) == 0]
    pool += [("valid", int(g)) for s, g, f in zip(split, groups, folds) if s == "valid" and f != 0]
    cohort = sorted(pool, key=lambda r: hashlib.sha256(f"full-ple-v1|{r[0]}|{r[1]}".encode()).hexdigest())[:256]
    ck = torch.load(R / "artifacts/runs/d256bc1k_f0/model_ep3.pt", map_location="cpu", weights_only=False)
    mean, std = ck["model"]["mean"].float(), ck["model"]["std"].float()
    pos = torch.from_numpy(np.arange(1024, dtype=np.int64) * 19999 // 1023).cuda()
    files = {s: SequenceFile(R / f"datasets/{s}.parquet") for s in ("train", "valid")}
    samples = np.empty((256, 1024, 211), np.float32)
    for start in range(0, 256, 16):
        raw = torch.from_numpy(np.stack([files[s].read(g, targets=False, mask=False).features for s, g in cohort[start:start + 16]])).cuda()
        with torch.no_grad():
            norm = (T.build_inputs(raw) - mean.cuda()) / std.cuda()
            samples[start:start + 16] = norm.index_select(1, pos).cpu().numpy()
        print("features", start + 16, flush=True)
    assert np.isfinite(samples).all()
    knots = quantile_knots(samples.reshape(-1, 211))
    from ple_model import validate_knots
    validate_knots(knots, 211)
    np.savez(out, **knots, mean=mean.numpy(), std=std.numpy())
    print("knots", len(knots["feature"]), "intervals on", np.unique(knots["feature"]).size, "features ->", out)


if __name__ == "__main__":
    main()
