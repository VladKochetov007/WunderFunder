"""Native enc="conv" engine: parity, chunk-reset gap and per-row latency (CPU only).

    PYTHONPATH=submissions/_native_conv/v3 taskset -c 2 scripts/limited .venv/bin/python \
        scripts/check_conv_native.py --run ab61_conv_gru_f0 --linear ab60_final_ema4_p1_lpft --score-seqs 468

1. parity: native Predictor (fast and exact activations) vs one full-sequence torch forward (true
   causal conv, zeros only before row 0), rows >= 99, on --seqs validation row groups.
2. chunk reset: torch with the conv restarted every 4000 / 2000 rows (train_rnn.predict / TBPTT
   regime, GRU state carried) vs the true stream: gap by rows since the last restart, and global
   / per-sequence WP of both regimes on the first --score-seqs sequences of the held-out fold.
3. latency: per-row Predictor.step on real rows, conv member vs linear member from the same build,
   interleaved repetitions (min), one thread. Pin with taskset for stable numbers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

import wnn_native  # noqa: E402
from wnn.data import WARMUP, SequenceFile  # noqa: E402
from wnn.features import event_features  # noqa: E402
from wnn.gmetric import global_wp  # noqa: E402
from wnn.metric import score  # noqa: E402
from wnn.nn import RNNModel  # noqa: E402
from wnn.splits import valid_row_groups  # noqa: E402

ROOT = Path(os.environ.get("WNN_ROOT", Path(__file__).resolve().parents[1]))
VALID = ROOT / "datasets" / "valid.parquet"


def load_model(run: str) -> RNNModel:
    ck = torch.load(ROOT / "artifacts" / "runs" / run / "model.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    m = RNNModel(ck["n_in"], a["hidden"], a["layers"], 0.0, n_out=ck["model"]["head.2.bias"].shape[0],
                 enc=a.get("enc", "linear"), conv_k=a.get("conv_k", 16)).eval()
    m.load_state_dict(ck["model"])
    return m


def native_weights(model: RNNModel) -> dict[str, np.ndarray]:
    """Same arrays as package_submission.export writes (head sliced to the two targets)."""
    w = {k: np.ascontiguousarray(v.detach().float().numpy()) for k, v in model.state_dict().items()}
    w["head.2.weight"] = np.ascontiguousarray(w["head.2.weight"][:2])
    w["head.2.bias"] = np.ascontiguousarray(w["head.2.bias"][:2])
    return w


@torch.no_grad()
def torch_predict(model, x, chunk=None):
    """x (B, T, 112) raw rows -> (B, T, 2); chunk restarts the conv context (GRU state carried)."""
    xin = torch.from_numpy(np.concatenate([x, event_features(x)], -1))
    t = xin.shape[1]
    chunk = chunk or t
    out, h = [], None
    for s in range(0, t, chunk):
        y, h = model(xin[:, s:s + chunk], h)
        out.append(y[..., :2])
    return torch.cat(out, 1).numpy()


def fold_groups(fold=0, n_folds=4):
    dev, hold = valid_row_groups(VALID)
    all_valid = np.r_[dev, hold]
    cf = np.load(ROOT / "cache" / "chain_folds.npz")
    vm = cf["split"] == "valid"
    fmap = dict(zip(cf["row_group"][vm].tolist(), cf["fold"][vm].tolist()))
    groups = all_valid[np.array([fmap[int(g)] for g in all_valid]) % n_folds == fold]
    return np.random.default_rng(12345).permutation(groups)  # train_rnn's eval order


def parity(model, sf, seqs):
    w = native_weights(model)
    print("== parity: native stream vs full-sequence torch, rows >= 99")
    gaps = {}
    for g in seqs:
        x = sf.read(g, targets=False, mask=False).features
        want = torch_predict(model, x[None])[0]
        row = {}
        for fast in (True, False):
            p = wnn_native.Predictor(w, fast_activations=fast)
            got = np.stack([p.step(r) for r in x])
            d = np.abs(got - want)[WARMUP:]
            row["fast" if fast else "exact"] = f"max {d.max():.2e} mean {d.mean():.2e}"
        print(f"  seq {g} ({p.encoder}): {row}")
        for chunk in (4000, 2000):
            gaps.setdefault(chunk, []).append(np.abs(torch_predict(model, x[None], chunk)[0] - want))
        gaps.setdefault("scale", []).append(np.abs(want[WARMUP:]).mean())
    print("== chunk reset vs true stream (torch), |diff| by rows since restart; mean |pred| "
          f"{np.mean(gaps.pop('scale')):.3f}")
    for chunk, ds in gaps.items():
        d = np.stack(ds)[:, chunk:]  # rows after the first restart
        off = np.arange(chunk, d.shape[1] + chunk) % chunk
        parts = []
        for lo, hi in ((0, 15), (15, 100), (100, 1000), (1000, chunk)):
            sel = d[:, (off >= lo) & (off < hi)]
            parts.append(f"[{lo},{hi}) max {sel.max():.3f} mean {sel.mean():.1e}")
        print(f"  chunk {chunk}: " + "; ".join(parts) + f"; rows > 1e-2: {(d > 1e-2).mean():.2%}")


def metric_gap(model, sf, n, batch=16):
    groups = fold_groups()[:n]
    preds = {None: [], 4000: [], 2000: []}
    ys, ms = [], []
    t0 = time.perf_counter()
    for s in range(0, len(groups), batch):
        seqs = [sf.read(int(g)) for g in groups[s:s + batch]]
        x = np.stack([q.features for q in seqs])
        ys.append(np.stack([q.targets for q in seqs]))
        ms.append(np.stack([q.scored for q in seqs]))
        for chunk in preds:
            preds[chunk].append(torch_predict(model, x, chunk))
    y, m = np.concatenate(ys), np.concatenate(ms)
    print(f"== held-out fold 0, {len(groups)} sequences ({time.perf_counter() - t0:.0f} s)")
    for chunk, p in preds.items():
        p = np.concatenate(p)
        name = "true stream" if chunk is None else f"reset every {chunk}"
        print(f"  {name:18s} global WP {global_wp(y, p, m)['wp']:.5f}  per-seq WP {score(y, p, m)['weighted_pearson']:.5f}")


def latency(members, x, reps, block=500):
    """Paired blocks: every member steps the same `block` rows back to back (order alternating),
    so clock and co-tenant drift cancel in the per-block ratio."""
    print(f"== latency: Predictor.step per row, {len(x)} real rows x {reps} passes, paired {block}-row blocks")
    preds = {k: wnn_native.Predictor(w, fast_activations=True) for k, w in members.items()}
    names = list(preds)
    times = {k: [] for k in preds}
    rows = [np.ascontiguousarray(r) for r in x]
    for rep in range(reps):
        for p in preds.values():
            p.reset()
        for b, s in enumerate(range(0, len(rows), block)):
            chunk = rows[s:s + block]
            for k in names if (rep + b) % 2 == 0 else names[::-1]:
                step = preds[k].step
                t = time.perf_counter()
                for r in chunk:
                    step(r)
                times[k].append((time.perf_counter() - t) / len(chunk) * 1e6)
    base = np.array(times[names[0]])
    for k, v in times.items():
        v = np.array(v)
        q1, q3 = np.percentile(v / base - 1, [25, 75])
        print(f"  {k:32s} p10 {np.percentile(v, 10):6.2f}  median {np.median(v):6.2f} us/row; "
              f"paired ratio median {np.median(v / base) - 1:+.1%} (IQR {q1:+.1%} .. {q3:+.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ab61_conv_gru_f0")
    ap.add_argument("--linear", default="ab60_final_ema4_p1_lpft", help="linear run for the latency baseline")
    ap.add_argument("--seqs", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--score-seqs", type=int, default=0)
    ap.add_argument("--bench-rows", type=int, default=20_000)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(1)
    print("wnn_native:", wnn_native.__file__, getattr(wnn_native, "ENCODERS", None), file=sys.stderr)
    sf = SequenceFile(VALID)
    model = load_model(a.run)
    parity(model, sf, a.seqs)
    if a.score_seqs:
        metric_gap(model, sf, a.score_seqs)
    if a.bench_rows:
        x = sf.read(a.seqs[0], targets=False, mask=False).features[: a.bench_rows]
        lin = load_model(a.linear)
        latency({f"linear {a.linear}": native_weights(lin), f"conv {a.run}": native_weights(model)}, x, a.reps)


if __name__ == "__main__":
    main()
