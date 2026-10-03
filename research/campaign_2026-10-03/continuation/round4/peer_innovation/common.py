from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
PRIOR = ROOT.parents[1] / "round3/matched_usage"
sys.path.insert(0, str(REPO / "scripts"))


def dump(name: str, value: dict) -> None:
    (ROOT / name).write_text(json.dumps(value, indent=2))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def ema(x: np.ndarray, half: float) -> np.ndarray:
    alpha = 1 - 2 ** (-1 / half)
    out = np.empty(len(x), dtype=np.float64)
    state = 0.0
    for i, v in enumerate(x):
        state += alpha * (float(v) - state)
        out[i] = state
    return out


def signal_features(raw: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    own = np.r_[0.0, np.diff(raw[:, 0].astype(float))] / std[0]
    peer = np.r_[0.0, np.diff(raw[:, 52].astype(float))] / std[52]
    own_emas = np.column_stack([ema(own, h) for h in [4, 16, 64]])
    peer_ema = ema(peer, 16)
    event = np.r_[False, np.any(raw[1:, 52:74] != raw[:-1, 52:74], axis=1)]
    rows = np.arange(len(raw))
    age = rows - np.maximum.accumulate(np.where(event, rows, 0))
    return own_emas, peer_ema, age


def ridge_inputs(features: np.ndarray, weight: np.ndarray) -> dict:
    mean = np.average(features, axis=0, weights=weight)
    scale = np.sqrt(np.average((features - mean) ** 2, axis=0, weights=weight))
    scale = np.maximum(scale, 1e-6)
    block = np.r_[
        np.full(211, 1 / np.sqrt(211)),
        np.full(3, 1 / np.sqrt(3)),
        np.full(512, 1 / np.sqrt(512)),
        np.ones(features.shape[1] - 726),
    ]
    x = (features - mean) / scale * block
    xw = x * np.sqrt(weight[:, None])
    eigen, u = np.linalg.eigh(xw @ xw.T)
    eigen = np.maximum(eigen, 0.0)
    threshold = max(eigen.max() * 1e-10, 1e-12)
    rank = int(np.sum(eigen > threshold))
    return dict(mean=mean, scale=scale, block=block, xw=xw, eigen=eigen, u=u, rank=rank)


def ridge_fit(prep: dict, y: np.ndarray, weight: np.ndarray, df: int) -> dict:
    eigen = prep["eigen"]
    u = prep["u"]
    lo = 0.0
    hi = max(eigen.max(), 1.0) * len(eigen) / max(df, 1)
    for _ in range(80):
        mid = (lo + hi) / 2
        effective = float(np.sum(eigen / (eigen + mid)))
        if effective > df:
            lo = mid
        else:
            hi = mid
    penalty = (lo + hi) / 2
    intercept = float(np.average(y, weights=weight))
    rhs = (y - intercept) * np.sqrt(weight)
    dual = u @ ((u.T @ rhs) / (eigen + penalty))
    beta = prep["xw"].T @ dual
    hat = (u * (eigen / (eigen + penalty))) @ u.T
    return dict(
        mean=prep["mean"],
        scale=prep["scale"],
        block=prep["block"],
        beta=beta,
        intercept=intercept,
        lambda_=penalty,
        edf=float(np.trace(hat)),
        rank=prep["rank"],
        hat=hat,
    )


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    return ((x - model["mean"]) / model["scale"] * model["block"]) @ model["beta"] + model[
        "intercept"
    ]
