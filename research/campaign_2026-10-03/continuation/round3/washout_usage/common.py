from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
PRIOR = ROOT.parent / "matched_usage"
sys.path.insert(0, str(REPO / "scripts"))
train = importlib.import_module("train_rnn")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def parity(a: np.ndarray, b: np.ndarray) -> dict:
    return dict(
        max_abs=float(np.max(np.abs(a - b))),
        rms=rms(a - b),
        passed=bool(np.allclose(a, b, atol=2e-5, rtol=2e-5)),
    )


def donor(raw: np.ndarray, t: int, std: np.ndarray) -> dict:
    s, e = t - 352, t - 96
    candidates = []
    for lag in [64, 128, 256]:
        if s - lag >= 0:
            value = rms((raw[s:e, :52] - raw[s - lag : e - lag, :52]) / std[:52])
            if np.isfinite(value):
                candidates.append(dict(lag=lag, rms=value))
    chosen = min(candidates, key=lambda c: (c["rms"], c["lag"])) if candidates else None
    return dict(
        query=t,
        start=s,
        end=e,
        lag=chosen["lag"] if chosen else None,
        amplitude=chosen["rms"] if chosen else None,
        candidates=candidates,
    )


def raw_window(raw: np.ndarray, packed: np.ndarray, d: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    t, s, e, lag = d["query"], d["start"], d["end"], d["lag"]
    altered = raw[: t + 1].copy()
    altered[s:e, :52] = raw[s - lag : e - lag, :52]
    assert np.array_equal(altered[:s], raw[:s])
    assert np.array_equal(altered[e:], raw[e : t + 1])
    assert np.array_equal(altered[:, 52:], raw[: t + 1, 52:])
    rebuilt = train.build_inputs(torch.from_numpy(altered)[None])[0].numpy()
    assert np.array_equal(rebuilt[:s], packed[:s])
    suffix = rebuilt[t - 32 : t + 1] - packed[t - 32 : t + 1]
    finite = bool(np.isfinite(rebuilt).all())
    equal = bool(np.array_equal(rebuilt[t - 32 : t + 1], packed[t - 32 : t + 1]))
    gate = dict(
        accepted=finite and equal,
        suffix_max_abs=float(np.abs(suffix).max()),
        suffix_unequal_entries=int(np.sum(suffix != 0)),
        suffix_clock_unequal_entries=int(np.sum(suffix[:, [186, 187]] != 0)),
        suffix_held_unequal_entries=int(np.sum(suffix[:, 189:211] != 0)),
        finite=finite,
        reason="accepted_exact_suffix"
        if finite and equal
        else "nonfinite"
        if not finite
        else "engineered_suffix_mismatch",
    )
    return altered, rebuilt[s:], gate
