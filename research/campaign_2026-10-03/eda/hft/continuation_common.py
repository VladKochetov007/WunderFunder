from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[3]
PRICE0 = [f"i0_p{i}" for i in range(22)]
PRICE1 = [f"i1_p{i}" for i in range(22)]
TRADE0 = [f"i0_dv{i}" for i in range(4)]
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])


def save_json(name: str, value: object) -> None:
    """Write a JSON artifact only to the analyst's owned directory."""
    (OWN / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def set_phase(phase: str, **updates: object) -> None:
    """Preserve the existing state and record a probe transition."""
    state = json.loads((OWN / "STATE.json").read_text())
    state.update(updates)
    state["phase"] = phase
    save_json("STATE.json", state)


def sha256(path: Path) -> str:
    """Hash a local artifact in bounded chunks for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sampling_manifest() -> dict[str, object]:
    """Conservatively exclude every documented prior VALID selection."""
    prior_valid = {0, 1, 2, 3, 74, 111, 555}
    for stride in [7, 9, 23, 37, 47]:
        prior_valid.update(range(0, 1873, stride))
    selected = [g for g in range(1300, 1873) if g not in prior_valid][:20]
    train_table = pd.read_csv(OWN / "regime_stats.csv")
    train_ids = sorted(train_table.loc[train_table.split == "train", "g"].unique())
    decoder_train = np.random.default_rng(0).choice(10607, 60, replace=False)
    return {
        "selected_valid_row_groups": selected,
        "prior_valid_row_groups_conservative": sorted(prior_valid),
        "prior_regime_train_row_groups": [int(g) for g in train_ids],
        "prior_decoder_train_row_groups": sorted(int(g) for g in decoder_train),
        "own_prior_overlap": sorted(set(selected) & prior_valid),
        "scope": "Own documented prior exposure only; other campaign exposure unknown.",
        "caveat": "Script selections conservatively count as exposure even without logs.",
        "hashes": {
            name: sha256(OWN / name)
            for name in ["HYPOTHESES.json", "decoder.pkl", "regime_stats.csv"]
        },
    }


def change_mask(values: np.ndarray) -> np.ndarray:
    """Detect input changes causally, with no change at the initial row."""
    return np.r_[False, np.any(np.diff(values, axis=0) != 0, axis=1)]


def quantum_diagnostic(targets: np.ndarray) -> tuple[float, float, int]:
    """Recover a target quantum for diagnostics, never inference features."""
    nonzero = np.abs(targets[np.abs(targets) > 1e-6])
    initial = float(nonzero.min())
    label = int(np.argmin(np.abs(QUANTA - initial)))
    ticks = np.rint(nonzero / QUANTA[label])
    q = float(np.median(nonzero / np.maximum(ticks, 1)))
    residual = float(np.max(np.abs(targets / q - np.rint(targets / q))))
    if residual > 1e-3:
        raise ValueError(f"Target quantum reconstruction residual {residual}")
    return q, residual, label


def prefix_stats(frame: pd.DataFrame) -> dict[str, float]:
    """Exactly reproduce the legacy T=100 TRAIN feature definitions."""
    prefix = frame.iloc[:100]
    x0 = prefix[PRICE0].to_numpy()
    x1 = prefix[PRICE1].to_numpy()
    result = {
        "bk0": float(change_mask(x0)[1:].mean()),
        "bk1": float(change_mask(x1)[1:].mean()),
        "tr0": float(change_mask(prefix[TRADE0].to_numpy())[1:].mean()),
        "nst": len(np.unique(x0[:, 0])) / 100,
    }
    for name in ["a5", "a6", "a7"]:
        values = prefix[name].to_numpy()
        active = values > -5.19
        result[name + "fl"] = float((values < -5.19).mean())
        result[name + "m"] = float(values[active].mean()) if active.any() else -5.0
    for name in ["a2", "a3"]:
        result[name] = float(prefix[name].iloc[-1])
    for j in [0, 3, 11, 19]:
        result[f"p{j}m"] = float(x0[:, j].mean())
        result[f"p{j}sd"] = float(x0[:, j].std())
    result["T"] = 100.0
    return result


def load_decoder() -> dict[str, object]:
    """Load the analyst's preserved target-supervised research decoder."""
    with (OWN / "decoder.pkl").open("rb") as stream:
        return pickle.load(stream)


def decode_prices(
    prices: np.ndarray, models: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the old decoder to current/previous inputs without oracle scale."""
    events = np.flatnonzero(change_mask(prices))
    features = np.c_[
        prices[events - 1], prices[events], prices[events] - prices[events - 1]
    ]
    increments = []
    for name in ["da", "db"]:
        model = models[name]
        changes = np.zeros(len(prices), dtype=np.float64)
        if len(events):
            changes[events] = model.classes_[model.predict_proba(features).argmax(1)]
        increments.append(np.cumsum(changes))
    spread_model = models["sp"]
    spread = spread_model.predict_proba(prices) @ spread_model.classes_
    return increments[0], increments[1], spread


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Compute a causal EMA initialized from the first row."""
    alpha = 2 / (span + 1)
    return lfilter([alpha], [1, -(1 - alpha)], values, zi=[(1 - alpha) * values[0]])[0]


def moments(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return additive sufficient statistics for the exact clipped global WP."""
    y = np.clip(target.astype(np.float64), -2, 2)
    p = np.clip(prediction.astype(np.float64), -2, 2)
    w = np.abs(y)
    return np.array(
        [w.sum(), w @ y, w @ p, w @ (y * y), w @ (p * p), w @ (y * p)], dtype=np.float64
    )


def correlation(stats: np.ndarray) -> float | None:
    """Calculate WP from pooled sufficient statistics; flag degenerate variance."""
    weight, sy, sp, syy, spp, syp = stats
    if weight <= 0:
        return None
    variance = (syy - sy * sy / weight) * (spp - sp * sp / weight)
    if variance <= 0:
        return None
    return float((syp - sy * sp / weight) / np.sqrt(variance))
