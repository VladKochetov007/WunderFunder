"""TRANSHEAD-01: frozen causal four-code F/T features, without torch or data readers."""
from __future__ import annotations
import numpy as np

WIDTH = 68
SCHEMA = "transhead01-68-v1"
KAPPA = 16.0
RAW_SLICE = slice(48, 52)
LOG_COUNT = np.log(20001.0)
LOG_AGE = np.log(10001.0)


def validate_nuisance(nuisance):
    center = np.asarray(nuisance["centers"], dtype=np.float64)
    pi = np.asarray(nuisance["pi"], dtype=np.float64)
    R = np.asarray(nuisance["R"], dtype=np.float64)
    if center.shape != (4,) or pi.shape != (16,) or R.shape != (16, 16):
        raise ValueError("nuisance dimensions")
    if not all(np.isfinite(v).all() for v in (center, pi, R)) or (pi <= 0).any() or (R <= 0).any():
        raise ValueError("nuisance must be finite with positive probabilities")
    np.testing.assert_allclose(pi.sum(), 1, atol=1e-12)
    np.testing.assert_allclose(R.sum(1), 1, atol=1e-12)
    return center, pi, R


def detect(raw4):
    x = np.asarray(raw4, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError("expected rows x four raw columns")
    valid = np.isfinite(x).all(1) & (np.abs(x) < 5.19).all(1)
    return np.flatnonzero(valid[1:] & valid[:-1] & (x[1:] != x[:-1]).any(1)
                          & (np.ptp(x[1:], axis=1) > 1e-6)) + 1


def encode(values, centers):
    return ((np.asarray(values, dtype=np.float32) > centers) * (2 ** np.arange(4))).sum(1).astype(np.int64)


def event_probabilities(marks, pi, R, arm):
    """Vectorized cumulative counts, strictly before every observed mark."""
    m = np.asarray(marks, dtype=np.int64)
    if arm not in ("F", "T"):
        raise ValueError(arm)
    one = np.eye(16, dtype=np.float64)[m]
    p = (np.cumsum(one, axis=0) - one + KAPPA*pi)/(np.arange(len(m))[:, None] + KAPPA)
    if arm == "T" and len(m):
        p[0] = pi
        for context in range(16):
            positions = np.flatnonzero(m[:-1] == context) + 1
            o = one[positions]
            p[positions] = (np.cumsum(o, axis=0) - o + KAPPA*R[context]) / (np.arange(len(o))[:, None] + KAPPA)
    return p


def batch_features(raw4, nuisance, arm):
    """Complete sequence features; computation is vectorized but every row remains prefix-causal."""
    centers, pi, R = validate_nuisance(nuisance)
    x = np.asarray(raw4, dtype=np.float32)
    events = detect(x)
    marks = encode(x[events], centers)
    probabilities = event_probabilities(marks, pi, R, arm)
    # Row0 in held is the defined no-event state. Row j+1 follows event j.
    held = np.zeros((len(events)+1, WIDTH), dtype=np.float32)
    held[0, 35:51] = pi
    if len(events):
        one = np.eye(16, dtype=np.float64)[marks]
        held[1:, :16] = one
        held[2:, 16:32] = one[:-1]
        held[1:, 32] = np.log1p(np.minimum(np.arange(1, len(events)+1), 20000))/LOG_COUNT
        held[1:, 35:51] = probabilities
        held[1:, 51:67] = one-probabilities
        held[1:, 67] = np.minimum(-np.log(probabilities[np.arange(len(events)), marks]), 10)/10
    row_indices = np.arange(len(x))
    state_at = np.searchsorted(events, row_indices, side="right")
    out = held[state_at]
    last_row = np.r_[0, events][state_at]
    out[:, 33] = np.log1p(np.minimum(row_indices-last_row, 10000))/LOG_AGE
    out[events, 34] = 1
    return out


class StreamingFeatures:
    """Literal independent scalar state machine; caller must reset at each seq_ix change."""
    def __init__(self, nuisance, arm):
        self.centers, self.pi, self.R = validate_nuisance(nuisance)
        if arm not in ("F", "T"):
            raise ValueError(arm)
        self.arm = arm
        self.reset()

    def reset(self):
        self.prev_raw = None
        self.last_mark = None
        self.total = 0
        self.counts = np.zeros(16, dtype=np.float64)
        self.transitions = np.zeros((16, 16), dtype=np.float64)
        self.context = np.zeros(16, dtype=np.float64)
        self.row = 0
        self.last_event = 0
        self.out = np.zeros(WIDTH, dtype=np.float32)
        self.out[35:51] = self.pi

    def update(self, raw4):
        x = np.asarray(raw4, dtype=np.float32)
        if x.shape != (4,):
            raise ValueError("expected four raw columns")
        valid = np.isfinite(x).all() and (np.abs(x) < 5.19).all()
        previous_valid = self.prev_raw is not None and np.isfinite(self.prev_raw).all() and (np.abs(self.prev_raw) < 5.19).all()
        event = bool(valid and previous_valid and np.any(x != self.prev_raw) and np.ptp(x) > 1e-6)
        self.out[34] = float(event)
        if event:
            mark = int(sum((int(x[j] > self.centers[j]) << j) for j in range(4)))
            if self.arm == "F":
                p = (self.counts + KAPPA*self.pi)/(self.total + KAPPA)
            elif self.last_mark is None:
                p = self.pi.copy()
            else:
                p = (self.transitions[self.last_mark] + KAPPA*self.R[self.last_mark])/(self.context[self.last_mark] + KAPPA)
            self.out[16:32] = self.out[:16]
            self.out[:16] = 0
            self.out[mark] = 1
            self.out[35:51] = p
            residual = -p.copy()
            residual[mark] += 1
            self.out[51:67] = residual
            self.out[67] = min(-np.log(p[mark]), 10)/10
            if self.last_mark is not None:
                self.transitions[self.last_mark, mark] += 1
                self.context[self.last_mark] += 1
            self.counts[mark] += 1
            self.total += 1
            self.out[32] = np.log1p(min(self.total, 20000))/LOG_COUNT
            self.last_mark = mark
            self.last_event = self.row
        self.out[33] = np.log1p(min(self.row-self.last_event, 10000))/LOG_AGE
        self.prev_raw = x.copy()  # invalid/neutral rows also advance raw state
        self.row += 1
        return self.out.copy()
