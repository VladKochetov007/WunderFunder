"""Unexecuted diagnostic primitive; no data readers, model loading or WP functions.

Loss: sum_i,k a_i q_i |clip(y_ik)| (p_ik-clip(y_ik))^2 /
      max(sum_i,k a_i q_i |clip(y_ik)|, 1e-8).
Predictions are not clipped. a is the fixed row-sampling inclusion weight.
The final layer has two independent output rows and ONE common denominator.
"""
import numpy as np


def sufficient(hidden, prediction, target, selector, inclusion_weight):
    """Return gradient numerator (2,129), common denominator and loss numerator."""
    hidden = np.asarray(hidden, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    selector = np.asarray(selector, dtype=np.float64)
    inclusion_weight = np.asarray(inclusion_weight, dtype=np.float64)
    n = hidden.shape[0]
    assert hidden.shape == (n, 128) and prediction.shape == target.shape == (n, 2)
    assert selector.shape == inclusion_weight.shape == (n,)
    assert all(np.isfinite(x).all() for x in
               (hidden, prediction, target, selector, inclusion_weight))
    assert np.all((selector >= 0) & (selector <= 1)) and np.all(inclusion_weight > 0)
    c = np.clip(target, -2, 2)
    phi = np.concatenate([hidden, np.ones((n, 1), dtype=np.float64)], axis=1)
    w = inclusion_weight[:, None] * selector[:, None] * np.abs(c)
    residual = prediction - c
    numerator = 2 * np.einsum("ik,ij->kj", w * residual, phi, optimize=False)
    return numerator, float(w.sum()), float((w * residual**2).sum())


def gradient(numerator, denominator):
    """Aggregate chain numerators/denominators first; never average chain gradients."""
    return np.asarray(numerator, dtype=np.float64) / max(float(denominator), 1e-8)


def cross_panel_comparisons(panel_a, panel_b):
    """Input dictionaries contain pooled gradients for M, X, XY and SHUFFLED.

    Positive differences favor XY. This cross-product reduces row-noise bias;
    ratios retain finite-sample bias and this is not a WP or forecast-gain metric.
    """
    a = {key: np.asarray(panel_a[key]) - panel_a["M"] for key in ("X", "XY", "SHUFFLED")}
    b = {key: np.asarray(panel_b[key]) - panel_b["M"] for key in ("X", "XY", "SHUFFLED")}
    xy = float(np.sum(a["XY"] * b["XY"]))
    return {key: float(np.sum(a[key] * b[key])) - xy for key in ("X", "SHUFFLED")}
