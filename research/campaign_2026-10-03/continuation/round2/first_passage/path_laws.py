from __future__ import annotations

import numpy as np

FAMILIES = ("fixed_extrema", "competing_barriers", "trailing_extrema")


def functional(ask: np.ndarray, bid: np.ndarray, valid: np.ndarray, family: str) -> dict:
    """Evaluate one frozen diagnostic law; stop on unknown paths, never bridge gaps."""
    output = np.full(2, np.nan)
    lags = np.full(2, -1, dtype=int)
    reasons = ["pending", "pending"]
    if not valid[0]:
        return {"pair": output, "lags": lags, "reasons": ["unknown_origin"] * 2}
    limit = 16 if family == "fixed_extrema" else 32
    if family == "fixed_extrema":
        if len(ask) <= 16:
            reason = "sequence_end" if valid.all() else "unknown_transition"
            reasons = [reason] * 2
        elif not valid[:17].all():
            reasons = ["unknown_transition"] * 2
        else:
            output[:] = [bid[1:17].max() - ask[0], bid[0] - ask[1:17].min()]
            lags[:] = 16
            reasons = ["resolved"] * 2
    elif family in ("competing_barriers", "trailing_extrema"):
        high, low = bid[0], ask[0]
        for j in range(1, min(limit + 1, len(ask))):
            if not valid[j]:
                reasons = ["unknown_transition" if v == "pending" else v for v in reasons]
                break
            if family == "competing_barriers":
                returns = [bid[j] - ask[0], bid[0] - ask[j]]
                hit = [abs(v) >= 2 for v in returns]
            else:
                high, low = max(high, bid[j]), min(low, ask[j])
                returns = [high - ask[0], bid[0] - low]
                hit = [bid[j] <= high - 1, ask[j] >= low + 1]
            for side in range(2):
                if reasons[side] == "pending" and hit[side]:
                    output[side], lags[side], reasons[side] = returns[side], j, "resolved"
            if reasons == ["resolved", "resolved"]:
                break
        reason = "sequence_end" if len(ask) <= limit else "horizon_censored"
        reasons = [reason if v == "pending" else v for v in reasons]
    else:
        raise ValueError(f"Unknown family {family}")
    return {"pair": output, "lags": lags, "reasons": reasons}


def independent_reference(ask: np.ndarray, bid: np.ndarray, family: str) -> np.ndarray:
    """Scalar specification for complete synthetic paths, independent of engine state."""
    if family == "fixed_extrema":
        return np.array([max(bid[1:17]) - ask[0], bid[0] - min(ask[1:17])])
    outputs = []
    for side in range(2):
        payoff = np.nan
        for j in range(1, min(len(ask), 33)):
            if family == "competing_barriers":
                ret = bid[j] - ask[0] if side == 0 else bid[0] - ask[j]
                stopped = ret >= 2 or ret <= -2
            elif side == 0:
                peak = max(bid[: j + 1])
                stopped = peak - bid[j] >= 1
                ret = peak - ask[0]
            else:
                trough = min(ask[: j + 1])
                stopped = ask[j] - trough >= 1
                ret = bid[0] - trough
            if stopped:
                payoff = ret
                break
        outputs.append(payoff)
    return np.array(outputs)
