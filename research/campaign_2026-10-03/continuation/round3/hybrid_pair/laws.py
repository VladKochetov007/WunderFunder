from __future__ import annotations

import numpy as np

LAWS = ("independent_hybrid", "shared_profitable_exit", "extrema_only")


def returned_pair(stop: dict, side: int) -> np.ndarray:
    """Return selected side's extreme and opposite execution at one selected event."""
    pair = np.array(stop["execution_pair"], dtype=float)
    pair[side] = stop["extreme"]
    return pair


def run_law(ask: np.ndarray, bid: np.ndarray, valid: np.ndarray, law: str) -> dict:
    """Evaluate the frozen pair rule and never scan beyond an unknown transition."""
    if law not in LAWS:
        raise ValueError(law)
    pair = np.full(2, np.nan)
    reasons = ["pending", "pending"]
    lags = [-1, -1]
    stops: list[dict | None] = [None, None]
    selected_side, selected_lag, required_lag = -1, -1, -1
    if not valid[0]:
        return {
            "pair": pair,
            "reasons": ["unknown_origin"] * 2,
            "lags": lags,
            "selected_side": -1,
            "selected_lag": -1,
            "required_prefix_lag": -1,
        }
    high, low = bid[0], ask[0]
    terminal_reason = "sequence_end" if len(ask) <= 32 else "horizon_censored"
    for j in range(1, min(len(ask), 33)):
        if not valid[j]:
            terminal_reason = "unknown_transition"
            break
        high, low = max(high, bid[j]), min(low, ask[j])
        extreme = [high - ask[0], bid[0] - low]
        execution = [bid[j] - ask[0], bid[0] - ask[j]]
        hits = [bid[j] <= high - 1, ask[j] >= low + 1]
        new_stops = []
        for side in range(2):
            if stops[side] is None and hits[side]:
                stop = {
                    "lag": j,
                    "extreme": float(extreme[side]),
                    "execution": float(execution[side]),
                    "execution_pair": execution.copy(),
                }
                stops[side] = stop
                new_stops.append(side)
                if law != "shared_profitable_exit":
                    use_extreme = law == "extrema_only" or extreme[side] > 0
                    pair[side] = extreme[side] if use_extreme else execution[side]
                    reasons[side], lags[side] = "resolved", j
        if law == "shared_profitable_exit":
            profitable = [s for s in new_stops if stops[s]["extreme"] > 0]
            if profitable:
                selected_side = max(profitable, key=lambda s: (stops[s]["extreme"], -s))
                stop = stops[selected_side]
                pair = returned_pair(stop, selected_side)
                selected_lag = required_lag = j
                reasons, lags = ["resolved"] * 2, [j, j]
                break
            if all(s is not None for s in stops):
                selected_side = min(
                    range(2), key=lambda s: (stops[s]["lag"], -stops[s]["extreme"], s)
                )
                stop = stops[selected_side]
                pair = returned_pair(stop, selected_side)
                selected_lag, required_lag = stop["lag"], j
                reasons, lags = ["resolved"] * 2, [selected_lag, selected_lag]
                break
        elif reasons == ["resolved", "resolved"]:
            required_lag = max(lags)
            break
    reasons = [terminal_reason if r == "pending" else r for r in reasons]
    if np.isfinite(pair).all() and not valid[: required_lag + 1].all():
        raise AssertionError("Computed pair crossed an unknown step")
    return {
        "pair": pair,
        "reasons": reasons,
        "lags": lags,
        "selected_side": selected_side,
        "selected_lag": selected_lag,
        "required_prefix_lag": required_lag,
    }


def scalar_reference(ask: np.ndarray, bid: np.ndarray, law: str) -> np.ndarray:
    """Independently specify first reversals using complete prefix extrema."""
    stops = []
    for side in range(2):
        record = None
        for j in range(1, min(len(ask), 33)):
            if side == 0:
                e = max(bid[: j + 1]) - ask[0]
                hit = max(bid[: j + 1]) - bid[j] >= 1
                execution = bid[j] - ask[0]
            else:
                e = bid[0] - min(ask[: j + 1])
                hit = ask[j] - min(ask[: j + 1]) >= 1
                execution = bid[0] - ask[j]
            if hit:
                record = (j, e, execution)
                break
        stops.append(record)
    if law == "shared_profitable_exit":
        candidates = [(v[0], -v[1], s) for s, v in enumerate(stops) if v is not None and v[1] > 0]
        if candidates:
            chosen = min(candidates)[2]
        elif all(v is not None for v in stops):
            chosen = min((v[0], -v[1], s) for s, v in enumerate(stops))[2]
        else:
            return np.full(2, np.nan)
        j = stops[chosen][0]
        output = np.array([bid[j] - ask[0], bid[0] - ask[j]])
        output[chosen] = stops[chosen][1]
        return output
    return np.array(
        [
            np.nan if s is None else s[1] if law == "extrema_only" or s[1] > 0 else s[2]
            for s in stops
        ]
    )
