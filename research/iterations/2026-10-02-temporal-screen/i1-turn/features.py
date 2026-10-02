"""One fixed lag1 i1-code turn. Pure full-prefix streaming, no data I/O."""
import numpy as np
from numba import njit

# State: prev_raw4, prev_event_delta2, T,E,count,net_delta2,age,started.
@njit(cache=False, fastmath=False)
def update(raw, state):
    out = np.zeros((len(raw), 8), np.float64)
    for i in range(len(raw)):
        d0 = 0.; d1 = 0.
        if state[12] == 0:
            state[12] = 1.
        else:
            reset = (raw[i,0] != np.float32(state[0])) and (raw[i,1] != np.float32(state[1]))
            event = (raw[i,2] != np.float32(state[2])) or (raw[i,3] != np.float32(state[3]))
            d0 = np.float64(np.float32(raw[i,2] - np.float32(state[2])))
            d1 = np.float64(np.float32(raw[i,3] - np.float32(state[3])))
            if reset:
                for k in range(4,12): state[k] = 0.
            else:
                state[11] += 1.
            if event:
                if state[8] > 0:
                    state[6] += state[4]*d1 - state[5]*d0
                state[7] += d0*d0 + d1*d1
                state[8] += 1.
                state[9] += d0; state[10] += d1
                state[4] = d0; state[5] = d1
        for k in range(4): state[k] = raw[i,k]
        out[i,0] = d0; out[i,1] = d1
        out[i,2] = state[9]; out[i,3] = state[10]
        out[i,4] = state[7]; out[i,5] = state[8]; out[i,6] = state[11]
        out[i,7] = state[6]/state[7] if state[7] > 0 else 0.
    return out

def full(raw):
    raw = np.asarray(raw, np.float32)
    assert raw.ndim == 2 and raw.shape[1] == 4 and np.isfinite(raw).all()
    return update(raw, np.zeros(13, np.float64))
