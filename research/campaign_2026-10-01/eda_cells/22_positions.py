# Shared per-row event coordinates: rows since the last quote move (causal) and rows until the next one (non-causal),
# plus the length of the run each row belongs to. Used by the error budget and the mechanism plots.
SINCE = np.zeros(Y.shape[:2], np.int32); UNTIL = np.zeros(Y.shape[:2], np.int32); RUNLEN = np.zeros(Y.shape[:2], np.int32)
for _i in range(len(Y)):
    _e = np.flatnonzero(EVENT[_i]); _idx = np.arange(20000)
    _k = np.searchsorted(_e, _idx, side="right") - 1
    _prev = np.where(_k >= 0, _e[np.maximum(_k, 0)], 0)
    _nxt = np.where(_k + 1 < len(_e), _e[np.minimum(_k + 1, len(_e) - 1)], 20000)
    SINCE[_i] = _idx - _prev; UNTIL[_i] = _nxt - _idx; RUNLEN[_i] = _nxt - _prev
POS_BINS = np.array([0, 2, 8, 32, 128, 512, 20001]); POS_LABELS = ["<2", "2–8", "8–32", "32–128", "128–512", "≥512"]
