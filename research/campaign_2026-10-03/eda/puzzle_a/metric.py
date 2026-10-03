import numpy as np
def wcorr(pred, y):
    """Weighted Pearson as described in the abstract task: both clipped to [-2,2], weights |target|."""
    p = np.clip(np.asarray(pred, np.float64), -2, 2); t = np.clip(np.asarray(y, np.float64), -2, 2)
    w = np.abs(t)
    if w.sum() == 0: return np.nan
    mp = (w*p).sum()/w.sum(); mt = (w*t).sum()/w.sum()
    cov = (w*(p-mp)*(t-mt)).sum(); vp = (w*(p-mp)**2).sum(); vt = (w*(t-mt)**2).sum()
    return cov/np.sqrt(vp*vt) if vp > 0 and vt > 0 else 0.0
