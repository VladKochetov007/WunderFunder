def wsums(y, p, m):
    """|y|-weighted raw sums per target over selected rows (clip ±2): [w, wy, wp, wyy, wpp, wyp]."""
    yy = np.clip(y[m].astype(np.float64), -2, 2); pp = np.clip(p[m].astype(np.float64), -2, 2); w = np.abs(yy)
    return np.stack([w.sum(0), (w*yy).sum(0), (w*pp).sum(0), (w*yy*yy).sum(0), (w*pp*pp).sum(0), (w*yy*pp).sum(0)], -1)
def wp_from(s):
    n, sy, sp, syy, spp, syp = np.moveaxis(np.asarray(s), -1, 0)
    r = (syp/n - sy*sp/n**2) / np.sqrt((syy/n - (sy/n)**2) * (spp/n - (sp/n)**2))
    return float(np.mean(r))
SEQ_SOTA = np.stack([wsums(Y[i], SOTA[i], MASK[i]) for i in range(len(Y))])
SEQ_REF = np.stack([wsums(Y[i], REF[i], MASK[i]) for i in range(len(Y))])
