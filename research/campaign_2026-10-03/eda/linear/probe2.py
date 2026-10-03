import pyarrow.parquet as pq, numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=220)
f = pq.ParquetFile('datasets/valid.parquet')
for rg in [0, 1, 2, 3, 500, 1000]:
    tb = f.read_row_group(rg).to_pandas()
    y0 = tb.t0.to_numpy(np.float64); y1 = tb.t1.to_numpy(np.float64)
    nz0 = np.abs(y0[y0 != 0]); nz1 = np.abs(y1[y1 != 0])
    q0 = nz0.min(); q1 = nz1.min()
    r0 = y0 / q0; r1 = y1 / q1
    print(f"rg{rg} seq{tb.seq_ix[0]} q0={q0:.5f} q1={q1:.5f} int-resid0={np.abs(r0-np.round(r0)).max():.3f} int-resid1={np.abs(r1-np.round(r1)).max():.3f} std0={y0.std():.3f} std1={y1.std():.3f} corr={np.corrcoef(y0,y1)[0,1]:.3f} scored={tb.is_scored.mean():.3f}")
    sc = tb.is_scored.to_numpy()
    # scored vs |y|
    print('   mean|y0| scored/unscored', np.abs(y0[sc]).mean(), np.abs(y0[~sc & tb.need_prediction.to_numpy()]).mean(), ' mean|y1|', np.abs(y1[sc]).mean(), np.abs(y1[~sc& tb.need_prediction.to_numpy()]).mean())
    # does scored relate to target changes?
    d0 = np.r_[0, np.diff(y0)] != 0
    print('   P(target changes | scored)', d0[sc].mean(), ' | unscored', d0[~sc].mean())
    # scored vs feature change of i0 prices
    X = tb.iloc[:, 4:116].to_numpy(np.float32)
    pch = np.r_[False, np.any(np.diff(X[:, 0:11], axis=0) != 0, axis=1)]
    ach = np.r_[False, np.any(np.diff(X[:, 11:22], axis=0) != 0, axis=1)]
    tch = np.r_[False, np.any(np.diff(X[:, 44:48], axis=0) != 0, axis=1)]
    print('   P(bid chg|sc)', pch[sc].mean(), pch[~sc].mean(), ' P(ask chg|sc)', ach[sc].mean(), ach[~sc].mean(), ' P(trade px chg|sc)', tch[sc].mean(), tch[~sc].mean())
    # rank of bid prices, ask prices
    for nm, sl in [('bid p', slice(0, 11)), ('ask p', slice(11, 22)), ('bid v', slice(22, 33)), ('i0 dp', slice(44, 48)), ('i1 bid p', slice(52, 63))]:
        Z = X[:, sl].astype(np.float64); Z = Z - Z.mean(0)
        s = np.linalg.svd(Z, compute_uv=False)
        print(f'   sv {nm}', (s**2 / (s**2).sum())[:6])
    print('   a7 uniques', np.unique(X[:, 111])[:30])
    print('   steps where scored by step bucket', np.histogram(np.where(sc)[0], bins=10, range=(0, 20000))[0])
