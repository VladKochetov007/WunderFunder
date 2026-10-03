import numpy as np, pyarrow.parquet as pq
np.set_printoptions(linewidth=220, precision=3, suppress=True)
names = pq.ParquetFile('datasets/valid.parquet').schema_arrow.names
feat = names[4:116]
def wp(y,p):
    y=np.clip(y,-2,2); p=np.clip(p,-2,2); w=np.abs(y); my=(w*y).sum()/w.sum(); mp=(w*p).sum()/w.sum()
    return (w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum())
def load(path, gs, scored=False):
    pf=pq.ParquetFile(path); X=[];Y=[];M=[]
    for g in gs:
        t=pf.read_row_group(g, columns=feat+['t0','t1']+(['is_scored'] if scored else [])).to_pandas()
        X.append(t[feat].values.astype(np.float32)[99:]); Y.append(t[['t0','t1']].values.astype(np.float32)[99:])
        M.append(t.is_scored.values[99:] if scored else np.ones(len(t)-99,bool))
    return np.vstack(X),np.vstack(Y),np.concatenate(M)
Xtr,Ytr,_=load('datasets/train.parquet', range(0,10607,265))
Xva,Yva,Mva=load('datasets/valid.parquet', range(0,1873,47), True)
print(Xtr.shape, Xva.shape, Mva.mean())
def fit(cols):
    A=np.c_[Xtr[:,cols].astype(np.float64), np.ones(len(Xtr))]; w=np.abs(np.clip(Ytr,-2,2))
    out=[]
    for j in range(2):
        sw=np.sqrt(w[:,j]); b=np.linalg.lstsq(A*sw[:,None], Ytr[:,j]*sw, rcond=None)[0]
        p=np.c_[Xva[:,cols],np.ones(len(Xva))]@b; out.append(wp(Yva[Mva,j],p[Mva]))
    return out
idx={n:i for i,n in enumerate(feat)}
groups={'i0_p':[idx[f'i0_p{k}'] for k in range(22)], 'i0_p+v':[idx[f'i0_{g}{k}'] for g in 'pv' for k in range(22)],
 'i0 all':list(range(52)),'i1 all':list(range(52,104)),'a':list(range(104,112)),'all':list(range(112))}
for k,v in groups.items(): r=fit(v); print('%-8s WP t0 %.4f t1 %.4f mean %.4f'%(k,r[0],r[1],np.mean(r)))
