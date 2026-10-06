import numpy as np,pyarrow.parquet as pq
R="/home/grinchenkoeg/WunderFunder-creative/"
cls=np.load(R+"research/artifacts/market-classes-v1.npz")["valid"]
pf=pq.ParquetFile(R+"datasets/valid.parquet");names=pf.schema_arrow.names;fc=names[4:116]
def load(c,n=8):
    ids=np.where(cls==c)[0][:n]
    return np.concatenate([np.column_stack([x.to_numpy() for x in pf.read_row_group(int(g),columns=fc).columns]).astype(np.float64) for g in ids])
for c in (3,0):
    X=load(c)
    print(f"=== class {c}")
    for blk,(b0,a0) in {"i0 price":(0,11),"i0 vol":(22,33),"i1 price":(52,63),"i1 vol":(74,85)}.items():
        B=X[:,b0:b0+11];A=X[:,a0:a0+11]
        # marginal distribution match: best matching ask col for each bid col, using quantiles, both A and -A
        qs=np.linspace(0.02,0.98,25)
        qb=np.quantile(B,qs,axis=0).T;qa=np.quantile(A,qs,axis=0).T;qna=np.quantile(-A,qs,axis=0).T
        d_pos=np.abs(qb[:,None,:]-qa[None,:,:]).mean(-1);d_neg=np.abs(qb[:,None,:]-qna[None,:,:]).mean(-1)
        print(f" {blk}: bid-col best match dist same-sign {d_pos.min(1).mean():.3f} neg-sign {d_neg.min(1).mean():.3f}; argmin(neg) {d_neg.argmin(1).tolist()}")
        print(f"   bid col means {B.mean(0).round(2).tolist()}\n   ask col means {A.mean(0).round(2).tolist()}")
    T=X[:,44:52];print(" trades dp0..3 means",T[:,:4].mean(0).round(2),"dv0..3",T[:,4:].mean(0).round(2))
