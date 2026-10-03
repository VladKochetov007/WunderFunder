import pyarrow.parquet as pq, numpy as np, pandas as pd, sys
sys.path.insert(0,'research/campaign_2026-10-03/eda/hft')
from decode_lib import decode, ema, P0
from scipy.signal import lfilter
pd.set_option('display.width',250)
pf=pq.ParquetFile('datasets/valid.parquet')
sc=pd.read_csv('research/campaign_2026-10-03/eda/hft/scan_seq.csv')
def ewm(x,span):
    a=2/(span+1); return lfilter([a],[1,-(1-a)],x,zi=[(1-a)*x[0]])[0]
recs=[]
for g in sc.g.values:
    t=pf.read_row_group(int(g),columns=P0+['t0','t1','is_scored']).to_pandas()
    y0=t.t0.values.astype(np.float64); y1=t.t1.values.astype(np.float64)
    v=np.unique(np.abs(np.r_[np.diff(y0),np.diff(y1),y0,y1])); q=v[v>1e-6].min()
    A,B,sp,bk=decode(t[P0].values)
    M=(A+B)/2
    # decoder consistency: implied F0 = A + k0 should be smooth; check implied-label agreement
    k0=y0/q; k1=y1/q
    d=dict(g=g,q=round(q,4))
    df=pd.DataFrame(dict(k0=k0,k1=k1,sp=sp,sc=t.is_scored.values))
    df['f_sp']=-sp/2
    for s in [5,20,100,400,1600]:
        df[f'mr{s}']=-(M-ewm(M,s))
    df['g']=g; df['q']=q
    recs.append(df[df.sc])
R=pd.concat(recs)
R.to_parquet('research/campaign_2026-10-03/eda/hft/decode_eval_rows.parquet')
def wcorr(y,p):
    y=np.clip(y,-2,2); p=np.clip(p,-2,2); w=np.abs(y)
    ym=(w*y).sum()/w.sum(); pm=(w*p).sum()/w.sum()
    return (w*(y-ym)*(p-pm)).sum()/np.sqrt((w*(y-ym)**2).sum()*(w*(p-pm)**2).sum())
feats=['f_sp']+[c for c in R.columns if c.startswith('mr')]
for qv,G in [('all',R)]+list(R.groupby('q')):
    line=[f'{qv}: n={len(G)}']
    for f in feats:
        line.append(f'{f}: c0={np.corrcoef(G.k0,G[f])[0,1]:.3f} c1={np.corrcoef(G.k1,G[f])[0,1]:.3f}')
    print(' | '.join(line))
# pooled WP in target units with oracle q scaling
for f in feats:
    p0=R[f]*R.q; 
    print(f,'WP t0 %.3f  t1(neg feature for mr) %.3f'%(wcorr(R.k0*R.q,p0), wcorr(R.k1*R.q, (R[f] if f=='f_sp' else -R[f])*R.q)))
