"""GLOBAL-HEAD probe, step 2: joint weighted-ridge readout refit toward the equal-weight ensemble; chain-half cross-fit, clipped pooled WP."""
import numpy as np
S="/home/grinchenkoeg/WunderFunder-creative/";members=["d256","d256s4","ref","refs4","c500"]
def wp(y,p):
    y=np.clip(y,-2,2);p=np.clip(p,-2,2);w=np.abs(y);my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum()
    return (w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum())
tot={}
for fold in (0,1):
    d=np.load(S+f"cache/globalhead_f{fold}.npz")
    Phi=np.concatenate([d[f"phi_{m}"] for m in members]+[np.ones((len(d["y"]),1),np.float32)],1).astype(np.float64)
    B0=np.concatenate([d[f"W_{m}"].T/5 for m in members]+[np.mean([d[f"b_{m}"] for m in members],0)[None]],0)  # (D+1,2)
    p0=Phi@B0;avg=np.mean([d[f"pred_{m}"] for m in members],0)
    print(f"fold{fold} rows {len(Phi)} dims {Phi.shape[1]} | readout repro max|d| {np.abs(p0-avg).max():.4f}")
    y=d["y"].astype(np.float64);ch=d["chain"];half=(np.array([hash((int(c),7))%2 for c in ch]))
    sd=Phi.std(0);sd[sd<1e-6]=1.0;sd[-1]=1.0
    for lam in (1e3,1e4,1e5,1e6):
        for A in (0,1):
            a=half==A;b=~a;res=[]
            for j in range(2):
                yc=np.clip(y[:,j],-2,2);w=np.abs(yc)
                Pa=Phi[a];G=(Pa*w[a,None]).T@Pa;r=(Pa*w[a,None]).T@(yc[a]-p0[a,j])
                G+=lam*np.diag(sd**2);delta=np.linalg.solve(G,r)
                res.append((wp(y[b,j],p0[b,j]),wp(y[b,j],p0[b,j]+Phi[b]@delta)))
            base=np.mean([r[0] for r in res]);new=np.mean([r[1] for r in res])
            tot.setdefault(lam,[]).append(new-base)
            print(f"  fold{fold} lam {lam:.0e} fit-half {A}: base {base:.4f} refit {new:.4f} delta {new-base:+.4f}")
print({f"{k:.0e}":round(float(np.mean(v)),5) for k,v in tot.items()})
