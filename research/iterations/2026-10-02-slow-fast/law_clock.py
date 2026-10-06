import sys,pickle,numpy as np
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"src");sys.path.insert(0,S+"scripts")
from wnn.data import SequenceFile
from wnn.ticks import reconstruct,CLASS3_QUANTUM as q
moves,spreads=pickle.load(open(S+"cache/class3_moves_full.pkl","rb"))
t=np.load(S+"cache/valid_targets.npz");rg=t["row_group"];cls=t["market_class"];sf=SequenceFile(S+"datasets/valid.parquet")
Ls=[10,25,50,100,200,400,800]
stats={}
def add(k,m):stats.setdefault(k,[0,0]);stats[k][0]+=m.sum();stats[k][1]+=m.size
for i in np.where(cls==3)[0][:60]:
    s=sf.read(int(rg[i]));p=s.features[:,0:22];y=np.rint(s.targets/q).astype(int)
    a,b,unk=reconstruct(p,moves,spreads)
    if unk:continue
    n=len(y);mid2=a+b  # 2*mid in ticks
    sp=a-b
    # implied F (ticks) from uncapped side: inside/above -> t0=F-a ; below -> t1=b-F
    F0=y[:,0]+a;F1=b-y[:,1]
    # choose: if t1 == -(sp+2) (short capped) use F0 ; if t0==-(sp+2) use F1 ; else both should agree
    capS=y[:,1]==-(sp+2);capL=y[:,0]==-(sp+2)
    F=np.where(capS,F0,np.where(capL,F1,F0));agree=(~capS&~capL&(F0==F1))|capS|capL
    ev=np.r_[0,np.flatnonzero((np.diff(a)!=0)|(np.diff(b)!=0))+1];nxt=np.searchsorted(ev,np.arange(n),side="right")
    d4=np.where(nxt+3<len(ev),ev[np.minimum(nxt+3,len(ev)-1)]-np.arange(n),n)
    cm=np.r_[0,np.cumsum(mid2[ev])]
    k=nxt  # index of first future event
    Fev=np.where(k+4<=len(ev),np.rint((cm[np.minimum(k+4,len(ev))]-cm[np.minimum(k,len(ev))])/8.0),np.nan)
    cmr=np.r_[0,np.cumsum(mid2)]
    rows=np.arange(99,n-800)
    far=rows[d4[rows]>=64];near=rows[d4[rows]<16]
    for name,rr in(("near",near),("far",far)):
        ok=agree[rr];add((name,"implied-F consistent"),ok)
        add((name,"law 4 events"),(Fev[rr]==F[rr])[ok])
        for L in Ls:
            add((name,f"mid at t+{L}"),(np.rint(mid2[rr+L]/2.0)==F[rr])[ok])
            add((name,f"mean mid next {L} rows"),(np.rint((cmr[rr+1+L]-cmr[rr+1])/(2.0*L))==F[rr])[ok])
for k,(h,nn) in sorted(stats.items()):print(f"{k[0]:4s} {k[1]:28s} {h/nn:.3f} (n={nn})")
