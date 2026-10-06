import sys,pickle,numpy as np
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"src")
from wnn.data import SequenceFile
from wnn.ticks import reconstruct,CLASS3_QUANTUM as q
moves,spreads=pickle.load(open(S+"cache/class3_moves_full.pkl","rb"))
t=np.load(S+"cache/valid_targets.npz");rg=t["row_group"];cls=t["market_class"];sf=SequenceFile(S+"datasets/valid.parquet")
Ks=[1,2,3,4,5,6,8];hit={k:0 for k in Ks};hit_cur={k:0 for k in Ks};anyk=0;nfail=0;nf=0;cover=0
for i in np.where(cls==3)[0][:60]:
    s=sf.read(int(rg[i]));p=s.features[:,0:22];y=np.rint(s.targets/q).astype(int)
    a,b,unk=reconstruct(p,moves,spreads)
    if unk:continue
    n=len(y);mid2=a+b;sp=a-b
    F0=y[:,0]+a;F1=b-y[:,1];capL=y[:,0]==-(sp+2);F=np.where(capL,F1,F0)
    ev=np.r_[0,np.flatnonzero((np.diff(a)!=0)|(np.diff(b)!=0))+1];nxt=np.searchsorted(ev,np.arange(n),side="right")
    cm=np.r_[0,np.cumsum(mid2[ev])];m=len(ev)
    def FK(K,with_cur=False):
        hi=np.minimum(nxt+K,m);lo=nxt
        s_=cm[hi]-cm[lo];cnt=hi-lo
        if with_cur:s_=s_+mid2;cnt=cnt+1
        return np.where((nxt+K<=m)|with_cur,np.rint(s_/(2.0*np.maximum(cnt,1))),np.nan)
    F4=FK(4);rows=np.arange(99,n);d4=np.where(nxt+3<m,ev[np.minimum(nxt+3,m-1)]-np.arange(n),n)
    far=rows[(d4[rows]>=64)];fail=far[(F4[far]!=F[far])]
    nf+=len(far);nfail+=len(fail);matched=np.zeros(len(fail),bool)
    for K in Ks:
        h=FK(K)[fail]==F[fail];hit[K]+=h.sum();matched|=h
        hc=FK(K,True)[fail]==F[fail];hit_cur[K]+=hc.sum();matched|=hc
    anyk+=matched.sum()
print(f"far rows {nf}, law-4 failures {nfail} ({nfail/nf:.3f})")
print("among failures, F equals mean of next K event mids:",{K:round(hit[K]/nfail,3) for K in Ks})
print("  ... including current mid in the average:",{K:round(hit_cur[K]/nfail,3) for K in Ks})
print("explained by some K variant: %.3f"%(anyk/nfail))
