import sys,pickle,numpy as np
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"scripts");sys.path.insert(0,S+"src")
from online_c3 import rebuilt_labels
from wnn.data import SequenceFile
from wnn.ticks import reconstruct,CLASS3_QUANTUM
moves,spreads=pickle.load(open(S+"cache/class3_moves_full.pkl","rb"))
t=np.load(S+"cache/valid_targets.npz");rg=t["row_group"];cls=t["market_class"]
sf=SequenceFile(S+"datasets/valid.parquet")
acc=[];cov0=[];cov1=[];unk1=[];ev1=[];stat=[]
for i in np.where(cls==3)[0][:40]:
    s=sf.read(int(rg[i]));X=s.features;y=s.targets
    a,b,u=reconstruct(X[:,0:22],moves,spreads);lab,known=rebuilt_labels(a,b);k=known<len(y)
    acc.append((np.abs(lab[k]-y[k])<1e-3).all(1).mean());cov0.append(k.mean())
    X1=X[:,52:74];a1,b1,u1=reconstruct(X1,moves,spreads);lab1,known1=rebuilt_labels(a1,b1);k1=known1<len(y)
    ne=int(((X1[1:]!=X1[:-1]).any(1)).sum());cov1.append(k1.mean());unk1.append(u1/max(ne,1));ev1.append(ne)
    stat.append([np.mean(lab1[k1]/CLASS3_QUANTUM,0).round(2).tolist() if k1.any() else None,(a1-b1).min(),(a1-b1).max()])
print("i0 rebuilt exact on known rows %.3f, label coverage %.3f"%(np.mean(acc),np.mean(cov0)))
print("i1 label coverage %.3f, unknown-transition share %.3f, price events/seq median %d"%(np.mean(cov1),np.mean(unk1),np.median(ev1)))
print("i1 sample stats (mean label ticks, spread min/max):",stat[:5])
print("i0 label mean ticks (true):",np.round(np.mean([sf.read(int(rg[i])).targets.mean(0) for i in np.where(cls==3)[0][:10]],0)/CLASS3_QUANTUM,2))
