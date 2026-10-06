import sys,pickle,numpy as np
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"src")
from i1swap_aug import labels_with_mask,swap_instruments
from wnn.data import SequenceFile
moves,spreads=pickle.load(open(S+"cache/class3_moves_full.pkl","rb"))
t=np.load(S+"cache/valid_targets.npz");rg=t["row_group"];cls=t["market_class"];sf=SequenceFile(S+"datasets/valid.parquet")
ex=[];okf=[];ok1=[];w_ok=[]
for i in np.where(cls==3)[0][:40]:
    s=sf.read(int(rg[i]));lab,ok=labels_with_mask(s.features[:,0:22],moves,spreads)
    ex.append((np.abs(lab[ok]-s.targets[ok])<1e-3).all(1).mean());okf.append(ok.mean())
    w=np.abs(np.clip(s.targets,-2,2)).sum(1);sc=s.scored;w_ok.append((w*sc*ok).sum()/max((w*sc).sum(),1e-9))
    xs=swap_instruments(s.features);lab1,ok_1=labels_with_mask(xs[:,0:22],moves,spreads);ok1.append(ok_1.mean())
print("i0: reliable-row share %.3f, exact on reliable %.3f, scored weight covered %.3f"%(np.mean(okf),np.mean(ex),np.mean(w_ok)))
print("i1 swapped: reliable-row share %.3f (median %.3f)"%(np.mean(ok1),np.median(ok1)))
