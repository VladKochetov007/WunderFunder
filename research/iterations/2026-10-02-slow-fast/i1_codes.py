import sys,pickle,numpy as np,pyarrow.parquet as pq
sys.path.insert(0,"/home/grinchenkoeg/WunderFunder-creative/src")
from wnn.ticks import reconstruct,_keys
R="/home/grinchenkoeg/WunderFunder-creative/"
obj=pickle.load(open(R+"cache/class3_moves_full.pkl","rb"));print(type(obj[0]),len(obj[0]),type(obj[1]),len(obj[1]))
moves,spreads=obj
known_codes=set(k for pair in moves for k in pair)
cls=np.load(R+"research/artifacts/market-classes-v1.npz")["valid"]
pf=pq.ParquetFile(R+"datasets/valid.parquet");fc=pf.schema_arrow.names[4:116]
for c in (3,2,0):
    res=[]
    for g in np.where(cls==c)[0][:15]:
        X=np.column_stack([x.to_numpy() for x in pf.read_row_group(int(g),columns=fc).columns]).astype(np.float32)
        r=[]
        for name,sl in (("i0",slice(0,22)),("i1",slice(52,74))):
            x=X[:,sl];ev=np.flatnonzero((x[1:]!=x[:-1]).any(1))+1
            k=_keys(x);codes=set(k);_,_,unk=reconstruct(x,moves)
            r+= [len(codes),len(codes&known_codes)/max(len(codes),1),1-unk/max(len(ev),1),len(ev)/len(x)]
        res.append(r)
    a=np.array(res).mean(0)
    print(f"class{c}: i0 codes {a[0]:.0f} known {a[1]:.2f} transitions-known {a[2]:.2f} ev-rate {a[3]:.3f} | i1 codes {a[4]:.0f} known {a[5]:.2f} transitions-known {a[6]:.2f} ev-rate {a[7]:.3f}")
