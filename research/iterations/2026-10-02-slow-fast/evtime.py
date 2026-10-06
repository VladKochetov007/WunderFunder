import numpy as np,pyarrow.parquet as pq
R="/home/grinchenkoeg/WunderFunder-creative/"
cls=np.load(R+"research/artifacts/market-classes-v1.npz")["valid"];Z=np.load("quiet_z.npz")["valid"]
pf=pq.ParquetFile(R+"datasets/valid.parquet");names=pf.schema_arrow.names
i0=[n for n in names if n.startswith("i0_")];i1=[n for n in names if n.startswith("i1_")];aa=[n for n in names if n[0]=="a" and n[1:].isdigit()]
for c in (3,2,0):
    ids=np.where(cls==c)[0][:6];acc=[]
    for g in ids:
        t=pf.read_row_group(int(g),columns=i0+i1+aa+["t0","t1"]);X=np.column_stack([x.to_numpy() for x in t.columns]).astype(float)
        n0,n1=len(i0),len(i1);ch=np.diff(X,axis=0)!=0
        ev0=ch[:,:n0].any(1);ev0p=ch[:,:22].any(1);ev1=ch[:,n0:n0+n1].any(1);yc=ch[:,-2:].any(1)
        acc.append([ev0.mean(),ev0p.mean(),yc.mean(),(yc&~ev0).sum()/max(yc.sum(),1),(yc&~ev0p).sum()/max(yc.sum(),1),(yc&~(ev0|ev1)).sum()/max(yc.sum(),1)])
    a=np.mean(acc,0)
    print(f"class{c}: rows with i0 change {a[0]:.3f}, i0 price-part change {a[1]:.3f}, target change {a[2]:.3f}; target changes w/o any i0 change {a[3]:.3f}, w/o i0 price change {a[4]:.3f}, w/o any i0/i1 change {a[5]:.3f}")
