import sys,json,numpy as np,pyarrow.parquet as pq
sys.path.insert(0,"pkg_e7/v3");import wnn_cpp
d="pkg_pair_fullft";mem=json.load(open(f"{d}/members.json"))
W=[{k:np.ascontiguousarray(v,dtype=np.float32) for k,v in np.load(d+"/"+m["weights"]).items()} for m in mem];C=[m["coef"] for m in mem]
mods={"int8":wnn_cpp.Ensemble(W,C,fast_activations=True,dynamic_int8=True,quant_backend="auto"),"fp32":wnn_cpp.Ensemble(W,C,fast_activations=False,dynamic_int8=False)}
pf=pq.ParquetFile("/home/grinchenkoeg/WunderFunder-creative/datasets/valid.parquet")
rng=np.random.default_rng(0);ids=rng.choice(pf.num_row_groups,int(sys.argv[1]),replace=False)
P={k:[] for k in mods};Ys=[];Ms=[]
for i in ids:
    t=pf.read_row_group(int(i));feat=[c for c in t.schema.names if c not in ("seq_ix","step_in_seq","need_prediction","is_scored","t0","t1")]
    X=np.stack([t.column(c).to_numpy() for c in feat],1).astype(np.float32);Y=np.stack([t.column("t0").to_numpy(),t.column("t1").to_numpy()],1);m=t.column("is_scored").to_numpy().astype(bool);m[:99]=False
    for k,mod in mods.items():
        mod.reset();P[k].append(np.array([mod.step(x) for x in X])[m])
    Ys.append(Y[m])
Y=np.concatenate(Ys)
def wp(p):
    o=[]
    for j in range(2):
        y=np.clip(Y[:,j],-2,2);q=np.clip(p[:,j],-2,2);w=np.abs(y);my=(w*y).sum()/w.sum();mp=(w*q).sum()/w.sum()
        o.append((w*(y-my)*(q-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(q-mp)**2).sum()))
    return np.mean(o)
a=np.concatenate(P["int8"]);b=np.concatenate(P["fp32"])
print("rows",len(Y),"WP int8 %.5f fp32 %.5f diff %+.5f corr %.6f maxdiff %.4f"%(wp(a),wp(b),wp(a)-wp(b),np.corrcoef(a[:,0],b[:,0])[0,1],np.abs(a-b).max()))
