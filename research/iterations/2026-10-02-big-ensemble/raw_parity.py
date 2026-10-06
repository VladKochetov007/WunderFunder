import sys,numpy as np,pyarrow.parquet as pq
sys.path.insert(0,"pkg_e7/v3");import wnn_cpp


W={k:np.ascontiguousarray(v,dtype=np.float32) for k,v in np.load("/tmp/raw_test.npz").items()}
m=wnn_cpp.Ensemble([W],[1.0],fast_activations=True,dynamic_int8=True,quant_backend="auto")
pf=pq.ParquetFile("/home/grinchenkoeg/WunderFunder-creative/datasets/valid.parquet");tv=np.load("/home/grinchenkoeg/WunderFunder-creative/cache/valid_targets.npz")["row_group"]
P=np.load("/home/grinchenkoeg/WunderFunder-creative/cache/pred_cmp_rawcorr_s3_valid.npy",mmap_mode="r")
for idx in (0,7):
    rg=int(tv[idx]);t=pf.read_row_group(rg);feat=[c for c in t.schema.names if c not in ("seq_ix","step_in_seq","need_prediction","is_scored","t0","t1")]
    X=np.stack([t.column(c).to_numpy() for c in feat],1).astype(np.float32)[:4000];m.reset();q=np.array([m.step(x) for x in X])
    p=np.asarray(P[idx,:4000]);print("seq",idx,"corr",np.corrcoef(p[99:,0],q[99:,0])[0,1],"maxdiff",np.abs(p[99:]-q[99:]).max())
