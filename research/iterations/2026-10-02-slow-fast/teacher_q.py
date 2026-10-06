import sys,numpy as np
S="/home/grinchenkoeg/WunderFunder-creative/"
rgs=np.load(S+"cache/i1swap_train_rg.npy");Y=np.load(S+"cache/i1swap_train_y.npy",mmap_mode="r");OK=np.load(S+"cache/i1swap_train_ok.npy",mmap_mode="r")
for t in sys.argv[1:]:
    P=np.load(S+f"cache/pred_{t}_i1swap.npy",mmap_mode="r");out=[]
    for j in range(2):
        ys=[];ps=[]
        for i in range(0,400):
            m=np.asarray(OK[i]).copy();m[:99]=False;ys.append(np.asarray(Y[i])[m,j]);ps.append(np.asarray(P[i])[m,j])
        y=np.clip(np.concatenate(ys),-2,2);p=np.concatenate(ps);w=np.abs(y)
        my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum();out.append((w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum()))
    print(t,"teacher WP vs rebuilt labels on reliable rows of swapped i1 (400 seqs): %.4f"%np.mean(out),"pred std",np.asarray(P[:50]).std().round(3))
