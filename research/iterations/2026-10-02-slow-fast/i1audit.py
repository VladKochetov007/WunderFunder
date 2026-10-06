import sys,numpy as np,torch
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"scripts");sys.path.insert(0,S+"src")
import train_rnn
from train_rnn import make_model
from predict_run import predict
from i1swap_aug import swap_instruments
from wnn.data import SequenceFile
rgs=np.load(S+"cache/i1swap_train_rg.npy");Yr=np.load(S+"cache/i1swap_train_y.npy",mmap_mode="r");OK=np.load(S+"cache/i1swap_train_ok.npy",mmap_mode="r")
tt=np.load(S+"cache/train_targets.npz");rgt=tt["row_group"];Yt=tt["y"];gi={g:i for i,g in enumerate(rgt)}
sf=SequenceFile(S+"datasets/train.parquet");n=int(sys.argv[1]);sel=np.arange(0,len(rgs),max(1,len(rgs)//n))[:n]
def wp(Y,P,W):
    o=[]
    for j in range(2):
        y=np.clip(Y[...,j],-2,2).ravel().astype(np.float64);p=np.clip(P[...,j],-2,2).ravel().astype(np.float64);w=np.abs(y)*W.ravel()
        my=(w*y).sum()/w.sum();mp=(w*p).sum()/w.sum();o.append((w*(y-my)*(p-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(p-mp)**2).sum()))
    return np.mean(o)
dev=torch.device("cuda")
for run in sys.argv[2:]:
    name,ep=run.split("@");ck=torch.load(S+f"artifacts/runs/{name}/model_ep{ep}.pt",map_location="cpu",weights_only=False)
    m=make_model(ck["args"],ck["n_in"],ck["model"]["head.2.weight"].shape[0]).to(dev);m.load_state_dict(ck["model"]);m.eval()
    train_rnn.A_MODE=ck["args"].get("a_mode") or "none"
    raw=[sf.read(int(rgs[i])).features for i in sel]
    pr=lambda L:np.concatenate([predict(m,torch.from_numpy(np.stack(L[k:k+8])),4000,dev)[...,:2] for k in range(0,len(L),8)])
    Ps=pr([swap_instruments(x) for x in raw]);Pi=pr(raw)
    W=np.asarray(OK[sel]).astype(float);W[:,:99]=0
    Yi=np.stack([Yt[gi[int(rgs[i])]] for i in sel]);Wi=np.ones(W.shape);Wi[:,:99]=0
    print(f"{run:16s} a_mode={train_rnn.A_MODE:5s} WP i1-swapped vs rebuilt (reliable rows) {wp(np.asarray(Yr[sel]),Ps,W):.4f} | WP i0 same seqs all rows {wp(Yi,Pi,Wi):.4f}",flush=True)
