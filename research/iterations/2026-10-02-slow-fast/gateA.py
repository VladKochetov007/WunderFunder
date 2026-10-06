"""Gate A: is swapping bid<->ask blocks (prices+volumes, i0 and i1) a mirror? corr(f(Mx)_t0, f(x)_t1) on class-3 / class-0 valid seqs."""
import sys,numpy as np,torch
sys.path.insert(0,"/home/grinchenkoeg/WunderFunder-creative/scripts");sys.path.insert(0,"/home/grinchenkoeg/WunderFunder-creative/src")
import train_rnn
from train_rnn import make_model,Sequences
from predict_run import predict
R="/home/grinchenkoeg/WunderFunder-creative/"
run=sys.argv[1]
ck=torch.load(R+f"artifacts/runs/{run}/model_ep3.pt",map_location="cpu",weights_only=False)
dev=torch.device("cuda");model=make_model(ck["args"],ck["n_in"],ck["model"]["head.2.weight"].shape[0]).to(dev);model.load_state_dict(ck["model"]);model.eval()
train_rnn.A_MODE=ck["args"].get("a_mode") or "none"
t=np.load(R+"cache/valid_targets.npz");rg=t["row_group"];cls=t["market_class"];Y=t["y"];M=t["mask"]
def swap(x,variant):
    x=x.clone()
    blocks=[(0,11),(22,33),(52,63),(74,85)] if variant!="price" else [(0,11),(52,63)]
    for b,a in blocks:
        tmp=x[...,b:b+11].clone();x[...,b:b+11]=x[...,a:a+11];x[...,a:a+11]=tmp
    if variant=="neg":
        for b,a in [(0,11),(52,63)]:x[...,b:b+11]*=-1;x[...,a:a+11]*=-1
    return x
for c in (3,0):
    ids=np.where(cls==c)[0][:12];ds=Sequences("valid",rg[ids]);x=torch.stack([ds[i][0] for i in range(len(ids))])
    p=predict(model,x,4000,dev)[...,:2]
    for v in ("all","price","neg"):
        q=predict(model,swap(x,v),4000,dev)[...,:2]
        m=M[ids].copy();m[:,:99]=False
        c01=np.corrcoef(q[...,0][m],p[...,1][m])[0,1];c10=np.corrcoef(q[...,1][m],p[...,0][m])[0,1];c00=np.corrcoef(q[...,0][m],p[...,0][m])[0,1]
        # WP of swapped-mirror prediction vs truth
        yy=np.clip(Y[ids],-2,2)
        def wp1(pp,y):
            w=np.abs(y);my=(w*y).sum()/w.sum();mp=(w*pp).sum()/w.sum();return (w*(y-my)*(pp-mp)).sum()/np.sqrt((w*(y-my)**2).sum()*(w*(pp-mp)**2).sum())
        orig=np.mean([wp1(p[...,j][m],yy[...,j][m]) for j in range(2)]);mir=np.mean([wp1(q[...,1-j][m],yy[...,j][m]) for j in range(2)])
        print(f"class{c} swap={v:5s} corr(f(Mx)t0,f(x)t1)={c01:.3f} corr(f(Mx)t1,f(x)t0)={c10:.3f} corr(t0,t0)={c00:.3f} | WP orig {orig:.3f} mirrored-swapped {mir:.3f}")
