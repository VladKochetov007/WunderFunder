"""GLOBAL-HEAD probe, step 1: penultimate activations (head GELU output) of 5 fold members on scored rows of their held-out fold."""
import sys,os,numpy as np,torch
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"scripts");sys.path.insert(0,S+"src")
import train_rnn
from train_rnn import make_model,Sequences,build_inputs
from wnn.data import WARMUP
fold=int(sys.argv[1]);members=["d256","d256s4","ref","refs4","c500"]
t=np.load(S+"cache/valid_targets.npz");rg=t["row_group"];Mk=t["mask"];Y=t["y"];cls=t["market_class"]
cf=np.load(S+"cache/chain_folds.npz");v=cf["split"]=="valid";fo=dict(zip(cf["row_group"][v],cf["fold"][v]));ch=dict(zip(cf["row_group"][v],cf["chain"][v]))
idx=np.array([i for i in range(len(rg)) if fo[rg[i]]==fold]);dev=torch.device("cuda")
out={}
rows=[(i,np.flatnonzero(Mk[i]&(np.arange(20000)>=WARMUP))) for i in idx]
out["y"]=np.concatenate([Y[i][r] for i,r in rows]);out["cls"]=np.concatenate([np.full(len(r),cls[i]) for i,r in rows])
out["chain"]=np.concatenate([np.full(len(r),ch[rg[i]]) for i,r in rows])
for mname in members:
    run=f"{mname}_f{fold}";p=S+f"artifacts/runs/{run}/model_ep3.pt"
    if not os.path.exists(p):p=f"/home/grinchenkoeg/WunderFunder/artifacts/runs/{run}/model_ep3.pt"
    ck=torch.load(p,map_location="cpu",weights_only=False)
    m=make_model(ck["args"],ck["n_in"],ck["model"]["head.2.weight"].shape[0]).to(dev);m.load_state_dict(ck["model"]);m.eval()
    train_rnn.A_MODE=ck["args"].get("a_mode") or "none"
    cap=[];hk=m.head[1].register_forward_hook(lambda mod,i,o:cap.append(o.detach().float()))
    feats=[];preds=[]
    ds=Sequences("valid",rg[idx])
    with torch.no_grad():
        for j in range(0,len(idx),8):
            x=torch.stack([ds[q][0] for q in range(j,min(j+8,len(idx)))]).to(dev);inp=build_inputs(x);h=None;F=[];P=[]
            for s in range(0,20000,4000):
                cap.clear()
                with torch.autocast("cuda",dtype=torch.float16):
                    o,h=m(inp[:,s:s+4000],h)
                F.append(cap[-1].cpu());P.append(o[...,:2].float().cpu())
            F=torch.cat(F,1).numpy();P=torch.cat(P,1).numpy()
            for b in range(F.shape[0]):
                r=rows[j+b][1];feats.append(F[b][r].astype(np.float32));preds.append(P[b][r])
    hk.remove()
    out[f"phi_{mname}"]=np.concatenate(feats);out[f"pred_{mname}"]=np.concatenate(preds)
    out[f"W_{mname}"]=ck["model"]["head.2.weight"][:2].float().numpy();out[f"b_{mname}"]=ck["model"]["head.2.bias"][:2].float().numpy()
    print(mname,out[f"phi_{mname}"].shape,flush=True)
np.savez(S+f"cache/globalhead_f{fold}.npz",**out);print("saved",fold)
