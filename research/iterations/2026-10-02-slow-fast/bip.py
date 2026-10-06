"""Bidirectional (smoothing) scoring-propensity P(S_t | X_1:T) for TRAIN WEIGHTS ONLY (never used at inference).
   bip.py FOLD [--write-cache]  : train on valid chains outside FOLD, report AUC on FOLD vs causal prop_fFOLD;
   with --write-cache also write cache/propcache_bip_fFOLD_{train,valid}.npy (format of cache_propensity.py)."""
import sys,json,hashlib,shutil,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"scripts")
from torch.utils.data import DataLoader
from train_rnn import Sequences,build_inputs
from wnn.data import N_FEATURES,WARMUP,CORRUPT_TRAIN_ROW_GROUPS
from wnn.features import N_EVENT_FEATURES
from wnn.propensity import load_propensity,propensity
from wnn.resources import configure_torch
fold=int(sys.argv[1]);write="--write-cache" in sys.argv
torch.manual_seed(0);dev=torch.device("cuda");configure_torch(gpu_fraction=0.4)
class BiProp(nn.Module):
    """causal logit (frozen prop_fK) + backward-GRU correction initialised at zero."""
    def __init__(s,n_in,causal,h=64):
        super().__init__();s.causal=causal
        for q in s.causal.parameters():q.requires_grad_(False)
        s.register_buffer("mean",torch.zeros(n_in));s.register_buffer("std",torch.ones(n_in))
        s.inp=nn.Linear(n_in,h);s.b=nn.GRU(h,h,batch_first=True);s.out=nn.Linear(h+1,1)
        nn.init.zeros_(s.out.weight);nn.init.zeros_(s.out.bias)
    def forward(s,x):
        with torch.no_grad():
            cl,_=s.causal(x)
        z=F.gelu(s.inp((x-s.mean)/s.std));ob,_=s.b(torch.flip(z,[1]));ob=torch.flip(ob,[1])
        return cl.float()+s.out(torch.cat([ob,cl.float()[...,None]],-1)).squeeze(-1)
rg=np.load(S+"cache/valid_targets.npz")["row_group"];cf=np.load(S+"cache/chain_folds.npz");v=cf["split"]=="valid"
fo=dict(zip(cf["row_group"][v].tolist(),cf["fold"][v].tolist()));f=np.array([fo[int(g)] for g in rg])
tr,te=rg[f!=fold],rg[f==fold]
L=DataLoader(Sequences("valid",tr),batch_size=8,shuffle=True,num_workers=2,prefetch_factor=1)
T=DataLoader(Sequences("valid",te),batch_size=8,num_workers=2,prefetch_factor=1)
n_in=N_FEATURES+N_EVENT_FEATURES;causal=load_propensity(S+f"artifacts/runs/prop_f{fold}/model.pt",dev);m=BiProp(n_in,causal).to(dev)
with torch.no_grad():
    x,_,_=next(iter(L));fx=build_inputs(x.to(dev))[...,:n_in];m.mean.copy_(fx.reshape(-1,n_in).mean(0));m.std.copy_(fx.reshape(-1,n_in).std(0).clamp_min(1e-3))
opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-2)
def auc(s,l):
    o=np.argsort(s);r=np.empty(len(s));r[o]=np.arange(1,len(s)+1);p=l.astype(bool);a,b=p.sum(),(~p).sum();return float((r[p].sum()-a*(a+1)/2)/(a*b))
causal=load_propensity(S+f"artifacts/runs/prop_f{fold}/model.pt",dev)
def evaluate():
    m.eval();ps,cs,ms=[],[],[]
    with torch.no_grad():
        for x,_,mk in T:
            fx=build_inputs(x.to(dev))
            with torch.autocast("cuda",dtype=torch.float16):p=torch.sigmoid(m(fx[...,:n_in]).float())
            ps.append(p[:,WARMUP:].cpu().numpy().ravel());cs.append(propensity(causal,fx[...,:n_in])[:,WARMUP:].cpu().numpy().ravel());ms.append(mk[:,WARMUP:].numpy().ravel())
    p,c,mk=map(np.concatenate,(ps,cs,ms));sub=np.random.default_rng(0).choice(len(p),2_000_000,replace=False)
    # calibration-free weight quality: weighted share of true-scored mass captured by p^2 weights
    return dict(auc_bi=round(auc(p[sub],mk[sub]),4),auc_causal=round(auc(c[sub],mk[sub]),4),
                brier_bi=round(float(((p-mk)**2).mean()),4),brier_causal=round(float(((c-mk)**2).mean()),4))
for ep in range(8):
    m.train()
    for x,_,mk in L:
        fx=build_inputs(x.to(dev))[...,:n_in];tg=mk.to(dev).float()
        with torch.autocast("cuda",dtype=torch.float16):lg=m(fx)
        loss=F.binary_cross_entropy_with_logits(lg.float()[:,WARMUP:],tg[:,WARMUP:])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    print("BIP",fold,ep,json.dumps(evaluate()),flush=True)
torch.save({k:v for k,v in m.state_dict().items() if not k.startswith("causal.")},S+f"artifacts/runs/bip_model_f{fold}.pt")
if write:
    import os;d=S+f"artifacts/runs/bip_f{fold}";os.makedirs(d,exist_ok=True);shutil.copy(S+f"artifacts/runs/prop_f{fold}/model.pt",d+"/model.pt")
    sha=hashlib.sha256(open(d+"/model.pt","rb").read()).hexdigest();m.eval()
    for split in ("train","valid"):
        g=np.load(S+f"cache/{split}_targets.npz")["row_group"] if split=="valid" else np.array([q for q in range(10607) if q not in CORRUPT_TRAIN_ROW_GROUPS])
        out=np.lib.format.open_memmap(S+f"cache/propcache_bip_f{fold}_{split}.npy",mode="w+",dtype=np.float32,shape=(10607 if split=="train" else 1873,20000))
        ds=Sequences(split,g);DL=DataLoader(ds,batch_size=8,num_workers=4,prefetch_factor=2);i=0
        with torch.no_grad():
            for x,_,_ in DL:
                fx=build_inputs(x.to(dev))[...,:n_in]
                with torch.autocast("cuda",dtype=torch.float16):p=torch.sigmoid(m(fx).float()).cpu().numpy()
                out[np.asarray(g[i:i+len(p)])]=p;i+=len(p)
        out.flush()
        json.dump({"model_sha256":sha,"note":"bidirectional smoother probabilities (bip.py); model.pt is a causal placeholder for loading"},open(S+f"cache/propcache_bip_f{fold}_{split}.json","w"))
    print("cache written",fold)
