"""Data selection by gradient alignment with the true-mask valid objective (per train sequence).
   ds_score.py FOLD : uses fold model muon6_fFOLD (ep3); target gradient = wMSE on scored rows (true mask) of valid sequences
   in the training pool (folds != FOLD); per-train-sequence gradient = wMSE with training weights |y|*(p^2+0.03).
   Writes research/.../ds_cos_fFOLD.npy (10607 cosines, nan for corrupt)."""
import sys,numpy as np,torch
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"scripts");sys.path.insert(0,S+"src")
import train_rnn
from train_rnn import make_model,Sequences,build_inputs
from wnn.data import WARMUP,CORRUPT_TRAIN_ROW_GROUPS
fold=int(sys.argv[1]);dev=torch.device("cuda")
ck=torch.load(S+f"artifacts/runs/muon6_f{fold}/model_ep3.pt",map_location="cpu",weights_only=False)
m=make_model(ck["args"],ck["n_in"],ck["model"]["head.2.weight"].shape[0]).to(dev);m.load_state_dict(ck["model"]);m.train()
params=[p for p in m.parameters() if p.requires_grad]
def seq_grad(x,y,w):
    """x (20000,112) y (20000,2) w (20000,) -> flat gradient of sum w*(p-y)^2 / sum w (chunked TBPTT, state detached)."""
    for p in params:p.grad=None
    inp=build_inputs(x[None].to(dev));h=None;W=w.sum().clamp_min(1e-8)
    for s in range(0,20000,2000):
        with torch.autocast("cuda",dtype=torch.float16):
            o,h=m(inp[:,s:s+2000],h)
        h=tuple(t.detach() for t in h) if isinstance(h,tuple) else h.detach()
        l=(w[s:s+2000,None].to(dev)*(o[0,:,:2].float()-y[s:s+2000].to(dev).clamp(-2,2))**2).sum()/W
        l.backward()
    return torch.cat([p.grad.flatten() for p in params]).double()
t=np.load(S+"cache/valid_targets.npz");vrg=t["row_group"];cf=np.load(S+"cache/chain_folds.npz");v=cf["split"]=="valid"
fo=dict(zip(cf["row_group"][v].tolist(),cf["fold"][v].tolist()))
pool=[g for g in vrg if fo[int(g)]!=fold]
vs=Sequences("valid",pool);G=None
for i in range(len(vs)):
    x,y,mk=vs[i];yc=y[:,:2].clamp(-2,2);w=yc.abs().sum(1)*mk.float();w[:WARMUP]=0
    g=seq_grad(x,y[:,:2],w)*float(w.sum());G=g if G is None else G+g
G=G/torch.linalg.norm(G);print("target gradient ready",len(pool),flush=True)
prop=np.load(S+f"cache/propcache_prop_f{fold}_train.npy",mmap_mode="r")
ts=Sequences("train",list(range(10607)));cos=np.full(10607,np.nan)
for g in range(10607):
    if g in CORRUPT_TRAIN_ROW_GROUPS:continue
    x,y,_=ts[g];p=torch.from_numpy(np.array(prop[g]));w=y[:,:2].clamp(-2,2).abs().sum(1)*(p**2+0.03);w[:WARMUP]=0
    gi=seq_grad(x,y[:,:2],w);cos[g]=float(gi@G/torch.linalg.norm(gi).clamp_min(1e-12))
    if g%1000==0:print(g,flush=True)
np.save(S+f"research/iterations/2026-10-02-slow-fast/ds_cos_f{fold}.npy",cos);print("done",np.nanpercentile(cos,[5,25,50,75,95]))
