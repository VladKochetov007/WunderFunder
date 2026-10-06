"""Context distillation teachers: fold model muon6_fFOLD predicted cold (state reset at each sequence) and warm
(state carried from the chain predecessor sequence, when one exists). Writes cache/pred_mucold_fFOLD_{split}.npy and
cache/pred_muwarm_fFOLD_{split}.npy in the row order of cache/{split}_targets.npz (distillation cache format).
Training-time targets only; the deployed student never sees neighbouring sequences."""
import sys,numpy as np,torch
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"scripts");sys.path.insert(0,S+"src")
import train_rnn
from train_rnn import make_model,build_inputs
from wnn.data import SequenceFile
fold=int(sys.argv[1]);dev=torch.device("cuda")
ck=torch.load(S+f"artifacts/runs/muon6_f{fold}/model_ep3.pt",map_location="cpu",weights_only=False)
m=make_model(ck["args"],ck["n_in"],ck["model"]["head.2.weight"].shape[0]).to(dev);m.load_state_dict(ck["model"]);m.eval()
c=np.load(S+"cache/chains.npz");prev={}
for i,n in enumerate(c["next"]):
    if n>=0:prev[(str(c["split"][n]),int(c["row_group"][n]))]=(str(c["split"][i]),int(c["row_group"][i]))
files={sp:SequenceFile(S+f"datasets/{sp}.parquet") for sp in ("train","valid")}
def run(x,h=None):
    inp=build_inputs(torch.from_numpy(x)[None].to(dev));outs=[]
    with torch.no_grad():
        for s in range(0,20000,4000):
            with torch.autocast("cuda",dtype=torch.float16):
                o,h=m(inp[:,s:s+4000],h)
            outs.append(o[0,:,:2].float().cpu().numpy())
    return np.concatenate(outs),h
nw=0
for sp in ("valid","train"):
    rg=np.load(S+f"cache/{sp}_targets.npz")["row_group"]
    cold=np.lib.format.open_memmap(S+f"cache/pred_mucold_f{fold}_{sp}.npy",mode="w+",dtype=np.float32,shape=(len(rg),20000,2))
    warm=np.lib.format.open_memmap(S+f"cache/pred_muwarm_f{fold}_{sp}.npy",mode="w+",dtype=np.float32,shape=(len(rg),20000,2))
    for i,g in enumerate(rg):
        x=files[sp].read(int(g)).features;pc,_=run(x);cold[i]=pc
        p=prev.get((sp,int(g)))
        if p is None:warm[i]=pc
        else:
            _,h=run(files[p[0]].read(p[1]).features);pw,_=run(x,h);warm[i]=pw;nw+=1
        if i%1000==0:print(sp,i,flush=True)
    cold.flush();warm.flush()
print("done fold",fold,"warm sequences",nw)
