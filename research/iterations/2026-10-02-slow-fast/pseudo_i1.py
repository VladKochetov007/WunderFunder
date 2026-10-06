"""Teacher pseudo-labels on role-swapped class-3 train sequences: cache/pred_<teacher>_i1swap.npy (row order = cache/i1swap_train_rg.npy)."""
import sys,numpy as np,torch
S="/home/grinchenkoeg/WunderFunder-creative/";sys.path.insert(0,S+"scripts");sys.path.insert(0,S+"src")
import train_rnn
from train_rnn import make_model
from predict_run import predict
from i1swap_aug import swap_instruments
from wnn.data import SequenceFile
teacher=sys.argv[1]
ck=torch.load(S+f"artifacts/runs/{teacher}/model_ep3.pt",map_location="cpu",weights_only=False)
dev=torch.device("cuda");model=make_model(ck["args"],ck["n_in"],ck["model"]["head.2.weight"].shape[0]).to(dev);model.load_state_dict(ck["model"]);model.eval()
train_rnn.A_MODE=ck["args"].get("a_mode") or "none"
rgs=np.load(S+"cache/i1swap_train_rg.npy");sf=SequenceFile(S+"datasets/train.parquet")
out=np.lib.format.open_memmap(S+f"cache/pred_{teacher}_i1swap.npy",mode="w+",dtype=np.float32,shape=(len(rgs),20000,2))
B=16
for s in range(0,len(rgs),B):
    x=torch.from_numpy(np.stack([swap_instruments(sf.read(int(g)).features) for g in rgs[s:s+B]]))
    out[s:s+len(x)]=predict(model,x,4000,dev)[...,:2]
    if s%800==0:print(s,flush=True)
out.flush();print("done",teacher,float(np.abs(np.asarray(out[:50])).mean()))
