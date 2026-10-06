"""WiSE interpolation: theta = parent + alpha*(ft - parent). wise.py PARENT_RUN@EP FT_RUN@EP OUT_RUN ALPHA"""
import sys,os,torch
R=os.path.expanduser("~/WunderFunder-creative/artifacts/runs/")
def load(s):
    n,e=s.split("@");e=e.replace("ep","");return torch.load(f"{R}{n}/model_ep{e}.pt" if e!="final" else f"{R}{n}/model.pt",map_location="cpu",weights_only=False)
p,f=load(sys.argv[1]),load(sys.argv[2]);a=float(sys.argv[4]);out=sys.argv[3]
ck=dict(f);ck["model"]={k:(p["model"][k].float()+a*(f["model"][k].float()-p["model"][k].float())).to(f["model"][k].dtype) if f["model"][k].is_floating_point() else f["model"][k] for k in f["model"]}
os.makedirs(R+out,exist_ok=True);torch.save(ck,f"{R}{out}/model_ep1.pt");torch.save(ck,f"{R}{out}/model.pt")
import shutil;src=sys.argv[2].split("@")[0];shutil.copy(f"{R}{src}/history.json",f"{R}{out}/history.json")
print("wrote",out,a)
