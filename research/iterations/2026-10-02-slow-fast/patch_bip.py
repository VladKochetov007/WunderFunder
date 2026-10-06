from pathlib import Path
p=Path("bip.py");s=p.read_text()
old=s[s.index("class BiProp"):s.index("rg=np.load")]
new='''class BiProp(nn.Module):
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
'''
s=s.replace(old,new)
s=s.replace('n_in=N_FEATURES+N_EVENT_FEATURES;m=BiProp(n_in).to(dev)','n_in=N_FEATURES+N_EVENT_FEATURES;causal=load_propensity(S+f"artifacts/runs/prop_f{fold}/model.pt",dev);m=BiProp(n_in,causal).to(dev)')
s=s.replace("for ep in range(4):","for ep in range(8):")
s=s.replace("torch.save(m.state_dict()",'torch.save({k:v for k,v in m.state_dict().items() if not k.startswith("causal.")}')
p.write_text(s)
