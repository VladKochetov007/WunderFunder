"""Two causal views trained against either their individual or combined risk."""
import torch
from wnn.nn import RNNModel, weighted_mse_loss

class JointModel(RNNModel):
    def __init__(self,n_in,hidden,layers,dropout,n_out,mode):
        super().__init__(n_in,hidden,layers,dropout,n_out)
        self.other=RNNModel(n_in,hidden,layers,dropout,n_out)
        self.mode=mode
    def forward(self,x,h=None):
        h0,h1=(None,None) if h is None else h
        a,h0=super().forward(x,h0)
        raw=torch.cat([x[...,:112],torch.zeros_like(x[...,112:])],-1)
        b,h1=self.other(raw,h1)
        self.members=(a,b)
        return .5*(a+b),(h0,h1)

def joint_loss(model,y,valid,objective):
    a,b=model.members
    la=objective(a[...,:2],y,valid);lb=objective(b[...,:2],y,valid)
    if model.mode=='independent':return .5*(la+lb)
    # Average risk plus a weak individual anchor prevents cancellation-only solutions.
    lc=objective(.5*(a[...,:2]+b[...,:2]),y,valid)
    yy=y.float().clamp(-2,2);w=yy.abs()*valid[...,None].float()
    ea=a[...,:2].float()-yy;eb=b[...,:2].float()-yy
    cov=(w*ea*eb).sum((0,1))
    norm=((w*ea.square()).sum((0,1))*(w*eb.square()).sum((0,1))).sqrt().clamp_min(1e-8)
    diversity=.01*torch.relu(cov/norm).mean()
    return lc+.05*(la+lb)+diversity
