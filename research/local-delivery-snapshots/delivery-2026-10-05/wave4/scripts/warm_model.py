"""Causal warmup-summary correction, applied once after exactly 99 observations."""
import torch
from torch import nn
from wnn.nn import RNNModel

class WarmModel(RNNModel):
    def __init__(self,n_in,hidden,layers,dropout,n_out,mode):
        super().__init__(n_in,hidden,layers,dropout,n_out)
        self.mode=mode;self.hidden=hidden;self.layers=layers
        self.warm_adapter=nn.Sequential(nn.Linear((layers+1)*hidden,32),nn.GELU(),nn.Linear(32,layers*hidden))
        nn.init.zeros_(self.warm_adapter[2].weight);nn.init.zeros_(self.warm_adapter[2].bias)

    def forward(self,x,state=None):
        z=self.encode((x-self.mean)/self.std)
        b,t,_=z.shape
        if state is None:
            h=None;total=torch.zeros(b,self.hidden,device=z.device,dtype=torch.float32);count=0
        else:
            h,total,c=state;count=int(c.item())
        first=min(max(99-count,0),t)
        parts=[]
        if first:
            a,h=self.gru(z[:,:first],h);parts.append(a)
            total=total+z[:,:first].float().sum(1)
            if count+first==99:
                pooled=total/99 if self.mode=='pool' else total*0
                f=torch.cat([h.transpose(0,1).reshape(b,-1).float(),pooled],-1)
                delta=self.warm_adapter(f).reshape(b,self.layers,self.hidden).transpose(0,1)
                h=(h+delta.to(h.dtype)).contiguous()
        if first<t:
            a,h=self.gru(z[:,first:],h);parts.append(a)
        o=torch.cat(parts,1)
        out=self.head(torch.cat([o,z],-1))
        return out,(h,total,torch.tensor(count+t,device=z.device))
