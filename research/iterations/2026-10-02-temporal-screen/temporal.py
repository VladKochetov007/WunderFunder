"""Training-only gradient cut; identical parameter keys and serving recurrence."""
import torch
from wnn.nn import RNNModel

class SegmentedRNN(RNNModel):
 def __init__(self,*args,gradient_horizon=2000,segment=500,**kwargs):
  super().__init__(*args,**kwargs)
  if gradient_horizon not in (segment,4*segment):raise ValueError('Only fixed candidate/control horizons')
  self.gradient_horizon=gradient_horizon;self.segment=segment
 def forward(self,x,h=None):
  outputs=[]
  for start in range(0,x.shape[1],self.segment):
   if start and start%self.gradient_horizon==0:
    h=h.detach() if not isinstance(h,tuple) else tuple(v.detach() for v in h)
   out,h=super().forward(x[:,start:start+self.segment],h);outputs.append(out)
  return torch.cat(outputs,dim=1),h
