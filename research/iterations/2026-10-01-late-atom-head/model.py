"""Frozen ordinary H2 plus a zero-gain, corrected categorical residual."""
from __future__ import annotations
import torch
from torch import nn

def corrected_decode(logits, atoms):
 # FP32 explicitly, including the two small moment matrix multiplies under AMP.
 with torch.autocast(device_type=logits.device.type,enabled=False):
  c=atoms.float().clamp(-2,2);a=c.abs().mean(-1)
  u=torch.softmax(logits.float(),-1)/a
  return (u@(c*c.abs()))/(u@c.abs()).clamp_min(1e-6)

class Residual(nn.Module):
 def __init__(self,atoms,features=512,seed=3):
  super().__init__()
  assert atoms.ndim==2 and atoms.shape[1]==2 and torch.isfinite(atoms).all()
  assert (atoms.clamp(-2,2).abs().mean(-1)>0).all()
  self.register_buffer('atoms',atoms.detach().float().clone())
  # CPU construction only. Preserve global RNG; all arms get identical new weights.
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(seed);self.mlp=nn.Sequential(nn.Linear(features,64),nn.GELU(),nn.Linear(64,len(atoms)))
  self.gain=nn.Parameter(torch.zeros(()))
 def forward(self,features,base_prediction,p3):
  logits=self.mlp(features.detach()).float();d=corrected_decode(logits,self.atoms)
  p=base_prediction.float();return p+self.gain*p3.detach()*(d-p),logits

class LateAtom(nn.Module):
 def __init__(self,baseline,atoms,class_start=17):
  super().__init__();self.baseline=baseline;self.class_start=int(class_start)
  for p in self.baseline.parameters():p.requires_grad_(False)
  self.baseline.eval();self.residual=Residual(atoms,2*baseline.gru.hidden_size)
 def train(self,mode=True):
  super().train(mode);self.baseline.eval();return self
 @torch.no_grad()
 def context(self,x,state=None):
  b=self.baseline;z=b.encode((x-b.mean)/b.std);o,state=b.gru(z,state)
  features=torch.cat([o,z],-1);out=b.head(features).float()
  p3=torch.softmax(out[...,self.class_start:self.class_start+4],-1)[...,3:4]
  return features,out[...,:2],p3,state
 def forward(self,x,state=None):
  features,p,p3,state=self.context(x,state);pred,logits=self.residual(features,p,p3)
  return pred,logits,state
