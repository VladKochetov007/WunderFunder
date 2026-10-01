"""SCALARPLE01 fixed scalar representation; no data loading or fitting on import."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from wnn.nn import RNNModel

SCHEMA='scalarple01-44x16-r16-v1'
RAW_INDICES=tuple(range(22))+tuple(range(52,74))
TABLE_SEED=2026100103
INV_SQRT44=float(np.float32(1/math.sqrt(44)))

def canonicalize_knots(quantiles):
 q=np.asarray(quantiles,dtype=np.float64)
 if q.shape!=(44,16) or not np.isfinite(q).all():raise ValueError('finite44x16 quantiles required')
 q=q.astype(np.float32)
 if not np.isfinite(q).all():raise ValueError('knots overflow float32')
 q[q==0]=np.float32(0)
 padded=np.empty((44,16),np.float32);counts=np.empty(44,np.int64)
 for j in range(44):
  k=np.unique(q[j]);n=len(k)
  if not 1<=n<=16 or not np.all(np.diff(k.astype(np.float64))>0):raise ValueError('knot order')
  counts[j]=n;padded[j,:n]=k;padded[j,n:]=k[-1]
 return padded,counts

def validate_knots(knots,counts):
 k=np.asarray(knots,dtype=np.float32);c=np.asarray(counts)
 if k.shape!=(44,16) or c.shape!=(44,) or not np.isfinite(k).all():raise ValueError('knot shape/finite')
 if not np.all(np.isfinite(c)) or not np.all(c==c.astype(np.int64)):raise ValueError('counts integer')
 c=c.astype(np.int64)
 for j,n in enumerate(c):
  if not 1<=n<=16 or not np.all(np.diff(k[j,:n].astype(np.float64))>0):raise ValueError('unique knot widths/count')
  if not np.all(k[j,n:]==k[j,n-1]):raise ValueError('padding must repeat last knot')
 return k,c

def _interpolate(raw,knots,table,counts,save):
 n=raw.shape[0];total=torch.zeros((n,16),dtype=torch.float32,device=raw.device)
 indices=torch.empty((n,44),dtype=torch.int16,device=raw.device) if save else None
 fractions=torch.empty((n,44),dtype=torch.float32,device=raw.device) if save else None
 for j,count in enumerate(counts):
  k=knots[j,:count];x=raw[:,j].contiguous()
  lo=(torch.searchsorted(k,x,right=True)-1).clamp(0,count-1)
  hi=(lo+1).clamp_max(count-1)
  kd=k.double();xd=x.double().clamp(float(k[0]) if raw.device.type=='cpu' else k[0].double(),float(k[-1]) if raw.device.type=='cpu' else k[-1].double())
  left=kd[lo];right=kd[hi];width=torch.where(lo==hi,torch.ones_like(left),right-left)
  alpha=((xd-left)/width).clamp(0,1).float()
  a=table[j].index_select(0,lo);b=table[j].index_select(0,hi)
  difference=b-a;product=difference*alpha[:,None];value=a+product
  total.add_(value)
  if save:indices[:,j]=lo.to(torch.int16);fractions[:,j]=alpha
 total.mul_(INV_SQRT44)
 return total,indices,fractions

class _InterpolationSum(torch.autograd.Function):
 @staticmethod
 def forward(ctx,raw,knots,table,counts):
  out,indices,fractions=_interpolate(raw,knots,table,counts,True)
  ctx.save_for_backward(indices,fractions);ctx.counts=counts
  return out
 @staticmethod
 def backward(ctx,gradient):
  indices,fractions=ctx.saved_tensors;g=gradient.float()*INV_SQRT44
  table_grad=torch.zeros((44,16,16),device=g.device,dtype=torch.float32)
  for j,count in enumerate(ctx.counts):
   lo=indices[:,j].long();hi=(lo+1).clamp_max(count-1)
   right=g*fractions[:,j,None];left=g-right
   table_grad[j].index_add_(0,lo,left);table_grad[j].index_add_(0,hi,right)
  return None,None,table_grad,None

class ScalarAdapter(nn.Module):
 def __init__(self,knots,counts,hidden=128):
  super().__init__();k,c=validate_knots(knots,counts)
  if hidden!=128:raise ValueError('SCALARPLE01 fixes hidden128')
  self.register_buffer('knots',torch.from_numpy(k.copy()))
  self.register_buffer('counts',torch.from_numpy(c.copy()))
  self.register_buffer('raw_indices',torch.tensor(RAW_INDICES,dtype=torch.long))
  self.count_tuple=tuple(int(n) for n in c)
  generator=torch.Generator(device='cpu').manual_seed(TABLE_SEED)
  self.table=nn.Parameter(torch.randn((44,16,16),generator=generator,dtype=torch.float32))
  self.projection=nn.Parameter(torch.zeros((hidden,16),dtype=torch.float32))
 def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
  for name in ['knots','counts','raw_indices']:
   key=prefix+name
   if key in state_dict and not torch.equal(state_dict[key].detach().cpu(),getattr(self,name).detach().cpu()):
    error_msgs.append(f'{key}: immutable scalar fit/indices mismatch')
  for name in ['table','projection','knots']:
   key=prefix+name
   if key in state_dict and state_dict[key].dtype!=torch.float32:error_msgs.append(f'{key}: requires float32 state')
  super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)
 def aggregated(self,x):
  raw=x.reshape(-1,x.shape[-1]).index_select(1,self.raw_indices).float()
  if raw.requires_grad:raise ValueError('SCALARPLE01 data inputs are not learnable')
  if raw.device.type=='cpu':
   if not torch.isfinite(raw).all():raise ValueError('nonfinite scalar inputs')
  else:torch._assert_async(torch.isfinite(raw).all(),'nonfinite scalar inputs')
  if self.table.dtype!=torch.float32 or self.knots.dtype!=torch.float32 or self.projection.dtype!=torch.float32:raise ValueError('scalar parameters must stayfloat32')
  if torch.is_grad_enabled() and self.table.requires_grad:
   value=_InterpolationSum.apply(raw,self.knots,self.table,self.count_tuple)
  else:value=_interpolate(raw,self.knots,self.table,self.count_tuple,False)[0]
  return value.reshape(*x.shape[:-1],16)
 def forward(self,x):
  with torch.autocast(device_type=x.device.type,enabled=False):
   return F.linear(self.aggregated(x),self.projection)

class ScalarRNNModel(RNNModel):
 def __init__(self,n_in,hidden=128,layers=2,dropout=0.,n_out=23,*,knots,counts):
  if (n_in,hidden,layers,dropout,n_out)!=(211,128,2,0.,23):raise ValueError('fixed SCALARPLE01 architecture')
  super().__init__(n_in,hidden,layers,dropout,n_out=n_out)
  self.scalar=ScalarAdapter(knots,counts,hidden)
 def forward(self,x,h=None):
  u=self.inp((x-self.mean)/self.std)
  residual=self.scalar(x).to(dtype=u.dtype)
  z=F.gelu(u+residual)
  o,h=self.gru(z,h)
  return self.head(torch.cat([o,z],dim=-1)),h

def load_scalar_model(checkpoint):
 state=checkpoint['model'];args=checkpoint['args']
 if args.get('scalar_schema')!=SCHEMA:raise ValueError('missing scalar schema')
 k=state['scalar.knots'].detach().cpu().numpy();c=state['scalar.counts'].detach().cpu().numpy()
 model=ScalarRNNModel(checkpoint['n_in'],args['hidden'],args['layers'],args['dropout'],state['head.2.bias'].numel(),knots=k,counts=c)
 model.load_state_dict(state,strict=True)
 if not torch.equal(model.scalar.raw_indices,torch.tensor(RAW_INDICES)):raise ValueError('raw indices changed')
 return model
