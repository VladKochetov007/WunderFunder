"""Tiny generated CUDA checks against independently frozen literal CPU oracle."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from scalar_model import ScalarAdapter, RAW_INDICES, INV_SQRT44

def check_cuda_scalar(directory):
 directory=Path(directory)
 with np.load(directory/'weights.npz',allow_pickle=False) as z:
  k=z['scalar.knots'].copy();c=z['scalar.counts'].astype(np.int64);table=z['scalar.table'].copy()
 raw=np.load(directory/'raw.npy');expected=np.load(directory/'expected.npy')
 adapter=ScalarAdapter(k,c).cuda()
 with torch.no_grad():adapter.table.copy_(torch.from_numpy(table).cuda())
 x=torch.from_numpy(raw).cuda()
 observed=adapter.aggregated(x)
 assert np.array_equal(observed.detach().cpu().numpy().view(np.uint32),expected.view(np.uint32)), 'CUDA literal boundary oracle mismatch'
 with torch.no_grad():other=adapter.aggregated(x)
 assert torch.equal(observed,other)
 adapter.table.requires_grad_(False)
 assert torch.equal(other,adapter.aggregated(x))
 adapter.table.requires_grad_(True)
 rng=np.random.default_rng(881729);records=[]
 for name,arr in [('boundary',raw),('repeated_knots',np.repeat(raw[9:10],2048,axis=0))]:
  n=len(arr);basis=np.zeros((44,n,16),np.float64)
  for j,col in enumerate(RAW_INDICES):
   count=int(c[j])
   for row in range(n):
    value=min(max(float(arr[row,col]),float(k[j,0])),float(k[j,count-1]));lo=0
    for upper in range(1,count):
     if value>=float(k[j,upper]):lo=upper
    hi=min(lo+1,count-1)
    alpha=float(np.float32(0 if hi==lo else (value-float(k[j,lo]))/(float(k[j,hi])-float(k[j,lo]))))
    basis[j,row,lo]+=1-alpha;basis[j,row,hi]+=alpha
  upstream=rng.normal(size=(n,16)).astype(np.float32)
  adapter.zero_grad(set_to_none=True)
  value=adapter.aggregated(torch.from_numpy(arr).cuda())
  (value*torch.from_numpy(upstream).cuda()).sum().backward()
  oracle=np.einsum('jnk,nr->jkr',basis,upstream.astype(np.float64),optimize=False)*INV_SQRT44
  actual=adapter.table.grad.detach().cpu().numpy().astype(np.float64)
  diff=actual-oracle;maximum=float(np.abs(diff).max());relative=float(np.sqrt(np.mean(diff**2))/max(np.sqrt(np.mean(oracle**2)),1e-12))
  assert np.isfinite(actual).all() and maximum<2e-4 and relative<2e-5,(name,maximum,relative)
  for j,count in enumerate(c):assert np.count_nonzero(actual[j,count:])==0
  records.append({'case':name,'rows':n,'gradient_dense_float64_max_abs':maximum,'gradient_relative_RMS':relative})
 result={'status':'PASS','literal_boundary_bitexact':True,'rows':len(raw),'all_grad_modes_same':True,'gradient_cases':records,'fixture_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 del adapter,observed,other,value,x;torch.cuda.empty_cache()
 return result
