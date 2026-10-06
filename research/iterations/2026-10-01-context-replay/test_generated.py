"""Generated NumPy recurrence only: no Torch/checkpoints/data imports or GPU."""
import importlib.util,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('replay',HERE/'replay.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def clone(h):
 if isinstance(h,tuple) and h[0]=='equal':return bool(np.array_equal(h[1],h[2]))
 return None if h is None else h.copy()

def forward(x,lo,hi,h):
 h=np.zeros((x.shape[0],2)) if h is None else h.copy();values=[]
 for t in range(lo,hi):
  h=np.tanh(.87*h + x[:,t,:2]);values.append((h+.03*x[:,t,2:4]).astype(np.float32))
 return np.stack(values,axis=1),h

def main():
 rng=np.random.default_rng(11);x=rng.normal(size=(3,101,4));x[:,:,3]=np.arange(101) # fullprefix feature, never reset
 starts=(31,57,85);length=8;contexts=(7,19)
 full,rows,checks=r.core(forward,clone,x,101,starts,length,contexts,True)
 direct,_=forward(x,0,101,None);assert np.array_equal(full,direct);assert all(a and b for a,b in checks)
 assert rows.shape==(3,24,3,2)
 for j,s in enumerate(starts):
  sl=slice(j*length,(j+1)*length);assert np.array_equal(rows[:,sl,0],direct[:,s:s+length])
  for ai,c in enumerate(contexts,1):
   literal,_=forward(x,s-c,s+length,None);assert np.array_equal(rows[:,sl,ai],literal[:,-length:])
 y=rng.normal(size=(3,101,2));m=rng.integers(0,2,size=(3,101));y[:]=100;m[:]=1 # irrelevant to core
 untouched=x.copy();xx=x.copy();xx[:,93:]+=1000
 f2,o2,_=r.core(forward,clone,xx,101,starts,length,contexts,False)
 assert np.array_equal(full[:,:93],f2[:,:93]) and np.array_equal(rows,o2)
 assert np.array_equal(x,untouched)
 assert r.STEPS.size==2048 and r.EXPECTED_ROWS==1761280 and len(np.unique(r.STEPS))==2048
 bs=r.boundaries();assert bs[0][0]==0 and bs[-1][1]==20000 and all(a[1]==b[0] for a,b in zip(bs,bs[1:]))
 assert all((s,s+512) in bs for s in r.STARTS) and all(hi-lo<=4000 for lo,hi in bs)
 # Signed zero is intentionally not accepted by the real byte-equality output control.
 assert np.array_equal(np.array([-0.],np.float32),np.array([0.],np.float32))
 assert np.array([-0.],np.float32).tobytes()!=np.array([0.],np.float32).tobytes()
 print(json.dumps({'status':'PASS','generated_only':True,'checks':['full_carry_direct_equality','exact_incoming_state_reproduction','literal_reset_context_equivalence','fixed_block_alignment_and_count','same_fullprefix_feature_rows','no_future_dependence','no_input_mutation','signed_zero_byte_claim'],'source_sha256':r.sha(HERE/'replay.py')}))
if __name__=='__main__':main()
