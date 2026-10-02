"""Generated-only checks; no original data, labels, predictions or WP queries."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from features import full,update
from diagnostic import design,crossfit,summarize,DIM,QY,QP,CONTROL,ARM_INDICES,H,sign

def scalar_oracle(raw):
 prev=None;prev_event=None;turn=energy=count=age=0.;net=np.zeros(2);rows=[]
 for row in raw:
  d=np.zeros(2)
  if prev is not None:
   reset=bool(row[0]!=prev[0] and row[1]!=prev[1])
   if reset:prev_event=None;turn=energy=count=age=0.;net=np.zeros(2)
   else:age+=1
   d=(row[2:]-prev[2:]).astype(np.float32).astype(np.float64)
   if np.any(row[2:]!=prev[2:]):
    if prev_event is not None:turn+=prev_event[0]*d[1]-prev_event[1]*d[0]
    energy+=d[0]*d[0]+d[1]*d[1];count+=1;net+=d;prev_event=d.copy()
  rows.append([*d,*net,energy,count,age,turn/energy if energy>0 else 0.]);prev=row.copy()
 return np.asarray(rows)

def main():
 rng=np.random.default_rng(20261002)
 raw=np.cumsum(rng.integers(-2,3,size=(141,4)).astype(np.float32),axis=0)
 raw[::4,0]=raw[np.maximum(np.arange(0,len(raw),4)-1,0),0]
 got=full(raw);np.testing.assert_array_equal(got,scalar_oracle(raw))
 fractional=np.cumsum(rng.normal(size=(111,4)).astype(np.float32),axis=0);np.testing.assert_array_equal(full(fractional),scalar_oracle(fractional))
 state=np.zeros(13);parts=[update(raw[a:b],state) for a,b in [(0,1),(1,17),(17,17),(17,91),(91,141)]]
 np.testing.assert_array_equal(np.concatenate(parts),got)
 for end in (1,2,33,140):np.testing.assert_array_equal(full(raw[:end]),got[:end])
 state=np.zeros(13);np.testing.assert_array_equal(np.concatenate([update(row[None],state) for row in raw]),got)
 # Two ordered increments give±1/2 despite same net endpoint/energy/count.
 a=np.array([[0,0,0,0],[0,0,1,0],[0,0,1,1]],np.float32)
 b=np.array([[0,0,0,0],[0,0,0,1],[0,0,1,1]],np.float32)
 fa,fb=full(a),full(b);assert fa[-1,7]==.5 and fb[-1,7]==-.5
 np.testing.assert_array_equal(fa[-1,2:7],fb[-1,2:7])
 # Simultaneous reset + i1 event is first event, no cross-interval determinant.
 c=np.vstack([a,[1,1,2,1],[1,1,2,2]]).astype(np.float32);fc=full(c)
 assert fc[-2,7]==0 and fc[-2,5]==1 and fc[-2,6]==0 and fc[-1,7]==.5
 empty=full(np.zeros((17,4),np.float32));assert not np.any(empty[:,[0,1,2,3,4,5,7]]);np.testing.assert_array_equal(empty[:,6],np.arange(17))
 col=np.c_[np.zeros((20,2)),np.arange(20),2*np.arange(20)].astype(np.float32)
 assert np.all(full(col)[:,7]==0)
 # Float32 subtraction really occurs before double products, including large codes.
 huge=np.array([[0,0,1e8,1e8],[0,0,1e8+8,1e8],[0,0,1e8+8,1e8+16]],np.float32)
 np.testing.assert_array_equal(full(huge),scalar_oracle(huge))
 # Separate raw-p predictor and clipped-p response slots, plus unchanged labels convention.
 hist=np.zeros((9,12));hist[:,8:11]=2;hist[:,11]=rng.uniform(-1,1,9)
 pred=rng.normal(size=(9,8,2)).astype(np.float32)*4;y=rng.normal(size=(9,2)).astype(np.float32)*5
 Z=design(hist,pred,y,13);assert Z.shape==(9,22)
 np.testing.assert_array_equal(Z[:,1:3],pred[:,0]);np.testing.assert_array_equal(Z[:,QP:QP+2],np.clip(pred[:,0],-2,2));np.testing.assert_array_equal(Z[:,QY:QY+2],np.clip(y,-2,2));np.testing.assert_array_equal(Z[:,17],sign(13)*Z[:,16])
 # Generated chain Gram crossfit smoke; rowwise SSE equals stored quadratic form.
 keys=np.array([(f,100*f+k) for f in (0,1) for k in range(8)]);cc=np.ones((len(keys),1),int)*40
 grams=np.zeros((3,len(keys),2,DIM,DIM));rows=[]
 for k in range(len(keys)):
  n=40;hist=rng.normal(size=(n,12));hist[:,8:11]=np.abs(hist[:,8:11]);pr=rng.normal(size=(n,8,2));yy=.3*pr[:,0]+.2*hist[:,11,None]+rng.normal(size=(n,2));z=design(hist,pr,yy,int(keys[k,1]));w=abs(z[:,QY:QY+2]);rows.append((z,w))
  for t in (0,1):grams[0,k,t]=z.T@(z*w[:,t,None]);grams[1,k,t]=grams[0,k,t]
 sse,const,halves,recs=crossfit(grams,keys,cc,np.array([3]),bootstrap=0)
 for rec in recs:
  f,h,t=rec['fold'],rec['heldout_half'],rec['target'];te=(keys[:,0]==f)&(halves==h);tr=(keys[:,0]==f)&(halves!=h);alpha,_=H.q_definition(grams[0,tr,t].sum(0),t,prediction_index=QP+t,target_index=QY+t);ai=('control','turn','placebo').index(rec['arm'])
  assert not set(rec['training_chains'])&set(rec['heldout_chains'])
  for k in np.flatnonzero(te):
   z,w=rows[k];direct=np.sum(w[:,t]*(z@(alpha-rec['coefficients']))**2)
   np.testing.assert_allclose(sse[0,k,t,ai],direct,rtol=1e-10,atol=1e-10)
 assert summarize(sse,const,np.ones(len(keys),bool))['defined']
 files=[HERE/'features.py',HERE/'diagnostic.py',Path(__file__)]
 print(json.dumps(dict(status='PASS_GENERATED_ONLY',checks=['independent_scalar_oracle','prefix_chunk_empty_one_row_parity','order_sensitive_same_endpoint','simultaneous_reset','empty_and_collinear_zero','float32_delta_before_float64_accumulation','separate_raw_and_clipped_prediction_slots','placebo_exact_chain_sign','same_outerfold_chain_separation','rowwise_SSE_equals_Gram'],source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},real_data_reads=0,metrics=0)))
if __name__=='__main__':main()
