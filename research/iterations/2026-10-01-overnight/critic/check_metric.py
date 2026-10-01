"""Independent synthetic metric-definition check; never reads prediction/target caches."""
from pathlib import Path
import importlib.util,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
fe=load(ROOT/'scripts/fold_eval.py','metric_immutable')
gr=load(HERE.parent/'grader/evaluate.py','guarded_metric')
def literal(y,p,mask):
 values=[]
 for k in range(2):
  yy=np.clip(y[...,k][mask].astype(np.float64),-2,2);pp=np.clip(p[...,k][mask].astype(np.float64),-2,2);w=np.abs(yy)
  if w.sum()==0:values.append(0.);continue
  yy=yy-np.average(yy,weights=w);pp=pp-np.average(pp,weights=w)
  a=np.dot(w,yy*yy);b=np.dot(w,pp*pp)
  values.append(float(np.dot(w,yy*pp)/np.sqrt(a*b)) if a>1e-16 and b>1e-16 else 0.)
 return np.array(values)
rng=np.random.default_rng(17033);y=rng.normal(size=(5,431,2)).astype(np.float32);p=rng.normal(size=y.shape).astype(np.float32);q=rng.normal(size=y.shape).astype(np.float32)*7
mask=rng.uniform(size=y.shape[:-1])>.3;mask[:,:99]=False
y[0]+=1.8;p[0]+=1.7;y[1]*=3;p[1]+=y[1]
blended=(p+q)/2
stats=fe.sums(y,blended,mask).sum(0)
a=literal(y,blended,mask);b=np.array(gr.target_scores(stats));err=float(np.max(abs(a-b)))
assert err<1e-12
assert abs(fe.pooled(stats)-a.mean())<1e-12
seqmean=np.mean([literal(y[i:i+1],blended[i:i+1],mask[i:i+1]).mean() for i in range(5)])
wrongclip=literal(y,(np.clip(p,-2,2)+np.clip(q,-2,2))/2,mask).mean()
assert abs(seqmean-a.mean())>1e-4
assert abs(wrongclip-a.mean())>1e-4
zero=fe.sums(np.zeros_like(y),p,mask).sum(0);assert gr.target_scores(zero)==[0.,0.]
print(json.dumps({'status':'PASS','per_target_max_abs_error':err,'true_pooled':a.mean(),'wrong_sequence_mean':seqmean,'wrong_clip_before_blend':wrongclip,'degenerate_zero_mass':'PASS','scope':'synthetic formula check only; no empirical WP query','wrapper_sha256':hashlib.sha256((HERE.parent/'grader/evaluate.py').read_bytes()).hexdigest(),'scorer_sha256':hashlib.sha256((ROOT/'scripts/fold_eval.py').read_bytes()).hexdigest()},indent=2))
