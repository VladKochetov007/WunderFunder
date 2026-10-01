"""Independent synthetic/scalar audit only. Never invokes the empirical entry point."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,sys,struct,tempfile
import numpy as np
HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent/'renewal'
spec=importlib.util.spec_from_file_location('renewal_core_under_review',ACTOR/'core.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def scalar_events(dv,a):
 reset=[];marked=[];old=[]
 for t,x in enumerate(dv):
  valid=all(np.isfinite(v) and abs(v)<np.float32(5.19) for v in x)
  nonneutral=np.float32(max(x)-min(x))>np.float32(1e-6) if valid else False
  r=t>0 and np.isfinite(a[t]) and np.isfinite(a[t-1]) and a[t] < a[t-1]-1e-6
  prevvalid=t>0 and all(np.isfinite(v) and abs(v)<np.float32(5.19) for v in dv[t-1])
  d=t>0 and valid and prevvalid and any(v!=w for v,w in zip(x,dv[t-1])) and nonneutral
  reset.append(r);marked.append(r and valid and nonneutral);old.append(d)
 return tuple(np.array(v,dtype=bool) for v in [reset,marked,old])

def fixture(seed,n=640):
 rng=np.random.default_rng(seed);dt=rng.uniform(.5,1.5,n);age=np.zeros((n,2));mark=np.array([.6,-.4,.2,-.8],np.float32);dv=np.zeros((n,4),np.float32)
 for t in range(1,n):
  for j,period in enumerate([13,17]):age[t,j]=0. if (t+seed)%period==0 else age[t-1,j]+dt[t]
 age[:,0]=.2*age[:,0]-5.199;age[:,1]=.3*age[:,1]-5.199
 for t in range(n):
  if t%37==0:mark=rng.uniform(-1,1,4).astype(np.float32)
  dv[t]=mark
 return {'chain':seed,'class':seed%4,'half':seed%2,'age':age,'dv':dv,'dp':np.tile(np.array([.2,-.3,.7,-.4],np.float32),(n,1))}

def independent_design(records,fit):
 knots=fit['knots'];ns=[len(k)-1 for k in knots];dim=sum(ns)
 def basis(x,k):
  return np.array([[0. if v<=lo else 1. if v>=hi else (v-lo)/(hi-lo) for lo,hi in zip(k[:-1],k[1:])] for v in x])
 H=np.zeros((dim,dim));positive=[];As=[]
 for r in records:
  a=r['age'];ids=[]
  for t in range(len(a)-1):
   if all(np.isfinite(a[t:t+2,j]).all() and a[t+1,j]>=a[t,j] and a[t,j]>=knots[j][0] and a[t+1,j]<=knots[j][-1] for j in range(2)):ids.append(t)
  ids=np.array(ids,dtype=int)
  if len(ids)>1024:ids=ids[np.floor(np.linspace(0,len(ids)-1,1024)).astype(int)]
  A=np.column_stack([basis(a[ids+1,0],knots[0])-basis(a[ids,0],knots[0]),-(basis(a[ids+1,1],knots[1])-basis(a[ids,1],knots[1]))]);As.append(A)
  delta=(a[ids+1,0]-a[ids,0])/fit['input_iqr'][0];positive.extend(delta[delta>1e-12])
 scale=np.median(positive)
 for A in As:H+=A.T@A/(len(records)*len(A)*scale**2)
 offset=0
 for k,iqr,n in zip(knots,fit['input_iqr'],ns):
  for b in range(n-1):
   row=np.zeros(dim);row[offset+b]=-iqr/(k[b+1]-k[b]);row[offset+b+1]=iqr/(k[b+2]-k[b+1]);H+=.001*np.outer(row,row)/(2*(n-1))
  offset+=n
 c=np.zeros(dim);q5,q95=fit['quantiles_05_25_50_75_95'][0][[0,-1]];c[:ns[0]]=(basis([q95],knots[0])-basis([q5],knots[0]))[0]
 return H,c

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();checks=[]
 rng=np.random.default_rng(92031);dv=rng.uniform(-1,1,(500,4)).astype(np.float32);dv[1::4]=dv[::4][:len(dv[1::4])];dv[10]=np.nan;dv[11]=[.2,-.3,.1,.4];dv[20]=np.float32(5.19);dv[30]=0.;dv[40]=[1e-6,0,0,0];dv[41]=[np.nextafter(np.float32(1e-6),np.float32(np.inf)),0,0,0]
 age=rng.uniform(-5.2,2.,len(dv));age[25]=np.nan;age[50]=.1;age[51]=age[50]-1e-6;age[52]=np.nextafter(age[51]-1e-6,-np.inf)
 expected=scalar_events(dv,age);actual=core.events(dv,age)
 for e,b in zip(expected,actual):assert np.array_equal(e,b)
 pair=core.pair_masks(np.column_stack([age,np.arange(len(age),dtype=float)]));assert np.array_equal(pair['reset5'],expected[0][1:])
 checks.append('independent scalar event oracle, strict float32 mark thresholds, invalid bridge, reset-threshold boundary agreement')
 median=np.array([.1,-.1,.2,-.2]);iqr=np.array([.5,.4,.3,.6]);got=core.mark_states(dv,age,median,iqr);rr=np.zeros(4);dd=np.zeros(4);ref=[[],[]];decay=2**(-1/16)
 for t in range(len(dv)):
  u=np.clip((dv[t]-median)/np.maximum(iqr,.001),-4,4)
  if expected[1][t]:rr=decay*rr+(1-decay)*u
  if expected[2][t]:dd=decay*dd+(1-decay)*u
  ref[0].append(rr.copy());ref[1].append(dd.copy())
 for g,r in zip(got,ref):assert np.array_equal(g,np.array(r))
 for n in [1,17,51,53,499]:
  short=core.mark_states(dv[:n],age[:n],median,iqr)
  for g,s in zip(got,short):assert np.array_equal(g[:n],s)
 checks.append('independent four-state recurrence, missing-mark hold, all reset semantics and causal prefixes')
 # Simple declared maps permit literal denominators independent of the fitter.
 fit={'knots':[np.array([0.,1.,2.])]*2,'increments':[np.array([1.,1.])]*2,'reset_origin':0.,'complete_run_excursion_median':2.}
 ages=np.array([[0,0],[1,1],[1,1],[1-.5e-6,1],[0,0],[3,3],[0,0],[np.nan,0],[0,0]],float);r={'age':ages,'dv':np.tile([.5,-.5,.2,-.2],(9,1)).astype(np.float32),'dp':np.zeros((9,4)), 'chain':1,'class':0,'half':0};report=core.audit_clock([r],fit)
 assert report['counts']['all']==8 and report['counts']['finite_nonreset']==4 and report['counts']['usable']==1 and report['coverage']==.25
 assert report['counts']['both_raw_ties']==1 and report['counts']['microdrop_nonreset']==1 and report['counts']['outside_support_nonreset']==1
 assert report['reset']['total']==2 and report['reset']['near_origin']==2
 checks.append('literal all-row/finite/nonreset/tie/microdrop/outside-support denominators and reset origins')
 records=[fixture(i) for i in range(8)];fitted=core.fit_clock(records);H,c=independent_design(records,fitted);x=np.concatenate(fitted['increments'])*fitted['positive_increment_normalizer'];Hs=H/np.linalg.norm(H,2);g=Hs@x;free=x>1e-7;lam=-np.dot(c[free],g[free])/np.dot(c[free],c[free]);dual=g+lam*c
 kkt=max(float(np.max(abs(dual[free]))),float(max(0.,-dual[~free].min())) if (~free).any() else 0.,float(abs(c@x-1)),float(max(0.,-x.min())))
 assert kkt<2e-6,(kkt,dual,x)
 # Independent equality-constrained analytic oracle applies if all coefficients are interior.
 analytical=None
 invc=np.linalg.solve(Hs,c);unconstrained=invc/(c@invc)
 if unconstrained.min()>0:
  analytical=float(np.max(abs(unconstrained-x)));assert analytical<.005
 audit=core.audit_clock([fixture(i) for i in range(20,28)],fitted);assert all(audit['gates'].values()),audit
 checks.append('independently rebuilt chain-equal QP, nonzero range and KKT oracle; known shared-clock held-chain synthetic passes')
 tree=ast.parse((ACTOR/'run_inputs.py').read_text());assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALLOWED_COLUMNS' for t in n.targets));whitelist=eval(compile(ast.Expression(assign.value),'whitelist','eval'),{'range':range});assert whitelist==['a5','a6']+[f'i0_dv{i}' for i in range(4)]+[f'i0_dp{i}' for i in range(4)]
 reads=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='read_row_group'];assert len(reads)==1;kw={k.arg:ast.unparse(k.value) for k in reads[0].keywords};assert kw['columns']=='ALLOWED_COLUMNS' and kw['use_threads']=='False'
 current=json.loads((ACTOR/'selected-chains.json').read_text());prior=ACTOR.parents[1]/'2026-10-01-quant-controls/prior-inputs.json';assert sha(prior)==current['source_manifest_sha256'];assert current['selected']==json.loads(prior.read_text())['selected']
 checks.append('exact ten input-column parser whitelist and exact historical 256-chain record reuse; no empirical entry point invoked')
 run_spec=importlib.util.spec_from_file_location('renewal_entry_under_review',ACTOR/'run_inputs.py');entry=importlib.util.module_from_spec(run_spec);run_spec.loader.exec_module(entry)
 metadata=b'SYNTHETIC-FOOTER-NO-COMPETITION-DATA';tail=struct.pack('<I',len(metadata))+b'PAR1'
 with tempfile.TemporaryDirectory(prefix='review-renewal-',dir=HERE) as folder:
  fake=Path(folder)/'fake.bin';fake.write_bytes(b'PAR1synthetic body'+metadata+tail);fingerprint=entry.fingerprint(fake)
  assert fingerprint['footer_bytes']==len(metadata)+8
  assert fingerprint['footer_sha256']==hashlib.sha256(metadata+tail).hexdigest()
 checks.append('synthetic footer fingerprint independently matches historical metadata-plus-eight-byte-trailer convention')
 result={'status':'PASS','checks':checks,'kkt_residual_independent':kkt,'analytical_interior_max_error':analytical,'synthetic_clock_error':audit['error'],'source_sha256':{n:sha(ACTOR/n) for n in ['core.py','run_inputs.py','selected-chains.json']},'script_sha256':sha(__file__),'real_data_reads':0,'forecasting_fits':0,'WP_queries':0}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
