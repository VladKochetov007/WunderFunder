"""Generated arrays only: independent literal next-change oracle + NPZ mapping."""
from pathlib import Path
import importlib.util,json,tempfile
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('target_coverage',HERE/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

def main():
 cases=0
 for d in ([0,0,0,0],[0,1,1,1],[0,0,0,1],[0,1,0,1],[0,0,1,1,0,0]):
  y=np.column_stack((d,np.zeros(len(d)))).astype(np.float32);m=C.masks(y);expected=[]
  for t in range(len(d)):
   future=[j for j in range(t+1,len(d)) if d[j]!=d[j-1]]
   expected.append(not future)
  np.testing.assert_array_equal(m['censored'],expected)
  assert not m['informative'][-1] and m['lower_bound'][-1]==-1.5
  assert np.all(m['trainer_sentinel']>m['lower_bound'])
  assert bool(m['no_event_sequence'].all())==(len(set(d))==1)
  cases+=1
 y=np.zeros((105,2),np.float32);y[-3:,0]=2;total=C.empty();C.update(total,y,C.masks(y));s=C.summarize(total)
 assert s['all']['eligible']['rows']==6 and s['all']['eligible']['target_mass']==6
 assert s['all']['informative_eventful_tail']['rows']==2 and s['all']['informative_eventful_tail']['target_mass']==4
 assert s['all']['informative_eventful_tail']['hypothetical_since_factor_fraction_lower']==4/12
 assert s['all']['informative_eventful_tail']['hypothetical_since_factor_fraction_upper']==16/18
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'generated.npz';expected=np.arange(120,dtype=np.float32).reshape(3,20,2)
  np.savez(p,y=expected,mask=np.zeros((3,20),bool))
  mapped=C.stored_array(p,'y');np.testing.assert_array_equal(mapped,expected);assert isinstance(mapped,np.memmap)
 print(json.dumps(dict(status='PASS_GENERATED_ONLY_CENSOR_CHECK',literal_cases=cases,weighted_count_case=True,stored_NPZ_memmap_case=True,data_reads=0,coverage_source_sha256=C.sha(HERE/'coverage.py'),protocol_sha256=C.sha(HERE/'protocol.json'))))
if __name__=='__main__':main()
