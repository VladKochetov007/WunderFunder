"""Full final-ZIP callbacks and raw-only exact reference, including long reset trace."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib,json,os,resource,sys,tempfile,time,zipfile
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--rows',type=int,choices=[100000,1000000],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
a.zip=a.zip.resolve();a.raw=a.raw.resolve();a.reference=a.reference.resolve();a.output=a.output.resolve();assert sorted(os.sched_getaffinity(0))==[18]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def counters():return {line.split()[0]:list(map(int,line.split()[1:9]))for line in Path('/proc/stat').read_text().splitlines()if line.split()[0]in('cpu18','cpu58')}
def delta(b,e):
 result={}
 for key in b:
  d=[y-x for x,y in zip(b[key],e[key])];total=sum(d);result[key]={'total_ticks':total,'busy_fraction':1-(d[3]+d[4])/total if total else 0}
 return result
class NoTorch:
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch'or fullname.startswith('torch.'):raise AssertionError('Torch runtime forbidden')
sys.meta_path.insert(0,NoTorch())
with tempfile.TemporaryDirectory(prefix='baseline3-timing-')as tmp:
 d=Path(tmp)/'package';d.mkdir();cwd=Path(tmp)/'empty-cwd';cwd.mkdir();os.chdir(cwd);t=time.perf_counter()
 with zipfile.ZipFile(a.zip)as z:z.extractall(d)
 extract_s=time.perf_counter()-t;sys.path.insert(0,str(d));t=time.perf_counter();module=importlib.import_module('solution');model=module.PredictionModel();startup=time.perf_counter()-t
 assert module.LEVEL=='v3'and model.member_count==3;startup_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 x=np.load(a.raw,mmap_mode='r');reference=np.load(a.reference,mmap_mode='r');sequence=[];digest=hashlib.sha256();expected_digest=hashlib.sha256();whole_before=counters()
 for k in range(a.rows//20000):
  seq=np.asarray(x[k%5]);y=np.zeros((len(seq),2),np.float32);point=SimpleNamespace(seq_ix=10000+k,step_in_seq=0,need_prediction=False,state=None)
  b=counters();cpu0=time.process_time();t=time.perf_counter()
  for j,row in enumerate(seq):
   point.step_in_seq=j;point.need_prediction=j>=99;point.state=row;value=model.predict(point)
   if j<99:assert value is None
   else:y[j]=value
  wall=time.perf_counter()-t;cpu=time.process_time()-cpu0;e=counters();assert np.isfinite(y).all();expected=np.asarray(reference[k%5]).copy();expected[:99]=0;np.testing.assert_array_equal(y,expected);digest.update(y.tobytes());expected_digest.update(expected.tobytes())
  sequence.append({'index':k,'rows':len(seq),'wall_seconds':wall,'cpu_seconds':cpu,'activity':delta(b,e)})
 activity=delta(whole_before,counters());wall=sum(s['wall_seconds']for s in sequence);cpu=sum(s['cpu_seconds']for s in sequence)
 assert digest.hexdigest()==expected_digest.hexdigest();assert not any(n=='torch'or n.startswith('torch.')for n in sys.modules)
 status=dict(line.split(':',1)for line in Path('/proc/self/status').read_text().splitlines()if':'in line)
 cpuinfo=[z for z in Path('/proc/cpuinfo').read_text().split('\n\n')if z.startswith('processor\t: 18\n')]
 r={'status':'PASS','policy':'fixed3 equal coefficients, no governor; fallback pending final selection','rows':a.rows,'sequence_count':len(sequence),'wall_seconds':wall,'process_cpu_seconds':cpu,'cpu_to_wall_ratio':cpu/wall,'us_per_row':wall/a.rows*1e6,'projected_minutes_39_4M':wall/a.rows*39400000/60,'aggregate_cpu_activity':activity,'smt_sibling_below_5pct':activity['cpu58']['busy_fraction']<.05,'per_sequence':sequence,'startup_import_model_seconds':startup,'extract_seconds':extract_s,'startup_max_rss_KiB':startup_rss,'max_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'final_vm_rss':status.get('VmRSS','').strip(),'threads':int(status['Threads']),'affinity':sorted(os.sched_getaffinity(0)),'cpuinfo':cpuinfo[0]if cpuinfo else None,'cpu_level':module.LEVEL,'zip_sha256':sha(a.zip),'zip_bytes':a.zip.stat().st_size,'solution_sha256':sha(d/'solution.py'),'native_sha256':sha(module.wnn_cpp.__file__),'prediction_sha256':digest.hexdigest(),'archived_reference_bitexact':True,'torch_runtime_absent':True,'targets_read':False,'WP_queries':0,'python':sys.version,'harness_sha256':sha(__file__)}
 a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in['rows','us_per_row','projected_minutes_39_4M','smt_sibling_below_5pct','startup_import_model_seconds','max_rss_KiB','prediction_sha256']}))
