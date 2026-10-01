"""After root freeze: complete callbacks, fixed Q3; no targets, metric, or tuning."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse, hashlib, importlib, io, json, os, resource, sys, tempfile, time, zipfile
import numpy as np

p=argparse.ArgumentParser()
for name in ("zip","raw","reference","output"):p.add_argument("--"+name,type=Path,required=True)
p.add_argument("--rows",type=int,choices=[100000,1000000],required=True)
p.add_argument("--backend",choices=["vnni","avx2"],default="vnni")
a=p.parse_args()
for key in ("zip","raw","reference","output"):setattr(a,key,getattr(a,key).resolve())
assert not a.output.exists() and sorted(os.sched_getaffinity(0))==[18]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def counters():
 return {line.split()[0]:list(map(int,line.split()[1:9])) for line in Path("/proc/stat").read_text().splitlines() if line.split()[0] in ("cpu18","cpu58")}
def activity(before,after):
 result={}
 for key in before:
  d=[y-x for x,y in zip(before[key],after[key])];total=sum(d);busy=total-d[3]-d[4]
  result[key]={"total_ticks":total,"busy_ticks":busy,"busy_fraction":busy/total if total else 0.}
 return result

def errors(candidate,reference):
 c=np.asarray(candidate,dtype=np.float64).reshape(-1,20000,2)[:,99:].reshape(-1,2)
 r=np.asarray(reference,dtype=np.float64).reshape(-1,20000,2)[:,99:].reshape(-1,2)
 result=[]
 for k in range(2):
  d=c[:,k]-r[:,k];rmse=float(np.sqrt(np.mean(d*d)));std=float(np.std(r[:,k]));v={"target":k,"rmse":rmse,"reference_population_std":std,"relative_rmse":rmse/max(std,1e-8),"p99_abs":float(np.quantile(np.abs(d),.99)),"max_abs":float(np.max(np.abs(d)))}
  assert v["relative_rmse"]<=.03 and v["p99_abs"]<=.05 and v["max_abs"]<=.5,v
  result.append(v)
 return result

class NoTorch:
 def find_spec(self,fullname,path=None,target=None):
  if fullname=="torch" or fullname.startswith("torch."):raise AssertionError("Torch runtime forbidden")
sys.meta_path.insert(0,NoTorch())
raw_hash=sha(a.raw);reference_hash=sha(a.reference);zip_hash=sha(a.zip)
assert raw_hash=="1e90c52f1332950b70d573618adad29324fb6705675e74cb75c7ca6efda993b4"
with tempfile.TemporaryDirectory(prefix="q3-timing-") as tmp:
 d=Path(tmp)/"package";d.mkdir();cwd=Path(tmp)/"empty-cwd";cwd.mkdir();os.chdir(cwd);t=time.perf_counter()
 with zipfile.ZipFile(a.zip) as z:
  assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist());z.extractall(d)
 extract_s=time.perf_counter()-t;sys.path.insert(0,str(d));t=time.perf_counter()
 original_open=open
 def forced_open(path,*args,**kwargs):
  return io.StringIO("flags : avx avx2 fma\n") if str(path)=="/proc/cpuinfo" else original_open(path,*args,**kwargs)
 if a.backend=="avx2":
  with patch("builtins.open",forced_open):module=importlib.import_module("solution")
  assert module.LEVEL=="avx2" and module.wnn_cpp.AVX2_QUANT_SUPPORTED
  original_constructor=module.wnn_cpp.Ensemble;constructed=[]
  def force_backend(*args,**kwargs):
   assert kwargs["dynamic_int8"] and kwargs["quant_backend"]=="auto"
   kwargs["quant_backend"]="avx2";constructed.append(kwargs.copy());return original_constructor(*args,**kwargs)
  with patch.object(module.wnn_cpp,"Ensemble",force_backend):model=module.PredictionModel()
  assert len(constructed)==1;effective_backend=1
 else:
  module=importlib.import_module("solution");model=module.PredictionModel()
  assert module.LEVEL=="v3" and model.quant_backend==2;effective_backend=2
 startup=time.perf_counter()-t;assert model.member_count==3
 startup_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 x=np.load(a.raw,mmap_mode="r");reference=np.load(a.reference,mmap_mode="r")
 assert x.shape==(5,20000,112) and reference.shape==(5,20000,2)
 pre_before=counters();time.sleep(2);pre=activity(pre_before,counters())
 if pre["cpu58"]["busy_fraction"]>=.05:
  r={"status":"NOISY_PRECHECK","backend":a.backend,"precheck_activity":pre,"zip_sha256":zip_hash,"raw_sha256":raw_hash,"reference_sha256":reference_hash,"targets_read":False,"WP_queries":0}
  a.output.write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r));sys.exit(3)
 sequence=[];digest=hashlib.sha256();expected_digest=hashlib.sha256();first_five=[];all_bitexact=True;whole_before=counters()
 for k in range(a.rows//20000):
  seq=np.asarray(x[k%5]);y=np.zeros((20000,2),np.float32);point=SimpleNamespace(seq_ix=10000+k,step_in_seq=0,need_prediction=False,state=None)
  before=counters();cpu0=time.process_time();t=time.perf_counter()
  for j,row in enumerate(seq):
   point.step_in_seq=j;point.need_prediction=j>=99;point.state=row;value=model.predict(point)
   if j<99:assert value is None
   else:y[j]=value
  wall=time.perf_counter()-t;cpu=time.process_time()-cpu0;after=counters()
  assert np.isfinite(y).all();expected=np.asarray(reference[k%5]).copy();expected[:99]=0
  bitexact=y.tobytes()==expected.tobytes();all_bitexact &= bitexact
  if a.backend=="vnni":np.testing.assert_array_equal(y,expected)
  numerical=errors(y,expected) if a.backend=="avx2" else None
  if k<5:first_five.append(y.copy())
  digest.update(y.tobytes());expected_digest.update(expected.tobytes())
  sequence.append({"index":k,"rows":20000,"wall_seconds":wall,"cpu_seconds":cpu,"activity":activity(before,after),"reference_bitexact":bitexact,"reference_errors":numerical})
 whole_activity=activity(whole_before,counters());timed={}
 for key in ("cpu18","cpu58"):
  total=sum(s["activity"][key]["total_ticks"] for s in sequence);busy=sum(s["activity"][key]["busy_ticks"] for s in sequence)
  timed[key]={"total_ticks":total,"busy_ticks":busy,"busy_fraction":busy/total if total else 0.}
 wall=sum(s["wall_seconds"] for s in sequence);cpu=sum(s["cpu_seconds"] for s in sequence)
 pooled_errors=errors(np.stack(first_five),reference) if a.backend=="avx2" else None
 if a.backend=="vnni":assert digest.hexdigest()==expected_digest.hexdigest()
 assert not any(n=="torch" or n.startswith("torch.") for n in sys.modules)
 status=dict(line.split(":",1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
 assert int(status["Threads"])==1
 cpuinfo=[z for z in Path("/proc/cpuinfo").read_text().split("\n\n") if z.startswith("processor\t: 18\n")]
 projected=wall/a.rows*39400000/60;quiet=timed["cpu58"]["busy_fraction"]<.05
 r={"status":"NOISY" if not quiet else "PASS" if projected<=50 else "FAIL_RUNTIME","policy":"Q3 fixed d256+s3+c500 equal1/3 dynamic-int8; no governor","rows":a.rows,"sequence_count":len(sequence),"wall_seconds":wall,"process_cpu_seconds":cpu,"cpu_to_wall_ratio":cpu/wall,"us_per_row":wall/a.rows*1e6,"projected_minutes_39_4M":projected,"precheck_activity":pre,"aggregate_cpu_activity":timed,"whole_window_cpu_activity":whole_activity,"smt_sibling_below_5pct":quiet,"per_sequence":sequence,"startup_import_model_seconds":startup,"extract_seconds":extract_s,"startup_max_rss_KiB":startup_rss,"max_rss_KiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"final_vm_rss":status.get("VmRSS","").strip(),"threads":int(status["Threads"]),"affinity":sorted(os.sched_getaffinity(0)),"cpuinfo":cpuinfo[0] if cpuinfo else None,"cpu_level":module.LEVEL,"effective_integer_backend":a.backend,"effective_integer_backend_code":effective_backend,"forced_module_and_backend_test_only":a.backend=="avx2","zip_sha256":zip_hash,"zip_bytes":a.zip.stat().st_size,"raw_sha256":raw_hash,"reference_sha256":reference_hash,"solution_sha256":sha(d/"solution.py"),"native_sha256":sha(module.wnn_cpp.__file__),"prediction_sha256":digest.hexdigest(),"reference_prediction_sha256":expected_digest.hexdigest(),"frozen_v3_quant_reference_bitexact":all_bitexact,"pooled_cross_isa_errors":pooled_errors,"runtime_gate_pass":projected<=50 and quiet,"quality_qualification":"not established by runtime; non-VNNI WP remains unqualified","torch_runtime_absent":True,"targets_read":False,"WP_queries":0,"python":sys.version,"harness_sha256":sha(__file__)}
 a.output.write_text(json.dumps(r,indent=2)+"\n");print(json.dumps({k:r[k] for k in ["status","rows","cpu_level","effective_integer_backend","us_per_row","projected_minutes_39_4M","smt_sibling_below_5pct","startup_import_model_seconds","max_rss_KiB","prediction_sha256"]}))
