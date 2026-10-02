"""Root-allocated single native compile/check/benchmark, total60s, no retry."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--cpu',type=int,required=True);a=ap.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))=={a.cpu}
 out=a.output_dir.resolve();assert out.is_relative_to(P) and out!=P and not out.exists();out.mkdir();start=time.monotonic()
 fixed=[P/n for n in ('prepare.py','check_benchmark.py','native.cpp','README.md','source-origins.json','model.diff')]
 fixed.extend(sorted((P/'include').rglob('*.hpp')))
 fixed.extend([P.parent/'interval_memory.py',P.parents[3]/'src/wnn/nn.py',P.parents[3]/'src/wnn/__init__.py'])
 assert sha(P.parent/'interval_memory.py')=='7ef61881baf1ad0befe229ea087c958f9aad6aa65240c0b3d8580dd90015d241'
 sources={os.path.relpath(p,P):sha(p) for p in fixed}
 (out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n');records=[];result=dict(status='FAILED_NO_RETRY')
 compile=['g++','-O3','-march=x86-64-v3','-mtune=generic','-fno-trapping-math','-fno-math-errno','-std=c++17','-shared','-fPIC','-I'+str(P/'include'),str(P/'native.cpp'),'-o',str(out/'probe.so')]
 stages=[('compile',compile),('check_benchmark',[sys.executable,str(P/'check_benchmark.py'),'--output-dir',str(out),'--cpu',str(a.cpu)])]
 try:
  for name,argv in stages:
   remaining=60-(time.monotonic()-start);assert remaining>1
   t=time.monotonic()
   with (out/(name+'.stdout')).open('x') as o,(out/(name+'.stderr')).open('x') as e:r=subprocess.run(argv,stdout=o,stderr=e,timeout=remaining)
   row=dict(stage=name,argv=argv,exit_code=r.returncode,seconds=time.monotonic()-t);records.append(row);print(json.dumps(row),flush=True)
   assert r.returncode==0,('stage failed',name)
  for rel,h in sources.items():assert sha(P/rel)==h,rel
  result['status']='COMPLETE_GENERATED_NATIVE_PREPARATION'
 except BaseException as exc:result['failure']=repr(exc);raise
 finally:
  result.update(stages=records,seconds=time.monotonic()-start,sources_sha256=sha(out/'sources.json'))
  (out/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
