"""Root-allocated CPU-only compile, generated checks and fixed paired benchmark. No retries."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--cpu',type=int,required=True);a=ap.parse_args()
 out=a.output_dir.resolve();assert out.is_relative_to(P) and out!=P and not out.exists();out.mkdir()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))=={a.cpu}
 sources={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file() and not p.is_relative_to(out)}
 (out/'source-before.json').write_text(json.dumps(sources,indent=2)+'\n');started=time.monotonic()
 commands=[('compile_reference',['g++','-O3','-march=x86-64-v3','-mtune=generic','-fno-trapping-math','-fno-math-errno','-std=c++17','-shared','-fPIC',str(P/'reference/minimal_native.cpp'),'-o',str(out/'reference.so')],120),
 ('compile_optimized',['g++','-O3','-march=x86-64-v3','-mtune=generic','-fno-trapping-math','-fno-math-errno','-std=c++17','-shared','-fPIC','-I'+str(P/'include'),str(P/'native.cpp'),'-o',str(out/'optimized.so')],120),
 ('check_benchmark',[sys.executable,str(P/'check_benchmark.py'),'--output-dir',str(out),'--cpu',str(a.cpu)],300)]
 receipts=[]
 try:
  for name,argv,limit in commands:
   t=time.monotonic()
   with (out/(name+'.stdout')).open('x') as stdout,(out/(name+'.stderr')).open('x') as stderr:r=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=limit,check=False)
   q=dict(name=name,argv=argv,exit_code=r.returncode,seconds=time.monotonic()-t);receipts.append(q);(out/(name+'.json')).write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q),flush=True)
   assert r.returncode==0,('stage failed',name)
  for name,h in sources.items():assert sha(P/name)==h,('source changed',name)
  receipt=dict(status='COMPLETE_GENERATED_NATIVE_READINESS_ONLY',stages=receipts,source_inventory_sha256=sha(out/'source-before.json'),seconds=time.monotonic()-started)
 except BaseException as exc:
  receipt=dict(status='FAILED_NO_RETRY',exception=repr(exc),stages=receipts,seconds=time.monotonic()-started);(out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n');raise
 (out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
