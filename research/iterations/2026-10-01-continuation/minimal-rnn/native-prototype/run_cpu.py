"""Bounded isolated compile/check controller; no data/GPU/engine access."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

P=Path(__file__).resolve().parent
ROOT=P.parents[4]
OUT=P/'generated-01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def cpu_sample():
    out={}
    for line in Path('/proc/stat').read_text().splitlines():
        label,*nums=line.split()
        if label in ['cpu34','cpu36','cpu74','cpu76']:
            v=list(map(int,nums));out[label]=(sum(v[:8]),v[3]+v[4])
    return out


def main():
    OUT.mkdir(exist_ok=False)
    begin=time.monotonic();a=cpu_sample();time.sleep(1);b=cpu_sample()
    busy={k:1-(b[k][1]-a[k][1])/max(1,b[k][0]-a[k][0]) for k in a}
    (OUT/'preflight.json').write_text(json.dumps({'CPU_samples_busy':busy,'reserved_CUDA_VISIBLE_DEVICES':'','source_sha256':{p.name:sha(p) for p in P.iterdir() if p.is_file()}},indent=2)+'\n')
    assert len(busy)==4 and max(busy.values()) < .2, 'allocated CPU/sibling currently busy; no auto substitution'
    env=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','NUMBA_NUM_THREADS=1']
    stages=[('compile','34,36',['g++','-O3','-march=x86-64-v3','-mtune=generic','-fno-trapping-math','-fno-math-errno','-std=c++17','-shared','-fPIC',str(P/'minimal_native.cpp'),'-o',str(P/'libminimal_fp32.so')]),
            ('check','34',[str(ROOT/'.venv/bin/python'),str(P/'check_native.py')])]
    for name,cpus,program in stages:
        unit='cont-mingru-native-fp32-'+name+'-01'
        cmd=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=200%','-p','MemoryMax=4G','-p','MemorySwapMax=0','--','taskset','-c',cpus,'nice','-n','10',*env,'/usr/bin/timeout','--signal=TERM','--kill-after=5s','300s','/usr/bin/time','-v',*program]
        t=time.monotonic()
        with (OUT/(name+'.stdout')).open('x') as stdout,(OUT/(name+'.stderr')).open('x') as stderr:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
            try: code=process.wait(timeout=310)
            except BaseException:
                subprocess.run(['systemctl','--user','stop',unit+'.scope'],timeout=10)
                process.wait(timeout=10)
                raise
        release=subprocess.run(['systemctl','--user','show',unit+'.scope','-p','ActiveState','-p','SubState'],capture_output=True,text=True,timeout=5)
        receipt={'status':'PASS' if code==0 else 'FAILED_NO_RETRY','exit_code':code,'seconds':time.monotonic()-t,'scope':unit,'scope_release':release.stdout,'argv':cmd}
        (OUT/(name+'-execution.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
        if code: return
    (OUT/'execution.json').write_text(json.dumps({'status':'PASS','seconds':time.monotonic()-begin,'controller_sha256':sha(Path(__file__)),'GPU_used':False,'forecast_fit_or_metric':False},indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():(OUT/'controller-failure.json').write_text(json.dumps({'status':'FAILED_NO_RETRY','exception':repr(exc)},indent=2)+'\n')
        raise
