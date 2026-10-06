"""Single CPU-only bounded direct-scan generated probe, no retries."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

P=Path(__file__).resolve().parent;ROOT=P.parents[4];OUT=P/'generated-01'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def sample():
    out={}
    for line in Path('/proc/stat').read_text().splitlines():
        label,*nums=line.split()
        if label in ['cpu34','cpu36','cpu74','cpu76']:
            v=list(map(int,nums));out[label]=(sum(v[:8]),v[3]+v[4])
    return out


def main():
    OUT.mkdir(exist_ok=False);begin=time.monotonic();a=sample();time.sleep(1);b=sample()
    busy={k:1-(b[k][1]-a[k][1])/max(1,b[k][0]-a[k][0]) for k in a}
    (OUT/'preflight.json').write_text(json.dumps({'CPU_samples_busy':busy,'source_sha256':{p.name:sha(p) for p in P.iterdir() if p.is_file()}},indent=2)+'\n')
    assert len(busy)==4 and max(busy.values())<.2,'allocated CPU/sibling busy; no auto substitution'
    unit='cont-mingru-directscan-cpu-01'
    cmd=['systemd-run','--user','--scope','--quiet','--unit',unit,'-p','CPUQuota=200%','-p','MemoryMax=4G','-p','MemorySwapMax=0','--','taskset','-c','34','nice','-n','10','env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','NUMBA_NUM_THREADS=1','/usr/bin/timeout','--signal=TERM','--kill-after=5s','300s','/usr/bin/time','-v',str(ROOT/'.venv/bin/python'),str(P/'check_direct.py')]
    with (OUT/'check.stdout').open('x') as stdout,(OUT/'check.stderr').open('x') as stderr:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
        try:code=process.wait(timeout=310)
        except BaseException:
            subprocess.run(['systemctl','--user','stop',unit+'.scope'],timeout=10);process.wait(timeout=10);raise
    release=subprocess.run(['systemctl','--user','show',unit+'.scope','-p','ActiveState','-p','SubState'],capture_output=True,text=True,timeout=5)
    result=json.loads((OUT/'result.json').read_text()) if (OUT/'result.json').exists() else {}
    receipt={'execution_status':'COMPLETE' if code==0 else 'FAILED_NO_RETRY','numerical_status':result.get('status'),'exit_code':code,'seconds':time.monotonic()-begin,'scope':unit,'scope_release':release.stdout,'argv':cmd,'controller_sha256':sha(Path(__file__))}
    assert sum(p.stat().st_size for p in P.rglob('*') if p.is_file())<=100*(1<<20)
    (OUT/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():(OUT/'controller-failure.json').write_text(json.dumps({'status':'FAILED_NO_RETRY','exception':repr(exc)},indent=2)+'\n')
        raise
