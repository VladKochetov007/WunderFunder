"""Wait for the authorized pipeline, then bind and grade exactly once in a CPU scope.

Default is source-only dry-run. --execute starts the bounded waiter; root launches
it detached from SSH. No training, inference, metric retry or alternative profile.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ALLOCATION=HERE/'allocation-01.json'
ALLOCATION_SHA='5bd5235593168d22fc3883d3c7168b2bfeee205690e700f9d009bee583e2dde6'
EVALUATOR=HERE/'evaluate.py'
EVALUATOR_SHA='c0ea2c2b122f5ab0b758942bc2068cce3089dfa2b8d18295fc6818fc48749d33'
BUILDER=HERE/'build_bindings_v2.py'
BUILDER_SHA='f808b4477bfe9cf700e5610930373888764c4e3f527c871398e5ab0fe861e61f'
PIPELINE=HERE/'execution-01/result.json'
FINISH=HERE/'finish-01'
BINDINGS=HERE/'evaluation-bindings-01.json'
EVALUATION=HERE/'evaluation-01'
SUCCESS='COMPLETED_4_FITS_2_OOF_NO_METRICS'


def utc():return datetime.now(timezone.utc)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def validate_sources():
    for p,h in ((ALLOCATION,ALLOCATION_SHA),(EVALUATOR,EVALUATOR_SHA),(BUILDER,BUILDER_SHA)):
        assert sha(p)==h,('source/allocation changed',str(p))
    allocation=json.loads(ALLOCATION.read_text())
    assert allocation['execution_authorized'] is True
    assert allocation['execution_directory']==str((HERE/'execution-01').relative_to(ROOT))
    deadline=datetime.fromisoformat(allocation['deadline_utc'])
    assert deadline.utcoffset() is not None
    return deadline+timedelta(seconds=60)


def scope(unit):
    assert unit.startswith('onorm-finish-') and len(unit)==len('onorm-finish-')+32
    p=subprocess.run(['systemctl','--user','show',unit+'.scope','-p','ControlGroup','-p','ActiveState','-p','SubState'],
                     text=True,capture_output=True,timeout=10)
    fields=dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
    pids=set();limits={}
    if fields.get('ControlGroup'):
        base=Path('/sys/fs/cgroup')/fields['ControlGroup'].lstrip('/')
        for cp in base.rglob('cgroup.procs'):
            try:pids.update(map(int,cp.read_text().split()))
            except FileNotFoundError:pass
        for name in ('cpu.max','memory.max','memory.swap.max'):
            try:limits[name]=(base/name).read_text().strip()
            except FileNotFoundError:pass
    return fields,pids,limits


def builder_argv():
    return [str(ROOT/'.venv/bin/python'),str(BUILDER),'--allocation',str(ALLOCATION),
            '--allocation-sha256',ALLOCATION_SHA,'--source-sha256',BUILDER_SHA,
            '--completion',str(PIPELINE),'--checkpoint-receipt-dir',str(HERE/'execution-01'),
            '--output',str(BINDINGS)]


def worker(source_hash):
    assert source_hash==sha(__file__)
    validate_sources()
    unit=os.environ['WNN_FINISH_OWN_UNIT']
    fields,pids,limits=scope(unit)
    assert os.getpid() in pids,('worker outside owned scope',unit)
    assert limits.get('memory.max')==str(10*2**30) and limits.get('memory.swap.max')=='0'
    q,period=map(int,limits['cpu.max'].split());assert q==2*period
    assert set(os.sched_getaffinity(0)).issubset({35,36})
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert FINISH.exists() and not BINDINGS.exists() and not EVALUATION.exists()
    pipeline=json.loads(PIPELINE.read_text())
    assert pipeline['status']==SUCCESS and pipeline['allocation_sha256']==ALLOCATION_SHA
    result={'status':'RUNNING','source_sha256':source_hash,'allocation_sha256':ALLOCATION_SHA,
            'pipeline_receipt_sha256':sha(PIPELINE),'scope':unit,'scope_limits':limits,'steps':[]}
    start=time.monotonic()
    try:
        for name in ('builder','evaluate'):
            validate_sources();assert sha(__file__)==source_hash
            if name=='builder':argv=builder_argv()
            else:
                assert BINDINGS.exists() and not EVALUATION.exists()
                argv=[str(ROOT/'.venv/bin/python'),str(EVALUATOR),'--bindings',str(BINDINGS),
                      '--bindings-sha256',sha(BINDINGS),'--output-dir',str(EVALUATION)]
            with (FINISH/f'{name}.log').open('x') as stream:
                completed=subprocess.run(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            result['steps'].append({'name':name,'argv':argv,'exit_code':completed.returncode})
            assert completed.returncode==0,(name,completed.returncode)
        er=json.loads((EVALUATION/'receipt.json').read_text())
        assert er['status']=='COMPLETE_ONE_FIXED_ONORM_REPLACEMENT'
        result.update(status='COMPLETE_ONE_AUTHORIZED_EVALUATION',bindings_sha256=sha(BINDINGS),
                      evaluation_receipt_sha256=sha(EVALUATION/'receipt.json'),
                      evaluation_report_sha256=sha(EVALUATION/'report.json'))
    except BaseException as exc:
        result.update(status='FAILED_STOP_NO_RETRY',failure=repr(exc));raise
    finally:
        result['seconds']=time.monotonic()-start
        write(FINISH/'worker-result.json',result)


def execute():
    deadline=validate_sources();source_hash=sha(__file__)
    assert not FINISH.exists() and not BINDINGS.exists() and not EVALUATION.exists(), 'output collision; no resume or retry'
    FINISH.mkdir()
    start=time.monotonic();unit=None;process=None
    result={'status':'WAITING_FOR_PIPELINE','source_sha256':source_hash,'allocation_sha256':ALLOCATION_SHA,
            'evaluator_sha256':EVALUATOR_SHA,'builder_v2_sha256':BUILDER_SHA,
            'started_utc':utc().isoformat(),'wait_deadline_utc':deadline.isoformat(),
            'GPU_allocations':0,'training_or_inference_jobs':0,'metric_retries':0}
    write(FINISH/'started.json',result)
    try:
        while True:
            assert utc()<=deadline,'pipeline receipt wait deadline exceeded'
            if PIPELINE.exists():
                try:pipeline=json.loads(PIPELINE.read_text())
                except json.JSONDecodeError:
                    # The runner writes a single terminal receipt; allow that write
                    # to finish, without ever launching a partial or failed result.
                    time.sleep(2);continue
                if pipeline.get('status')!=SUCCESS:
                    result.update(status='ABORTED_PIPELINE_NOT_SUCCESSFUL',pipeline_status=pipeline.get('status'),
                                  pipeline_receipt_sha256=sha(PIPELINE));return
                assert pipeline['allocation_sha256']==ALLOCATION_SHA
                break
            time.sleep(5)
        validate_sources();assert sha(__file__)==source_hash
        assert not BINDINGS.exists() and not EVALUATION.exists(),'output appeared during wait; no resume'
        unit='onorm-finish-'+uuid.uuid4().hex
        _,existing,_=scope(unit);assert not existing
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
            PYTHONUNBUFFERED='1',WNN_FINISH_OWN_UNIT=unit,WNN_THREADS='2')
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS'):env[name]='2'
        argv=['systemd-run','--user','--scope','--quiet','--unit',unit,
              '-p','CPUQuota=200%','-p',f'MemoryMax={10*2**30}',
              '-p','MemorySwapMax=0','--','taskset','-c','35,36','nice','-n','10',
              '/usr/bin/timeout','--signal=TERM','--kill-after=10s','900s','/usr/bin/time','-v',
              str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--worker','--source-sha256',source_hash]
        result.update(status='RUNNING_ONE_CPU_EVALUATION',unit=unit,argv=argv,
                      pipeline_receipt_sha256=sha(PIPELINE))
        write(FINISH/'cpu-launch.json',result)
        with (FINISH/'scope.log').open('x') as stream:
            process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            result['launcher_pid']=process.pid
            rc=process.wait(timeout=930)
        result['exit_code']=rc
        assert rc==0,('CPU stage exit',rc)
        wr=json.loads((FINISH/'worker-result.json').read_text())
        assert wr['status']=='COMPLETE_ONE_AUTHORIZED_EVALUATION'
        result.update(status=wr['status'],worker_receipt_sha256=sha(FINISH/'worker-result.json'),
                      evaluation_receipt_sha256=wr['evaluation_receipt_sha256'])
    except BaseException as exc:
        result.update(status='FAILED_STOP_NO_RETRY',failure=repr(exc));raise
    finally:
        try:
            if unit is not None:
                fields,pids,_=scope(unit)
                if pids or (process is not None and process.poll() is None):
                    # Only this fresh owned UUID scope is signalled, never training/foreign jobs.
                    subprocess.run(['systemctl','--user','stop',unit+'.scope'],timeout=15,check=False)
                    if process is not None:
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            subprocess.run(['systemctl','--user','kill','--signal=KILL',unit+'.scope'],timeout=15,check=False)
                            process.wait(timeout=15)
                    fields,pids,_=scope(unit)
                result.update(scope_state=fields,remaining_scope_pids=sorted(pids))
                if pids:result.update(status='FAILED_SCOPE_NOT_RELEASED',cleanup_failure='owned CPU scope still has PIDs')
        except BaseException as exc:
            # Retain the primary failure and a terminal receipt even if systemd
            # itself becomes unavailable during cleanup. Never claim release.
            result.update(status='FAILED_SCOPE_RELEASE_UNVERIFIED',cleanup_failure=repr(exc),remaining_scope_pids=None)
        result.update(seconds=time.monotonic()-start,finished_utc=utc().isoformat())
        write(FINISH/'result.json',result)
        print(json.dumps(result),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--source-sha256',help=argparse.SUPPRESS)
    a=ap.parse_args();assert not (a.worker and a.execute)
    if a.worker:assert a.source_sha256;worker(a.source_sha256)
    elif a.execute:execute()
    else:
        deadline=validate_sources()
        print(json.dumps({'status':'SOURCE_ONLY_DRY_RUN_NO_PIPELINE_OR_DATA_READ',
            'source_sha256':sha(__file__),'wait_deadline_utc':deadline.isoformat(),
            'builder_argv':builder_argv(),'CPUQuota':200,'cpu_ids':[35,36],
            'memory_GiB':10,'swap':0,'worker_timeout_seconds':900,'automatic_retries':0},indent=2))


if __name__=='__main__':main()
