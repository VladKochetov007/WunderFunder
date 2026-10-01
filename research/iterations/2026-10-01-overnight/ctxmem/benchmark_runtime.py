"""Single fresh-process fixed3 callback timing on a frozen raw-only fixture; no labels."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def counters():
    return {parts[0]:list(map(int,parts[1:9])) for parts in
            (line.split() for line in Path('/proc/stat').read_text().splitlines())
            if parts[0] in ('cpu18','cpu58')}


def difference(a,b):
    result={}
    for key in a:
        d=[y-x for x,y in zip(a[key],b[key])];total=sum(d)
        busy=total-d[3]-d[4]
        result[key]={'ticks':total,'busy_ticks':busy,'busy_fraction':busy/total if total else 0.}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--condition',choices=['base','memory','zero'],required=True);p.add_argument('--trial',type=int,required=True)
    a=p.parse_args();a.plan=a.plan.resolve();assert 0<=a.trial<=4
    assert sorted(os.sched_getaffinity(0))==[18]
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):assert os.environ.get(key)=='1'
    plan=json.loads(a.plan.read_text());assert plan['id']=='CTXMEM01-native-cost-preflight-v1'
    for name,expected in plan['files_sha256'].items():assert sha(HERE/name)==expected,name
    directory=HERE/'runtime-prototype';output=directory/f'{a.condition}-r{a.trial}.json'
    assert not output.exists();package=(directory/a.condition).resolve();raw=ROOT/plan['raw_fixture']
    preparation=json.loads((directory/'preparation.json').read_text())
    assert preparation['plan_sha256']==sha(a.plan) and sha(raw)==plan['raw_fixture_sha256']
    for name,expected in preparation['package_files_sha256'][a.condition].items():assert sha(package/name)==expected,name
    class NoTorch:
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Torch runtime forbidden')
    sys.meta_path.insert(0,NoTorch())
    before=counters();time.sleep(2.);precheck=difference(before,counters())
    if a.condition!='zero' and precheck['cpu58']['busy_fraction']>=.05:
        result={'status':'NOISY_PRECHECK','condition':a.condition,'trial':a.trial,'precheck':precheck,'plan_sha256':sha(a.plan)}
        with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result));return
    with tempfile.TemporaryDirectory(prefix='ctx-native-cwd-') as empty:
        os.chdir(empty);sys.path.insert(0,str(package));start=time.perf_counter()
        module=importlib.import_module('solution');model=module.PredictionModel();startup=time.perf_counter()-start
        assert model.member_count==3 and module.LEVEL=='v3'
        imported_native=Path(module.wnn_cpp.__file__).resolve()
        expected_native=(plan['native_module_sha256'] if a.condition!='base' else
                         preparation['package_files_sha256']['base'][str(imported_native.relative_to(package))])
        assert sha(imported_native)==expected_native
        threads=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('Threads:')))
        assert threads==1
        x=np.load(raw,mmap_mode='r');assert x.shape==(5,20000,112)
        observations=[];outputs=[];whole=counters()
        for k,seq in enumerate(x):
            values=np.zeros((20000,2),np.float32);point=SimpleNamespace(seq_ix=10000+k,step_in_seq=0,need_prediction=False,state=None)
            b=counters();cpu0=time.process_time();t0=time.perf_counter()
            for j,row in enumerate(seq):
                point.step_in_seq=j;point.need_prediction=j>=99;point.state=row;value=model.predict(point)
                if j<99:assert value is None
                else:values[j]=value
            wall=time.perf_counter()-t0;cpu=time.process_time()-cpu0
            observations.append({'sequence':k,'wall_seconds':wall,'cpu_seconds':cpu,'activity':difference(b,counters())})
            assert np.isfinite(values).all();outputs.append(values)
        wholetrace_activity=difference(whole,counters())
        activity={}
        for key in ('cpu18','cpu58'):
            ticks=sum(row['activity'][key]['ticks'] for row in observations)
            busy=sum(row['activity'][key]['busy_ticks'] for row in observations)
            assert ticks>0
            activity[key]={'ticks':ticks,'busy_ticks':busy,'busy_fraction':busy/ticks}
        array=np.stack(outputs)
        reference=directory/f'reference-{a.condition}.npy'
        if a.condition=='zero':
            reference=directory/'reference-base.npy';np.testing.assert_array_equal(array,np.load(reference,allow_pickle=False))
        elif reference.exists():np.testing.assert_array_equal(array,np.load(reference,allow_pickle=False))
        else:np.save(reference,array)
        wall=sum(x['wall_seconds'] for x in observations);cpu=sum(x['cpu_seconds'] for x in observations)
        clean=activity['cpu58']['busy_fraction']<.05
        assert int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('Threads:')))==1
        result={'status':'PASS' if clean or a.condition=='zero' else 'NOISY_TRACE','condition':a.condition,'trial':a.trial,
                'rows':100000,'member_count':3,'weights':'fixed equal1/3','wall_seconds':wall,'cpu_seconds':cpu,
                'cpu_to_wall_ratio':cpu/wall,'us_per_row':wall/100000*1e6,'projected_minutes_39_4M':wall/100000*39400000/60,
                'precheck':precheck,'activity':activity,'wholetrace_activity':wholetrace_activity,
                'activity_qualification':'Sum timed-sequence-region busy_ticks / sum timed-sequence-region ticks',
                'sequences':observations,'sibling58_below_5pct':clean,
                'startup_seconds':startup,'max_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'prediction_digest_for_parity_only':hashlib.sha256(array.tobytes()).hexdigest(),
                'zero_extra_archived_baseline_bitexact':True if a.condition=='zero' else None,
                'qualification_only':a.condition=='zero',
                'raw_fixture_sha256':sha(raw),'plan_sha256':sha(a.plan),'harness_sha256':sha(Path(__file__)),
                'native_sha256':sha(imported_native),'solution_sha256':sha(package/'solution.py'),'threads':threads,
                'affinity':sorted(os.sched_getaffinity(0)),'torch_runtime_absent':not any(n=='torch' or n.startswith('torch.') for n in sys.modules),
                'targets_read':False,'masks_read':False,'WP_queries':0,'python':sys.version}
        with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps({k:result[k] for k in ['status','condition','trial','us_per_row','projected_minutes_39_4M','sibling58_below_5pct']}))


if __name__=='__main__':main()
