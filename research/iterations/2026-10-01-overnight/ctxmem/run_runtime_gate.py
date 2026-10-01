"""Execute only root-allocated CTXMEM native cost preflight; never score forecasts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--expected-plan-sha256',required=True)
    args=parser.parse_args();args.plan=args.plan.resolve()
    assert sha(args.plan)==args.expected_plan_sha256
    plan=json.loads(args.plan.read_text())
    assert plan['id']=='CTXMEM01-native-cost-preflight-v1'
    for name,expected in plan['files_sha256'].items():assert sha(HERE/name)==expected,name
    directory=HERE/'runtime-prototype';assert not directory.exists()
    started=time.time();attempts=[];pairs=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    receipt=HERE/'runtime-gate-result.json';assert not receipt.exists()
    try:
        prep=subprocess.run([sys.executable,str(HERE/'prepare_runtime.py'),'--plan',str(args.plan)],
                            env=env,text=True,capture_output=True,timeout=120)
        (HERE/'runtime-preparation.stdout').write_text(prep.stdout)
        (HERE/'runtime-preparation.stderr').write_text(prep.stderr)
        assert prep.returncode==0,f'Preparation failed, exit {prep.returncode}'
        def run(condition,trial):
            assert time.time()-started<780,'Overall preflight time limit approached'
            command=['timeout','--signal=TERM','--kill-after=5s','120s','prlimit','--as=4294967296',
                     'taskset','-c','18',plan['scorer_python'],str(HERE/'benchmark_runtime.py'),
                     '--plan',str(args.plan),'--condition',condition,'--trial',str(trial)]
            with (directory/f'{condition}-r{trial}.stdout').open('x') as stdout, \
                 (directory/f'{condition}-r{trial}.stderr').open('x') as stderr:
                p=subprocess.run(command,env=env,stdout=stdout,stderr=stderr,timeout=130)
            assert p.returncode==0,f'{condition} trial {trial} failed exit {p.returncode}'
            path=directory/f'{condition}-r{trial}.json';row=json.loads(path.read_text())
            attempts.append({'condition':condition,'trial':trial,'status':row['status'],'receipt_sha256':sha(path)})
            return row
        # Whole-pair noise retries only. Never drop a clean unfavorable pair.
        for trial in range(5):
            order=('base','memory') if trial%2==0 else ('memory','base')
            results={condition:run(condition,trial) for condition in order}
            clean=all(r['status']=='PASS' for r in results.values())
            pair={'trial':trial,'order':list(order),'clean':clean,
                  'ratio':results['memory']['wall_seconds']/results['base']['wall_seconds'] if clean else None,
                  'base_seconds':results['base'].get('wall_seconds'),
                  'memory_seconds':results['memory'].get('wall_seconds')}
            pairs.append(pair)
            if sum(p['clean'] for p in pairs)==3:break
        clean=[p for p in pairs if p['clean']]
        if len(clean)!=3:
            result={'status':'INCONCLUSIVE_NOISE','reason':'Fewer than three clean whole pairs within five attempts'}
        else:
            qualification=run('zero',0)
            assert qualification['zero_extra_archived_baseline_bitexact'] is True
            median=statistics.median(p['ratio'] for p in clean)
            result={'status':'PASS' if median<=1.03 else 'FAIL_RUNTIME',
                    'primary_median_paired_ratio':median,'primary_overhead_fraction':median-1,
                    'gate_max_ratio':1.03,'zero_extra_archived_baseline_bitexact':True,
                    'diagnostic_ratio_of_condition_medians':statistics.median(p['memory_seconds'] for p in clean)/statistics.median(p['base_seconds'] for p in clean),
                    'diagnostic_memory_projected_minutes_39_4M':statistics.median(p['memory_seconds'] for p in clean)/100000*39400000/60,
                    'diagnostic_base_projected_minutes_39_4M':statistics.median(p['base_seconds'] for p in clean)/100000*39400000/60}
        result.update({'plan_sha256':sha(args.plan),'pairs':pairs,'attempts':attempts,
                       'seconds':time.time()-started,'targets_read':False,'masks_read':False,'WP_queries':0,
                       'forecast_fits':0,'does_not_authorize_training':True,'final_complete_zip_50minute_gate_still_required':True})
    except BaseException as error:
        result={'status':'FAIL_PREFLIGHT','error':repr(error),'plan_sha256':sha(args.plan),
                'pairs':pairs,'attempts':attempts,'seconds':time.time()-started}
        with receipt.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        raise
    with receipt.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
