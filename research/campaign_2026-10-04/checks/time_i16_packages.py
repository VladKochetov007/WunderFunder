"""Measure fixed scorer callbacks against the public q16x reference package."""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path.cwd()
    campaign = root/'research/campaign_2026-10-04'
    verification = json.loads((campaign/'checks/e412_i16_package_verification_v2.json').read_text())
    if not verification['passed']:
        raise ValueError('Numerical and reset gates must pass before timing')
    packages = {
        'combo_i16': root/'submissions/c20261004_combo_s3_i16/submission.zip',
        'public_q16x_reference': root/'artifacts/submission_quota_2026-10-03/3_precision_q16x_D256.zip',
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in packages.items()}
    if hashes['combo_i16'] != verification['package_sha256']:
        raise ValueError('Verified package changed')
    if 15 not in os.sched_getaffinity(0):
        raise ValueError('Timing requires the fixed E409 CPU15 affinity')
    os.sched_setaffinity(0, {15})
    output = campaign/'checks/e412_clean_package_timing_v1.json'
    trials_file = campaign/'checks/e412_clean_package_timing_v1_trials.jsonl'
    if output.exists() or trials_file.exists():
        raise FileExistsError('Timing output identity already used')
    environment = dict(os.environ)
    environment.pop('WNN_PRECISION', None)  # Each exact package owns its precision file.
    environment['WNN_NATIVE_LEVEL']='v3'
    trials=[]
    for repeat in range(5):
        order=list(packages) if repeat%2==0 else list(reversed(packages))
        for position,name in enumerate(order):
            command=[sys.executable,'scripts/engine_fp16_check.py','calib',str(packages[name]),
                     '--data','cache/valid_subset20.parquet','--seqs','3']
            completed=subprocess.run(command,env=environment,text=True,capture_output=True,check=True)
            result=json.loads(completed.stdout.strip().splitlines()[-1])
            if result['kcycles_per_row']<=0:
                raise RuntimeError('CPU cycle counter unavailable')
            trial=dict(case=name,repeat=repeat,position=position,**result,
                       stderr=completed.stderr,loadavg=Path('/proc/loadavg').read_text())
            trials.append(trial)
            with trials_file.open('a') as f:
                f.write(json.dumps(trial)+'\n')
            print(json.dumps({key:trial[key] for key in ('case','repeat','us_per_row','kcycles_per_row')}),flush=True)
    summaries={name:{metric:{'median':statistics.median(t[metric] for t in trials if t['case']==name),
                              'all':[t[metric] for t in trials if t['case']==name]}
                     for metric in ('us_per_row','kcycles_per_row')} for name in packages}
    ratios={metric:summaries['combo_i16'][metric]['median']/summaries['public_q16x_reference'][metric]['median']
            for metric in ('us_per_row','kcycles_per_row')}
    output.write_text(json.dumps(dict(packages=hashes,trials=trials,summaries=summaries,ratios=ratios,
        affinity=sorted(os.sched_getaffinity(0)),rows_per_trial=60000,
        scope='Measured local complete callback cost; public extrapolation is separate and uncertain'),indent=2)+'\n')


if __name__=='__main__':
    main()
