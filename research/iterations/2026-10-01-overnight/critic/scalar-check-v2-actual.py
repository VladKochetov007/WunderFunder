"""Target-free completed generated-preflight receipt auditor; no model execution."""
from pathlib import Path
import datetime,hashlib,json,math,statistics
P=Path(__file__).resolve().parent.parent;ROOT=P.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
A=P/'scalar/gpu-v2-allocation.json';a=json.loads(A.read_text())
E=P/'scalar/gpu-v2-execution/execution.json';e=json.loads(E.read_text())
assert a['SCALARPLE_generated_GPU_probe_v2_authorized'] is True and a['competition_inputs_authorized'] is False
assert e['allocation_sha256']==sha(A)
for f,h in a['sources'].items():assert sha(ROOT/f)==h,f
assert e['proposal_sha256']==sha(P/'scalar/gpu-v2-proposal.json')
assert e['freeze_sha256']==sha(P/'scalar/GPU-V2-PREP-FREEZE.json')
assert e['driver_sha256']==sha(P/'scalar/gpu-v2-execution/driver.py')
observations=[];results={}
for job in e['jobs']:
 arm=job['arm'];assert arm in ['control','ple']
 assert job['prelaunch']['free_MiB']>=6144 and job['prelaunch']['processes']==[]
 assert job['identity']['pgrp']==job['identity']['pid']
 assert job['seconds']<=242
 assert job['resource_peaks']['GPU_MiB']<=5632 and job['resource_peaks']['RSS_KiB']*1024<=8*(1<<30)
 samples=[json.loads(line) for line in (P/f'scalar/gpu-v2-execution/{arm}-attempt01-samples.jsonl').read_text().splitlines()]
 assert all(all(v['pid']==job['identity']['pid'] for v in row['NVML']['processes']) for row in samples)
 log=P/f'scalar/gpu-v2-execution/{arm}-attempt01.log'
 logged=[json.loads(line)['diagnostic_step'] for line in log.read_text().splitlines() if line.startswith('{"diagnostic_step":')]
 observations.append({'arm':arm,'exit_code':job['exit_code'],'raw_step_count':len(logged),'log_sha256':sha(log),'samples_sha256':sha(P/f'scalar/gpu-v2-execution/{arm}-attempt01-samples.jsonl')})
 if job['exit_code']!=0:continue
 file=P/f'scalar/gpu-v2-{arm}-attempt01.json';r=json.loads(file.read_text())
 assert sha(file)==job['result_sha256'] and r['source_sha256']==sha(P/'scalar/gpu_probe_v2.py') and r['allocation_sha256']==sha(A)
 assert r['status']=='PASS' and r['measured_steps']==list(range(3,10)) and len(r['steps'])==10 and len(logged)==10
 completed=0
 for i,(s,l) in enumerate(zip(r['steps'],logged)):
  assert s['step']==i and all(s[k]==v for k,v in l.items() if math.isfinite(v) if isinstance(v,(int,float)))
  assert math.isfinite(s['loss']) and s['audit']['PASS']
  skip=s['scale_after']<s['scale_before'];assert skip==s['audit']['skipped_by_scaler']
  if skip:assert i<3 and s['audit']['parameters_unchanged'] and s['audit']['moments_unchanged_or_new_zero']
  else:assert math.isfinite(s['gradient_norm']) and not s['audit']['parameters_unchanged']
  assert s['audit']['completed_updates_before']==completed
  completed+=int(not skip)
  assert s['audit']['completed_updates_after']==completed
  assert s['audit']['step_counter_after_range']==[completed,completed]
 assert all(r['final_state_audit'][key] for key in ['model_finite','EMA_finite','optimizer_finite','all_step_counters_completed_updates'])
 assert r['final_state_audit']['completed_updates']==completed
 assert r['peak_reserved_bytes']<=5.5*(1<<30)
 assert abs(statistics.median(s['wall_seconds'] for s in r['steps'][3:])-r['median_wall_seconds'])<1e-12
 assert abs(statistics.median(s['device_ms'] for s in r['steps'][3:])-r['median_device_ms'])<1e-10
 assert r['scalar_CUDA_oracle']['status']=='PASS' and r['scalar_CUDA_oracle']['literal_boundary_bitexact']
 assert r['initial_AMP_parity']['ordinary_parameter_gradients_exact_numerical'] and r['initial_AMP_parity']['CPU_CUDA_constructor_RNG_exact']
 if arm=='ple':assert all(v==(k in ['scalar.table','scalar.projection']) for k,v in r['scalar_state_changes'].items())
 results[arm]=r
if e['status'] in ['PASS','FAIL_COST_GATE']:
 assert [j['arm'] for j in e['jobs']]==['control','ple'] and set(results)=={'control','ple'}
 estimate=1.25*15000*(2*results['control']['median_wall_seconds']+3*results['ple']['median_wall_seconds'])+5400
 assert abs(estimate-e['cost']['estimate_seconds'])<1e-8
 assert e['cost']['PASS']==(estimate<=e['cost']['remaining_seconds'])
 assert (e['status']=='PASS')==e['cost']['PASS']
else:assert e['status']=='FAIL'
report={'status':'ACTUAL_RECEIPTS_AUDITED','stage_outcome':e['status'],'allocation_sha256':sha(A),'execution_sha256':sha(E),'checker_sha256':sha(__file__),'all_frozen_source_hashes_match':True,'jobs':observations,'cost':e['cost'],'arms':{arm:{'warmup_skip_steps':r['warmup_skip_steps'],'median_wall_seconds':r['median_wall_seconds'],'median_device_ms':r['median_device_ms'],'peak_reserved_bytes':r['peak_reserved_bytes'],'completed_updates':r['final_state_audit']['completed_updates'],'loop_wall_seconds':r['seconds'],'loop_minus_timed_region_seconds':r['seconds']-sum(s['wall_seconds'] for s in r['steps'])} for arm,r in results.items()},'no_forecast_or_runtime_allocation_inferred':True,'empirical_data_reads':0,'reviewer_GPU_runs':0,'forecast_fits':0,'WP_queries':0}
(P/'critic/scalar-v2-actual-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
