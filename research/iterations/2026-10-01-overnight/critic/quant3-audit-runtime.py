"""Independent arithmetic/provenance audit of completed receipts only; no arrays."""
from pathlib import Path
import hashlib,json,math
HERE=Path(__file__).resolve().parent;P=HERE.parent/'quant3';D=HERE/'quant3-runtime-audit-input'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
f=read(P/'FREEZE.json');a=read(P/'runtime-allocation.json');summary=read(D/'summary.json');reference=read(D/'reference/receipt.json')
assert sha(P/'FREEZE.json')==a['freeze_sha256']==summary['freeze_sha256']
assert sha(P/'runtime-allocation.json')==summary['allocation_sha256']=='3aea8e74620e07b10f920174bc139a959246cdff48c40b3f511cb45a9949517c'
assert sha(HERE/'quant3-preparation-review.json')==a['critic_review_sha256']
assert summary['zip_sha256']==a['zip_sha256']==f['zip_sha256']
assert summary['status']=='VNNI_RUNTIME_PASS_NO_QUALITY_READOUT' and summary['seconds']<=900
assert summary['forecasting_fits']==summary['WP_queries']==0 and not summary['targets_read']
ledger=[json.loads(line) for line in (D/'execution-ledger.jsonl').read_text().splitlines()]
started=[r for r in ledger if r['event']=='started'];finished=[r for r in ledger if r['event']=='finished']
assert [r['name'] for r in started]==[r['name'] for r in finished]
names=['reference','correctness-v3','correctness-avx2','correctness-v2','vnni-100000-r0-attempt0','vnni-100000-r1-attempt0','vnni-100000-r2-attempt0','vnni-1000000-r0-attempt0','avx2-100000-r0-attempt0']
assert [r['name'] for r in started]==names
for r in finished:assert r['returncode']==0 and sha(D/(r['name']+'.log'))==r['log_sha256']
for r in ledger:
    if r['event']=='runtime_result':assert sha(D/(r['name']+'.json'))==r['receipt_sha256']
assert ledger[-1]['summary_sha256']==sha(D/'summary.json')
assert sha(D/'driver.py')==summary['driver_sha256']==ledger[0]['driver_sha256']
prefix='research/iterations/2026-10-01-overnight/quant3/'
assert reference['script_sha256']==f['source_and_binaries'][prefix+'make_runtime_reference.py']
assert reference['raw_sha256']==f['empirical_fixture']['sha256'] and reference['original_zip_sha256']=='ea494921497d9f3cf53b19b1b68285133983560a249b38ef40738b3d846d319d'
assert reference['native_sha256']=='4cb8a76f9b3010b506515ee2012e3088ab2f066edbddf0b6f63a7101bba4c84f'
correctness={}
for level in ['v3','avx2','v2']:
    r=read(D/f'correctness-{level}.json');assert r['status']=='PASS'
    assert r['zip_sha256']==f['zip_sha256'] and r['raw_sha256']==f['empirical_fixture']['sha256'] and r['reference_sha256']==reference['quant']['sha256']
    assert r['native_sha256']==f['source_and_binaries'][prefix+f'native/build/{level}/wnn_cpp.cpython-311-x86_64-linux-gnu.so']
    assert len(set(r['within_engine_backend_hashes'].values()))==1 and r['reset_prefix_stream_masks'] and r['torch_absent']
    assert r['rows']==(100000 if level=='v3' else 2560)
    if level=='v3':assert r['reference_bitexact']
    for key in ['reference_errors','fp16_same_composition_errors']:
        for e in r.get(key) or []:
            assert e['relative_rmse']<=.03 and e['p99_abs']<=.05 and e['max_abs']<=.5
            assert math.isclose(e['relative_rmse'],e['rmse']/max(e['reference_population_std'],1e-8),abs_tol=1e-14)
    correctness[level]={'rows':r['rows'],'reference_bitexact':r['reference_bitexact'],'receipt_sha256':sha(D/f'correctness-{level}.json')}
timings=[]
for name in names[4:]:
    r=read(D/(name+'.json'));seq=r['per_sequence'];assert len(seq)==r['rows']//20000==r['sequence_count']
    assert [s['index'] for s in seq]==list(range(len(seq))) and all(s['rows']==20000 for s in seq)
    wall=sum(s['wall_seconds'] for s in seq);cpu=sum(s['cpu_seconds'] for s in seq);projection=wall/r['rows']*39400000/60
    assert math.isclose(wall,r['wall_seconds'],abs_tol=1e-12) and math.isclose(cpu,r['process_cpu_seconds'],abs_tol=1e-12)
    assert math.isclose(projection,r['projected_minutes_39_4M'],abs_tol=1e-12) and math.isclose(cpu/wall,r['cpu_to_wall_ratio'],abs_tol=1e-12)
    for core in ['cpu18','cpu58']:
        total=sum(s['activity'][core]['total_ticks'] for s in seq);busy=sum(s['activity'][core]['busy_ticks'] for s in seq);v=r['aggregate_cpu_activity'][core]
        assert total==v['total_ticks'] and busy==v['busy_ticks'] and math.isclose(busy/total,v['busy_fraction'],abs_tol=1e-15)
    assert r['precheck_activity']['cpu58']['busy_fraction']<.05 and r['aggregate_cpu_activity']['cpu58']['busy_fraction']<.05
    assert r['threads']==1 and r['affinity']==[18] and r['max_rss_KiB']<4*2**20
    assert r['raw_sha256']==reference['raw_sha256'] and r['reference_sha256']==reference['quant']['sha256'] and r['zip_sha256']==f['zip_sha256']
    assert r['solution_sha256']==f['source_and_binaries'][prefix+'solution_quant3.py'] and r['harness_sha256']==f['source_and_binaries'][prefix+'benchmark_zip.py']
    level='avx2' if name.startswith('avx2') else 'v3'
    assert r['native_sha256']==f['source_and_binaries'][prefix+f'native/build/{level}/wnn_cpp.cpython-311-x86_64-linux-gnu.so']
    assert r['cpu_level']==level and r['effective_integer_backend_code']==(1 if level=='avx2' else 2)
    assert r['prediction_sha256']==r['reference_prediction_sha256'] and r['frozen_v3_quant_reference_bitexact']
    assert r['torch_runtime_absent'] and not r['targets_read'] and r['WP_queries']==0
    assert r['runtime_gate_pass']==(projection<=50) and r['status']==('PASS' if projection<=50 else 'FAIL_RUNTIME')
    timings.append({'name':name,'projected_minutes':projection,'us_per_row':wall/r['rows']*1e6,'SMT_busy_fraction':r['aggregate_cpu_activity']['cpu58']['busy_fraction'],'max_RSS_KiB':r['max_rss_KiB'],'receipt_sha256':sha(D/(name+'.json'))})
assert all(x['projected_minutes']<=50 for x in timings[:4]) and timings[4]['projected_minutes']>50
assert len(summary['vnni'])==4 and len(summary['avx2'])==1 and summary['avx2_runtime_gate']=='UNQUALIFIED'
result={'status':'VNNI_RUNTIME_AND_CORRECTNESS_PASS','allocation_sha256':sha(P/'runtime-allocation.json'),'freeze_sha256':sha(P/'FREEZE.json'),'summary_sha256':sha(D/'summary.json'),'ledger_sha256':sha(D/'execution-ledger.jsonl'),'reference_receipt_sha256':sha(D/'reference/receipt.json'),'zip_sha256':f['zip_sha256'],'correctness':correctness,'timings':timings,'all_VNNI_repetitions_under_50min':True,'VNNI_max_projected_minutes':max(x['projected_minutes'] for x in timings[:4]),'VNNI_long_projected_minutes':timings[3]['projected_minutes'],'no_contamination_retries_or_extra_runs':True,'AVX2_pilot_failed_and_stopped':True,'OOF_gate_decision':'Eligible for root to separately allocate only frozen VNNI/FP16 same-composition OOF production and the two fixed WP comparisons. This receipt grants no execution allocation.','AVX2_fullfold_audit_gate':'NOT ELIGIBLE: runtime failed; no extra audit/readout','deployment_limit':'VNNI host only. Forced AVX2 is125.18min and fails; natural v3+AVX2 and v2 runtimes are unmeasured. Unknown public CPU prevents universal delivery claim.','trace_limit':'Long trace repeats five frozen20k sequences ten times with fresh sequence IDs; not one continuous1M-state stream.','startup_limit':'Reported import/model time excludes NumPy already imported by harness and process interpreter startup.','prediction_arrays_opened_by_critic':False,'raw_fixture_opened_by_critic':False,'targets_read':False,'forecast_fits':0,'WP_queries':0}
out=HERE/'quant3-runtime-independent-review.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
