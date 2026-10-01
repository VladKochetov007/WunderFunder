"""Post-run descriptive receipt audit only; reads no raw or prediction arrays."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import math
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def main():
    worker=load(HERE/'execution/worker-result.json')
    launch=load(HERE/'execution/launch-result.json')
    outer=load(HERE/'outer-result.json')
    plan=load(HERE/'plan.json')
    rows=[json.loads(line) for line in (HERE/'execution/sequences.jsonl').read_text().splitlines()]
    bindings={'plan':sha(HERE/'plan.json'),'allocation':sha(HERE/'allocation.json'),'worker_result':sha(HERE/'execution/worker-result.json'),'launch_result':sha(HERE/'execution/launch-result.json'),'outer_result':sha(HERE/'outer-result.json'),'sequences':sha(HERE/'execution/sequences.jsonl'),'summary_source':sha(__file__)}
    assert launch['worker_result_sha256']==bindings['worker_result']
    assert launch['log_sha256']==sha(HERE/'execution/worker.log')
    assert outer['log_sha256']==sha(HERE/'outer-launch.log')
    assert all(x['plan_sha256']==bindings['plan'] for x in [launch,worker,outer])
    assert all(x['allocation_sha256']==bindings['allocation'] for x in [launch,worker,outer])
    callbacks=sum(r['callbacks'] for r in rows);emitted=sum(r['emitted'] for r in rows)
    assert callbacks==worker['callback_count'] and emitted==worker['emitted_count']
    assert len(rows)==worker['completed_sequences']
    assert [r['row_group'] for r in rows]==(list(range(1873))+list(range(97)))[:len(rows)]
    assert len({r['seq_ix'] for r in rows})==len(rows)
    for i,r in enumerate(rows):
        assert r['schedule_index']==i and r['callbacks']==20000 and r['emitted']==19901
        assert r['governor']['rows_before']==i*20000 and r['governor']['rows_after']==(i+1)*20000
        assert r['governor']['govern_calls_after']==i
        assert 1<=r['governor']['k_after_sequence']==r['governor']['k_after_first_callback']<=r['governor']['k_before_boundary']<=4
        assert r['governor']['active_members']==plan['artifact']['members'][:r['governor']['k_after_sequence']]
        if i:assert rows[i-1]['governor']['k_after_sequence']==r['governor']['k_before_boundary']
    totals={name:sum(r[key] for r in rows) for name,key in [('callback_wall','callback_loop_wall_seconds'),('callback_cpu','callback_loop_cpu_seconds'),('io_decode','io_decode_conversion_seconds'),('validation_checksum','validation_checksum_seconds')]}
    activity={}
    for cpu in ['cpu18','cpu58']:
        total=sum(r['callback_activity'][cpu]['total_ticks'] for r in rows)
        busy=sum(r['callback_activity'][cpu]['busy_ticks'] for r in rows)
        activity[cpu]={'total_ticks':total,'busy_ticks':busy,'busy_fraction':busy/total if total else 0.0}
    if worker['status']=='COMPLETE':
        for name,key in [('callback_wall','callback_loop_wall_seconds'),('callback_cpu','callback_loop_cpu_seconds'),('io_decode','io_decode_conversion_seconds'),('validation_checksum','validation_checksum_seconds')]:assert math.isclose(totals[name],worker[key],rel_tol=0,abs_tol=1e-7)
        assert activity==worker['callback_activity']
        assert callbacks==39400000 and emitted==39204970 and len(rows)==1970
        assert worker['warmup_none_count']==195030 and worker['repeated_raw_rows']==1940000
        assert worker['input_identity_before']==worker['input_identity_after']
        assert worker['input_identity_before']['sha256']==plan['input']['sha256']
    count_by_k=collections.Counter(r['governor']['k_after_sequence'] for r in rows)
    wall_by_k=collections.defaultdict(float)
    for r in rows:wall_by_k[r['governor']['k_after_sequence']]+=r['callback_loop_wall_seconds']
    transitions=[{'schedule_index':r['schedule_index'],'model_rows_before':r['governor']['rows_before'],'from':r['governor']['k_before_boundary'],'to':r['governor']['k_after_sequence'],'sample_utc':r['sample_utc']} for r in rows if r['governor']['k_before_boundary']!=r['governor']['k_after_sequence']]
    status_samples=[s for r in rows for s in [r['status_before'],r['status_after']]]+[worker[k] for k in ['process_status_after_dependencies','final_process_status','postguard_process_status'] if k in worker]
    peak=max([worker.get('max_rss_KiB',0),worker.get('max_observed_rss_KiB',0),*[s.get('vm_rss_KiB',0) for s in status_samples],*[s.get('vm_hwm_KiB',0) for s in status_samples]])
    gates={'functional':launch['functional_complete'],'callback_sibling_quiet':worker.get('callback_smt_clean',False),'whole_trace_sibling_quiet':worker.get('whole_trace_smt_clean',False),'actual_full_process_at_most_60min':launch['actual_full_process_at_most_60min'],'observed_hard_deadline':launch['process_deadline_observed'],'no_timeout':not launch['hard_timeout']}
    qualified=all(gates.values());assert qualified==launch['delivery_runtime_qualified']
    result={'status':'COMPLETE_DESCRIPTIVE_AUDIT' if worker['status']=='COMPLETE' else 'INCOMPLETE_PRESERVED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bindings':bindings,'artifact_sha256':plan['artifact']['sha256'],'callback_count':callbacks,'emitted_count':emitted,'complete_sequences':len(rows),'natural_ISA':worker.get('natural_native_level'),'governor_transitions':transitions,'sequence_counts_by_active_k':dict(count_by_k),'callback_seconds_by_active_k':dict(wall_by_k),'total_seconds':totals,'worker_total_seconds':worker['worker_total_seconds'],'controller_full_process_seconds':launch['process_wall_seconds'],'outer_including_systemd_seconds':outer['outer_wall_seconds'],'outer_also_at_most_60min':outer['outer_wall_seconds']<=3600,'callback_activity':activity,'whole_trace_activity':worker.get('whole_trace_activity'),'max_observed_rss_KiB':peak,'max_observed_OS_threads':max(s['threads'] for s in status_samples),'gates':gates,'delivery_runtime_qualified':qualified,'targets_decoded':False,'prediction_arrays_read':False,'WP_queries':0,'limitations':['Unchanged historical primary ZIP only; no new artifact/composition selection.','Governor trace is this machine/harness policy, not historical public trace or quality evidence.','1970sequences include97 repeated raw sequences with fresh callback IDs.','Warm-cache natural-ISA single attempt; no retry or alternative-hardware guarantee.']}
    (HERE/'outcome-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='evidence-manifest.json' and '__pycache__' not in p.parts}
    (HERE/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
