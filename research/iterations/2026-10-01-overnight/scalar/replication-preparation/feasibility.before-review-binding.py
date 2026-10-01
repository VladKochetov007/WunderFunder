"""Prospective remaining-cost constraints, using completed duration receipts only."""
from pathlib import Path
import datetime,hashlib,json
UTC=datetime.timezone.utc

def calculation(records,launch,control_median,ple_median):
 jobs={r['label']:r for r in records};assert len(jobs)==len(records) and all(r['exit_code']==0 and r['failure'] is None for r in records)
 def wall(name):
  x=float(jobs[name]['seconds']);assert x>0;return x
 def maximum(pattern):return max(wall(pattern.format(f)) for f in [0,1])
 c=maximum('on_plectl_s3_f{}_parent');p=maximum('on_ple_s3_f{}_parent');ch=maximum('on_plectl_s3_f{}');before=wall('inputs-before');after=wall('inputs-after')
 cc=maximum('on_plectl_s3_f{}_parent-check');hc=maximum('on_plectl_s3_f{}-check')
 fixed=1.25*15000*(control_median+2*ple_median)+5400
 actual=1.25*(c+2*p)+5400
 ple_start_delay=1.25*(before+c+ch+cc+hc)
 paths=[]
 for fold in [0,1]:
  names=[f'on_plectl_s3_f{fold}_parent',f'on_plectl_s3_f{fold}',f'on_ple_s3_f{fold}_parent',f'on_ple_s3_f{fold}']
  paths.append(sum(wall(n)+wall(n+'-check') for n in names)+sum(wall(n+'-OOF') for n in [f'on_plectl_s3_f{fold}',f'on_ple_s3_f{fold}']))
 stage=1.25*(before+max(paths)+after)
 deadline=datetime.datetime(2026,10,1,8,23,30,tzinfo=UTC);parent_cutoff=datetime.datetime(2026,10,1,6,tzinfo=UTC);screen_stop=deadline-datetime.timedelta(seconds=4800)
 latest={'fixed_cost':deadline-datetime.timedelta(seconds=fixed),'actual_cost':deadline-datetime.timedelta(seconds=actual),'all_parents_start':parent_cutoff-datetime.timedelta(seconds=ple_start_delay),'screen_complete':screen_stop-datetime.timedelta(seconds=stage)}
 gates={'fixed_remaining_cost':fixed<=(deadline-launch).total_seconds(),'actual_remaining_parent_cost':actual<=(deadline-launch).total_seconds(),'all_parents_projected_start_before06':launch+datetime.timedelta(seconds=ple_start_delay)<parent_cutoff,'screen_projected_complete_before_reserve':launch+datetime.timedelta(seconds=stage)<screen_stop}
 return {'PASS':all(gates.values()),'gates':gates,'launch_UTC':launch.isoformat(),'fixed_remaining_seconds':fixed,'actual_remaining_seconds':actual,'projected_PLE_parent_delay_seconds':ple_start_delay,'projected_screen_seconds':stage,'latest_start_by_constraint':{k:v.isoformat() for k,v in latest.items()},'latest_admissible_start_UTC':min(latest.values()).isoformat(),'observed_maxima_seconds':{'control_parent':c,'ple_parent':p,'control_head':ch,'control_parent_check':cc,'control_head_check':hc,'inputs_before':before,'inputs_after':after,'full_lane':max(paths)},'formulas':{'fixed':'1.25*15000*(control_median+2*PLE_median)+5400','actual_parent':'1.25*(max_control_parent+2*max_PLE_parent)+5400','late_parent_delay':'1.25*(inputs_before+max_control_parent+max_control_head+max_control_parent_check+max_control_head_check)','screen':'1.25*(inputs_before+max(sum8fit_and_check_jobs+2OOF_per_fold)+inputs_after)'},'uses_quality':False}

def approved_receipts(root,allocation,launch,control_median,ple_median):
 entry=allocation['completed_seed3'];path=root/entry['result_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['result_sha256'];data=json.loads(path.read_text());assert data['status']=='PASS_FIXED_FITS_AND_OOF_PENDING_INDEPENDENT_AUDIT'
 review=root/entry['independent_review_path'];assert hashlib.sha256(review.read_bytes()).hexdigest()==entry['independent_review_sha256'];assert allocation.get('seed3_both_quality_gates_PASS') is True
 result=calculation(data['jobs'],launch,control_median,ple_median);result['completed_seed3']=entry;return result
