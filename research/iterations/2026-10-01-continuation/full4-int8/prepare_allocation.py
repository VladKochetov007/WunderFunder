"""Bind the root-authorized one-point study after its generated/package smoke."""
from pathlib import Path
import datetime,hashlib,json
P=Path(__file__).resolve().parent;ROOT=P.parents[3];OLD=ROOT/'research/iterations/2026-10-01-overnight'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (P/'allocation.json').exists()
 smoke=json.loads((P/'synthetic-four.json').read_text());assert smoke['status']=='PASS_GENERATED_FOUR_MEMBERS'
 package=json.loads((P/'package/quant4_fixed.manifest.json').read_text());assert package['runtime_policy']=='fixed4'
 assert [Path(c['checkpoint']).parent.name for c in package['checkpoints']]==['find256_s3h2','fin128_s3h2','fin128_s4h2','finc500b_s3h2']
 assert all(c['epoch']==1 for c in package['checkpoints']) and [m['coef']for m in package['members']]==[.25]*4
 assert sha(P/'package/quant4_fixed.zip')==package['zip_sha256']
 base=json.loads((OLD/'v1-runtime-audit/plan.json').read_text())
 sources={str(f.relative_to(ROOT)):sha(f)for f in P.glob('*.py')}
 sources.update({k:sha(ROOT/k)for k in ['src/wnn/data.py','scripts/fold_eval.py']})
 plan={**base,'id':'FULL4-INT8-CONTINUATION-01','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authority':'Root explicit one-point actual allocation:3CPUlanes,0GPU,no retries,no blend tuning; current-host official60min threshold.',
       'source_sha256':sources,'artifact':{'path':str((P/'package/quant4_fixed.zip').relative_to(ROOT)),'sha256':package['zip_sha256'],'bytes':package['zip_bytes'],'files':{k:{**v,'is_dir':False}for k,v in package['files'].items()},'members':['find256_s3h2','fin128_s3h2','fin128_s4h2','finc500b_s3h2'],'member_spec':package['members'],'coefficients':[.25]*4,'governor':False,'ISA':'natural deterministic dispatch; VNNI on this measured host'},'runtime_cpu':39,'runtime_sibling':79,'hard_wall_seconds':4200,'official_full_process_seconds':3600,'quiet_sibling_diagnostic_only':True}
 # Remove old descriptive resource/timing fields that described a different artifact.
 for key in ['timing','resources','run','gates','notes','allocation_requirement','execution_policy']:
  plan.pop(key,None)
 plan['execution_authorized']=False
 (P/'runtime-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 allocation={'execution_authorized':True,'root_authority':'Explicit parent-agent allocation received2026-10-01:finish onefull4INT8 point through actualquality/runtime, no further gate required.',
  'full4_runtime_audit_authorized':True,'Q4_heldfold_predictions_authorized':True,'Q4_quality_evaluation_authorized':True,'runtime_gate':'ALLOCATED_SEPARATE_FINAL_RUNTIME','plan_sha256':sha(P/'runtime-plan.json'),'zip_sha256':package['zip_sha256'],'attempts':1,'runtime_cpu':39,'runtime_sibling':79,'OOF_lanes':{'0':30,'1':35},'runtime_host_bytes':16*(1<<30),'OOF_host_bytes_per_lane':4*(1<<30),'cpu_quota_each':'100%','swap_bytes':0,'runtime_wall_seconds':4200,'OOF_lane_wall_seconds':5400,'GPU_jobs':0,'max_disk_bytes':4*(1<<30),'source_sha256':sources,'quality':'onefixednativeINT8 versus same4nativeFP16 onpurefolds0/1; frozenfold_eval math; no weight/epoch/seed search','runtime':'actual39.4Mcallbacks,1970freshsequenceIDs, wholeprocess<=3600 evaluation; SMTreportednotveto','old_evidence_immutable':True}
 (P/'allocation.json').write_text(json.dumps(allocation,indent=2)+'\n');print(json.dumps({'allocation_sha256':sha(P/'allocation.json'),'runtime_plan_sha256':sha(P/'runtime-plan.json'),'zip_sha256':package['zip_sha256']}))
if __name__=='__main__':main()
