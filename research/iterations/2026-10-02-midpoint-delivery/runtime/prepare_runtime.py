"""Create an unallocated runtime plan after packaging/checking. Never authorizes a run."""
import argparse,datetime,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[3];D=P.parent
LEGACY=ROOT/'research/iterations/2026-10-01-continuation/student-runtime'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--cpu',type=int,required=True);ap.add_argument('--smt',type=int,required=True);ap.add_argument('--hard-seconds',type=int,default=3900);a=ap.parse_args()
 assert a.cpu!=a.smt and 3600<=a.hard_seconds<=4200 and not (P/'runtime-plan.json').exists()
 package=a.package.resolve();assert package.is_relative_to(D/'package')
 m=json.loads(package.with_suffix('.manifest.json').read_text());c=json.loads((P/'package-check.json').read_text())
 assert c['status']=='PASS' and c['zip_sha256']==m['zip_sha256']==sha(package)
 members=['oct02_wm_final_s3h2','fin128_s3h2','fin128_s4h2','finc500b_s3h2']
 assert [Path(x['checkpoint']).parent.name for x in m['checkpoints']]==members and [x['coef'] for x in m['members']]==[.25]*4
 inv=json.loads((D/'legacy-bindings.json').read_text());old=LEGACY/'runtime-plan.json'
 assert sha(old)==inv['source_sha256'][str(old.relative_to(ROOT))]
 prior=json.loads(old.read_text());plan={k:prior[k] for k in ['input','scorer_environment','environment_sha256','library_versions']}
 files=[*sorted(P.glob('*.py')),D/'package_student.py',D/'final_midpoint.py',D/'midpoint-binding.json',D/'commands.json',D/'legacy-bindings.json',ROOT/'src/wnn/data.py']
 plan.update(id='MIDPOINT-FINAL-INT8-RUNTIME-01',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_authorized=False,
  source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},
  artifact=dict(path=str(package.relative_to(ROOT)),sha256=m['zip_sha256'],bytes=m['zip_bytes'],files={k:{**v,'is_dir':False} for k,v in m['files'].items()},members=members,member_spec=m['members'],coefficients=[.25]*4,governor=False),
  callbacks=dict(count=39400000,emitted_count=39204970,sequence_resets=1970,warmup_none_count=195030,need_prediction='step>=99, no target/mask reads',schedule='valid0..1872 then0..96 with freshIDs',prediction_arrays_saved=False),
  resources=dict(CPU=a.cpu,SMT_sibling=a.smt,CPU_quota='100%',RAM_bytes=16*(1<<30),swap_bytes=0,hard_wall_seconds=a.hard_seconds,official_full_process_seconds=3600,GPU_jobs=0),
  package_check_sha256=sha(P/'package-check.json'),package_manifest_sha256=sha(package.with_suffix('.manifest.json')),
  limitation='One shared-host naturalv3/VNNI actualrun;97repeatedinputsequences,warm filesystemcache,SMTreportedwithout costsubtraction. ExactnewZIP required. No otherISA/hardware claim or automaticallocation.')
 with (P/'runtime-plan.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
 print(json.dumps(dict(status='UNALLOCATED_RUNTIME_PLAN_READY',plan_sha256=sha(P/'runtime-plan.json'),zip_sha256=m['zip_sha256'],required_allocation_fields=dict(midpoint_runtime_audit_authorized=True,attempts=1,plan_sha256='<above>'))))
if __name__=='__main__':main()
