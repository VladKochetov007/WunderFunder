import sys,json,tempfile,pathlib,hashlib
repo=pathlib.Path('/Users/nb/Documents/wunder/WunderFunder-maintenance')
op=repo/'research/campaign_2026-10-18/quality_gap_r1/export_preparation/preflight_execution_r2'
sys.path.insert(0,str(op))
import stage
from bindings import *
from local_common import *
record=get_record_dir('preflight-r2-20261018t1520z')
binding=read_json(record/'STAGE_BINDING_R2.json')
assert read_json(record/'STAGE_FAILURE.json')['phase']=='small_bundle_transfer'
assert not (record/'STAGE.json').exists()
pins=verify_local_prerequisites()
check="import pathlib,json; p=pathlib.Path("+repr(str(B_PROJECT_ROOT))+"); assert p.is_dir() and not p.is_symlink(); assert not list(p.iterdir()); print(json.dumps({'root':str(p),'empty':True}))"
r=run_ssh('beleriand',[str(B_PYTHON),'-B','-c',check],timeout=30)
assert r.returncode==0,r.stderr
write_bytes_exclusive(record/'EMPTY_ROOT_RECOVERY_OBSERVATION.raw',r.stdout)
with tempfile.TemporaryDirectory(prefix='source-stage-recovery-') as td:
    bundle=pathlib.Path(td)/'sources.tar.gz'
    size=stage._build_local_bundle(bundle,binding)
    command=['tar','--extract','--gzip','--no-same-owner','--keep-old-files','--file=-',f'--directory={B_PROJECT_ROOT}']
    write_json_exclusive(record/'TRANSPORT_RECOVERY_START.json',{'created_utc':utc_now(),'diagnosis':'gzip stdin without decompression flag; tar extracted no files; exact root independently observed empty','preserved_failure_sha256':sha256_file(record/'STAGE_FAILURE.json'),'recovery_script_sha256':sha256_file(pathlib.Path(__file__)),'bundle_bytes':size,'bundle_sha256':sha256_file(bundle),'command':command,'overwrite':False})
    r=run_ssh('beleriand',command,timeout=120,input_bytes=bundle.read_bytes())
    write_bytes_exclusive(record/'TRANSPORT_RECOVERY_STDERR.raw',r.stderr)
    assert r.returncode==0,r.stderr
helper=B_PROJECT_ROOT/EXECUTION_REL/'remote_stage.py'
verify_remote_file_sha256('beleriand',helper,binding['remote']['stage_helper']['sha256'])
r=run_ssh('beleriand',[str(B_PYTHON),'-B',str(helper),'--mode','stage','--attempt-id',binding['attempt_id'],'--root',str(B_PROJECT_ROOT)],timeout=180)
write_bytes_exclusive(record/'STAGE_RECOVERY_STDOUT.raw',r.stdout)
write_bytes_exclusive(record/'STAGE_RECOVERY_STDERR.raw',r.stderr)
assert r.returncode==0,r.stderr
rr=parse_remote_json(r.stdout)
assert rr['status']=='STAGED'
write_bytes_exclusive(record/'STAGE_REMOTE_RECEIPT.json',r.stdout)
write_json_exclusive(record/'STAGE.json',{'schema':'quality_gap.preflight_stage_execution.r2','status':'STAGED','attempt_id':binding['attempt_id'],'created_utc':utc_now(),'ssh_alias':'beleriand','remote_project_root':str(B_PROJECT_ROOT),'local_verified_pins':pins,'bundle_bytes':size,'remote_stage_receipt_sha256':sha256_file(record/'STAGE_REMOTE_RECEIPT.json'),'transport_recovery':'TRANSPORT_RECOVERY_START.json','prior_failure_preserved':True})
print(record)
