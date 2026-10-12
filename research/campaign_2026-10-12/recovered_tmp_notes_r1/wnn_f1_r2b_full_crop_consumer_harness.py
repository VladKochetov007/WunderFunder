from __future__ import annotations
import hashlib, importlib.util, json, os, shutil
from pathlib import Path

MAIN = Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
PACKET = Path('/home/vlad/development/wunder/wnn_oldz4_f1_core_hft_20261012/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_f1_core_hft_r2')
TMP = Path('/tmp/wnn_f1_r2b_full_crop_consumer_20261012')
TMP_PACKET = TMP / 'packet'

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

if TMP.exists():
    raise FileExistsError(TMP)
TMP.mkdir(parents=True)
shutil.copytree(PACKET, TMP_PACKET)

run_name = 'oldz4_h230_student_f1_s3_r4'
source = TMP / 'artifacts/runs' / run_name
source.mkdir(parents=True)
actual_source = MAIN / 'artifacts/runs' / run_name
for name in ('model_ep1.pt', 'provenance.json', 'training_state.pt'):
    (source / name).symlink_to(actual_source / name)

runner_path = TMP_PACKET / 'run_f1_core_hft_stage_r2.py'
runner = import_file(runner_path, 'r2b_full_crop_consumer_fixture')
os.environ.update({
    'WNN_LOADER_WORKERS':'0','WNN_THREADS':'1','OMP_NUM_THREADS':'1',
    'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONHASHSEED':'0',
    'PYTHONDONTWRITEBYTECODE':'1'
})
runner._require_environment()
fit_contract = runner.read_object(runner.R4_STUDENT_FIT)
cp = Path(fit_contract['expected_output'])
prov = cp.parent / 'provenance.json'
state = cp.parent / 'training_state.pt'
fit_sha = sha(runner.R4_STUDENT_FIT)
cp_sha, prov_sha, state_sha = sha(cp), sha(prov), sha(state)
student_gate = {
    'schema':'wnn.root.f1-student-cp1-terminal-gate.r1','status':'pass',
    'fixture_only':True,'run':fit_contract['run'],'epoch':1,
    'fit_contract_sha256':fit_sha,'checkpoint_sha256':cp_sha,
    'provenance_sha256':prov_sha,'training_state_sha256':state_sha
}
gate_path = TMP_PACKET / 'ROOT_F1_STUDENT_CP1_TERMINAL_GATE_R2.json'
gate_path.write_text(json.dumps(student_gate,sort_keys=True,indent=2)+'\n')
contract_path = TMP_PACKET / 'CORE_CROP_CONTRACT_R2.json'
contract_sha = sha(contract_path)
admission = {
    'schema':'wnn.f1-core-crop-admission.r1','decision':'PASS',
    'fixture_only':True,'crop_contract_sha256':contract_sha,
    'student_terminal_gate_sha256':sha(gate_path),
    'source_checkpoint_sha256':cp_sha,'source_provenance_sha256':prov_sha
}
(TMP_PACKET/'ROOT_F1_CORE_CROP_ADMISSION_R2.json').write_text(json.dumps(admission,sort_keys=True,indent=2)+'\n')

result = runner._run_crop(TMP)
receipt_path = TMP_PACKET / 'CORE_CROP_PRODUCER_RECEIPT_R2.json'
receipt = runner.read_object(receipt_path)
out_cp = TMP / receipt['output_checkpoint_relative_path']
out_prov = TMP / receipt['output_provenance_relative_path']
if not out_cp.is_file() or not out_prov.is_file():
    raise RuntimeError('crop consumer did not produce checkpoint and provenance in temp project')
if out_cp.stat().st_size > runner.read_object(contract_path)['maximum_checkpoint_bytes']:
    raise RuntimeError('crop output exceeded contract bound')
summary = {
 'schema':'wnn.oldz4-f1-full-crop-consumer-fixture.r2b',
 'status':'PASS_FULL_CROP_CONSUMER_TEMPORARY_PROJECT',
 'fixture_only':True,'root_authorization_used':False,'main_artifacts_written':False,
 'device_started':False,'production_crop_admission':False,
 'invocation':'Copied packet runner exact _run_crop(project_root=temporary project); _require_environment enforced. Outer run_stage MAIN-root identity guard intentionally not invoked because outputs must remain temporary.',
 'temporary_project_root':str(TMP),'temporary_packet_root':str(TMP_PACKET),
 'actual_source_run':run_name,'source_checkpoint_sha256':cp_sha,
 'source_checkpoint_bytes':cp.stat().st_size,'source_provenance_sha256':prov_sha,
 'source_training_state_sha256':state_sha,
 'crop_contract_sha256':contract_sha,
 'contract_maximum_checkpoint_bytes':runner.read_object(contract_path)['maximum_checkpoint_bytes'],
 'fixture_student_gate_sha256':sha(gate_path),
 'fixture_crop_admission_sha256':sha(TMP_PACKET/'ROOT_F1_CORE_CROP_ADMISSION_R2.json'),
 'consumer_result':result,'consumer_receipt_sha256':sha(receipt_path),
 'output_checkpoint_sha256':sha(out_cp),'output_checkpoint_bytes':out_cp.stat().st_size,
 'output_provenance_sha256':sha(out_prov),
 'producer_verification':receipt['verification'],
 'actual_membership':receipt['source_qualified_membership']
}
print(json.dumps(summary,sort_keys=True))
