from pathlib import Path
import json
import shutil
import hashlib
import zipfile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave20'
tag='distilled_h256_pair';dst=D/tag;assert not dst.exists();dst.mkdir()
src=R/'research/iterations/2026-10-05-wave2/blend_twostream/solution.py'
s=src.read_text();s=s.replace('Native GRU pair (dynamic int8) + organizer baseline member in fp32 (exact float path; its inputs carry a +16 shift).','Heterogeneous-ensemble distillation: two h256 causal GRUs, final all-data seeds 5 and 6, dynamic int8.')
(dst/'solution.py').write_text(s)
for level in ('v2','avx2','v3'):shutil.copytree(R/'research/iterations/2026-10-05-wave16/native-sparse/build'/level,dst/level)
members=[]
for seed in (5,6):
 name=f'weights_s{seed}.npz';shutil.copy2(D/f'student_final_s{seed}.npz',dst/name)
 members.append({'weights':name,'coef':.5,'name':f'w18final_distilled_s{seed}@ep1'})
(dst/'members.json').write_text(json.dumps(members,indent=2))
manifest={'candidate':tag,'training':'Final all-data heterogeneous distillation; fixed epoch1; seed5/6 parents. No final-training scores treated as holdout.','validation':'Fold0 seed3/4 pair .71815029; native int8 .71806187 on 425 sequences. Independent train-half0 composite replication: valid delta +.00174293 and train diagnostic +.00168444 vs respective teacher; bootstrap conditional on fitted models.','status':'built_pending_actual_zip_audit_not_submitted','files':{str(f.relative_to(dst)):hashlib.sha256(f.read_bytes()).hexdigest() for f in dst.rglob('*') if f.is_file()}}
(D/'candidate-manifest.json').write_text(json.dumps(manifest,indent=2))
archive=D/(tag+'.zip');assert not archive.exists()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(dst.rglob('*')):
  if f.is_file():z.write(f,f.relative_to(dst))
assert archive.stat().st_size<20_000_000
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}),flush=True)
