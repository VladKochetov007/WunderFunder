from pathlib import Path
import json,shutil,hashlib,zipfile
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave47';base=R/'research/iterations/2026-10-05-wave20/distilled_h256_pair'
manifest=[]
for mode,tag in [('pair','distilled_short_h256_pair'),('four','distilled_short_h256_four')]:
 dst=D/tag;dst.mkdir();shutil.copy2(base/'solution.py',dst/'solution.py')
 for level in ('v2','avx2','v3'):shutil.copytree(base/level,dst/level)
 members=[]
 for seed in (5,6):
  name=f'weights_s{seed}.npz';shutil.copy2(D/f'student_final_s{seed}.npz',dst/name);members.append({'weights':name,'coef':.5 if mode=='pair' else .25,'name':f'w47final_short_s{seed}@ep1'})
 if mode=='four':
  for seed in (5,6):
   name=f'weights_old_s{seed}.npz';shutil.copy2(base/f'weights_s{seed}.npz',dst/name);members.append({'weights':name,'coef':.25,'name':f'w18final_distilled_s{seed}@ep1'})
 (dst/'members.json').write_text(json.dumps(members,indent=2));archive=D/(tag+'.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in sorted(dst.rglob('*')):
   if f.is_file():z.write(f,f.relative_to(dst))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 assert archive.stat().st_size<20_000_000
 manifest.append({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'members':members,'status':'prepared_pending_native_quality_selection_and_functional_audit','validation':'wave45 fixedep1 two-seed full/composite; final all-data training not holdout','runtime':'No additional timing per user instruction'})
(D/'candidate-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
