from pathlib import Path
import json,zipfile,shutil,hashlib
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave44';dst=D/'four_h256_lab';dst.mkdir()
base=R/'research/iterations/2026-10-05-wave20/distilled_h256_pair'
shutil.copy2(base/'solution.py',dst/'solution.py')
for level in ('v2','avx2','v3'):shutil.copytree(base/level,dst/level)
members=[];refs=[];probe_x=None
for wave,tag in [(20,'old'),(21,'new')]:
 src=R/f'research/iterations/2026-10-05-wave{wave}'
 for seed in (5,6):
  name=f'weights_{tag}_s{seed}.npz';shutil.copy2(src/f'student_final_s{seed}.npz',dst/name)
  members.append({'weights':name,'coef':.25,'name':f'wave{wave}_seed{seed}_LAB_ONLY'})
  p=np.load(src/f'student_final_s{seed}.probe.npz');print(wave,seed,p.files,flush=True)
  if probe_x is None:probe_x=p['x']
  else:assert np.array_equal(probe_x,p['x'])
  refs.append(p['reference'])
(dst/'members.json').write_text(json.dumps(members))
archive=D/'four_h256_lab.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(dst.rglob('*')):
  if f.is_file():z.write(f,f.relative_to(dst))
x=np.load(R/'research/iterations/2026-10-05-wave22/wide-probe.npz')['x'];assert np.array_equal(x[:len(probe_x)],probe_x)
np.savez_compressed(D/'probe.npz',x=x,reference=sum(refs)*.25)
assert archive.stat().st_size<20_000_000
(D/'lab-manifest.json').write_text(json.dumps({'scope':'Four-member cost feasibility only, NOT promoted submission candidate','members':members,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))
