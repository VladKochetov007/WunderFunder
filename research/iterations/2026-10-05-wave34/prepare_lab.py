from pathlib import Path
import zipfile,json,shutil,hashlib
import numpy as np
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave34';dst=D/'class_oof_lab';dst.mkdir()
shutil.copy2(R/'research/iterations/2026-10-05-wave21/distilled_twostream_h256_pair/solution.py',dst/'solution.py')
for level in ('v2','avx2','v3'):shutil.copytree(R/'research/iterations/2026-10-05-wave26/native-class/build'/level,dst/level)
members=[]
for seed in (3,4):
 name=f'weights_s{seed}.npz';shutil.copy2(R/f'research/iterations/2026-10-05-wave26/native_soft_s{seed}.npz',dst/name)
 members.append({'weights':name,'coef':.5,'name':f'w23_soft_s{seed}@ep1 LAB_ONLY_OOF'})
(dst/'members.json').write_text(json.dumps(members))
archive=D/'class_oof_lab.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(dst.rglob('*')):
  if f.is_file():z.write(f,f.relative_to(dst))
x=np.load(R/'research/iterations/2026-10-05-wave22/wide-probe.npz')['x']
p=[np.load(R/f'research/iterations/2026-10-05-wave26/native_soft_s{s}.probe.npz')['reference'] for s in (3,4)]
np.savez_compressed(D/'probe.npz',x=x,reference=.5*(p[0]+p[1]))
(D/'lab-manifest.json').write_text(json.dumps({'scope':'OOF pair native feasibility, NOT final submission candidate','bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))
