from pathlib import Path
import json,shutil,hashlib,zipfile,sys
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave46';tag=sys.argv[1] if len(sys.argv)>1 else 'distilled_h256_four_seeds';dst=D/tag;dst.mkdir()
mixed=tag=='distilled_twostream_seed_four';base=R/('research/iterations/2026-10-05-wave21/distilled_twostream_h256_pair' if mixed else 'research/iterations/2026-10-05-wave20/distilled_h256_pair');shutil.copy2(base/'solution.py',dst/'solution.py')
for level in ('v2','avx2','v3'):shutil.copytree(base/level,dst/level)
members=[]
for seed in (5,6,15,16):
 name=f'weights_s{seed}.npz';src=base/name if seed in (5,6) else D/f'student_final_s{seed}.npz';shutil.copy2(src,dst/name)
 members.append({'weights':name,'coef':.25,'name':f'w{(21 if mixed else 18) if seed in (5,6) else 46}final_distilled_s{seed}@ep1'})
(dst/'members.json').write_text(json.dumps(members,indent=2))
archive=D/(tag+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(dst.rglob('*')):
  if f.is_file():z.write(f,f.relative_to(dst))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
assert archive.stat().st_size<20_000_000
manifest={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'members':members,'status':'prepared_pending_quality_selection_and_functional_audit','training':'Same original distillation, fixed ep1, new student seeds15/16 from parents5/6. Alltraining data+allvalid; no finalmodel holdout claim.','quality_evidence':('Wave48 fixed equal twostreamteacher+newseed blend, native full425/composite436+train282.' if mixed else 'Wave43 full425 and composite436/282 fixed fourmean. Must review before user delivery.'),'runtime':'Per user instruction no further timing benchmark. Separate fourh256 deployment lab in DataSphere ratio1.979 vs publicpair, projection49.38min, not scorer guarantee.'}
(D/(tag+'-manifest.json')).write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
