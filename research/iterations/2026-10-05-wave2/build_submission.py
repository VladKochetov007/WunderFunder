"""Replace raw pair with full-data two-stream, preserving the other public incumbent components."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

R=Path.home()/'WunderFunder-creative'
D=R/'research/iterations/2026-10-05-wave2'
def main():
    lite='--lite' in sys.argv
    source=R/'research/iterations/2026-10-02-big-ensemble/run_vlad_pair_bw'
    tag='blend_twostream_lite' if lite else 'blend_twostream'
    dst=D/tag
    assert not dst.exists(),dst
    dst.mkdir()
    for name in ('solution.py','v2','v3','avx2'):
        src=source/name
        if src.is_dir():shutil.copytree(src,dst/name)
        else:shutil.copy2(src,dst/name)
    members=json.loads((source/'members.json').read_text())
    kept=[]
    for m in members:
        if m['name'].startswith('raw:'):continue
        if lite and m['name'].startswith('run:ffull_'):continue
        if lite and m['name'].startswith('vlad:'):m['coef']=.5
        shutil.copy2(source/m['weights'],dst/m['weights'])
        kept.append(m)
    shutil.copy2(D/'tstream_final.npz',dst/'weights_twostream.npz')
    kept.insert(-1,{'weights':'weights_twostream.npz','coef':.25,'name':'twostream:w2fts_s3ft'})
    (dst/'members.json').write_text(json.dumps(kept,indent=2))
    manifest={'parent':'blend_rw.zip (recorded public .6664)' if lite else 'blend_pair_rw.zip (recorded public .6666)',
              'change':'replace raw-wMSE pair total coefficient .25 with single full-data twostream seed3 coefficient .25',
              'files':{str(f.relative_to(dst)):hashlib.sha256(f.read_bytes()).hexdigest() for f in dst.rglob('*') if f.is_file()},
              'status':'built_pending_runtime_audit_not_submitted'}
    (D/('submission-manifest-lite.json' if lite else 'submission-manifest.json')).write_text(json.dumps(manifest,indent=2))
    archive=D/(tag+'.zip')
    assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(dst.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(dst))
    assert archive.stat().st_size<20_000_000,archive.stat().st_size
    print(json.dumps({'zip':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
