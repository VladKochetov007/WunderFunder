"""Allocated runtime preflight only; opens fixed package/old raw fixture after plan review."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from memory import fixed_projection,metadata

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args()
    plan=json.loads(a.plan.read_text());assert plan['id']=='CTXMEM01-native-cost-preflight-v1'
    for name,expected in plan['files_sha256'].items():assert sha(HERE/name)==expected,name
    source=ROOT/plan['baseline_zip'];raw=ROOT/plan['raw_fixture']
    assert sha(source)==plan['baseline_zip_sha256'];assert sha(raw)==plan['raw_fixture_sha256']
    native=HERE/plan['native_module'];assert sha(native)==plan['native_module_sha256']
    out=HERE/'runtime-prototype';out.mkdir(exist_ok=False)
    base=out/'base';base.mkdir()
    with zipfile.ZipFile(source) as archive:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist())
        archive.extractall(base)
    members=json.loads((base/'members.json').read_text())
    assert len(members)==3 and all(m['coef']==1/3 for m in members)
    rng=np.random.default_rng(20261009)
    extra=[rng.normal(0,.03,(128,32)).astype(np.float32) for _ in range(2)]
    identities=[]
    for condition in ('memory','zero'):
        dest=out/condition;shutil.copytree(base,dest)
        shutil.copy2(native,dest/'v3'/native.name)
        config=metadata('memory')
        own_members=[dict(m) for m in members]
        for i in range(2):
            original={k:v for k,v in np.load(base/members[i]['weights'],allow_pickle=False).items()}
            assert original['inp.weight'].shape==(128,211)
            expanded={k:v.copy() for k,v in original.items()}
            expanded['ctx_projection']=fixed_projection()
            expanded['head_extra.weight']=extra[i] if condition=='memory' else np.zeros((128,32),np.float32)
            np.savez(dest/members[i]['weights'],**expanded)
            reloaded=dict(np.load(dest/members[i]['weights'],allow_pickle=False))
            assert all(np.array_equal(original[k],reloaded[k]) for k in original)
            own_members[i]['feature_config']=f'ctx_{i}.json'
            (dest/f'ctx_{i}.json').write_text(json.dumps(config,indent=2)+'\n')
            identities.append({'condition':condition,'member':i,'all_original_arrays_bitexact':True,
                               'added_keys':['ctx_projection','head_extra.weight'],
                               'extra_weight_sha256':hashlib.sha256(reloaded['head_extra.weight'].tobytes()).hexdigest()})
        assert sha(base/members[2]['weights'])==sha(dest/members[2]['weights'])
        (dest/'members.json').write_text(json.dumps(own_members,indent=2)+'\n')
        assert (dest/'solution.py').read_bytes()==(base/'solution.py').read_bytes()
    manifests={condition:{str(path.relative_to(out/condition)):sha(path)
                           for path in sorted((out/condition).rglob('*')) if path.is_file()}
               for condition in ('base','memory','zero')}
    receipt={'status':'PASS','plan_sha256':sha(a.plan),'baseline_zip_sha256':sha(source),'raw_fixture_sha256':sha(raw),
             'native_module_sha256':sha(native),'fixed_composition':'plain128s3,plain128s4,c500; equal1/3',
             'extra_weights':'PCG64 seed20261009; Gaussian sd.03; two nonzero128x32 matrices; no tuning',
             'identities':identities,'package_files_sha256':manifests,
             'new_datasets_read':False,'targets_read':False,'mask_read':False,'WP_queries':0,
             'script_sha256':sha(Path(__file__))}
    (out/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':'PASS','receipt_sha256':sha(out/'preparation.json')}))


if __name__=='__main__':main()
