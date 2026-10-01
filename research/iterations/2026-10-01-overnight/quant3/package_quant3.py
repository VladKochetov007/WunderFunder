"""Version2: package a declared fixed2/3/4 ensemble into a fresh deterministic ZIP.

Original package_fixed.py is preserved. This tool packages only a supplied composition;
it does not score, select, or train members. Final ZIP verification remains separate.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,tempfile,zipfile
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
MAX_BYTES=20_000_000

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--members',type=Path,required=True);ap.add_argument('--native-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists() or a.output.with_suffix('.manifest.json').exists():raise FileExistsError(a.output)
    specification=json.loads(a.members.read_text());assert len(specification)==3, 'Q3 is exactly three members'
    assert [Path(x['checkpoint']).parent.name for x in specification]==['find256_s3h2','fin128_s3h2','finc500b_s3h2'], 'fixed Q3 identities'
    assert [float(x['coef']) for x in specification]==[1/3]*3
    coefs=[float(x['coef']) for x in specification];assert np.isfinite(coefs).all() and min(coefs)>=0 and abs(sum(coefs)-1)<1e-8
    with tempfile.TemporaryDirectory(prefix='wunder-fixed-package-') as tmp:
        d=Path(tmp);shutil.copy(HERE/'solution_quant3.py',d/'solution.py');members=[];provenance=[]
        for i,member in enumerate(specification):
            path=Path(member['checkpoint']).resolve();ck=torch.load(path,map_location='cpu',weights_only=False)
            assert int(ck['epoch'])==1,'fixed2epoch endpoint required'
            assert not ck['args'].get('lookahead'),'noncausal teacher cannot serve'
            cfg=ck['args']
            assert cfg.get('arch','gru')=='gru' and cfg.get('cell','gru')=='gru' and cfg.get('enc','linear')=='linear','native package requires plain linear GRU'
            assert not any(cfg.get(k) for k in ['class_heads','heuristics','quant_feats','prop_input','atom_head']),'unsupported inference representation'
            assert ck['n_in'] in (211,323),'unsupported native input width'
            arrays={k:v.detach().cpu().float().numpy() for k,v in ck['model'].items()}
            arrays['head.2.weight']=arrays['head.2.weight'][:2].copy();arrays['head.2.bias']=arrays['head.2.bias'][:2].copy()
            assert all(np.isfinite(x).all() for x in arrays.values())
            assert (arrays['std']>0).all(), 'normalization std must be positive'
            weight_name=f'weights_{i}.npz';np.savez(d/weight_name,**arrays)
            m={'weights':weight_name,'coef':float(member['coef'])}
            meta=ck['args'].get('transition_features')
            assert bool(meta)==('head_extra.weight' in arrays),'feature metadata/weight mismatch'
            if meta:
                assert meta['schema']=='transhead01-68-v1' and meta['arm'] in ['F','T']
                m['feature_config']=f'feature_{i}.json';(d/m['feature_config']).write_text(json.dumps(meta,allow_nan=False,indent=2)+'\n')
            members.append(m);provenance.append({'checkpoint':str(path),'sha256':sha(path),'epoch':ck['epoch'],'args':ck['args']})
        (d/'members.json').write_text(json.dumps(members,indent=2)+'\n')
        for level in ['v2','avx2','v3']:
            bins=list((a.native_dir/level).glob('wnn_cpp.cpython-311-*.so'));assert len(bins)==1,(level,bins)
            header=bins[0].read_bytes()[:20]
            assert header[:5]==b'\x7fELF\x02' and int.from_bytes(header[18:20],'little')==62,'native module must be x86_64 ELF'
            (d/level).mkdir();shutil.copy(bins[0],d/level/bins[0].name)
        files={str(p.relative_to(d)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(d.rglob('*')) if p.is_file()}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in files:z.write(d/name,name)
        assert a.output.stat().st_size<=MAX_BYTES
    receipt={'output':str(a.output.resolve()),'zip_sha256':sha(a.output),'zip_bytes':a.output.stat().st_size,'runtime_policy':f'fixed{len(members)}','members':members,'checkpoints':provenance,'files':files,'source_spec_sha256':sha(a.members),'packager_sha256':sha(__file__),'status':'packaged; needsfreshZIPparityandruntimeverification'}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ['output','zip_sha256','zip_bytes','runtime_policy','status']},indent=2))
if __name__=='__main__':main()
