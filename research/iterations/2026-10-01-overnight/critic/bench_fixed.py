"""Raw-only fresh-package fixed-members runtime/parity probe; no target or metric access."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib,json,os,shutil,sys,tempfile,time,zipfile
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def prepare(destination):
    import pyarrow.parquet as pq
    assert not destination.exists();destination.mkdir()
    source=Path('/home/grinchenkoeg/WunderFunder-cpp/submissions/candV1_ft2/candV1_ft2.zip')
    with zipfile.ZipFile(source) as z:
        names=z.namelist();assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        z.extractall(destination/'fixed_v1')
    shutil.copy(HERE/'solution_fixed.py',destination/'fixed_v1/solution.py')
    shutil.copytree(destination/'fixed_v1',destination/'fixed_recovered')
    shutil.copytree(destination/'fixed_v1',destination/'fixed_feature')
    for name,base in [('fixed_recovered',HERE/'recovery/native_cpp/build'),('fixed_feature',HERE/'native_transition/build')]:
        for level in ['v2','avx2','v3']:
            for old in (destination/name/level).glob('*.so'):old.unlink()
            binaries=list((base/level).glob('wnn_cpp.cpython-311-*.so'));assert len(binaries)==1,(base,level)
            shutil.copy(binaries[0],destination/name/level/binaries[0].name)
    rng=np.random.default_rng(192736)
    nuisance={'centers':[0.,0.,0.,0.],'pi':rng.dirichlet(np.ones(16)).tolist(),'R':rng.dirichlet(np.ones(16),size=16).tolist()}
    m=json.loads((destination/'fixed_feature/members.json').read_text())
    for idx in [1,2]:
        wpath=destination/'fixed_feature'/m[idx]['weights'];w=dict(np.load(wpath));h=w['head.0.weight'].shape[0]
        w['head_extra.weight']=rng.normal(0,.03,(h,68)).astype(np.float32);np.savez(wpath,**w)
        m[idx]['feature_config']=f'feature_{idx}.json'
        (destination/'fixed_feature'/m[idx]['feature_config']).write_text(json.dumps({'schema':'transhead01-68-v1','arm':'T','nuisance':nuisance}))
    (destination/'fixed_feature/members.json').write_text(json.dumps(m))
    pf=pq.ParquetFile(ROOT/'datasets/valid.parquet')
    cols=[n for n in pf.schema_arrow.names if n.startswith(('i0_','i1_')) or (n.startswith('a') and n[1:].isdigit())]
    assert len(cols)==112
    groups=[0,468,936,1404,1872];x=[];seq=[]
    for g in groups:
        t=pf.read_row_group(g,columns=['seq_ix']+cols,use_threads=False)
        x.append(np.column_stack([t[c].to_numpy() for c in cols]).astype(np.float32));seq.append(t['seq_ix'][0].as_py())
    np.save(destination/'raw.npy',np.stack(x))
    receipt={'groups':groups,'sequence_ids':seq,'rows':len(groups)*20000,'columns':cols,'raw_sha256':sha(destination/'raw.npy'),'source_v1_sha256':sha(source),'feature_weights':'syntheticnonzeroblock; latencyproxyonly','targets_read':False,'masks_read':False,'scored_metrics':0}
    (destination/'inputs.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

def run(folder,condition,repeat):
    package=folder/condition
    assert (package/'solution.py').exists()
    # Each run in fresh process imports only the extracted package plus scorer image deps.
    os.chdir(tempfile.mkdtemp(prefix='wunder-fixed-cwd-'))
    sys.path.insert(0,str(package.resolve()))
    t=time.perf_counter();module=importlib.import_module('solution');model=module.PredictionModel();load=time.perf_counter()-t
    x=np.load(folder/'raw.npy',mmap_mode='r');output=[];times=[]
    for k,seq in enumerate(x):
        y=np.zeros((20000,2),np.float32);point=SimpleNamespace(seq_ix=k,step_in_seq=0,need_prediction=False,state=None)
        t=time.perf_counter()
        for j,row in enumerate(seq):
            point.step_in_seq=j;point.need_prediction=j>=99;point.state=row
            value=model.predict(point)
            if j<99:assert value is None
            else:y[j]=value
        times.append(time.perf_counter()-t);assert np.isfinite(y).all();output.append(y)
    out=np.stack(output);elapsed=sum(times);digest=hashlib.sha256(out.tobytes()).hexdigest()
    np.save(folder/f'{condition}-r{repeat}-pred.npy',out)
    import resource
    result={'condition':condition,'repeat':repeat,'rows':int(np.prod(x.shape[:2])),'member_count':model.member_count,'cpu_level':module.LEVEL,'load_seconds':load,'callback_seconds':elapsed,'us_row':elapsed/np.prod(x.shape[:2])*1e6,'projected_minutes_39_4M':elapsed/np.prod(x.shape[:2])*39.4e6/60,'sequence_seconds':times,'prediction_sha256':digest,'max_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'solution_sha256':sha(package/'solution.py'),'native_sha256':sha(Path(module.wnn_cpp.__file__)),'policy':'fixed4; no wall-clock pruning','metrics_computed':0,'python':sys.version}
    (folder/f'{condition}-r{repeat}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','run']);a.add_argument('--folder',type=Path,required=True);a.add_argument('--condition',choices=['fixed_v1','fixed_recovered','fixed_feature']);a.add_argument('--repeat',type=int,default=0);args=a.parse_args()
    folder=args.folder.resolve()
    prepare(folder) if args.mode=='prepare' else run(folder,args.condition,args.repeat)
if __name__=='__main__':main()
