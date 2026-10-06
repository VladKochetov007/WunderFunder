"""Bounded correctness check of the exact final ZIP, without labels or scoring."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,ctypes,hashlib,json,resource,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from protocol import extract_exact

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    start=time.perf_counter()
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);a=ap.parse_args()
    assert sys.version_info[:2]==(3,11) and sorted(os.sched_getaffinity(0))==[18]
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[2].lstrip('/')
    limits={k:(cg/k).read_text().strip() for k in ['cpu.max','memory.max','memory.swap.max']}
    assert limits=={'cpu.max':'100000 100000','memory.max':str(4*(1<<30)),'memory.swap.max':'0'}
    resource.setrlimit(resource.RLIMIT_AS,(4*(1<<30),4*(1<<30)))
    class NoTorch:
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Torch inference forbidden')
    sys.meta_path.insert(0,NoTorch())
    import numpy as np
    import pyarrow as pa
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    manifest=json.loads(a.package.with_suffix('.manifest.json').read_text())
    assert sha(a.package)==manifest['zip_sha256'] and a.package.stat().st_size==manifest['zip_bytes']
    for c in manifest['checkpoints']:assert sha(c['checkpoint'])==c['sha256']
    with tempfile.TemporaryDirectory(prefix='student-final-check-') as tmp:
        dest=Path(tmp)
        extract_exact(a.package,dest,{k:{**v,'is_dir':False}for k,v in manifest['files'].items()})
        sys.path.insert(0,str(dest));import solution
        model=solution.PredictionModel()
        assert solution.LEVEL=='v3' and model.quant_backend==2 and model.member_count==4
        weights=[]
        for i in range(4):
            with np.load(dest/f'weights_{i}.npz',allow_pickle=False) as z:
                weights.append({k:np.ascontiguousarray(z[k],dtype=np.float32)for k in z.files})
        assert 'gru.weight_ih_l1' not in weights[0]
        assert all('gru.weight_ih_l1' in w for w in weights[1:])
        sys.path.insert(0,str(ROOT/'src'));from wnn.data import SequenceFile
        seq=SequenceFile(ROOT/'datasets/valid.parquet').read(0,targets=False,mask=False)
        assert seq.targets is None and seq.scored is None
        prefix=seq.features[:256]
        quant=solution.wnn_cpp.Ensemble(weights,[.25]*4,fast_activations=True,dynamic_int8=True,quant_backend='auto')
        expected=quant.run(prefix);assert expected.shape==(256,2) and np.isfinite(expected).all()
        assert quant.run(prefix).tobytes()==expected.tobytes()
        quant.reset();assert np.stack([quant.step(x)for x in prefix]).tobytes()==expected.tobytes()
        assert quant.run(prefix[:137]).tobytes()==expected[:137].tobytes()
        fma=ctypes.CDLL('libm.so.6').fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
        individual=np.zeros_like(expected)
        for w in weights:
            part=solution.wnn_cpp.Predictor(w,fast_activations=True,dynamic_int8=True,quant_backend='auto').run(prefix)
            for i in range(256):
                for k in range(2):individual[i,k]=fma(.25,float(part[i,k]),float(individual[i,k]))
        assert individual.tobytes()==expected.tobytes()
        for seq_id,alternate in [(10000,False),(10001,True),(10000,False)]:
            for j,row in enumerate(prefix):
                need=j>=99 and (not alternate or j%3!=0)
                y=model.predict(SimpleNamespace(seq_ix=seq_id,step_in_seq=j,need_prediction=need,state=row))
                if need:assert y.tobytes()==expected[j].tobytes()
                else:assert y is None
        native_sha=sha(solution.wnn_cpp.__file__)
        assert native_sha==manifest['files']['v3/wnn_cpp.cpython-311-x86_64-linux-gnu.so']['sha256']
    assert sha(a.package)==manifest['zip_sha256']
    result={'status':'PASS','zip_sha256':manifest['zip_sha256'],'native_sha256':native_sha,
        'rows':256,'repeat_reset_stream_prefix_wrapper_mask_exact':True,'ordered_member_fmaf_exact':True,
        'targets_read':False,'masks_read':False,'WP_queries':0,'runtime_qualification':False,
        'elapsed_seconds':time.perf_counter()-start,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'raw_prefix_sha256':hashlib.sha256(prefix.tobytes()).hexdigest(),
        'prediction_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),'checker_sha256':sha(__file__),'cgroup_limits':limits}
    with (HERE/'package-check.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
