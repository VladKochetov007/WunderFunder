"""One fixed five-sequence student runtime prefix, without labels or score queries."""
import time
START=time.perf_counter()
import ctypes, hashlib, importlib.util, json, os, resource, signal, sys, zipfile
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def counters():
    return {x[0]:list(map(int,x[1:9])) for x in (s.split() for s in Path('/proc/stat').read_text().splitlines()) if x and x[0] in ('cpu18','cpu58')}
def activity(a,b):
    out={}
    for k in a:
        d=[y-x for x,y in zip(a[k],b[k])]; total=sum(d);busy=total-d[3]-d[4]
        out[k]={'total_ticks':total,'busy_ticks':busy,'busy_fraction':busy/total if total else 0.}
    return out
def identity(p):
    s=p.stat();return {'resolved':str(p.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}

def main():
    assert sys.version_info[:2]==(3,11)
    assert sorted(os.sched_getaffinity(0))==[18]
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS']:assert os.environ.get(k)=='1'
    assert all(k not in os.environ for k in ['WNN_NATIVE_LEVEL','WNN_EXPECTED_ROWS','WNN_BUDGET_S'])
    cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[2].lstrip('/')
    limits={k:(cg/k).read_text().strip() for k in ['cpu.max','memory.max','memory.swap.max']}
    assert limits=={'cpu.max':'100000 100000','memory.max':str(4*(1<<30)),'memory.swap.max':'0'}
    resource.setrlimit(resource.RLIMIT_AS,(4*(1<<30),4*(1<<30)))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('180 second probe deadline')))
    plan=json.loads((HERE/'plan.json').read_text())
    assert plan['attempts']==1 and plan['callbacks']==100000 and plan['authorised_by_root'] is True
    for path,digest in plan['source_sha256'].items():assert sha(ROOT/path)==digest,path
    class NoTorch:
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Torch inference forbidden')
    sys.meta_path.insert(0,NoTorch())
    import numpy as np
    import pyarrow as pa
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    package=HERE/'student_fold0.zip';manifest=json.loads(package.with_suffix('.manifest.json').read_text())
    assert sha(package)==manifest['zip_sha256'] and package.stat().st_size==manifest['zip_bytes']
    for checkpoint in manifest['checkpoints']:assert sha(checkpoint['checkpoint'])==checkpoint['sha256']
    extraction=HERE/'extracted';extraction.mkdir(exist_ok=False)
    phase=time.perf_counter()
    with zipfile.ZipFile(package) as z:
        entries=z.infolist();assert {e.filename for e in entries}==set(manifest['files']) and len(entries)==len(manifest['files'])
        for e in entries:
            p=Path(e.filename);assert not p.is_absolute() and '..' not in p.parts
            b=z.read(e);m=manifest['files'][e.filename];assert hashlib.sha256(b).hexdigest()==m['sha256'] and len(b)==m['bytes']
            dest=extraction/p;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(b)
    extraction_s=time.perf_counter()-phase
    phase=time.perf_counter();sys.path.insert(0,str(extraction));import solution
    model=solution.PredictionModel();assert solution.LEVEL=='v3' and model.quant_backend==2 and model.member_count==4
    assert sha(solution.wnn_cpp.__file__)==plan['expected_native_sha256']
    startup_s=time.perf_counter()-phase
    weights=[]
    for i in range(4):
        with np.load(extraction/f'weights_{i}.npz',allow_pickle=False) as z:weights.append({k:np.ascontiguousarray(z[k],dtype=np.float32) for k in z.files})
    assert 'gru.weight_ih_l0' in weights[0] and 'gru.weight_ih_l1' not in weights[0]
    assert all('gru.weight_ih_l1' in w for w in weights[1:])
    sys.path.insert(0,str(ROOT/'src'));from wnn.data import SequenceFile
    raw=ROOT/'datasets/valid.parquet';raw_before=identity(raw);data=SequenceFile(raw)
    sequence=data.read(0,targets=False,mask=False);assert sequence.targets is None and sequence.scored is None
    prefix=sequence.features[:256];quant=solution.wnn_cpp.Ensemble(weights,[.25]*4,fast_activations=True,dynamic_int8=True,quant_backend='auto')
    expected=quant.run(prefix);assert expected.shape==(256,2) and np.isfinite(expected).all()
    assert quant.run(prefix).tobytes()==expected.tobytes()
    quant.reset();assert np.stack([quant.step(x) for x in prefix]).tobytes()==expected.tobytes()
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
    half=solution.wnn_cpp.Ensemble(weights,[.25]*4,fast_activations=True,half_weights=True)
    reference=half.run(prefix);assert np.isfinite(reference).all();error=expected[99:].astype(np.float64)-reference[99:].astype(np.float64)
    errors={'prefix_rows':256,'emitted_prefix_rows':157,'rmse':np.sqrt(np.mean(error**2,axis=0)).tolist(),'max_abs':np.max(np.abs(error),axis=0).tolist(),'p99_abs':np.quantile(np.abs(error),.99,axis=0).tolist(),'policy':'Descriptive 256-row FP16 comparison only; no metric or full quality claim.'}
    proof={'status':'PASS','one_layer_student':True,'repeat_reset_stream_prefix_wrapper_mask_exact':True,'ordered_individual_member_fmaf_exact':True,'int8_vs_fp16':errors,'raw_prefix_sha256':hashlib.sha256(prefix.tobytes()).hexdigest(),'quant_prefix_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),'seconds_before_trace':time.perf_counter()-START}
    write(HERE/'prefix-proof.json',proof)
    before=counters();time.sleep(2);pre=activity(before,counters())
    rows=[];digest=hashlib.sha256();raw_digest=hashlib.sha256();whole_before=counters();trace_start=time.perf_counter();trace_cpu=time.process_time()
    for group in range(5):
        total_start=time.perf_counter();io_start=time.perf_counter()
        seq=data.read(group,targets=False,mask=False);io_s=time.perf_counter()-io_start
        assert seq.targets is None and seq.scored is None and seq.features.shape==(20000,112) and seq.features.dtype==np.float32
        raw_digest.update(seq.features.tobytes());out=np.zeros((20000,2),np.float32)
        point=SimpleNamespace(seq_ix=20000+group,step_in_seq=0,need_prediction=False,state=None)
        b=counters();cpu0=time.process_time();t=time.perf_counter()
        for j,row in enumerate(seq.features):
            point.step_in_seq=j;point.need_prediction=j>=99;point.state=row;value=model.predict(point)
            if j<99:assert value is None
            else:out[j]=value
        wall=time.perf_counter()-t;cpu=time.process_time()-cpu0;e=counters()
        assert np.isfinite(out).all() and model.member_count==4
        digest.update(out.tobytes())
        item={'row_group':group,'raw_seq_ix':seq.seq_ix,'callback_seq_ix':point.seq_ix,'callbacks':20000,'emitted':19901,'callback_wall_seconds':wall,'callback_cpu_seconds':cpu,'io_seconds':io_s,'sequence_seconds':time.perf_counter()-total_start,'cpu_activity':activity(b,e)}
        rows.append(item);print(json.dumps(item),flush=True)
    trace_s=time.perf_counter()-trace_start;trace_cpu_s=time.process_time()-trace_cpu;whole=activity(whole_before,counters())
    callback_s=sum(r['callback_wall_seconds'] for r in rows);aggregate={}
    for k in ('cpu18','cpu58'):
        total=sum(r['cpu_activity'][k]['total_ticks'] for r in rows);busy=sum(r['cpu_activity'][k]['busy_ticks'] for r in rows);aggregate[k]={'total_ticks':total,'busy_ticks':busy,'busy_fraction':busy/total if total else 0.}
    assert raw_before==identity(raw)
    for path,digest_expected in plan['source_sha256'].items():assert sha(ROOT/path)==digest_expected,path
    for checkpoint in manifest['checkpoints']:assert sha(checkpoint['checkpoint'])==checkpoint['sha256']
    assert sha(package)==manifest['zip_sha256']
    assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
    task_affinities={p.name:sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()};assert all(v==[18] for v in task_affinities.values())
    disk_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());assert disk_bytes<150*(1<<20)
    result={'status':'COMPLETE_PREFIX_PROBE','attempts':1,'package_kind':'fold0 student exploratory only; not final weights','callbacks':100000,'emitted':99505,'trace_wall_seconds':trace_s,'trace_cpu_seconds':trace_cpu_s,'callback_wall_seconds':callback_s,'callback_cpu_seconds':sum(r['callback_cpu_seconds'] for r in rows),'callback_us':callback_s/100000*1e6,'callback_projection_minutes_39_4M':callback_s/100000*39400000/60,'trace_projection_minutes_39_4M':trace_s/100000*39400000/60,'whole_worker_before_result_seconds':time.perf_counter()-START,'extraction_seconds':extraction_s,'model_import_startup_seconds':startup_s,'prefix_proof':proof,'precheck_cpu_activity':pre,'trace_cpu_activity':whole,'callback_cpu_activity':aggregate,'quiet_sibling_callback_below5pct':aggregate['cpu58']['busy_fraction']<.05,'cpu':18,'sibling':58,'cgroup_limits':limits,'task_affinities':task_affinities,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'disk_bytes_before_result':disk_bytes,'per_sequence':rows,'native_sha256':sha(solution.wnn_cpp.__file__),'solution_sha256':sha(extraction/'solution.py'),'package_sha256':sha(package),'manifest_sha256':sha(package.with_suffix('.manifest.json')),'plan_sha256':sha(HERE/'plan.json'),'harness_sha256':sha(__file__),'raw_file_stat':raw_before,'raw_trace_sha256':raw_digest.hexdigest(),'prediction_sha256':digest.hexdigest(),'targets_decoded':False,'mask_decoded':False,'WP_queries':0,'full_runtime_qualification':False,'limitation':'100k callbacks on first five exposed validation sequences; projections are extrapolations, not a 39.4M/full-process runtime or predictive-quality qualification.'}
    write(HERE/'result.json',result);print(json.dumps({k:result[k] for k in ['status','callback_us','callback_projection_minutes_39_4M','trace_projection_minutes_39_4M','whole_worker_before_result_seconds','quiet_sibling_callback_below5pct']}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        path=HERE/'failure.json'
        if not path.exists():write(path,{'status':'FAILED','error_type':type(exc).__name__,'error':str(exc),'elapsed_seconds':time.perf_counter()-START,'retry_allowed':False})
        raise
