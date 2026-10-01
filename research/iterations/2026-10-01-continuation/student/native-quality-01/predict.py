"""Exactly one native INT8 heldfold pass, with no target/mask decode or metric."""
import argparse,hashlib,json,os,resource,signal,sys,time
from pathlib import Path
from common import P,ROOT,NAMES,NATIVE_SHA,sha,write,identity,guards,rowmaps
import numpy as np
import pyarrow as pa

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[0,1],required=True);a=ap.parse_args();start=time.monotonic();cpu=(30,35)[a.fold]
    allocation,limits=guards(cpu);assert sys.version_info[:2]==(3,11)
    write(P/f'fold{a.fold}/worker.json',{'PID':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),'cpu':cpu,'cgroup':Path('/proc/self/cgroup').read_text().strip(),'limits':limits})
    pa.set_cpu_count(1);pa.set_io_thread_count(1);resource.setrlimit(resource.RLIMIT_AS,(4*(1<<30),4*(1<<30)))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('1800-second lane cutoff')))
    prepared=json.loads((P/'prepared.json').read_text());manifest=P/f'fold{a.fold}/checkpoint.json';manifest_sha=sha(manifest);m=json.loads(manifest.read_text());assert prepared['fold_manifests'][str(a.fold)]==manifest_sha
    assert m['allocation_sha256']==sha(P/'allocation.json') and m['predictor_sha256']==sha(__file__) and m['fold']==a.fold
    for rel,record in m['inputs'].items():assert identity(ROOT/rel)=={k:record[k] for k in ['resolved','bytes','mtime_ns']}
    groups,info,orderhash=rowmaps();assert orderhash==m['source_row_group_order_sha256'] and sha(ROOT/'cache/chain_folds.npz')==m['chain_folds_sha256']
    selected=[i for i,g in enumerate(groups) if info[int(g)][0]==a.fold];assert selected==m['rows']
    weights=[]
    def verify():
        assert sha(manifest)==manifest_sha and sha(m['native'])==NATIVE_SHA
        for i,c in enumerate(m['children']):
            assert c['run']==f'{NAMES[i]}_f{a.fold}' and c['coef']==.25
            assert sha(c['checkpoint'])==c['checkpoint_sha256'] and sha(c['export'])==c['export_sha256']
    verify()
    for c in m['children']:
        with np.load(c['export'],allow_pickle=False) as z:arr={k:np.ascontiguousarray(z[k]) for k in z.files}
        for k,v in arr.items():assert str(v.dtype)==c['arrays'][k]['dtype'] and list(v.shape)==c['arrays'][k]['shape'] and hashlib.sha256(v.tobytes()).hexdigest()==c['arrays'][k]['sha256']
        weights.append(arr)
    sys.path.insert(0,str(Path(m['native']).parent));import wnn_cpp
    assert sha(wnn_cpp.__file__)==NATIVE_SHA and wnn_cpp.VNNI_SUPPORTED
    engine=wnn_cpp.Ensemble(weights,[.25]*4,fast_activations=True,half_weights=False,dynamic_int8=True,quant_backend='auto');assert wnn_cpp.quant_backend()==2 and len(engine)==4
    sys.path.insert(0,str(ROOT/'src'));from wnn.data import SequenceFile
    data=SequenceFile(ROOT/'datasets/valid.parquet');directory=P/'cache';directory.mkdir(exist_ok=True);path=directory/f'pred_student_quant_f{a.fold}_valid.npy';partial=path.with_suffix('.npy.partial')
    assert not path.exists() and not partial.exists() and not path.with_suffix('.json').exists()
    out=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(1873,20000,2));out[:]=np.nan
    raw_hash=hashlib.sha256()
    for count,row in enumerate(selected):
        sequence=data.read(int(groups[row]),targets=False,mask=False);assert sequence.targets is None and sequence.scored is None
        raw_hash.update(sequence.features.tobytes());pred=engine.run(sequence.features)
        assert pred.shape==(20000,2) and pred.dtype==np.float32 and np.isfinite(pred).all();out[row]=pred
        if count%50==0:print(json.dumps({'fold':a.fold,'completed':count+1,'total':len(selected),'seconds':time.monotonic()-start}),flush=True)
    out.flush();del out;verify();assert sha(P/'allocation.json')==m['allocation_sha256']
    for rel,record in m['inputs'].items():assert identity(ROOT/rel)=={k:record[k] for k in ['resolved','bytes','mtime_ns']}
    partial.replace(path)
    receipt={'status':'COMPLETE_ONE_NATIVE_INT8_OOF_PASS','run':f'student_quant_f{a.fold}','fold':a.fold,'epoch':1,'rows':selected,'row_groups':groups[selected].tolist(),'source_row_group_order_sha256':orderhash,'chain_folds_sha256':m['chain_folds_sha256'],'checkpoint':str(manifest),'checkpoint_sha256':manifest_sha,'prediction_path':str(path),'prediction_sha256':sha(path),'child_checkpoints':m['children'],'coefficients':[.25]*4,'native_sha256':NATIVE_SHA,'native_backend':'natural auto VNNI (code2)','allocation_sha256':m['allocation_sha256'],'source_sha256':sha(__file__),'raw_selected_features_sha256':raw_hash.hexdigest(),'seconds':time.monotonic()-start,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':limits,'cpu':cpu,'targets_decoded':False,'masks_decoded':False,'WP_queries':0,'nonselected_rows':'NaN'}
    write(path.with_suffix('.json'),receipt);print(json.dumps({k:receipt[k] for k in ['status','fold','seconds','peak_RSS_KiB','prediction_sha256']}),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        print(json.dumps({'status':'FAIL_NO_RETRY','error':repr(e)}),flush=True);raise
