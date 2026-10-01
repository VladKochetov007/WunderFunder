"""Allocated raw-only prototype/reference integrity; no targets, fitting or scoring."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact(actual,expected):
 assert actual.shape==expected.shape and actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes(),'byte parity'
ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);ap.add_argument('--level',choices=['v2','avx2','v3'],required=True);a=ap.parse_args();alloc=json.loads(a.allocation.read_text());assert alloc.get('SCALARPLE_native_pilot_authorized') is True
for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
rawpath=P.parent/'critic/runtime-probe/raw.npy';assert sha(rawpath)=='1e90c52f1332950b70d573618adad29324fb6705675e74cb75c7ca6efda993b4'
prototype=P/'native-pilot/scalar_nonzero_prototype.zip';package=json.loads((P/'native-pilot/package.json').read_text());assert sha(prototype)==package['zip_sha256']
basepath=ROOT/'submissions/overnight_baseline3/baseline3_fixed.zip';assert sha(basepath)==package['baseline_zip_sha256']
def weights(path):
 with zipfile.ZipFile(path) as z:
  members=json.loads(z.read('members.json'));return [{k:v.copy() for k,v in np.load(io.BytesIO(z.read(m['weights'])),allow_pickle=False).items()} for m in members]
base=weights(basepath);candidate=weights(prototype)
for i in range(3):
 for key,value in base[i].items():assert np.array_equal(value,candidate[i][key]),(i,key)
sys.path.insert(0,str(P/'native_v2/build'/a.level));import wnn_cpp
x=np.load(rawpath,mmap_mode='r');oldref=np.load(P.parent/f'delivery-baseline3/reference-{a.level}-real.npy',mmap_mode='r');receipt=json.loads((P.parent/f'delivery-baseline3/reference-{a.level}.json').read_text());assert hashlib.sha256(np.asarray(oldref).tobytes()).hexdigest()==receipt['real_prediction_sha256']
kwargs=dict(fast_activations=True,half_weights=True);ordinary=wnn_cpp.Ensemble(base,[1/3]*3,**kwargs);engine=wnn_cpp.Ensemble(candidate,[1/3]*3,**kwargs)
zero=[dict(w) for w in candidate];zero[0]=dict(zero[0]);zero[0]['scalar.projection']=np.zeros_like(zero[0]['scalar.projection']);zeroengine=wnn_cpp.Ensemble(zero,[1/3]*3,**kwargs)
# All3 ISAs on real512-row fixture prefix; full100k only on v3 for timing reference.
test=np.asarray(x[0,:512]);expected=np.asarray(oldref[0,:512]);exact(ordinary.run(test),expected);exact(zeroengine.run(test),expected)
y=engine.run(test);assert np.isfinite(y).all() and not np.array_equal(y,expected)
engine.reset();exact(np.stack([engine.step(row) for row in test]),y)
for stop in [1,2,99,100,319,511]:exact(engine.run(test[:stop]),y[:stop])
future=test.copy();future[100:]*=-7;exact(engine.run(future)[:100],y[:100]);exact(engine.run(test),y)
reference_file=None
if a.level=='v3':
 refs=[]
 for i,seq in enumerate(x):
  exact(ordinary.run(seq),oldref[i]);exact(zeroengine.run(seq),oldref[i]);pred=engine.run(seq);assert np.isfinite(pred).all();refs.append(pred)
 reference_file=P/'native-pilot/prototype-reference-v3.npy';assert not reference_file.exists();np.save(reference_file,np.stack(refs))
r={'status':'PASS','level':a.level,'native_sha256':sha(wnn_cpp.__file__),'raw_sha256':sha(rawpath),'baseline_reference_sha256':sha(P.parent/f'delivery-baseline3/reference-{a.level}-real.npy'),'baseline_reference_array_sha256':receipt['real_prediction_sha256'],'baseline_zip_sha256':sha(basepath),'prototype_zip_sha256':sha(prototype),'original_arrays_exact':True,'plain_and_zero_extra_vs_archived_exact':True,'nonzero_changes_predictions':True,'reset_step_prefix_future_exact':True,'rows':100000 if a.level=='v3' else 512,'prototype_reference_sha256':sha(reference_file) if reference_file else None,'targets_read':False,'WP_queries':0,'source_sha256':sha(__file__)}
(P/f'native-pilot/reference-{a.level}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
