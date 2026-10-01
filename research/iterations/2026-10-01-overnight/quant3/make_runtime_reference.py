"""After root freeze only: raw-fixture reference from frozen old member predictors."""
from pathlib import Path
import argparse,ctypes,hashlib,json,sys,zipfile,io
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--original-zip',type=Path,required=True);ap.add_argument('--old-native',type=Path,required=True);ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(a.raw)=='1e90c52f1332950b70d573618adad29324fb6705675e74cb75c7ca6efda993b4'
assert sha(a.original_zip)=='ea494921497d9f3cf53b19b1b68285133983560a249b38ef40738b3d846d319d'
assert sha(a.old_native)=='4cb8a76f9b3010b506515ee2012e3088ab2f066edbddf0b6f63a7101bba4c84f'
sys.path.insert(0,str(a.old_native.resolve().parent));import wnn_cpp
assert sha(wnn_cpp.__file__)==sha(a.old_native)
lib=ctypes.CDLL('libm.so.6');fma=lib.fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float;coef=np.float32(1/3)
with zipfile.ZipFile(a.original_zip) as z:
 members=json.loads(z.read('members.json'));assert [members[i]['source'] for i in [0,1,3]]==['find256_s3h2','fin128_s3h2','finc500b_s3h2']
 weights=[]
 for i in [0,1,3]:
  with np.load(io.BytesIO(z.read(members[i]['weights'])),allow_pickle=False) as w:weights.append({k:np.ascontiguousarray(w[k]) for k in w.files})
x=np.load(a.raw,mmap_mode='r');assert x.shape==(5,20000,112) and x.dtype==np.float32;receipt={}
for mode in ['quant','fp16']:
 predictors=[wnn_cpp.Predictor(w,fast_activations=True,**({'dynamic_int8':True} if mode=='quant' else {'half_weights':True})) for w in weights]
 result=[]
 for sequence in x:
  child=[p.run(np.ascontiguousarray(sequence)) for p in predictors];blend=np.zeros((20000,2),np.float32)
  for j in range(20000):
   for k in range(2):
    value=0.
    for pred in child:value=fma(coef,pred[j,k],value)
    blend[j,k]=value
  assert np.isfinite(blend).all();result.append(blend)
 result=np.stack(result);path=a.output_dir/f'{mode}.npy';np.save(path,result);receipt[mode]={'sha256':sha(path),'prediction_sha256':hashlib.sha256(result.tobytes()).hexdigest(),'rows':100000}
receipt.update({'status':'PASS','original_zip_sha256':sha(a.original_zip),'native_sha256':sha(a.old_native),'raw_sha256':sha(a.raw),'reference_arithmetic':'independent per-member frozen old native + libc fmaf original order coefficient float32(1/3)','targets_read':False,'WP_queries':0,'script_sha256':sha(__file__)})
(a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
