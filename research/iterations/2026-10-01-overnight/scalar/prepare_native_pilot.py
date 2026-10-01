"""Allocation-gated fixed nonzero cost prototype; never a trained candidate."""
from pathlib import Path
import argparse,hashlib,io,json,shutil,tempfile,zipfile
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--allocation',type=Path,required=True);a=ap.parse_args();alloc=json.loads(a.allocation.read_text());assert alloc.get('SCALARPLE_native_pilot_authorized') is True
for rel,h in alloc['sources'].items():assert sha(ROOT/rel)==h,rel
baseline=ROOT/'submissions/overnight_baseline3/baseline3_fixed.zip';assert sha(baseline)=='9471821e81e7c3f80a088147512183b72e0ab52bc593a1136d2becc229d0d098'
fit=json.loads((P/'knots/fit.json').read_text());assert fit['status']=='PASS' and not fit['targets_read'] and not fit['validation_chains_read'];assert fit['allocation_sha256']==sha(a.allocation)
kp=ROOT/fit['knots_path'];assert sha(kp)==fit['knots_file_sha256']
with np.load(kp,allow_pickle=False) as z:knots=z['knots'].copy();counts=z['counts'].copy()
assert hashlib.sha256(knots.tobytes()).hexdigest()==fit['knots_array_sha256'] and hashlib.sha256(counts.tobytes()).hexdigest()==fit['counts_array_sha256']
rng=np.random.default_rng(2026100159);table=rng.normal(0,1,(44,16,16)).astype(np.float32);projection=rng.normal(0,.01,(128,16)).astype(np.float32)
assert np.isfinite(table).all() and np.isfinite(projection).all() and np.count_nonzero(projection)==2048
out=P/'native-pilot';out.mkdir(exist_ok=False)
with zipfile.ZipFile(baseline) as z:
 contents={name:z.read(name) for name in z.namelist()};assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in contents)
 members=json.loads(contents['members.json']);assert len(members)==3 and [m['coef'] for m in members]==[1/3]*3
 with np.load(io.BytesIO(contents[members[0]['weights']]),allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
 weights=dict(original);weights.update({'scalar.knots':knots,'scalar.counts':counts,'scalar.raw_indices':np.array(list(range(22))+list(range(52,74)),np.int64),'scalar.table':table,'scalar.projection':projection})
 for key,value in original.items():assert np.array_equal(weights[key],value)
 b=io.BytesIO();np.savez(b,**weights);contents[members[0]['weights']]=b.getvalue()
 for level in ['v2','avx2','v3']:
  binary=list((P/'native_v2/build'/level).glob('wnn_cpp.cpython-311-*.so'));assert len(binary)==1
  paths=[n for n in contents if n.startswith(level+'/') and n.endswith('.so')];assert len(paths)==1;contents[paths[0]]=binary[0].read_bytes()
 dest=out/'scalar_nonzero_prototype.zip'
 with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for name,blob in sorted(contents.items()):
   info=zipfile.ZipInfo(name,date_time=(2026,10,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,blob)
 assert dest.stat().st_size<20_000_000
 receipt={'status':'PREPARED_NONTRAINED_COST_PROTOTYPE','allocation_sha256':sha(a.allocation),'source_sha256':sha(__file__),'baseline_zip_sha256':sha(baseline),'fit_receipt_sha256':sha(P/'knots/fit.json'),'fit_knots_sha256':sha(kp),'zip_sha256':sha(dest),'zip_bytes':dest.stat().st_size,'files':{n:hashlib.sha256(b).hexdigest() for n,b in contents.items()},'member_order':'exactbaseline3 firstplainseed3 receivesoneadapter; plainseed4+c500 unchanged','coefficients':[1/3]*3,'original_weight_arrays_exact':True,'prototype_seed':2026100159,'prototype_table':'NumPydefault_rng N(0,1) 44x16x16','prototype_projection':'NextdrawsN(0,.01)128x16 allnonzero','prototype_not_actual_training_initialization':True,'trainable_model_fits':0,'WP_queries':0,'targets_read':False}
 (out/'package.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
